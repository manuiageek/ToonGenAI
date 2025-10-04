import json
import os
import time
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
import csv
from typing import List, Optional, Tuple
import sqlite3
import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

# ==================== CONFIG ====================
# Chemins modèle / tags
MODEL_ONNX_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.onnx"
TAGS_CSV_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.csv"

# Inférence
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.55
INCLUDE_RATING_TAGS = True
REMOVE_UNDERSCORES = True
TOPK_OUTPUT = 20
BATCH_SIZE = 48

# Pipeline asynchrone (30GB RAM dispo) - MODE BEAST 🔥
PREFETCH_BATCHES = 104  # Précharge 104 batches en avance (4992 images * ~2MB = ~10GB RAM)
PREPROC_WORKERS = 96  # Threads pour préprocessing (I/O bound = 3x cores physiques)

# Prétraitement
TARGET_SIZE = (448, 448)
CHANNEL_ORDER = "BGR"
NORMALIZATION = "none"

# Providers
CUDA_PROVIDER_OPTIONS = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",  # Meilleure gestion mémoire
    "gpu_mem_limit": 6 * 1024 * 1024 * 1024,  # 6GB (laisse marge pour Windows)
    "cudnn_conv_algo_search": "HEURISTIC",
    "do_copy_in_default_stream": 1,
    "cudnn_conv_use_max_workspace": 0,  # Réduit l'utilisation mémoire
}
DEFAULT_PROVIDERS = [("CUDAExecutionProvider", CUDA_PROVIDER_OPTIONS), "CPUExecutionProvider"]
FORCE_CPU = False

# SQLite
DB_FILENAME = "SQLLITE.db"
SQLITE_QUERY = "select image_path from images"
SQLITE_TABLE = "images"
SQLITE_TAGS_COLUMN = "detect_wdtags"
BATCH_COMMIT_SIZE = 1000  # 0 = commit after each batch

# Characters dir + extensions images
CHARACTERS_DIR_NAME = "_characters"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ==================== LOGGING ====================
# Mettre DEBUG pour voir le pipeline en action, INFO pour production
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wd14")

# ==================== CUDA / cuDNN ====================
CUDNN_DIR = r"C:\Program Files\NVIDIA\CUDNN\v9.9\bin\12.9"
CUDA_DIR = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

def _setup_cuda_paths() -> None:
    # Ajout des dossiers DLL
    for d in [CUDNN_DIR, CUDA_DIR]:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                logger.info("DLL directory ajouté: %s", d)
            except Exception as e:
                logger.warning("Echec add_dll_directory pour %s: %s", d, e)
        else:
            logger.warning("Répertoire DLL introuvable: %s", d)
    # Préchargement opportuniste
    try:
        ort.preload_dlls(cudnn=True, directory=CUDNN_DIR)
    except Exception as e:
        logger.debug("preload_dlls(cudnn) indisponible/échec: %s", e)
    try:
        ort.preload_dlls(cuda=True, directory=CUDA_DIR)
    except Exception as e:
        logger.debug("preload_dlls(cuda) indisponible/échec: %s", e)
    # Debug ORT si dispo
    try:
        ort.print_debug_info()
    except Exception:
        pass

_setup_cuda_paths()

# ==================== TAGGER ====================
class WD14Tagger:
    MODEL_ONNX_PATH = MODEL_ONNX_PATH
    TAGS_CSV_PATH = TAGS_CSV_PATH

    def __init__(self):
        self.model_path = self.MODEL_ONNX_PATH
        self.tags_csv_path = self.TAGS_CSV_PATH

        # Session ORT optimisée pour 5950X (16 cores)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0  # 0 = auto (utilise tous les cores)
        so.inter_op_num_threads = 0  # 0 = auto

        req_providers = ["CPUExecutionProvider"] if FORCE_CPU else list(DEFAULT_PROVIDERS)
        logger.info("Modèle ONNX: %s", os.path.abspath(self.model_path))
        logger.info("CSV Tags: %s", os.path.abspath(self.tags_csv_path))
        self.tags = self._load_tags(self.tags_csv_path)
        self.session = self._create_session_with_fallback(req_providers, so)
        logger.info("Session ORT prête avec providers: %s", self.session.get_providers())
        self.input_layout = self._infer_input_layout()
        logger.info("Layout entrée modèle: %s", self.input_layout)

    def _load_tags(self, csv_path: str) -> List[str]:
        # Chargement des tags
        tags: List[str] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header is None:
                raise ValueError("CSV des tags vide")
            if "name" in header:
                name_idx = header.index("name")
            elif "tag_name" in header:
                name_idx = header.index("tag_name")
            else:
                name_idx = 1 if len(header) > 1 else 0
            for row in reader:
                if len(row) > name_idx and row[name_idx]:
                    tags.append(row[name_idx])
        if not tags:
            raise ValueError("Aucun tag chargé depuis le CSV")
        return tags

    def _create_session_with_fallback(self, providers: List, so: ort.SessionOptions):
        # Création session avec fallback GPU->CPU
        available = set(ort.get_available_providers())

        def normalize(prov_list: List, with_options: bool) -> List:
            out = []
            for p in prov_list:
                if isinstance(p, tuple):
                    name, opts = p
                    if name in available:
                        out.append((name, opts) if with_options else name)
                else:
                    if p in available:
                        out.append(p)
            return out or ["CPUExecutionProvider"]

        prov_with_opts = normalize(providers, True)
        prov_plain = normalize(providers, False)

        logger.info("Providers demandés: %s", providers)
        logger.info("Providers dispo: %s", list(available))
        try:
            return ort.InferenceSession(self.model_path, sess_options=so, providers=prov_with_opts)
        except Exception as e:
            logger.warning("Echec providers avec options, tentative sans options: %s", e)
            try:
                return ort.InferenceSession(self.model_path, sess_options=so, providers=prov_plain)
            except Exception as e2:
                if "CPUExecutionProvider" not in prov_plain:
                    logger.warning("CUDA indisponible, bascule en CPU")
                    return ort.InferenceSession(self.model_path, sess_options=so, providers=["CPUExecutionProvider"])
                raise

    def _infer_input_layout(self) -> str:
        # Détection layout entrée
        input_info = self.session.get_inputs()[0]
        shape = input_info.shape
        if len(shape) == 4:
            c_last = shape[3] if isinstance(shape[3], int) else None
            c_first = shape[1] if isinstance(shape[1], int) else None
            if c_last == 3:
                return "NHWC"
            if c_first == 3:
                return "NCHW"
        return "NHWC"

    def _preprocess(self, image_path: str) -> np.ndarray:
        # Chargement + resize + normalisation
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image introuvable: {os.path.abspath(image_path)}")
        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        except UnidentifiedImageError:
            raise ValueError(f"Fichier non reconnu comme image: {os.path.abspath(image_path)}")

        arr = np.asarray(image, dtype=np.float32)
        if CHANNEL_ORDER.upper() == "BGR":
            arr = arr[:, :, ::-1]

        if NORMALIZATION == "zero_to_one":
            arr = arr / 255.0
        elif NORMALIZATION == "neg_one_to_one":
            arr = (arr - 127.5) / 127.5
        elif NORMALIZATION == "none":
            pass
        else:
            raise ValueError(f"NORMALIZATION inconnu: {NORMALIZATION}")

        if self.input_layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)
        else:
            arr = np.expand_dims(arr, axis=0)

        return arr.astype(np.float32)

    def _preprocess_no_batch(self, image_path: str) -> np.ndarray:
        # Préproc sans dimension batch
        arr = self._preprocess(image_path)
        return arr[0]

    def _apply_thresholds(self, tag: str, score: float) -> bool:
        # Seuils par famille
        if tag.startswith("character:"):
            return score >= CHARACTER_THRESHOLD
        if tag.startswith("rating:"):
            if not INCLUDE_RATING_TAGS:
                return False
            return score >= GENERAL_THRESHOLD
        return score >= GENERAL_THRESHOLD

    def _clean_tag(self, tag: str) -> str:
        # Nettoyage format tag
        t = tag.split(":", 1)[1] if ":" in tag else tag
        if REMOVE_UNDERSCORES:
            t = t.replace("_", " ")
        return t

    def process(self, image_path: str) -> List[str]:
        # Inférence single
        input_tensor = self._preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        probs = outputs[0]
        if hasattr(probs, "ndim") and probs.ndim > 1 and probs.shape[0] == 1:
            probs = probs[0]
        scored = []
        for tag, score in zip(self.tags, probs):
            s = float(score)
            if self._apply_thresholds(tag, s):
                scored.append((self._clean_tag(tag), s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored]

    def _preprocess_batch_sync(self, image_paths: List[str]) -> Tuple[np.ndarray, List[int], List[Optional[List[str]]]]:
        """Prétraitement synchrone d'un batch - retourne (batch_tensor, index_map, results_template)"""
        preproc_ok = []
        index_map = []
        results: List[Optional[List[str]]] = [None] * len(image_paths)

        def worker(item):
            idx, p = item
            try:
                arr = self._preprocess_no_batch(p)
                return idx, arr
            except Exception as e:
                logger.warning("Préproc échouée pour %s: %s", p, e)
                return idx, None

        with ThreadPoolExecutor(max_workers=PREPROC_WORKERS) as ex:
            for idx, arr in ex.map(worker, enumerate(image_paths)):
                if arr is not None:
                    preproc_ok.append(arr)
                    index_map.append(idx)

        if not preproc_ok:
            return None, index_map, results

        batch_input = np.stack(preproc_ok, axis=0).astype(np.float32)
        return batch_input, index_map, results

    def _postprocess_batch(self, probs: np.ndarray, index_map: List[int], results: List[Optional[List[str]]]) -> List[Optional[List[str]]]:
        """Post-traitement des résultats d'inférence"""
        for out_idx, orig_idx in enumerate(index_map):
            row = probs[out_idx]
            scored = []
            for tag, score in zip(self.tags, row):
                s = float(score)
                if self._apply_thresholds(tag, s):
                    scored.append((self._clean_tag(tag), s))
            scored.sort(key=lambda x: x[1], reverse=True)
            tags = [t for t, _ in scored]
            if TOPK_OUTPUT and TOPK_OUTPUT > 0:
                tags = tags[:TOPK_OUTPUT]
            results[orig_idx] = tags
        return results

    def process_batch(self, image_paths: List[str]) -> List[Optional[List[str]]]:
        """Inférence par lot GPU (mode legacy sans pipeline)"""
        if not image_paths:
            return []
        input_name = self.session.get_inputs()[0].name

        batch_input, index_map, results = self._preprocess_batch_sync(image_paths)
        if batch_input is None:
            return results

        try:
            outputs = self.session.run(None, {input_name: batch_input})
        except Exception as e:
            logger.exception("Inférence ONNX échouée pour le lot")
            return results

        probs = outputs[0]
        return self._postprocess_batch(probs, index_map, results)

    def process_batch_pipeline_streaming(self, all_image_paths: List[str], callback=None):
        """Inférence avec pipeline asynchrone CPU/GPU overlapping - mode streaming avec callback temps réel"""
        if not all_image_paths:
            return

        input_name = self.session.get_inputs()[0].name

        # Queue pour batches prétraités (limitée pour éviter surcharge RAM)
        preprocessed_queue: Queue = Queue(maxsize=PREFETCH_BATCHES)

        # Thread producteur : prétraitement asynchrone
        def producer():
            for batch_start in range(0, len(all_image_paths), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(all_image_paths))
                batch_paths = all_image_paths[batch_start:batch_end]

                t_prep_start = time.perf_counter()
                batch_input, index_map, results_template = self._preprocess_batch_sync(batch_paths)
                t_prep = time.perf_counter() - t_prep_start

                logger.debug("[CPU] Batch preprocessed: %d images in %.3fs (queue size=%d)",
                            len(batch_paths), t_prep, preprocessed_queue.qsize())

                preprocessed_queue.put((batch_start, batch_paths, batch_input, index_map, results_template))

            # Signal de fin
            preprocessed_queue.put(None)
            logger.debug("[CPU] Producer finished")

        # Lancement du thread producteur
        producer_thread = Thread(target=producer, daemon=True)
        producer_thread.start()

        # Thread principal : consommation + inférence GPU
        batch_count = 0
        while True:
            item = preprocessed_queue.get()
            if item is None:  # Signal de fin
                break

            batch_start, batch_paths, batch_input, index_map, results_template = item
            batch_count += 1

            if batch_input is None:
                # Batch vide (toutes les images en erreur)
                if callback:
                    for i in range(len(results_template)):
                        callback(batch_paths[i], None, 0)
                continue

            # Inférence GPU (pendant que le producteur prépare le batch suivant)
            try:
                logger.debug("[GPU] Starting inference batch #%d (queue size=%d)", batch_count, preprocessed_queue.qsize())

                t1 = time.perf_counter()
                outputs = self.session.run(None, {input_name: batch_input})
                dt = time.perf_counter() - t1

                logger.debug("[GPU] Inference done: %.3fs for %d images (%.1f img/s)",
                            dt, len(batch_paths), len(batch_paths) / dt)

                probs = outputs[0]
                batch_results = self._postprocess_batch(probs, index_map, results_template)

                # Callback temps réel pour chaque image
                if callback:
                    for i, result in enumerate(batch_results):
                        callback(batch_paths[i], result, dt / max(1, len(batch_paths)))

            except Exception as e:
                logger.exception("Inférence ONNX échouée pour batch #%d", batch_count)
                if callback:
                    for path in batch_paths:
                        callback(path, None, 0)

        producer_thread.join()

# ==================== SQLITE ====================
def fetch_image_paths(conn: sqlite3.Connection, query: str) -> List[str]:
    # Récupération des chemins
    paths: List[str] = []
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    for r in rows:
        if not r:
            continue
        paths.append(str(r[0]))
    return paths

def update_detect_wdtag(conn: sqlite3.Connection, image_path: str, tags_json: str) -> int:
    # Mise à jour des tags
    cur = conn.cursor()
    cur.execute(f"UPDATE {SQLITE_TABLE} SET {SQLITE_TAGS_COLUMN} = ? WHERE image_path = ?", (tags_json, image_path))
    return cur.rowcount

# ==== Helpers fichiers images ====
def iter_image_files(dir_path: str):
    # Itération images du dossier
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        ext = os.path.splitext(entry.name)[1].lower()
        if ext in IMAGE_EXTS:
            yield entry.path

# ==== Lookalike helpers ===
def ensure_lookalike_table(conn: sqlite3.Connection) -> None:
    # Création table + index
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS "lookalike" ("character" TEXT PRIMARY KEY, "wdtags" TEXT NOT NULL)')
    cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS "ux_lookalike_character" ON "lookalike"("character")')
    conn.commit()

def upsert_lookalike(conn: sqlite3.Connection, character_name: str, tags_json: str) -> int:
    # UPSERT wdtags uniquement
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO "lookalike" ("character","wdtags") VALUES (?, ?) '
        'ON CONFLICT("character") DO UPDATE SET "wdtags"=excluded."wdtags"',
        (character_name, tags_json),
    )
    return cur.rowcount




# ==== Job _characters ====
def run_characters_dir_job(db_path: str, characters_dir: str) -> None:
    # Boucle _characters en batch
    logger.info("Début traitement lookalike pour: %s", characters_dir)
    if not os.path.isdir(characters_dir):
        logger.info("Dossier _characters absent, skip")
        return

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Base SQLite introuvable: {db_path}")

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        ensure_lookalike_table(conn)
        tagger = WD14Tagger()

        processed = 0
        upserts = 0
        errors = 0
        processed_since_commit = 0

        batch: List[str] = []

        def flush_batch(paths: List[str]) -> None:
            # Exécution d'un lot tolérant
            nonlocal processed, upserts, processed_since_commit, errors
            if not paths:
                return
            try:
                t1 = time.perf_counter()
                batch_tags = tagger.process_batch(paths)
                dt = time.perf_counter() - t1

                for fp, tags in zip(paths, batch_tags):
                    if tags is None:
                        errors += 1
                        continue

                    final_tags = tags
                    tags_json = json.dumps(final_tags, ensure_ascii=False)
                    character_name = os.path.splitext(os.path.basename(fp))[0]

                    rc = upsert_lookalike(conn, character_name, tags_json)
                    upserts += rc
                    processed += 1
                    processed_since_commit += 1

                    payload = {
                        "character": character_name,
                        "tags": final_tags,
                        "infer_sec": round(dt / max(1, len(paths)), 3),
                    }
                    print(json.dumps(payload, ensure_ascii=False))
            except Exception:
                errors += len(paths)
                logger.exception("Echec traitement batch lookalike")

        for fp in iter_image_files(characters_dir):
            batch.append(fp)
            if len(batch) >= BATCH_SIZE:
                flush_batch(batch)
                batch = []
                if processed_since_commit >= BATCH_COMMIT_SIZE:
                    conn.commit()
                    logger.info(
                        "Commit intermédiaire lookalike après %d fichiers (UPSERT cumulés=%d)",
                        processed_since_commit,
                        upserts,
                    )
                    processed_since_commit = 0

        flush_batch(batch)
        conn.commit()
        logger.info("Lookalike terminé. OK=%d, UPSERT=%d, erreurs=%d", processed, upserts, errors)
    finally:
        conn.close()

# ==================== MAIN LOOP ====================
def run_sqlite_job(db_path: str, use_pipeline: bool = True) -> None:
    # Boucle principale SQLite - mode pipeline par défaut
    logger.info("Début traitement SQLite (mode pipeline: %s)", use_pipeline)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Base SQLite introuvable: {db_path}")

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")

        paths = fetch_image_paths(conn, SQLITE_QUERY)
        logger.info("Chemins récupérés: %d", len(paths))

        tagger = WD14Tagger()

        processed = 0
        updated = 0
        errors = 0
        processed_since_commit = 0

        if use_pipeline:
            # MODE PIPELINE : traitement streaming avec affichage temps réel
            logger.info("Lancement pipeline asynchrone (prefetch=%d batches)", PREFETCH_BATCHES)
            t1 = time.perf_counter()

            # Callback pour affichage temps réel + écriture DB
            def on_image_processed(image_path: str, tags: Optional[List[str]], infer_time: float):
                nonlocal processed, updated, errors, processed_since_commit

                if tags is None:
                    errors += 1
                    return

                tags_json = json.dumps(tags, ensure_ascii=False)
                rc = update_detect_wdtag(conn, image_path, tags_json)
                updated += rc
                processed += 1
                processed_since_commit += 1

                # Affichage temps réel
                payload = {
                    "image_path": image_path,
                    "tags": tags,
                    "infer_sec": round(infer_time, 3),
                }
                print(json.dumps(payload, ensure_ascii=False))

                # Commit périodique
                if processed_since_commit >= BATCH_COMMIT_SIZE:
                    conn.commit()
                    logger.info(
                        "Commit intermédiaire après %d lignes OK (MAJ cumulées=%d)",
                        processed_since_commit,
                        updated,
                    )
                    processed_since_commit = 0

            # Exécution du pipeline avec callback
            tagger.process_batch_pipeline_streaming(paths, callback=on_image_processed)

            dt = time.perf_counter() - t1
            logger.info("Pipeline terminé en %.2fs (%.1f img/s)", dt, len(paths) / dt)

        else:
            # MODE LEGACY : batch par batch (dent de scie)
            for i in range(0, len(paths), BATCH_SIZE):
                batch_paths = paths[i:i + BATCH_SIZE]
                try:
                    t1 = time.perf_counter()
                    batch_tags = tagger.process_batch(batch_paths)
                    dt = time.perf_counter() - t1

                    for raw_p, tags in zip(batch_paths, batch_tags):
                        if tags is None:
                            errors += 1
                            continue
                        tags_json = json.dumps(tags, ensure_ascii=False)
                        rc = update_detect_wdtag(conn, raw_p, tags_json)
                        updated += rc
                        processed += 1
                        processed_since_commit += 1

                        payload = {
                            "image_path": raw_p,
                            "tags": tags,
                            "infer_sec": round(dt / max(1, len(batch_paths)), 3),
                        }
                        print(json.dumps(payload, ensure_ascii=False))

                    if processed_since_commit >= BATCH_COMMIT_SIZE:
                        conn.commit()
                        logger.info(
                            "Commit intermédiaire après %d lignes OK (MAJ cumulées=%d)",
                            processed_since_commit,
                            updated,
                        )
                        processed_since_commit = 0
                except Exception:
                    errors += len(batch_paths)
                    logger.exception("Echec traitement batch [%d:%d]", i, i + len(batch_paths))

        conn.commit()
        logger.info("Terminé. OK=%d, MAJ=%d, erreurs=%d", processed, updated, errors)
    finally:
        conn.close()

# ==================== CLI ====================
def build_db_path_from_dir(directory: str) -> str:
    # Concaténation dossier + nom fixe
    return os.path.join(directory, DB_FILENAME)

# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WD14 Tagger - Traitement via SQLite")
    parser.add_argument(
        "--directory",
        type=str,
        default=r"D:\SHUUMATSU NO WALKURE",
        help="Repertoire contenant SQLLITE.db"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Taille de lot pour l'inférence GPU"
    )
    parser.add_argument(
        "--chars-only",
        action="store_true",
        help="Ne traiter que le dossier _characters et la table lookalike"
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Désactive le pipeline asynchrone (mode legacy avec dent de scie)"
    )
    args = parser.parse_args()

    base_dir = os.path.normpath(args.directory)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Répertoire introuvable: {base_dir}")

    # Ajustement dynamique BATCH_SIZE
    BATCH_SIZE = max(1, int(args.batch_size))

    sqlite_db_path = build_db_path_from_dir(base_dir)
    logger.info("Base SQLite résolue: %s", sqlite_db_path)
    logger.info("Batch size: %d", BATCH_SIZE)

    characters_dir = os.path.join(base_dir, CHARACTERS_DIR_NAME)

    # Mode chars-only
    if args.chars_only:
        logger.info("Mode --chars-only activé: exécution uniquement du job lookalike")
        if os.path.isdir(characters_dir):
            logger.info("Dossier _characters détecté: %s", characters_dir)
            run_characters_dir_job(sqlite_db_path, characters_dir)
        else:
            logger.info("Dossier _characters non trouvé, aucune insertion lookalike")
    else:
        # Traitement principal
        use_pipeline = not args.no_pipeline
        run_sqlite_job(sqlite_db_path, use_pipeline=use_pipeline)
        # Traitement _characters -> table lookalike
        if os.path.isdir(characters_dir):
            logger.info("Dossier _characters détecté: %s", characters_dir)
            run_characters_dir_job(sqlite_db_path, characters_dir)
        else:
            logger.info("Dossier _characters non trouvé, aucune insertion lookalike")

