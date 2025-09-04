import json
import os
import time
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
import csv
from typing import List
import sqlite3
import argparse
from concurrent.futures import ThreadPoolExecutor

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
BATCH_SIZE = 32

# Prétraitement
TARGET_SIZE = (448, 448)
CHANNEL_ORDER = "BGR"
NORMALIZATION = "none"

# Providers
CUDA_PROVIDER_OPTIONS = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",
    "cudnn_conv_use_max_workspace": 1,
    "do_copy_in_default_stream": 1,
    "tunable_op_enable": 1,
    "tunable_op_tuning_enable": 1,
}
DEFAULT_PROVIDERS = [("CUDAExecutionProvider", CUDA_PROVIDER_OPTIONS), "CPUExecutionProvider"]
FORCE_CPU = False

# SQLite
DB_FILENAME = "SQLLITE.db"
SQLITE_QUERY = "select image_path from images"
SQLITE_TABLE = "images"
SQLITE_TAGS_COLUMN = "detect_wdtags"
BATCH_COMMIT_SIZE = 1000

# Characters dir + extensions images
CHARACTERS_DIR_NAME = "_characters"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Post-traitement IA
LOOKALIKE_AI_ENABLED = True

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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

        # Session ORT optimisée
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, (os.cpu_count() or 8))
        so.inter_op_num_threads = 1

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

    def process_batch(self, image_paths: List[str]) -> List[List[str]]:
        # Inférence par lot GPU
        if not image_paths:
            return []
        input_name = self.session.get_inputs()[0].name

        # Prétraitement concurrent
        max_workers = min(32, max(4, (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            batch = list(ex.map(self._preprocess_no_batch, image_paths))

        batch_input = np.stack(batch, axis=0).astype(np.float32)
        outputs = self.session.run(None, {input_name: batch_input})
        probs = outputs[0]

        results: List[List[str]] = []
        for row in probs:
            scored = []
            for tag, score in zip(self.tags, row):
                s = float(score)
                if self._apply_thresholds(tag, s):
                    scored.append((self._clean_tag(tag), s))
            scored.sort(key=lambda x: x[1], reverse=True)
            tags = [t for t, _ in scored]
            if TOPK_OUTPUT and TOPK_OUTPUT > 0:
                tags = tags[:TOPK_OUTPUT]
            results.append(tags)
        return results

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

# ==== Lookalike helpers ====
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

# ==== OpenAI post-traitement ====
def _sanitize_json_block(s: str) -> str:
    # Nettoyage éventuels fences
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def ai_refine_tags(tags: List[str]) -> List[str]:
    # Filtrage IA des tags
    if not LOOKALIKE_AI_ENABLED:
        return tags

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY manquante, post-traitement IA ignoré")
        return tags

    try:
        from openai import OpenAI
    except Exception:
        logger.warning("SDK OpenAI non disponible, post-traitement IA ignoré")
        return tags

    client = OpenAI(api_key=api_key)
    try:
        client = client.with_options(timeout=30)
    except Exception:
        pass

    system_msg = (
        "Tu es un assistant qui filtre des listes de tags. "
        "Retire les tags d'habillement, 'solo', '1girl', '1boy', "
        "ceux liés à la poitrine 'breast', l'état de la personne 'smile', 'open mouth','sensitive', "
        "les prises de vues 'looking at viewer', 'upper body', 'portrait' et autres, "
        "les tags relatifs au background 'outdoors' et autres. "
        "Réponds UNIQUEMENT avec un objet JSON valide."
    )
    user_msg = f"Filtre cette liste de tags : {json.dumps(tags, ensure_ascii=False)}"

    json_schema = {
        "name": "filtered_tags",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tags"]
        }
    }

    model = "gpt-4o-mini"

    for attempt in range(1, 4):
        try:
            content = None

            # Appel Responses prioritaire
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
                        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
                    ],
                    response_format={"type": "json_schema", "json_schema": json_schema},
                    temperature=0,
                    max_output_tokens=512,
                )
                if hasattr(resp, "output_text") and resp.output_text:
                    content = resp.output_text.strip()
                else:
                    try:
                        resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else getattr(resp, "__dict__", {})
                        content = resp_dict.get("output_text")
                        if not content:
                            output = resp_dict.get("output", []) or []
                            texts = []
                            for block in output:
                                typ = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                                if typ == "output_text":
                                    texts.append(block.get("text") if isinstance(block, dict) else getattr(block, "text", ""))
                            content = ("".join(texts)).strip() if texts else None
                    except Exception:
                        content = None

            except Exception as e_resp:
                logger.debug(f"Responses API indisponible (tentative {attempt}): {e_resp}")
                # Fallback Chat Completions propre
                try:
                    chat = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0,
                        max_tokens=512,
                    )
                    if chat and getattr(chat, "choices", None):
                        content = chat.choices[0].message.content.strip()
                except Exception as e_chat:
                    logger.debug(f"Echec Chat Completions fallback: {e_chat}")
                    raise

            if not content:
                raise RuntimeError("Réponse vide depuis l'API OpenAI")

            content = _sanitize_json_block(content)

            # Parsing JSON robuste
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    for key in ["tags", "filtered_tags", "result", "items"]:
                        val = data.get(key)
                        if isinstance(val, list):
                            return [str(x) for x in val if isinstance(x, str)]
                elif isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return data
            except json.JSONDecodeError:
                pass

            logger.warning(f"Réponse IA invalide tentative {attempt}: {content[:120]}...")

        except Exception as e:
            logger.warning(f"Erreur IA tentative {attempt}/3: {str(e)}")
            if attempt < 3:
                time.sleep(2)
            continue

    logger.warning("Post-traitement IA échoué, conservation des tags originaux")
    return tags



# ==== Job _characters ====
def run_characters_dir_job(db_path: str, characters_dir: str) -> None:
    # Boucle _characters en batch + IA
    logger.info("Début traitement lookalike pour: %s", characters_dir)
    if not os.path.isdir(characters_dir):
        logger.info("Dossier _characters absent, skip")
        return

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Base SQLite introuvable: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        ensure_lookalike_table(conn)
        tagger = WD14Tagger()

        processed = 0
        upserts = 0
        errors = 0
        processed_since_commit = 0

        batch: List[str] = []

        def flush_batch(paths: List[str]) -> None:
            # Exécution d'un lot
            nonlocal processed, upserts, processed_since_commit, errors
            if not paths:
                return
            try:
                t1 = time.perf_counter()
                batch_tags = tagger.process_batch(paths)
                dt = time.perf_counter() - t1

                for fp, tags in zip(paths, batch_tags):
                    final_tags = ai_refine_tags(tags) if LOOKALIKE_AI_ENABLED else tags
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
def run_sqlite_job(db_path: str) -> None:
    # Boucle principale SQLite en batch
    logger.info("Début traitement SQLite")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Base SQLite introuvable: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        paths = fetch_image_paths(conn, SQLITE_QUERY)
        logger.info("Chemins récupérés: %d", len(paths))

        tagger = WD14Tagger()

        processed = 0
        updated = 0
        errors = 0
        processed_since_commit = 0

        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i:i + BATCH_SIZE]
            try:
                t1 = time.perf_counter()
                batch_tags = tagger.process_batch(batch_paths)
                dt = time.perf_counter() - t1

                for raw_p, tags in zip(batch_paths, batch_tags):
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
        default=r"T:\_SELECT\READY\ARTE",
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
        help="Ne traiter que le dossier _characters et la table lookalike (avec post-traitement OpenAI)"
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
        run_sqlite_job(sqlite_db_path)
        # Traitement _characters -> table lookalike + IA
        if os.path.isdir(characters_dir):
            logger.info("Dossier _characters détecté: %s", characters_dir)
            run_characters_dir_job(sqlite_db_path, characters_dir)
        else:
            logger.info("Dossier _characters non trouvé, aucune insertion lookalike")
