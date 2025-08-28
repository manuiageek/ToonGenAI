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

# ==================== CONFIG ====================
# Chemins modèle / tags
MODEL_ONNX_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.onnx"
TAGS_CSV_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.csv"

# Inférence
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.75
INCLUDE_RATING_TAGS = True
REMOVE_UNDERSCORES = True
TOPK_OUTPUT = 20

# Prétraitement
TARGET_SIZE = (448, 448)
CHANNEL_ORDER = "BGR"
NORMALIZATION = "none"

# Providers
DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
FORCE_CPU = False

# SQLite
SQLITE_DB_PATH = r"T:\_SELECT\READY\GUY DOUBLE TARGET\SQLLITE.db"
SQLITE_QUERY = "select image_path from images limit 10"
SQLITE_TABLE = "images"
SQLITE_TAGS_COLUMN = "detect_wdtag"
BATCH_COMMIT_SIZE = 1000  # commit par lots

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
        req_providers = ["CPUExecutionProvider"] if FORCE_CPU else list(DEFAULT_PROVIDERS)
        logger.info("Modèle ONNX: %s", os.path.abspath(self.model_path))
        logger.info("CSV Tags: %s", os.path.abspath(self.tags_csv_path))
        self.tags = self._load_tags(self.tags_csv_path)
        self.session = self._create_session_with_fallback(req_providers)
        logger.info("Session ORT prête avec providers: %s", self.session.get_providers())
        self.input_layout = self._infer_input_layout()
        logger.info("Layout entrée modèle: %s", self.input_layout)

    def _load_tags(self, csv_path: str) -> List[str]:
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

    def _create_session_with_fallback(self, providers: List[str]):
        available = set(ort.get_available_providers())
        filtered = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        logger.info("Providers demandés: %s", providers)
        logger.info("Providers utilisés: %s", filtered)
        try:
            return ort.InferenceSession(self.model_path, providers=filtered)
        except Exception as e:
            if "CPUExecutionProvider" not in filtered:
                try:
                    logger.warning("CUDA indisponible, bascule en CPU")
                    return ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                except Exception as e2:
                    logger.exception("Echec création session ORT même en CPU")
                    raise RuntimeError(f"Echec création session ORT (CPU fallback). Cause initiale: {e}") from e2
            raise

    def _infer_input_layout(self) -> str:
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

    def _apply_thresholds(self, tag: str, score: float) -> bool:
        if tag.startswith("character:"):
            return score >= CHARACTER_THRESHOLD
        if tag.startswith("rating:"):
            if not INCLUDE_RATING_TAGS:
                return False
            return score >= GENERAL_THRESHOLD
        return score >= GENERAL_THRESHOLD

    def _clean_tag(self, tag: str) -> str:
        t = tag.split(":", 1)[1] if ":" in tag else tag
        if REMOVE_UNDERSCORES:
            t = t.replace("_", " ")
        return t

    def process(self, image_path: str) -> List[str]:
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

# ==================== SQLITE ====================
def ensure_column_exists(conn: sqlite3.Connection, table: str, column: str, col_def: str = "TEXT") -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
        conn.commit()

def fetch_image_paths(conn: sqlite3.Connection, query: str) -> List[str]:
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
    cur = conn.cursor()
    cur.execute(f"UPDATE {SQLITE_TABLE} SET {SQLITE_TAGS_COLUMN} = ? WHERE image_path = ?", (tags_json, image_path))
    return cur.rowcount

# ==================== MAIN LOOP ====================
def run_sqlite_job() -> None:
    logger.info("Début traitement SQLite")
    if not os.path.isfile(SQLITE_DB_PATH):
        raise FileNotFoundError(f"Base SQLite introuvable: {SQLITE_DB_PATH}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        ensure_column_exists(conn, SQLITE_TABLE, SQLITE_TAGS_COLUMN, "TEXT")
        paths = fetch_image_paths(conn, SQLITE_QUERY)
        logger.info("Chemins récupérés: %d", len(paths))

        tagger = WD14Tagger()

        processed = 0
        updated = 0
        errors = 0
        processed_since_commit = 0  # compteur lot

        for raw_p in paths:
            p = os.path.normpath(str(raw_p).strip())
            try:
                t1 = time.perf_counter()
                tags = tagger.process(p)
                if TOPK_OUTPUT and TOPK_OUTPUT > 0:
                    tags = tags[:TOPK_OUTPUT]
                dt = time.perf_counter() - t1
                tags_json = json.dumps(tags, ensure_ascii=False)

                rc = update_detect_wdtag(conn, raw_p, tags_json)
                updated += rc
                processed += 1
                processed_since_commit += 1

                payload = {"image_path": raw_p, "tags": tags, "infer_sec": round(dt, 3)}
                print(json.dumps(payload, ensure_ascii=False))

                # Commit intermédiaire par lots
                if processed_since_commit >= BATCH_COMMIT_SIZE:
                    conn.commit()
                    logger.info("Commit intermédiaire après %d lignes OK (MAJ cumulées=%d)", processed_since_commit, updated)
                    processed_since_commit = 0
            except Exception:
                errors += 1
                logger.exception("Echec traitement: %s", raw_p)

        # Commit final
        conn.commit()
        logger.info("Terminé. OK=%d, MAJ=%d, erreurs=%d", processed, updated, errors)
    finally:
        conn.close()

# ==================== MAIN ====================
if __name__ == "__main__":
    run_sqlite_job()