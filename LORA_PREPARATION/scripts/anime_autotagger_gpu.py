import json
import os
import time
import logging
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import csv
from fastapi import FastAPI, Query
from typing import List, Optional
import uvicorn

# ==================== CONFIG ====================
RUN_MODE = "api"  # "selftest" ou "api"

# Chemins
MODEL_ONNX_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.onnx"
TAGS_CSV_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.csv"
TEST_IMAGE_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\image\image1.jpg"

# Inférence
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.75
INCLUDE_RATING_TAGS = True
REMOVE_UNDERSCORES = True

# Prétraitement
TARGET_SIZE = (448, 448)
CHANNEL_ORDER = "BGR"
NORMALIZATION = "none"

# Providers
DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
FORCE_CPU = False

# Serveur FastAPI
FASTAPI_HOST = "localhost"
FASTAPI_PORT = 8000

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
    # Constantes de classe
    MODEL_ONNX_PATH = MODEL_ONNX_PATH
    TAGS_CSV_PATH = TAGS_CSV_PATH

    def __init__(self):
        # Chemins
        self.model_path = self.MODEL_ONNX_PATH
        self.tags_csv_path = self.TAGS_CSV_PATH

        # Providers
        req_providers = ["CPUExecutionProvider"] if FORCE_CPU else list(DEFAULT_PROVIDERS)

        # Log chemins
        logger.info("Modèle ONNX: %s", os.path.abspath(self.model_path))
        logger.info("CSV Tags: %s", os.path.abspath(self.tags_csv_path))

        # Tags
        self.tags = self._load_tags(self.tags_csv_path)

        # Session ORT
        self.session = self._create_session_with_fallback(req_providers)
        logger.info("Session ORT prête avec providers: %s", self.session.get_providers())

        # Mise en forme d'entrée
        self.input_layout = self._infer_input_layout()
        logger.info("Layout entrée modèle: %s", self.input_layout)

    # -------------------- chargement tags --------------------
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

    # -------------------- session ORT --------------------
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

    # -------------------- layout entrée --------------------
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

    # -------------------- prétraitement --------------------
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

    # -------------------- post-traitement --------------------
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

    # -------------------- inférence --------------------
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

# ==================== API ====================
app = FastAPI()
tagger = WD14Tagger()

class TagsRequest(BaseModel):
    # Corps JSON d'entrée
    image_path: str = Field(..., description="Chemin absolu de l'image")

class TagsResponse(BaseModel):
    tags: List[str]

@app.post("/getwdtags", response_model=TagsResponse)
def get_tags(req: TagsRequest) -> TagsResponse:
    # Lecture depuis le body JSON
    return TagsResponse(tags=tagger.process(req.image_path))

# ==================== SELF TEST ====================
def self_test() -> None:
    logger.info("Début self-test WD14")
    logger.info("ORT %s, providers disponibles: %s", ort.__version__, ort.get_available_providers())
    logger.info("Image de test: %s", TEST_IMAGE_PATH)
    try:
        t0 = time.perf_counter()
        local = WD14Tagger()
        t1 = time.perf_counter()
        logger.info("Session initialisée en %.2fs", t1 - t0)
        tags = local.process(TEST_IMAGE_PATH)
        t2 = time.perf_counter()
        logger.info("Inférence terminée en %.2fs, %d tags retenus", t2 - t1, len(tags))
        topk = 20
        if topk and topk > 0:
            tags = tags[:topk]
        payload = {"tags": tags}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("Self-test OK")
    except Exception:
        logger.exception("Self-test en échec")

# ==================== MAIN ====================
if __name__ == "__main__":
    if RUN_MODE == "api":
        uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)
    else:
        self_test()

