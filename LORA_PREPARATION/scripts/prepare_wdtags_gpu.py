import os
import argparse
import logging
import time
import csv
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

# ==================== CONFIG ====================
# Dossier par defaut si --directory n'est pas fourni
# Renseignez ce chemin pour eviter une longue ligne de commande
DEFAULT_DIRECTORY = r"E:\AI_WORK\TRAINED_LORA\SHUUMATSU NO WALKURE\aphrodite_snw\img"

# Modele / tags (memes chemins que detect_wdtags_gpu.py)
MODEL_ONNX_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.onnx"
TAGS_CSV_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.csv"

# Inference
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.55
INCLUDE_RATING_TAGS = True
REMOVE_UNDERSCORES = True
TOPK_OUTPUT = 0  # 0 = tous les tags retenus
BATCH_SIZE = 32

# Pre-traitement
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
FORCE_CPU = False

# Images / IO
IMAGE_EXTS = {".jpg", ".jpeg"}

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare-wdtags")

# ==================== CUDA / cuDNN ====================
# Identiques au script d'origine pour Windows
CUDNN_DIR = r"C:\Program Files\NVIDIA\CUDNN\v9.9\bin\12.9"
CUDA_DIR = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"


def _setup_cuda_paths() -> None:
    """Configure les chemins CUDA/cuDNN (Windows) pour permettre le chargement des DLL."""
    for d in [CUDNN_DIR, CUDA_DIR]:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


_setup_cuda_paths()


# ==================== TAGGER ====================
class WD14Tagger:
    MODEL_ONNX_PATH = MODEL_ONNX_PATH
    TAGS_CSV_PATH = TAGS_CSV_PATH

    def __init__(self):
        # Initialise la session ONNX Runtime et charge la liste des tags
        self.model_path = self.MODEL_ONNX_PATH
        self.tags_csv_path = self.TAGS_CSV_PATH

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, (os.cpu_count() or 8))
        so.inter_op_num_threads = 1

        logger.info("Modele ONNX: %s", os.path.abspath(self.model_path))
        logger.info("CSV Tags: %s", os.path.abspath(self.tags_csv_path))
        self.tags = self._load_tags(self.tags_csv_path)
        self.session = self._create_session_with_fallback(so)
        self.input_layout = self._infer_input_layout()

    def _load_tags(self, csv_path: str) -> List[str]:
        # Charge la liste des tags depuis le CSV fourni avec le modele
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
            raise ValueError("Aucun tag charge depuis le CSV")
        return tags

    def _create_session_with_fallback(self, so: ort.SessionOptions):
        # Cree une session ORT en privilegiant CUDA si disponible, sinon CPU
        available = set(ort.get_available_providers())
        if FORCE_CPU or "CUDAExecutionProvider" not in available:
            providers = ["CPUExecutionProvider"]
        else:
            providers = [("CUDAExecutionProvider", CUDA_PROVIDER_OPTIONS), "CPUExecutionProvider"]
        try:
            return ort.InferenceSession(self.model_path, sess_options=so, providers=providers)
        except Exception:
            return ort.InferenceSession(self.model_path, sess_options=so, providers=["CPUExecutionProvider"])

    def _infer_input_layout(self) -> str:
        # Deduit le layout d'entree attendu par le modele (NHWC ou NCHW)
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
        # Ouvre l'image, applique un resize fixe et normalise en float32
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image introuvable: {os.path.abspath(image_path)}")
        try:
            with Image.open(image_path) as im:
                # Appliquer l'orientation EXIF pour des entrees coherentes
                im = ImageOps.exif_transpose(im)
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

        # Ajout dimension batch selon layout
        if self.input_layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, axis=0)
        else:
            arr = np.expand_dims(arr, axis=0)
        return arr.astype(np.float32)

    def _preprocess_no_batch(self, image_path: str) -> np.ndarray:
        # Preprocess sans dimension batch (utilise pour le pre-traitement parallele)
        arr = self._preprocess(image_path)
        return arr[0]

    def _apply_thresholds(self, tag: str, score: float) -> bool:
        # Applique des seuils differents selon le type de tag (character/rating/general)
        if tag.startswith("character:"):
            return score >= CHARACTER_THRESHOLD
        if tag.startswith("rating:"):
            if not INCLUDE_RATING_TAGS:
                return False
            return score >= GENERAL_THRESHOLD
        return score >= GENERAL_THRESHOLD

    def _clean_tag(self, tag: str) -> str:
        # Nettoie le nom de tag (supprime le prefixe et remplace _ par espace)
        t = tag.split(":", 1)[1] if ":" in tag else tag
        if REMOVE_UNDERSCORES:
            t = t.replace("_", " ")
        return t

    def process_batch(self, image_paths: List[str]) -> List[Optional[List[str]]]:
        # Inference par lots avec pre-traitement multithread, tolerance aux erreurs
        if not image_paths:
            return []
        input_name = self.session.get_inputs()[0].name

        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(32, max(4, (os.cpu_count() or 8)))
        preproc_ok: List[np.ndarray] = []
        index_map: List[int] = []
        results: List[Optional[List[str]]] = [None] * len(image_paths)

        def worker(item):
            idx, p = item
            try:
                arr = self._preprocess_no_batch(p)
                return idx, arr
            except Exception as e:
                logger.warning("Preproc echouee pour %s: %s", p, e)
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, arr in ex.map(worker, enumerate(image_paths)):
                if arr is not None:
                    preproc_ok.append(arr)
                    index_map.append(idx)

        if not preproc_ok:
            return results

        batch_input = np.stack(preproc_ok, axis=0).astype(np.float32)
        try:
            outputs = self.session.run(None, {input_name: batch_input})
        except Exception:
            logger.exception("Inference ONNX echouee pour le lot")
            return results
        probs = outputs[0]

        for out_idx, orig_idx in enumerate(index_map):
            row = probs[out_idx]
            scored: List[Tuple[str, float]] = []
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


# ==================== FILE HELPERS ====================
def iter_image_files(dir_path: str) -> List[str]:
    """Liste les images d'un dossier (extensions supportees), triees par nom croissant."""
    files: List[str] = []
    for entry in os.scandir(dir_path):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(entry.path)
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files


def write_tag_files(finals: List[str], tag_lists: List[Optional[List[str]]]) -> None:
    """Cree un .txt par image avec les tags separes par ", "."""
    for dst, tags in zip(finals, tag_lists):
        base = os.path.splitext(os.path.basename(dst))[0]
        txt_path = os.path.join(os.path.dirname(dst), f"{base}.txt")
        line = "" if not tags else ", ".join(tags)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(line.strip())


# ==================== MAIN ====================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genere les .txt de tags pour chaque image presente dans le dossier fourni "
            "par --directory (sans renommage ni conversion)"
        )
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=DEFAULT_DIRECTORY,
        help="Dossier contenant directement les images a traiter (defaut: DEFAULT_DIRECTORY)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Taille de lot pour l'inference GPU")
    args = parser.parse_args()

    if not args.directory:
        raise FileNotFoundError("Aucun dossier fourni (--directory) et DEFAULT_DIRECTORY non defini.")
    base_dir = os.path.normpath(args.directory)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Repertoire introuvable: {base_dir}")

    # Utiliser directement le dossier fourni, sans sous-dossier 'img'
    img_dir = base_dir

    batch_size = max(1, int(args.batch_size))

    # 1) Lister les images a traiter (tri croissant)
    files = iter_image_files(img_dir)
    if not files:
        logger.info("Aucune image trouvee dans: %s", img_dir)
        return

    logger.info("Images detectees: %d", len(files))

    # 2) Detecter les tags
    tagger = WD14Tagger()
    all_tags: List[Optional[List[str]]] = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        t0 = time.perf_counter()
        res = tagger.process_batch(batch)
        dt = time.perf_counter() - t0
        logger.info("Batch %d-%d: %d images traitees en %.2fs", i + 1, i + len(batch), len(batch), dt)
        all_tags.extend(res)

    # 3) Ecriture des .txt sans renommage ni conversion
    write_tag_files(files, all_tags)
    logger.info("Fichiers .txt generes: %d", len(files))
    return


if __name__ == "__main__":
    main()
