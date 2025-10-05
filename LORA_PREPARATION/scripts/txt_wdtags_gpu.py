import os
import argparse
import logging
import time
import csv
import json
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

# ==================== CONFIG ====================
# Dossier par defaut si --directory n'est pas fourni
# Renseignez ce chemin pour eviter une longue ligne de commande
DEFAULT_DIRECTORY = r"D:\HIGH SCHOOL OF THE DEAD\_characters"

# Modele / tags (memes chemins que detect_wdtags_gpu.py)
MODEL_ONNX_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.onnx"
TAGS_CSV_PATH = r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\wd-vit-tagger-v3.csv"

# Inference
GENERAL_THRESHOLD = 0.3
CHARACTER_THRESHOLD = 0.4
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
DEFAULT_PROVIDERS = [("CUDAExecutionProvider", CUDA_PROVIDER_OPTIONS), "CPUExecutionProvider"]
FORCE_CPU = False

# Verbosite / logs ORT
LOG_STARTUP_DETAILS = False  # True pour logs detailles au demarrage
ONNXRUNTIME_LOG_SEVERITY_LEVEL = 3  # 0=VERBOSE,1=INFO,2=WARNING,3=ERROR,4=FATAL

# Filtrage de tags indesirables (apres nettoyage)
EXCLUDED_TAGS = {"1girl", "1boy", "looking at viewer"}
EXCLUDED_TAGS_LOWER = {t.lower().strip() for t in EXCLUDED_TAGS}

# Images / IO
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

# OpenAI post-traitement
LOOKALIKE_AI_ENABLED = True

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare-wdtags")

# ==================== CUDA / cuDNN ====================
# Identiques au script d'origine pour Windows
CUDNN_DIR = r"C:\Program Files\NVIDIA\CUDNN\v9.9\bin\12.9"
CUDA_DIR = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"


def _setup_cuda_paths() -> None:
    """Configure les chemins CUDA/cuDNN (Windows) et tente un prechargement des DLL."""
    for d in [CUDNN_DIR, CUDA_DIR]:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                if LOG_STARTUP_DETAILS:
                    logger.info("DLL directory ajoute: %s", d)
            except Exception as e:
                logger.debug("Echec add_dll_directory pour %s: %s", d, e)
        else:
            logger.debug("Repertoire DLL introuvable: %s", d)

    # Préchargement opportuniste des DLL (si disponible dans votre version d'onnxruntime)
    try:
        ort.preload_dlls(cudnn=True, directory=CUDNN_DIR)
    except Exception as e:
        logger.debug("preload_dlls(cudnn) indisponible/échec: %s", e)
    try:
        ort.preload_dlls(cuda=True, directory=CUDA_DIR)
    except Exception as e:
        logger.debug("preload_dlls(cuda) indisponible/échec: %s", e)

    # Infos debug ORT (tres verbeux) uniquement si demande
    if LOG_STARTUP_DETAILS:
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
        # Initialise la session ONNX Runtime et charge la liste des tags
        self.model_path = self.MODEL_ONNX_PATH
        self.tags_csv_path = self.TAGS_CSV_PATH

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, (os.cpu_count() or 8))
        so.inter_op_num_threads = 1
        try:
            so.log_severity_level = int(ONNXRUNTIME_LOG_SEVERITY_LEVEL)
            so.log_verbosity_level = 0
        except Exception:
            pass

        logger.info("Modele ONNX: %s", os.path.abspath(self.model_path))
        logger.info("CSV Tags: %s", os.path.abspath(self.tags_csv_path))
        self.tags = self._load_tags(self.tags_csv_path)
        req_providers = ["CPUExecutionProvider"] if FORCE_CPU else list(DEFAULT_PROVIDERS)
        self.session = self._create_session_with_fallback(req_providers, so)
        try:
            logger.info("Session ORT prête avec providers: %s", self.session.get_providers())
        except Exception:
            pass
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

    def _create_session_with_fallback(self, providers: List, so: ort.SessionOptions):
        # Crée une session en tentant CUDA puis bascule CPU si échec
        available = set(ort.get_available_providers())

        def normalize(prov_list: List, with_options: bool) -> List:
            out: List = []
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

        logger.debug("Providers demandés: %s", providers)
        logger.debug("Providers dispo: %s", list(available))
        try:
            return ort.InferenceSession(self.model_path, sess_options=so, providers=prov_with_opts)
        except Exception as e:
            logger.debug("Echec providers avec options, tentative sans options: %s", e)
            try:
                return ort.InferenceSession(self.model_path, sess_options=so, providers=prov_plain)
            except Exception as e2:
                if "CPUExecutionProvider" not in prov_plain:
                    logger.info("CUDA indisponible, bascule en CPU")
                    return ort.InferenceSession(self.model_path, sess_options=so, providers=["CPUExecutionProvider"])
                raise

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
            # Filtre les tags indesirables (insensible a la casse)
            if EXCLUDED_TAGS_LOWER:
                tags = [t for t in tags if t.lower() not in EXCLUDED_TAGS_LOWER]
            results[orig_idx] = tags

        return results


# ==================== OPENAI POST-TRAITEMENT ====================
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

    # Pour z.ai, utiliser directement l'API key (pas de base_url dans env)
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")  # ou "ZAI_API_KEY"
    if not auth_token:
        logger.warning("ANTHROPIC_AUTH_TOKEN manquant, post-traitement IA ignoré")
        return tags
    
    try:
        from openai import OpenAI
    except Exception:
        logger.warning("SDK OpenAI non disponible, post-traitement IA ignoré")
        return tags

    # URL correcte pour l'API compatible OpenAI de z.ai
    client = OpenAI(
        api_key=auth_token,
        base_url="https://api.z.ai/api/paas/v4/"  
    )
    
    try:
        client = client.with_options(timeout=30)
    except Exception:
        pass

    system_msg = (
        "Tu es un assistant qui filtre des listes de tags. "
        "Retire les tags d'habillement, 'solo', '1girl', '1boy', "
        "les tags liés à la poitrine 'breast', l'état de la personne 'smile', 'open mouth','sensitive', "
        "les tags liés aux prises de vues 'looking at viewer', 'upper body', 'portrait' et autres, "
        "les tags liés aux vues de photographie 'looking at viewer', 'upper body', 'portrait' et autres, "
        "les tags relatifs au background 'outdoors' et autres. "
        "Réponds UNIQUEMENT avec un objet JSON valide au format: {\"tags\": [\"tag1\", \"tag2\"]}"
    )
    user_msg = f"Filtre cette liste de tags : {json.dumps(tags, ensure_ascii=False)}"

    model = "glm-4.5-air"  # Modèle léger et économique

    for attempt in range(1, 4):
        try:
            # Appel Chat Completions standard
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=512,
            )
            
            if not chat or not chat.choices or len(chat.choices) == 0:
                raise RuntimeError("Réponse API vide")
            
            content = chat.choices[0].message.content
            if not content:
                raise RuntimeError("Message sans contenu")
            
            logger.info(f"Réponse brute IA: {content[:200]}")
            
            content = _sanitize_json_block(content)
            
            # Parsing JSON
            data = json.loads(content)
            if isinstance(data, dict):
                for key in ["tags", "filtered_tags", "result", "items"]:
                    val = data.get(key)
                    if isinstance(val, list):
                        return [str(x) for x in val if isinstance(x, str)]
            elif isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
            
            logger.warning(f"Format réponse invalide tentative {attempt}: {content[:120]}")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON invalide tentative {attempt}/3: {e}")
        except Exception as e:
            logger.warning(f"Erreur IA tentative {attempt}/3: {str(e)}")
        
        if attempt < 3:
            time.sleep(2)

    logger.warning("Post-traitement IA échoué, conservation des tags originaux")
    return tags



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
    """Cree un .txt par image avec les tags separes par ", " (avec post-traitement OpenAI optionnel)."""
    for dst, tags in zip(finals, tag_lists):
        if tags is None:
            tags = []
        else:
            # Appliquer le post-traitement OpenAI si activé
            if LOOKALIKE_AI_ENABLED:
                tags = ai_refine_tags(tags)

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
    parser.add_argument("--no-ai", action="store_true", help="Désactive le post-traitement OpenAI des tags")
    args = parser.parse_args()

    # Désactiver l'IA si demandé
    global LOOKALIKE_AI_ENABLED
    if args.no_ai:
        LOOKALIKE_AI_ENABLED = False

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
