import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional

# Dépendances image
try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

# Torch (optionnel pour l'upscale)
REAL_ESRGAN_IMPORT_ERROR: Optional[str] = None
try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    REAL_ESRGAN_IMPORT_ERROR = f"torch import failed: {e!r}"
    torch = None  # type: ignore

# Compat torchvision récent: certains basicsr importent un ancien module
try:
    import torchvision.transforms.functional_tensor as _tt  # type: ignore
except Exception:
    try:
        import types
        from torchvision.transforms import functional as _F  # type: ignore
        _shim_mod = types.ModuleType("torchvision.transforms.functional_tensor")
        setattr(_shim_mod, "rgb_to_grayscale", getattr(_F, "rgb_to_grayscale"))
        sys.modules["torchvision.transforms.functional_tensor"] = _shim_mod
    except Exception as e:
        REAL_ESRGAN_IMPORT_ERROR = (REAL_ESRGAN_IMPORT_ERROR or "") + f"; tv shim failed: {e!r}"

# realesrgan: support des deux APIs (RealESRGAN et RealESRGANer)
RealESRGAN = None  # type: ignore
RealESRGANer = None  # type: ignore
REAL_API_MODE = None  # 'simple' ou 'er'
try:
    from realesrgan import RealESRGAN as _RealESRGAN  # type: ignore
    RealESRGAN = _RealESRGAN
    REAL_API_MODE = "simple"
except Exception as e1:  # pragma: no cover
    try:
        from realesrgan import RealESRGANer as _RealESRGANer  # type: ignore
        RealESRGANer = _RealESRGANer
        REAL_API_MODE = "er"
    except Exception as e2:  # pragma: no cover
        REAL_ESRGAN_IMPORT_ERROR = f"realesrgan import failed: {e1!r}; fallback failed: {e2!r}"
        RealESRGAN = None  # type: ignore
        RealESRGANer = None  # type: ignore


# Convertit en RGB en gérant l'alpha (fond blanc)
def to_rgb_composite(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.getchannel("A") if "A" in img.getbands() else None
        if alpha is not None:
            if img.mode == "LA":
                img_rgb = img.convert("RGB")
                bg.paste(img_rgb, mask=alpha)
            else:
                bg.paste(img, mask=alpha)
        else:
            bg.paste(img)
        return bg
    if img.mode == "P":
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# Redimensionne (downscale) à max_side si le plus grand côté dépasse max_side
def downscale_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    ratio = max_side / float(max(w, h))
    new_size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
    return img.resize(new_size, Image.LANCZOS)


# Prépare et retourne un prédicteur .predict(PIL)->PIL (Real-ESRGAN)
def prepare_realesrgan(weights: Path):
    if Image is None:
        raise RuntimeError("Pillow est requis. Installez-le: pip install pillow")
    if ((RealESRGAN is None and RealESRGANer is None) or torch is None):
        msg = [
            "Real-ESRGAN indisponible (import échoué).",
            f"Interpréteur Python: {sys.executable}",
        ]
        if REAL_ESRGAN_IMPORT_ERROR:
            msg.append(f"Détail import: {REAL_ESRGAN_IMPORT_ERROR}")
        msg.append("Vérifiez: python -m pip install realesrgan torch torchvision")
        raise RuntimeError("\n".join(msg))

    if not weights.is_file():
        raise FileNotFoundError(
            f"Poids Real-ESRGAN introuvables: {weights}. Placez le fichier dans scripts/models ou utilisez --realesrgan-weights."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if REAL_API_MODE == "simple":
        model = RealESRGAN(device, scale=4)  # type: ignore[call-arg]
        model.load_weights(str(weights))
        print(f"Poids chargés: {weights}")
        return model

    # Fallback RealESRGANer: architecture explicite pour anime_6B
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except Exception as e_imp1:  # pragma: no cover
        try:
            from realesrgan.archs.rrdbnet_arch import RRDBNet  # type: ignore
        except Exception as e_imp2:
            raise RuntimeError(
                "Impossible d'importer RRDBNet (basicsr/realesrgan). Installez basicsr>=1.4.2. "
                f"Détails: {e_imp1!r} / {e_imp2!r}"
            )

    rrdb = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,       # anime_6B = 6 blocks
        num_grow_ch=32,
        scale=4,
    )
    model_er = RealESRGANer(  # type: ignore[call-arg]
        scale=4,
        model_path=str(weights),
        model=rrdb,
        device=device,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )

    class _PredictWrapper:
        def __init__(self, er_model):
            self.er_model = er_model

        def predict(self, pil_img: Image.Image) -> Image.Image:
            import numpy as np  # type: ignore
            rgb = pil_img.convert("RGB")
            arr = np.array(rgb)  # HWC RGB
            bgr = arr[:, :, ::-1]
            out_bgr, _ = self.er_model.enhance(bgr, outscale=1)
            out_rgb = out_bgr[:, :, ::-1]
            return Image.fromarray(out_rgb)

    print(f"Poids chargés: {weights}")
    return _PredictWrapper(model_er)


# Copie brute (octets) d'un fichier (pour JPEG déjà à la bonne taille)
def copy_as_is(src: Path, dst: Path) -> None:
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())


# Ouvre une image en sécurité; retourne None si échec
def load_image_safe(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


# Copie/convertit les images de <base>/dataset vers <base>/img
# - Recréation du dossier <base>/img à chaque exécution
# - Upscale si plus petit que max_side (puis downscale à max_side)
# - Downscale si plus grand que max_side
# - Copie brute (JPEG) si déjà exactement max_side
def copy_dataset_to_img(
    base_dir: Path,
    max_side: int = 1024,
    realesrgan_weights: Optional[Path] = None,
) -> None:

    if Image is None:
        raise RuntimeError("Pillow est requis. Installez-le: pip install pillow")

    # Par défaut: scripts/models/RealESRGAN_x4plus_anime_6B.pth (pas de guessing)
    if realesrgan_weights is None:
        realesrgan_weights = Path(__file__).resolve().parent / "models" / "RealESRGAN_x4plus_anime_6B.pth"
    print(f"Poids Real-ESRGAN attendus: {realesrgan_weights}")

    base_dir = base_dir.resolve()
    dataset_dir = base_dir / "dataset"
    dest_dir = base_dir / "img"

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de base introuvable: {base_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Répertoire 'dataset' introuvable: {dataset_dir}")

    # Recréation du dossier de sortie
    if dest_dir.is_dir():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = 0          # fichiers rencontrés (tous types)
    processed = 0      # images produites en sortie
    skipped = 0        # non-images ignorés
    resized = 0        # images réduites car > max_side
    upscaled = 0       # images upscalées car < max_side
    counter = 1        # compteur séquentiel pour nommer 00001.jpg, ...

    # Préparation (si échec, on reportera l'erreur au moment de l'upscale)
    try:
        realesrgan_model = prepare_realesrgan(realesrgan_weights)
        realesrgan_prepare_error = None
    except Exception as e:
        realesrgan_model = None
        realesrgan_prepare_error = str(e)

    # Parcours récursif de "dataset" (on ne modifie jamais dataset)
    for root, _, files in os.walk(dataset_dir):
        for name in sorted(files):  # ordre déterministe
            total += 1
            src = Path(root) / name

            img = load_image_safe(src)
            if img is None:
                skipped += 1
                continue

            dst_name = f"{counter:05d}.jpg"
            dst = dest_dir / dst_name

            fmt = (img.format or "").upper()
            try:
                # Correction d'orientation EXIF si présente
                base = ImageOps.exif_transpose(img)
                largest = max(base.size)

                # Cas 1: image petite -> upscale puis downscale à max_side
                if largest < max_side:
                    if realesrgan_model is None:
                        raise RuntimeError(realesrgan_prepare_error or "Real-ESRGAN indisponible.")
                    work = to_rgb_composite(base)
                    # passes x4 jusqu'à atteindre/dépasser max_side
                    current_max = largest
                    passes = 0
                    while current_max < max_side:
                        work = realesrgan_model.predict(work)
                        current_max = max(work.size)
                        passes += 1
                        if passes > 6:
                            break
                    work = downscale_to_max_side(work, max_side)
                    upscaled += 1

                # Cas 2: JPEG déjà à la bonne taille -> copie brute
                elif fmt == "JPEG" and largest == max_side:
                    copy_as_is(src, dst)
                    processed += 1
                    counter += 1
                    continue

                # Cas 3: conversion JPEG et downscale si nécessaire
                else:
                    work = to_rgb_composite(base)
                    if largest > max_side:
                        work = downscale_to_max_side(work, max_side)
                        resized += 1

                # Sauvegarde en JPEG (propage EXIF si présent)
                exif = img.info.get("exif")
                save_kwargs = {
                    "format": "JPEG",
                    "quality": 95,
                    "optimize": True,
                }
                if exif:
                    save_kwargs["exif"] = exif
                work.save(dst, **save_kwargs)
            finally:
                img.close()

            processed += 1
            counter += 1

    print(
        f"Terminé: trouvés={total} | produits(JPEG)={processed} | ignorés(non-images)={skipped} | "
        f"réduits(>={max_side})={resized} | upscalés(<{max_side})={upscaled}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copie <base>/dataset -> <base>/img en 00001.jpg, 00002.jpg, ... "
            "(downscale > max_side, upscale < max_side via Real-ESRGAN)"
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(r"E:\AI_WORK\TRAINED_LORA\SHUUMATSU NO WALKURE\aphrodite_snw"),
        help="Répertoire à analyser",
    )
    parser.add_argument(
        "--max-side",
        dest="max_side",
        type=int,
        default=1024,
        help="Dimension maximale du côté le plus long",
    )
    parser.add_argument(
        "--realesrgan-weights",
        dest="realesrgan_weights",
        type=Path,
        default=None,
        help="Chemin vers RealESRGAN_x4plus_anime_6B.pth (défaut: scripts/models/...)",
    )
    args = parser.parse_args()
    copy_dataset_to_img(args.directory, max_side=args.max_side, realesrgan_weights=args.realesrgan_weights)


if __name__ == "__main__":
    main()

