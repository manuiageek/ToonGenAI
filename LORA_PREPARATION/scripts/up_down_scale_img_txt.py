import os
import shutil
import argparse
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageOps # pyright: ignore[reportMissingImports]
except ImportError:
    Image = None  # type: ignore


# Copie/convertit les images de <base>/dataset vers <base>/img en les
# renommant séquentiellement sous la forme 00001.jpg, 00002.jpg, ...
# Règles:
# - Recrée <base>/img à chaque exécution (suppression préalable)
# - Ignore les fichiers non-images
# - Convertit en JPEG si nécessaire et aplatit l'arborescence dans <base>/img
def copy_dataset_to_img(base_dir: Path) -> None:

    if Image is None:
        raise RuntimeError("Pillow est requis. Installez-le avec: pip install pillow")

    base_dir = base_dir.resolve()
    dataset_dir = base_dir / "dataset"
    dest_dir = base_dir / "img"

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de base introuvable: {base_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Répertoire 'dataset' introuvable: {dataset_dir}")

    # RecrǸation du dossier de sortie
    if dest_dir.is_dir():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = 0          # fichiers rencontrǸs (tous types)
    processed = 0      # images produites en sortie
    skipped = 0        # non-images ignorǸs
    resized = 0        # images rǸduites car > 1024px
    counter = 1        # compteur sǸquentiel pour nommer 00001.jpg, ...

    # Ouvre une image en sécurité; retourne None si échec
    def load_image_safe(path: Path) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            img.load()
            return img
        except Exception:
            return None

    # Convertit en RGB en gǸrant l'alpha (fond blanc)
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

    # Redimensionne si le plus grand côté dépasse max_side
    def resize_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        ratio = max_side / float(max(w, h))
        new_size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
        return img.resize(new_size, Image.LANCZOS)

    # Parcours rǸcursif de "dataset"
    for root, _, files in os.walk(dataset_dir):
        for name in sorted(files):  # tri pour ordre dǸterministe
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
                # Correction d'orientation EXIF si prǸsente
                base = ImageOps.exif_transpose(img)

                need_resize = max(base.size) > 1024

                if fmt == "JPEG" and not need_resize:
                    # Copie directe, on impose juste le nom .jpg (pas de rǸ-encodage)
                    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                        fdst.write(fsrc.read())
                else:
                    work = to_rgb_composite(base)
                    if need_resize:
                        work = resize_if_needed(work, 1024)
                        resized += 1
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
        f"total={total} | processed(JPEG)={processed} | skipped(non-images)={skipped} | resized(>1024)={resized}"
    )


# Point d'entrée CLI: parse les arguments et lance le traitement
def main() -> None:
    parser = argparse.ArgumentParser(description="Copie <base>/dataset -> <base>/img en 00001.jpg, 00002.jpg�?�")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(r"E:\AI_WORK\TRAINED_LORA\SHUUMATSU NO WALKURE\aphrodite_snw"),
        help="Répertoire à analyser",
    )
    args = parser.parse_args()
    copy_dataset_to_img(args.directory)


if __name__ == "__main__":
    main()

