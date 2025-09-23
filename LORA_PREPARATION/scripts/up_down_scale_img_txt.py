import os
import shutil
import argparse
import hashlib
from pathlib import Path

# Script: copie toutes les images de <base>\dataset vers <base>\img.
# - Ne modifie jamais les originaux (lecture seule dans "dataset").
# - Avant de copier, supprime tous les fichiers déjà présents dans "img".
# - Si un nom existe déjà lors de la copie et que la taille diffère, ajoute un suffixe court.
# Exemple:
#   python up_down_scale_img_txt.py "E:\\AI_WORK\\TRAINED_LORA\\SHUUMATSU NO WALKURE\\aphrodite_snw"


# Copie tout le contenu fichier de <base>/dataset vers <base>/img (tout fichier est considéré image)
def copy_dataset_to_img(base_dir: Path) -> None:
    """Copie toutes les images de <base>/dataset vers <base>/img sans toucher aux originaux.
    Étapes: vérifs -> suppression du dossier <base>/img s'il existe -> création -> copie depuis dataset.
    Règles de collision: même taille=ignore; taille différente=suffixe court ajouté (hash du chemin source).
    """
    base_dir = base_dir.resolve()
    dataset_dir = base_dir / "dataset"
    dest_dir = base_dir / "img"

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de base introuvable: {base_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Répertoire 'dataset' introuvable: {dataset_dir}")

    # IMPORTANT: si <base>/img existe, on supprime entièrement ce dossier pour repartir propre
    if dest_dir.is_dir():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    skipped = 0

    # Parcours récursif de "dataset" (tout fichier trouvé est copié)
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            total += 1
            src = Path(root) / name
            dst = dest_dir / name

            # Gestion des collisions de nom dans "img"
            if dst.exists():
                try:
                    # Même nom ET même taille => déjà présent, on ignore
                    if src.stat().st_size == dst.stat().st_size:
                        skipped += 1
                        continue
                except OSError:
                    pass
                # Taille différente: on ajoute un suffixe court basé sur un hash du chemin source
                stem, ext = (Path(name).stem, Path(name).suffix)
                suffix = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
                dst = dest_dir / f"{stem}_{suffix}{ext}"

            # Copie avec métadonnées (dates, etc.)
            shutil.copy2(src, dst)
            copied += 1

    # Résumé final
    print(f"Terminé: total={total} | copiées={copied} | ignorées={skipped}")


# Point d'entrée CLI: argument --directory (type Path) avec une valeur par défaut
def main() -> None:
    parser = argparse.ArgumentParser(description="Copie <base>/dataset -> <base>/img")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(r"D:\\SHUUMATSU NO WALKURE"),
        help="Répertoire à analyser",
    )
    args = parser.parse_args()
    copy_dataset_to_img(args.directory)


if __name__ == "__main__":
    main()
