import os
import re
import sqlite3
import argparse
import logging
import shutil
import hashlib
from typing import Set, Dict, Optional

# ==================== CONFIG ====================
DB_FILENAME = "SQLLITE.db"
SQLITE_IMAGES_TABLE = "images"
SQLITE_LOOKALIKE_TABLE = "lookalike"
SQLITE_PATH_COLUMN = "image_path"
SQLITE_LOOKALIKE_COLUMN = "lookalike"
SQLITE_CHARACTER_COLUMN = "character"
READ_CHUNK_SIZE = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("create-folders-by-character")


# ==================== UTIL: chemins ====================
def build_db_path_from_dir(directory: str) -> str:
    return os.path.join(directory, DB_FILENAME)

def sanitize_component(name: str) -> str:
    name = name.strip()
    if not name:
        return "_"
    name = re.sub(r"[\\/]+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name)
    name = name.strip(" .")
    return name or "_"

def resolve_src_path(base_dir: str, image_path: str) -> str:
    if not image_path:
        return ""
    return image_path if os.path.isabs(image_path) else os.path.normpath(os.path.join(base_dir, image_path))


# ==================== FS: création/copier ====================
def ensure_dirs(dest_root: str, categories: Set[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for cat in sorted(categories):
        safe = sanitize_component(cat)
        dst = os.path.join(dest_root, safe)
        os.makedirs(dst, exist_ok=True)
        mapping[cat] = dst
    return mapping

def safe_copy(src: str, dst_dir: str, dry_run: bool = False) -> Optional[str]:
    if not os.path.isfile(src):
        return None
    base = os.path.basename(src)
    dest_path = os.path.join(dst_dir, base)
    if os.path.exists(dest_path):
        try:
            if os.path.getsize(src) == os.path.getsize(dest_path):
                return dest_path
        except OSError:
            pass
        name, ext = os.path.splitext(base)
        suffix = hashlib.sha1(src.encode("utf-8")).hexdigest()[:8]
        dest_path = os.path.join(dst_dir, f"{name}_{suffix}{ext}")
    if not dry_run:
        shutil.copy2(src, dest_path)
    return dest_path


# ==================== DB: requêtes ====================
def fetch_lookalike_categories(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.cursor()
    cur.execute(
        f"SELECT DISTINCT {SQLITE_CHARACTER_COLUMN} "
        f"FROM {SQLITE_LOOKALIKE_TABLE} "
        f"WHERE {SQLITE_CHARACTER_COLUMN} IS NOT NULL AND TRIM({SQLITE_CHARACTER_COLUMN}) <> ''"
    )
    return {str(r[0]).strip() for r in cur.fetchall()}


# ==================== MAIN ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="Créer dossiers par character et copier les images classées")
    parser.add_argument(
        "--directory",
        type=str,
        default=r"T:\_SELECT\READY\GUY DOUBLE TARGET",
        help="Répertoire contenant SQLLITE.db, base pour chemins relatifs"
    )
    args = parser.parse_args()

    base_dir = os.path.normpath(args.directory)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Répertoire introuvable: {base_dir}")

    # Destination locale
    dest_root = os.path.join(base_dir, "_classified")
    os.makedirs(dest_root, exist_ok=True)

    db_path = build_db_path_from_dir(base_dir)
    logger.info("Base SQLite: %s", db_path)
    logger.info("Destination: %s", dest_root)

    conn = sqlite3.connect(db_path)
    try:
        # Dossiers depuis lookalike.character
        categories = fetch_lookalike_categories(conn)
        if not categories:
            logger.warning("Aucune catégorie dans lookalike")
        dir_map = ensure_dirs(dest_root, categories)

        # Copie par lot
        cur = conn.cursor()
        cur.execute(f"SELECT {SQLITE_PATH_COLUMN}, {SQLITE_LOOKALIKE_COLUMN} FROM {SQLITE_IMAGES_TABLE}")

        processed = 0
        copied = 0
        missing = 0
        skipped_unknown = 0

        while True:
            rows = cur.fetchmany(READ_CHUNK_SIZE)
            if not rows:
                break

            for image_path, lookalike in rows:
                processed += 1

                category = str(lookalike).strip() if lookalike else ""
                if not category:
                    skipped_unknown += 1
                    continue

                dst_dir = dir_map.get(category)
                if not dst_dir:
                    skipped_unknown += 1
                    logger.warning("Catégorie inconnue (absente de lookalike): %s", category)
                    continue

                src = resolve_src_path(base_dir, str(image_path) if image_path else "")
                if not src or not os.path.isfile(src):
                    missing += 1
                    logger.warning("Source introuvable, ignorée: %s", src or "<vide>")
                    continue

                if safe_copy(src, dst_dir, dry_run=False):
                    copied += 1

            if processed % 5000 == 0:
                logger.info("Progression: processed=%d | copied=%d | missing=%d | skipped_unknown=%d",
                            processed, copied, missing, skipped_unknown)

        logger.info("Terminé | processed=%d | copied=%d | missing=%d | skipped_unknown=%d | categories=%d",
                    processed, copied, missing, skipped_unknown, len(categories))
    finally:
        conn.close()


if __name__ == "__main__":
    main()