import os
import json
import sqlite3
import argparse
import logging
from typing import List, Optional

# ==================== CONFIG ====================
DB_FILENAME = "SQLLITE.db"
SQLITE_TABLE = "images"
SQLITE_PATH_COLUMN = "image_path"
SQLITE_TAGS_COLUMN = "detect_wdtags"
SQLITE_LOOKALIKE_COLUMN = "lookalike"
SQLITE_QUERY = f"select {SQLITE_PATH_COLUMN}, {SQLITE_TAGS_COLUMN} from {SQLITE_TABLE}"
BATCH_COMMIT_SIZE = 1000
READ_CHUNK_SIZE = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wd14-update")

# ==================== UTIL ====================
def build_db_path_from_dir(directory: str) -> str:
    return os.path.join(directory, DB_FILENAME)

def parse_tags(tags_json: Optional[str]) -> List[str]:
    # JSON -> liste de tags
    if not tags_json:
        return []
    try:
        data = json.loads(tags_json)
        if isinstance(data, list):
            return [str(x).strip() for x in data if isinstance(x, str) and x.strip()]
    except Exception:
        logger.warning("JSON tags invalide, ignoré")
    return []

def contains_substr(tags: List[str], needle: str) -> bool:
    # Test sous-chaîne insensible à la casse
    n = needle.lower()
    for t in tags:
        if n in t.lower():
            return True
    return False

def update_lookalike(conn: sqlite3.Connection, image_path: str, value: str) -> int:
    # UPDATE idempotent
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE {SQLITE_TABLE}
        SET {SQLITE_LOOKALIKE_COLUMN} = ?
        WHERE {SQLITE_PATH_COLUMN} = ?
          AND ({SQLITE_LOOKALIKE_COLUMN} IS NULL OR {SQLITE_LOOKALIKE_COLUMN} <> ?)
        """,
        (value, image_path, value),
    )
    return cur.rowcount

def init_lookalike(conn: sqlite3.Connection) -> None:
    # Réinitialisation globale
    cur = conn.cursor()
    cur.execute(f"UPDATE {SQLITE_TABLE} SET {SQLITE_LOOKALIKE_COLUMN} = ''")
    conn.commit()
    logger.info("Initialisation lookalike = '' effectuée")

# ==================== MAIN ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="Règles lookalike: z_boy / z_closed_eyes / z_background")
    parser.add_argument(
        "--directory",
        type=str,
        default=r"T:\_SELECT\READY\GUY DOUBLE TARGET",
        help="Répertoire contenant SQLLITE.db"
    )
    args = parser.parse_args()

    base_dir = os.path.normpath(args.directory)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Répertoire introuvable: {base_dir}")

    db_path = build_db_path_from_dir(base_dir)
    logger.info("Base SQLite: %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        # Init globale
        init_lookalike(conn)

        # Lecture + traitement par lots
        cur = conn.cursor()
        cur.execute(SQLITE_QUERY)

        processed = 0
        updated = 0
        since_commit = 0

        while True:
            rows = cur.fetchmany(READ_CHUNK_SIZE)
            if not rows:
                break

            for r in rows:
                image_path = r[0]
                tags_json = r[1] if len(r) > 1 else None
                if image_path is None:
                    continue

                tags = parse_tags(tags_json)

                has_boy = contains_substr(tags, "boy")
                has_girl = contains_substr(tags, "girl")
                has_closed = contains_substr(tags, "closed_eyes")

                target: Optional[str] = None

                # R1: boy seul
                if has_boy and not has_girl:
                    target = "z_boy"
                # R2: si R1 non remplie et closed_eyes
                elif has_closed:
                    target = "z_closed_eyes"
                # R3: ni boy ni girl => fond
                elif not has_boy and not has_girl:
                    target = "z_background"
                # Sinon: laisser lookalike=""

                if target is not None:
                    rc = update_lookalike(conn, str(image_path), target)
                    updated += rc
                    since_commit += rc

                processed += 1

                if since_commit >= BATCH_COMMIT_SIZE:
                    conn.commit()
                    logger.info("Commit intermédiaire | updated=%d / processed=%d", updated, processed)
                    since_commit = 0

            if processed % 5000 == 0:
                logger.info("Progression: %d lignes traitées", processed)

        conn.commit()
        logger.info("Terminé | processed=%d | updated=%d", processed, updated)
    finally:
        conn.close()

if __name__ == "__main__":
    main()