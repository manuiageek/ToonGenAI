import os
import json
import sqlite3
import argparse
import logging
import random
from typing import List, Optional, Tuple, Set

# ==================== CONFIG ====================
DB_FILENAME = "SQLLITE.db"
SQLITE_TABLE = "images"
SQLITE_PATH_COLUMN = "image_path"
SQLITE_TAGS_COLUMN = "detect_wdtags"
SQLITE_LOOKALIKE_COLUMN = "lookalike"
SQLITE_QUERY = f"select {SQLITE_PATH_COLUMN}, {SQLITE_TAGS_COLUMN} from {SQLITE_TABLE}"
BATCH_COMMIT_SIZE = 1000
READ_CHUNK_SIZE = 2000

# Table lookalike
LOOKALIKE_TABLE = "lookalike"
LOOKALIKE_CHARACTER_COLUMN = "character"
LOOKALIKE_WDTAGS_COLUMN = "wdtags"

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

def norm_tag(s: str) -> str:
    # Normalisation tag
    return s.strip().lower().replace(" ", "_")

def normalize_tags(tags: List[str]) -> Set[str]:
    # Liste -> set normalisé
    return {norm_tag(t) for t in tags if t}

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
    logger.info("Initialisation lookalike")

# ==================== LOOKALIKE MATCHING ====================
Candidate = Tuple[str, Set[str], int]

def load_lookalike_candidates(conn: sqlite3.Connection) -> List[Candidate]:
    # Préchargement des candidats
    cur = conn.cursor()
    cur.execute(
        f"SELECT {LOOKALIKE_CHARACTER_COLUMN}, {LOOKALIKE_WDTAGS_COLUMN} FROM {LOOKALIKE_TABLE}"
    )
    rows = cur.fetchall()
    candidates: List[Candidate] = []
    for character, wdtags_json in rows:
        tags = parse_tags(wdtags_json)
        tagset = normalize_tags(tags)
        if not character or not tagset:
            continue
        candidates.append((str(character), tagset, len(tagset)))
    logger.info("Candidats lookalike chargés: %d", len(candidates))

    # Listing des characters
    characters = [c[0] for c in candidates]
    characters = sorted(dict.fromkeys(characters))
    if characters:
        lines = "\n".join(f"- {c}" for c in characters)
        logger.info("Characters lookalike:\n%s", lines)
    else:
        logger.info("Characters lookalike:\n<aucun>")

    return candidates

def match_lookalike(image_tags: List[str], candidates: List[Candidate], threshold: float) -> Optional[str]:
    # Parcours complet + sélection du meilleur
    imgset = normalize_tags(image_tags)
    if not imgset:
        return None

    best_ratio = -1.0
    best_chars: List[str] = []
    eps = 1e-9

    for character, tagset, count in candidates:
        if count == 0:
            continue
        inter_size = len(imgset & tagset)
        ratio = inter_size / count
        if ratio > best_ratio + eps:
            best_ratio = ratio
            best_chars = [character]
        elif abs(ratio - best_ratio) <= eps:
            best_chars.append(character)

    if best_ratio >= threshold and best_chars:
        return random.choice(best_chars)

    return None

def decide_target(image_tags: List[str], candidates: List[Candidate], threshold: float) -> str:
    # Règles de classement
    has_boy = contains_substr(image_tags, "boy")
    has_girl = contains_substr(image_tags, "girl")
    has_monochrome = contains_substr(image_tags, "monochrome")
    has_closed = contains_substr(image_tags, "closed_eyes")

    target: Optional[str] = None

    # R1: boy seul
    if has_boy and not has_girl:
        target = "z_boy"
    # R2: monochrome
    elif has_monochrome:
        target = "z_monochrome"
    # R3: closed_eyes
    elif has_closed:
        target = "z_closed_eyes"
    # R4: ni boy ni girl => fond
    elif not has_boy and not has_girl:
        target = "z_background"

    # R5: matching lookalike sinon
    if target is None:
        match = match_lookalike(image_tags, candidates, threshold)
        if match is not None:
            target = match
        else:
            target = "z_misc"

    return target

# ==================== MAIN ====================
def main() -> None:
    parser = argparse.ArgumentParser(description="Règles lookalike + matching par wdtags")
    parser.add_argument(
        "--directory",
        type=str,
        default=r"T:\_SELECT\READY\GUY DOUBLE TARGET",
        help="Répertoire contenant SQLLITE.db"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        help="Seuil [0..1] de correspondance entre lookalike.wdtags et images.detect_wdtags"
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

        # Préchargement des candidats
        candidates = load_lookalike_candidates(conn)

        # Lecture + traitement par lots
        cur = conn.cursor()
        cur.execute(SQLITE_QUERY)

        processed = 0
        updated = 0
        since_commit = 0
        matched_characters = 0

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

                # Décision centralisée
                target = decide_target(tags, candidates, args.match_threshold)

                if target is not None:
                    rc = update_lookalike(conn, str(image_path), target)
                    updated += rc
                    since_commit += rc
                    if target not in ("z_boy", "z_monochrome", "z_closed_eyes", "z_background", "z_misc"):
                        matched_characters += 1

                processed += 1

                if since_commit >= BATCH_COMMIT_SIZE:
                    conn.commit()
                    logger.info("Commit intermédiaire | updated=%d / processed=%d | matched_characters=%d", updated, processed, matched_characters)
                    since_commit = 0

            if processed % 5000 == 0:
                logger.info("Progression: %d lignes traitées", processed)

        conn.commit()
        logger.info("Terminé | processed=%d | updated=%d | matched_characters=%d", processed, updated, matched_characters)
    finally:
        conn.close()

if __name__ == "__main__":
    main()