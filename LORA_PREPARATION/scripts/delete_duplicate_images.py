#!/usr/bin/env python3
"""Detecteur/suppresseur de doublons d'images optimise - Version amelioree."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import lz4.frame

import imagehash
from PIL import Image

# Configuration optimisee
HASH_SIZE = 16  # Taille du hash perceptuel
PERCEPTUAL_THRESHOLD = 3  # Seuil de difference pour hash perceptuel (augmente)
SIZE_DIFFERENCE_RATIO = 0.1  # 10% de difference de taille max pour comparer
CACHE_FILE = Path("image_hashes_cache.lz4")
WORKERS = min(32, (os.cpu_count() or 4) * 2)
BATCH_SIZE = 500
MAX_GROUP_SIZE = 100
CACHE_VERSION = "v7"

# Types de hash pour meilleure detection
HASH_TYPES = ['average', 'phash', 'dhash']

class ImageHasher:
    """Classe pour calculer plusieurs types de hash d'image."""
    
    @staticmethod
    def compute_hashes(path: Path) -> Optional[Dict[str, str]]:
        """Calcule plusieurs types de hash pour une image."""
        try:
            with Image.open(path) as img:
                # Convertir en RGB si necessaire
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                hashes = {
                    'average': str(imagehash.average_hash(img, hash_size=HASH_SIZE)),
                    'phash': str(imagehash.phash(img, hash_size=HASH_SIZE//2)),  # Plus petit pour phash
                    'dhash': str(imagehash.dhash(img, hash_size=HASH_SIZE)),
                    'file_hash': ImageHasher._compute_file_hash(path)  # Hash MD5 du fichier
                }
                return hashes
        except Exception as e:
            logging.warning(f"Erreur hash {path}: {e}")
            return None
    
    @staticmethod
    def _compute_file_hash(path: Path, chunk_size: int = 8192) -> str:
        """Calcule le hash MD5 du fichier pour detection exacte."""
        hasher = hashlib.md5()
        try:
            with open(path, 'rb') as f:
                # Lire seulement les premiers et derniers chunks pour performance
                f.seek(0)
                hasher.update(f.read(chunk_size))
                
                file_size = path.stat().st_size
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)
                    hasher.update(f.read(chunk_size))
                    
            return hasher.hexdigest()
        except Exception:
            return ""

class EnhancedCache:
    """Cache ameliore avec validation plus robuste."""
    
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, str]] = {}  # path -> hashes dict
        self.stats: Dict[str, Tuple[float, int]] = {}  # path -> (mtime, size)
        self.version: str = CACHE_VERSION
        self._load()
    
    def _load(self) -> None:
        """Charge le cache depuis le disque."""
        if not CACHE_FILE.exists():
            return
            
        try:
            with CACHE_FILE.open("rb") as f:
                compressed = f.read()
            if compressed:
                data = pickle.loads(lz4.frame.decompress(compressed))
                
                if data.get('version') != CACHE_VERSION:
                    logging.info("Version de cache obsolete, reconstruction")
                    return
                
                self.data = data.get('hashes', {})
                self.stats = data.get('stats', {})
                logging.info(f"Cache charge: {len(self.data)} entrees")
        except Exception as e:
            logging.warning(f"Erreur chargement cache: {e}")
    
    def save(self) -> None:
        """Sauvegarde le cache."""
        try:
            data = {
                'version': self.version,
                'hashes': self.data,
                'stats': self.stats
            }
            compressed = lz4.frame.compress(pickle.dumps(data))
            CACHE_FILE.write_bytes(compressed)
            logging.info(f"Cache sauve: {len(self.data)} entrees")
        except Exception as e:
            logging.error(f"Erreur sauvegarde cache: {e}")
    
    def is_valid(self, path: Path) -> bool:
        """Verifie si le cache est valide pour ce fichier."""
        path_str = str(path)
        if path_str not in self.data or path_str not in self.stats:
            return False
            
        try:
            stat = path.stat()
            cached_mtime, cached_size = self.stats[path_str]
            # Verifier mtime ET taille
            return (abs(stat.st_mtime - cached_mtime) < 1.0 and 
                    stat.st_size == cached_size)
        except Exception:
            return False
    
    def get_or_compute(self, path: Path) -> Optional[Dict[str, str]]:
        """Recupere ou calcule les hashes d'une image."""
        # Verifier d'abord que le fichier existe
        if not path.exists():
            return None
            
        path_str = str(path)
        
        # Verifier le cache
        if self.is_valid(path):
            return self.data[path_str]
        
        # Calculer les nouveaux hashes
        hashes = ImageHasher.compute_hashes(path)
        if hashes:
            self.data[path_str] = hashes
            try:
                stat = path.stat()
                self.stats[path_str] = (stat.st_mtime, stat.st_size)
            except Exception:
                pass
            return hashes
        
        return None

class DuplicateFinder:
    """Classe principale pour trouver et gerer les doublons."""
    
    def __init__(self, cache: EnhancedCache, threshold: int = PERCEPTUAL_THRESHOLD):
        self.cache = cache
        self.removed_count = 0
        self.threshold = threshold
        
    def find_duplicates_in_group(self, images: List[Path]) -> List[Set[Path]]:
        """Trouve tous les groupes de doublons dans une liste d'images."""
        if len(images) < 2:
            return []
        
        # Filtrer les images qui existent encore
        existing_images = [img for img in images if img.exists()]
        if len(existing_images) < 2:
            return []
        
        # Calculer les hashes pour toutes les images existantes
        with ThreadPoolExecutor(max_workers=min(16, len(existing_images))) as pool:
            hash_results = list(pool.map(self.cache.get_or_compute, existing_images))
        
        # Creer un mapping pour comparaisons rapides
        image_data = [(img, hashes) for img, hashes in zip(existing_images, hash_results) if hashes]
        
        # Grouper les doublons
        duplicate_groups = []
        processed = set()
        
        for i, (img_a, hashes_a) in enumerate(image_data):
            if img_a in processed or not img_a.exists():
                continue
                
            current_group = {img_a}
            
            for j, (img_b, hashes_b) in enumerate(image_data):
                if i >= j or img_b in processed or not img_b.exists():
                    continue
                    
                if self._are_duplicates(img_a, hashes_a, img_b, hashes_b):
                    current_group.add(img_b)
                    processed.add(img_b)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed.add(img_a)
        
        return duplicate_groups
    
    def _are_duplicates(self, img_a: Path, hashes_a: Dict, 
                        img_b: Path, hashes_b: Dict) -> bool:
        """Determine si deux images sont des doublons."""
        
        # 1. Verification exacte par hash MD5
        if hashes_a.get('file_hash') and hashes_a['file_hash'] == hashes_b.get('file_hash'):
            return True
        
        # 2. Verification par taille (si trop differente, pas doublon)
        try:
            size_a = img_a.stat().st_size
            size_b = img_b.stat().st_size
            size_diff_ratio = abs(size_a - size_b) / max(size_a, size_b)
            if size_diff_ratio > SIZE_DIFFERENCE_RATIO:
                return False
        except Exception:
            pass
        
        # 3. Verification par hash perceptuel
        try:
            # Average hash
            avg_a = imagehash.hex_to_hash(hashes_a['average'])
            avg_b = imagehash.hex_to_hash(hashes_b['average'])
            if avg_a - avg_b <= self.threshold:
                return True
            
            # pHash (plus robuste aux modifications)
            if 'phash' in hashes_a and 'phash' in hashes_b:
                ph_a = imagehash.hex_to_hash(hashes_a['phash'])
                ph_b = imagehash.hex_to_hash(hashes_b['phash'])
                if ph_a - ph_b <= self.threshold // 2:
                    return True
            
            # dHash (sensible aux differences de detail)
            if 'dhash' in hashes_a and 'dhash' in hashes_b:
                dh_a = imagehash.hex_to_hash(hashes_a['dhash'])
                dh_b = imagehash.hex_to_hash(hashes_b['dhash'])
                if dh_a - dh_b <= self.threshold:
                    return True
                    
        except Exception as e:
            logging.debug(f"Erreur comparaison hash: {e}")
        
        return False
    
    def process_duplicate_group(self, group: Set[Path]) -> int:
        """Traite un groupe de doublons."""
        if len(group) < 2:
            return 0
        
        # Trier par nom pour garder le premier alphabetiquement
        sorted_group = sorted(group, key=lambda p: (p.parent.name, p.name))
        
        keep = sorted_group[0]
        to_remove = sorted_group[1:]
        
        removed = 0
        logging.info(f"Groupe de {len(group)} doublons trouve pour {keep.name}")
        
        for dup in to_remove:
            try:
                dup.unlink()
                removed += 1
                logging.info(f"  Supprime: {dup.parent.name}/{dup.name}")
                
                # Pause Windows
                if removed % 50 == 0:
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"  Echec suppression {dup}: {e}")
        
        return removed

def scan_directory(directory: Path) -> List[Path]:
    """Scan recursif du repertoire pour trouver les images."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    
    logging.info(f"Scan de {directory}...")
    
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            try:
                # Ignorer les fichiers trop petits (probablement des vignettes)
                if path.stat().st_size > 1024:
                    images.append(path)
            except Exception:
                continue
    
    return images

def deduplicate_images(images: List[Path], cache: EnhancedCache, threshold: int = PERCEPTUAL_THRESHOLD, dry_run: bool = False) -> int:
    """Deduplique une liste d'images."""
    if not images:
        return 0
    
    finder = DuplicateFinder(cache, threshold)
    total_removed = 0
    
    # Garder trace des fichiers deja supprimes
    already_deleted = set()
    
    # Strategie 1: Grouper par taille approximative
    size_groups = defaultdict(list)
    for img in images:
        if img in already_deleted:
            continue
        try:
            if img.exists():  # Verifier que le fichier existe encore
                size = img.stat().st_size
                # Grouper par tranches de 100KB
                size_bucket = size // (1024 * 100)
                size_groups[size_bucket].append(img)
        except Exception:
            continue
    
    # Ajouter aussi les images adjacentes
    all_groups = []
    processed_images = set()  # Pour eviter de traiter une image plusieurs fois
    
    for bucket in sorted(size_groups.keys()):
        current_group = []
        
        # Ajouter les images du bucket et des buckets adjacents
        for b in [bucket - 1, bucket, bucket + 1]:
            if b in size_groups:
                for img in size_groups[b]:
                    if img not in processed_images and img not in already_deleted:
                        current_group.append(img)
                        processed_images.add(img)
        
        if len(current_group) > 1:
            all_groups.append(current_group)
    
    logging.info(f"{len(images)} images dans {len(all_groups)} groupes potentiels")
    
    # Traiter les groupes
    for i, group in enumerate(all_groups, 1):
        # Filtrer les images deja supprimees
        group = [img for img in group if img not in already_deleted and img.exists()]
        
        if len(group) < 2:
            continue
            
        logging.info(f"Analyse groupe {i}/{len(all_groups)} ({len(group)} images)")
        
        # Limiter la taille des groupes
        if len(group) > MAX_GROUP_SIZE:
            # Traiter par sous-groupes
            for j in range(0, len(group), MAX_GROUP_SIZE):
                subgroup = group[j:j + MAX_GROUP_SIZE]
                duplicate_sets = finder.find_duplicates_in_group(subgroup)
                
                for dup_set in duplicate_sets:
                    # Filtrer les fichiers deja supprimes
                    dup_set = {img for img in dup_set if img not in already_deleted and img.exists()}
                    if len(dup_set) > 1:
                        if dry_run:
                            logging.info(f"[DRY-RUN] Trouverait {len(dup_set)} doublons")
                            for img in dup_set:
                                already_deleted.add(img)
                        else:
                            removed = finder.process_duplicate_group(dup_set)
                            total_removed += removed
                            # Marquer comme supprimes
                            for img in dup_set:
                                already_deleted.add(img)
        else:
            duplicate_sets = finder.find_duplicates_in_group(group)
            
            for dup_set in duplicate_sets:
                # Filtrer les fichiers deja supprimes
                dup_set = {img for img in dup_set if img not in already_deleted and img.exists()}
                if len(dup_set) > 1:
                    if dry_run:
                        logging.info(f"[DRY-RUN] Trouverait {len(dup_set)} doublons")
                        for img in dup_set:
                            already_deleted.add(img)
                    else:
                        removed = finder.process_duplicate_group(dup_set)
                        total_removed += removed
                        # Marquer comme supprimes
                        for img in dup_set:
                            already_deleted.add(img)
    
    # Strategie 2: Verification globale rapide par hash MD5
    logging.info("Verification finale des doublons exacts...")
    md5_groups = defaultdict(list)
    
    for img in images:
        if img in already_deleted:
            continue
        if img.exists():  # Verifier que l'image n'a pas ete supprimee
            hashes = cache.get_or_compute(img)
            if hashes and hashes.get('file_hash'):
                md5_groups[hashes['file_hash']].append(img)
    
    for md5_hash, group in md5_groups.items():
        group = [img for img in group if img not in already_deleted and img.exists()]
        if len(group) > 1:
            logging.info(f"Doublons exacts trouves: {len(group)} fichiers identiques")
            if dry_run:
                logging.info(f"[DRY-RUN] Supprimerait {len(group) - 1} doublons exacts")
            else:
                removed = finder.process_duplicate_group(set(group))
                total_removed += removed
    
    return total_removed

def main() -> None:
    parser = argparse.ArgumentParser(description="Detecteur de doublons d'images ameliore")
    parser.add_argument("--directory", type=Path, 
                       default=Path(r"T:\_SELECT\READY\TENCHI MUYO"),
                       help="Repertoire a analyser")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Efface le cache avant de commencer")
    parser.add_argument("--dry-run", action="store_true",
                       help="Mode simulation (ne supprime rien)")
    parser.add_argument("--threshold", type=int, default=PERCEPTUAL_THRESHOLD,
                       help=f"Seuil de difference pour hash perceptuel (defaut: {PERCEPTUAL_THRESHOLD})")
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    threshold = args.threshold
    
    start_time = datetime.now()
    
    # Effacer le cache si demande
    if args.clear_cache and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logging.info("Cache efface")
    
    # Scanner le repertoire
    images = scan_directory(args.directory)
    if not images:
        logging.warning("Aucune image trouvee")
        return
    
    logging.info(f"Trouve {len(images)} images a analyser")
    
    # Statistiques par dossier
    folder_stats = defaultdict(int)
    for img in images:
        folder_stats[img.parent.name] += 1
    
    logging.info("Distribution par dossier:")
    for folder, count in sorted(folder_stats.items())[:10]:
        logging.info(f"  {folder}: {count} images")
    
    # Charger le cache
    cache = EnhancedCache()
    
    # Dedupliquer
    if args.dry_run:
        logging.info("MODE SIMULATION - Aucune suppression reelle")
    
    removed = deduplicate_images(images, cache, threshold, args.dry_run)
    
    # Sauvegarder le cache
    cache.save()
    
    # Statistiques finales
    duration = (datetime.now() - start_time).total_seconds()
    logging.info(f"\n--- Resultats ---")
    logging.info(f"Images analysees: {len(images)}")
    logging.info(f"Doublons supprimes: {removed}")
    logging.info(f"Temps total: {duration:.1f}s")
    if duration > 0:
        logging.info(f"Vitesse: {len(images)/duration:.0f} images/seconde")

if __name__ == "__main__":
    main()