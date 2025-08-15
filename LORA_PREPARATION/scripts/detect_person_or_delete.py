import os
import sys
import cv2
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from typing import Generator, Tuple, List, Optional
import logging
from collections import deque
import time

# Force unbuffered output pour conda
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # Line buffering

# Configuration logging minimal
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUAnimeDetectionConfig:
    """Configuration centralisée pour traitement GPU exclusif RTX 3070"""
    def __init__(self):
        self.device = self._validate_gpu()
        self.num_processes = 16  # RESTAURÉ pour RTX 3070
        self.batch_size = 20  # RESTAURÉ - Optimisé pour RTX 3070
        self.max_batch_size = 24
        self.target_size = (640, 640)
        self.model_path = Path(r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\yolov8x6_animeface.pt")
        self.anime_class_id = 0
        
        # Configuration streaming optimisée RTX 3070 - RESTAURÉE
        self.buffer_size = 96
        self.chunk_size = 64
        
    def _validate_gpu(self) -> str:
        """Validation de la disponibilité GPU"""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU non disponible. Ce script nécessite CUDA.")
        return 'cuda'

class StreamingGPUAnimeDetector:
    """Détecteur d'anime avec traitement par flux continu optimisé RTX 3070"""
    
    def __init__(self, config: GPUAnimeDetectionConfig):
        self.config = config
        self.stats = {
            'processed': 0,
            'deleted': 0,
            'kept': 0,
            'errors': 0,
            'total_files': 0
        }
        self.model = self._load_model()
        self._optimize_gpu()
        self.last_update_time = time.time()
        
    def _load_model(self) -> YOLO:
        """Chargement et optimisation du modèle pour GPU"""
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.config.model_path}")
        
        model = YOLO(str(self.config.model_path))
        model.to(self.config.device)
        model.model.eval()
        
        return model
    
    def _optimize_gpu(self):
        """Configuration optimale pour GPU RTX 3070 - RESTAURÉE"""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        self.cuda_stream = torch.cuda.Stream()  # RESTAURÉ

    def load_and_resize_image(self, image_path: str) -> Tuple[Optional[np.ndarray], str]:
        """Chargement optimisé pour pipeline GPU"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None, image_path
            
            img_resized = cv2.resize(img, self.config.target_size, interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            return img_rgb, image_path
            
        except Exception as e:
            return None, image_path

    def process_batch_gpu(self, images_batch: List[np.ndarray], paths_batch: List[str]) -> None:
        """Traitement GPU optimisé RTX 3070 avec affichage temps réel"""
        try:
            # Utilisation du CUDA stream pour performances optimales - RESTAURÉ
            with torch.cuda.stream(self.cuda_stream):
                with torch.amp.autocast('cuda', enabled=True):
                    results = self.model(
                        images_batch, 
                        verbose=False, 
                        device=self.config.device
                    )

            torch.cuda.current_stream().wait_stream(self.cuda_stream)
            
            # Traitement des résultats avec affichage forcé
            self._process_results_realtime(results, paths_batch)
            
            # Nettoyage mémoire conservateur pour RTX 3070 - RESTAURÉ
            if torch.cuda.memory_allocated() > 6e9:
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError:
            self._handle_oom_batch(images_batch, paths_batch)
        except Exception as e:
            logger.error(f"Erreur batch GPU: {e}")

    def _process_results_realtime(self, results, paths_batch: List[str]) -> None:
        """Traitement avec affichage forcé immédiat"""
        for i, result in enumerate(results):
            try:
                anime_detected = False
                
                if result.boxes is not None and len(result.boxes) > 0:
                    classes_tensor = result.boxes.cls
                    if classes_tensor is not None:
                        anime_mask = classes_tensor == self.config.anime_class_id
                        anime_detected = anime_mask.any().item()

                if not anime_detected:
                    # Suppression et affichage FORCÉ
                    self._safe_delete_image(paths_batch[i])
                    
                    # Message simple avec flush forcé
                    print(f"SUPPRIME: {paths_batch[i]} (aucune personne detectee)", flush=True)
                    sys.stdout.flush()  # Double flush pour conda
                    
                    self.stats['deleted'] += 1
                else:
                    self.stats['kept'] += 1

                self.stats['processed'] += 1
                
                # Affichage progression toutes les 100 images
                if self.stats['processed'] % 100 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.last_update_time
                    speed = 100 / elapsed if elapsed > 0 else 0
                    self.last_update_time = current_time
                    
                    progress_pct = (self.stats['processed'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
                    print(f"--- Progression: {self.stats['processed']}/{self.stats['total_files']} ({progress_pct:.1f}%) | Vitesse: {speed:.1f} img/s | Supprimees: {self.stats['deleted']} ---", flush=True)
                    sys.stdout.flush()
                    
            except Exception as e:
                self.stats['errors'] += 1

    def _handle_oom_batch(self, images_batch: List[np.ndarray], paths_batch: List[str]) -> None:
        """Gestion OOM adaptée RTX 3070 - RESTAURÉE"""
        torch.cuda.empty_cache()
        reduced_size = max(1, len(images_batch) // 2)
        
        print("Memoire GPU insuffisante, reduction temporaire du batch...", flush=True)
        
        for i in range(0, len(images_batch), reduced_size):
            end_idx = min(i + reduced_size, len(images_batch))
            mini_batch_images = images_batch[i:end_idx]
            mini_batch_paths = paths_batch[i:end_idx]
            
            try:
                with torch.amp.autocast('cuda'):
                    results = self.model(mini_batch_images, verbose=False, device=self.config.device)
                self._process_results_realtime(results, mini_batch_paths)
            except Exception as e:
                logger.error(f"Erreur mini-batch: {e}")

    def _safe_delete_image(self, image_path: str) -> None:
        """Suppression sécurisée"""
        try:
            os.remove(image_path)
        except Exception as e:
            logger.error(f"Erreur suppression {image_path}: {e}")

    def get_streaming_image_generator(self, directory: str) -> Generator[Tuple[np.ndarray, str], None, None]:
        """Générateur streaming optimisé RTX 3070 - RESTAURÉ"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        # Comptage rapide
        print("Analyse du repertoire...", flush=True)
        total_files = sum(
            1 for root, _, files in os.walk(directory)
            for file in files
            if Path(file).suffix.lower() in image_extensions
        )
        self.stats['total_files'] = total_files
        print(f"{total_files} images trouvees", flush=True)
        print("-" * 50, flush=True)
        sys.stdout.flush()
        
        # Traitement par chunks streaming - RESTAURÉ
        for root, _, files in os.walk(directory):
            image_files = [
                os.path.join(root, file) for file in files
                if Path(file).suffix.lower() in image_extensions
            ]
            
            if not image_files:
                continue
            
            # Traiter le dossier par chunks optimisés
            for i in range(0, len(image_files), self.config.chunk_size):
                chunk_files = image_files[i:i + self.config.chunk_size]
                
                # Chargement parallèle du chunk - RESTAURÉ
                with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
                    future_to_path = {
                        executor.submit(self.load_and_resize_image, path): path 
                        for path in chunk_files
                    }
                    
                    # Buffer pour accumuler les images chargées - RESTAURÉ
                    buffer = deque(maxlen=self.config.buffer_size)
                    
                    for future in as_completed(future_to_path):
                        img_resized, image_path = future.result()
                        if img_resized is not None:
                            buffer.append((img_resized, image_path))
                            
                            # Yield quand le buffer est plein - RESTAURÉ
                            if len(buffer) >= self.config.buffer_size // 2:
                                while buffer:
                                    yield buffer.popleft()
                    
                    # Vider le buffer restant
                    while buffer:
                        yield buffer.popleft()

def main():
    """Fonction principale avec streaming optimisé RTX 3070"""
    parser = argparse.ArgumentParser(
        description="Detection anime GPU streaming RTX 3070"
    )
    parser.add_argument(
        "--directory", 
        type=str, 
        default=r"D:\ISEKAI MAOU\_temp2\07", 
        help="Repertoire a traiter"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Forcer une taille de batch (defaut: 20 pour RTX 3070)"
    )
    args = parser.parse_args()

    print("=" * 50, flush=True)
    print("DETECTION ANIME - NETTOYAGE AUTOMATIQUE", flush=True)
    print("=" * 50, flush=True)
    print(f"Repertoire: {args.directory}", flush=True)
    sys.stdout.flush()

    if not os.path.exists(args.directory):
        print(f"ERREUR: Repertoire introuvable: {args.directory}", flush=True)
        return

    try:
        config = GPUAnimeDetectionConfig()
        
        if args.batch_size:
            config.batch_size = args.batch_size
            print(f"Taille de batch forcee a: {config.batch_size}", flush=True)
        else:
            print(f"Batch optimise RTX 3070: {config.batch_size} images", flush=True)
        
        detector = StreamingGPUAnimeDetector(config)
        
        # Pipeline streaming optimisé
        images_batch = []
        paths_batch = []
        
        start_time = datetime.now()
        detector.last_update_time = time.time()
        
        # Traitement en flux continu avec optimisations RTX 3070
        for img, image_path in detector.get_streaming_image_generator(args.directory):
            images_batch.append(img)
            paths_batch.append(image_path)
            
            # Traiter dès qu'on a un batch complet
            if len(images_batch) == config.batch_size:
                detector.process_batch_gpu(images_batch, paths_batch)
                images_batch = []
                paths_batch = []
        
        # Dernier batch partiel
        if images_batch:
            detector.process_batch_gpu(images_batch, paths_batch)
    
    except KeyboardInterrupt:
        print("\nArret demande par l'utilisateur", flush=True)
    except Exception as e:
        print(f"ERREUR FATALE: {e}", flush=True)
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if 'detector' in locals():
            stats = detector.stats
            speed = stats['processed'] / duration.total_seconds() if duration.total_seconds() > 0 else 0
            
            # Format temps propre
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Résumé final
            print("=" * 50, flush=True)
            print("TRAITEMENT TERMINE", flush=True)
            print(f"Duree totale: {time_str}", flush=True)
            print(f"Images traitees: {stats['processed']}/{stats['total_files']}", flush=True)
            print(f"Images supprimees: {stats['deleted']}", flush=True)
            print(f"Images conservees: {stats['kept']}", flush=True)
            print(f"Vitesse moyenne: {speed:.1f} img/s", flush=True)
            
            if stats['errors'] > 0:
                print(f"Erreurs rencontrees: {stats['errors']}", flush=True)

if __name__ == '__main__':
    main()