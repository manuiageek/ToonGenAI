import os
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

# Configuration logging minimal
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUAnimeDetectionConfig:
    """Configuration centralisée pour traitement GPU exclusif"""
    def __init__(self):
        self.device = self._validate_gpu()
        self.num_processes = 16
        self.batch_size = 20  # Optimisé pour RTX 3070
        self.max_batch_size = 24
        self.target_size = (640, 640)
        self.model_path =  Path(r"E:\_DEV\ToonGenAI\LORA_PREPARATION\scripts\models\yolov8x6_animeface.pt")
        self.anime_class_id = 0
        
        # Configuration streaming optimisée RTX 3070
        self.buffer_size = 96
        self.chunk_size = 64
        
    def _validate_gpu(self) -> str:
        """Validation de la disponibilité GPU"""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU non disponible. Ce script nécessite CUDA.")
        
        return 'cuda'

class StreamingGPUAnimeDetector:
    """Détecteur d'anime avec traitement par flux continu"""
    
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
        
    def _load_model(self) -> YOLO:
        """Chargement et optimisation du modèle pour GPU"""
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.config.model_path}")
        
        model = YOLO(str(self.config.model_path))
        model.to(self.config.device)
        model.model.eval()
        
        return model
    
    def _optimize_gpu(self):
        """Configuration optimale pour GPU"""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        self.cuda_stream = torch.cuda.Stream()

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
        """Traitement GPU silencieux"""
        try:
            with torch.cuda.stream(self.cuda_stream):
                with torch.amp.autocast('cuda', enabled=True):
                    results = self.model(
                        images_batch, 
                        verbose=False, 
                        device=self.config.device
                    )

            torch.cuda.current_stream().wait_stream(self.cuda_stream)
            self._process_results_vectorized(results, paths_batch)
            
            # Nettoyage mémoire conservateur pour RTX 3070
            if torch.cuda.memory_allocated() > 6e9:
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError:
            self._handle_oom_batch(images_batch, paths_batch)
        except Exception as e:
            logger.error(f"Erreur batch GPU: {e}")

    def _process_results_vectorized(self, results, paths_batch: List[str]) -> None:
        """Traitement vectorisé silencieux"""
        for i, result in enumerate(results):
            try:
                anime_detected = False
                
                if result.boxes is not None and len(result.boxes) > 0:
                    classes_tensor = result.boxes.cls
                    if classes_tensor is not None:
                        anime_mask = classes_tensor == self.config.anime_class_id
                        anime_detected = anime_mask.any().item()

                if not anime_detected:
                    self._safe_delete_image(paths_batch[i])
                    self.stats['deleted'] += 1
                else:
                    self.stats['kept'] += 1
                
                self.stats['processed'] += 1
                    
            except Exception as e:
                self.stats['errors'] += 1

    def _handle_oom_batch(self, images_batch: List[np.ndarray], paths_batch: List[str]) -> None:
        """Gestion OOM adaptée RTX 3070"""
        torch.cuda.empty_cache()
        reduced_size = max(1, len(images_batch) // 2)
        
        for i in range(0, len(images_batch), reduced_size):
            end_idx = min(i + reduced_size, len(images_batch))
            mini_batch_images = images_batch[i:end_idx]
            mini_batch_paths = paths_batch[i:end_idx]
            
            try:
                with torch.amp.autocast('cuda'):
                    results = self.model(mini_batch_images, verbose=False, device=self.config.device)
                self._process_results_vectorized(results, mini_batch_paths)
            except Exception as e:
                logger.error(f"Erreur mini-batch: {e}")

    def _safe_delete_image(self, image_path: str) -> None:
        """Suppression sécurisée"""
        try:
            os.remove(image_path)
        except Exception as e:
            logger.error(f"Erreur suppression {image_path}: {e}")

    def get_streaming_image_generator(self, directory: str) -> Generator[Tuple[np.ndarray, str], None, None]:
        """Générateur streaming silencieux"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        # Comptage rapide
        total_files = sum(
            1 for root, _, files in os.walk(directory)
            for file in files
            if Path(file).suffix.lower() in image_extensions
        )
        self.stats['total_files'] = total_files
        
        # Traitement par chunks streaming
        for root, _, files in os.walk(directory):
            image_files = [
                os.path.join(root, file) for file in files
                if Path(file).suffix.lower() in image_extensions
            ]
            
            if not image_files:
                continue
            
            # Traiter le dossier par chunks
            for i in range(0, len(image_files), self.config.chunk_size):
                chunk_files = image_files[i:i + self.config.chunk_size]
                
                # Chargement parallèle du chunk
                with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
                    future_to_path = {
                        executor.submit(self.load_and_resize_image, path): path 
                        for path in chunk_files
                    }
                    
                    # Buffer pour accumuler les images chargées
                    buffer = deque(maxlen=self.config.buffer_size)
                    
                    for future in as_completed(future_to_path):
                        img_resized, image_path = future.result()
                        if img_resized is not None:
                            buffer.append((img_resized, image_path))
                            
                            # Yield quand le buffer est plein
                            if len(buffer) >= self.config.buffer_size // 2:
                                while buffer:
                                    yield buffer.popleft()
                    
                    # Vider le buffer restant
                    while buffer:
                        yield buffer.popleft()

def main():
    """Fonction principale streaming"""
    parser = argparse.ArgumentParser(
        description="Détection anime GPU streaming RTX 3070"
    )
    parser.add_argument(
        "--directory", 
        type=str, 
        default=r"D:\ISEKAI MAOU\_temp2\07", 
        help="Répertoire à traiter"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Forcer une taille de batch"
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        logger.error(f"Répertoire introuvable: {args.directory}")
        return

    try:
        config = GPUAnimeDetectionConfig()
        
        if args.batch_size:
            config.batch_size = args.batch_size
        
        detector = StreamingGPUAnimeDetector(config)
        
        # Pipeline streaming
        images_batch = []
        paths_batch = []
        
        start_time = datetime.now()
        
        # Traitement en flux continu
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
        print("\nArrêt demandé")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if hasattr(detector, 'stats'):
            stats = detector.stats
            speed = stats['processed'] / duration.total_seconds() if duration.total_seconds() > 0 else 0
            
            # Format temps propre
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            print(f"\n\nTIME {time_str}")
            print(f"IMAGES DONE: {stats['processed']}/{stats['total_files']}")
            print(f"DELETED : {stats['deleted']}")
            print(f"SPEED: {speed:.1f} img/s")

if __name__ == '__main__':
    main()