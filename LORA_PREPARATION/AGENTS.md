# Repository Guidelines

## Project Structure & Modules
- `scripts/`: Python tools for LORA data prep.
  - `delete_duplicate_images.py`: de-duplicates images using MD5 + perceptual hashes and a compressed cache (`scripts/image_hashes_cache.lz4`).
  - `detect_person_or_delete.py`: GPU YOLO-based anime/person detector; deletes non-detections.
- `scripts/models/`: Model weights (e.g., `yolov8x6_animeface.pt`).
- `environments/`: JSON presets for runtime/config.
- `*.hpl`, `*.hwf`: Workflow definitions orchestrating the preparation pipeline.
- `todo.txt`: High-level tasks and notes.

## Setup, Run, and Test
- Python 3.10+ recommended. CUDA GPU required for detection.
- Create venv and install deps:
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
  - `pip install pillow imagehash ultralytics torch opencv-python lz4`
- Duplicate finder (start with dry run):
  - `python scripts/delete_duplicate_images.py --directory "D:\path\to\images" --dry-run`
  - Then rerun without `--dry-run` to actually delete.
- Anime/person detector (GPU):
  - `python scripts/detect_person_or_delete.py --directory "D:\path\to\images" --batch-size 20`

## Coding Style & Naming
- Python, PEP 8, 4-space indentation; prefer type hints.
- Use `logging` (INFO/WARNING/ERROR) over prints; keep messages concise.
- Snake_case for modules/functions; descriptive names for configs and weights.
- Avoid hardcoded absolute paths; expose via CLI args or config.

## Testing Guidelines
- No formal tests yet. Validate on a small folder before full runs.
- Use `--dry-run` for duplicates; confirm logs, then apply.
- Detector prints throughput and counts at end; ensure CUDA is available.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
- PRs include: clear description, sample commands, linked issue, and log snippets or screenshots when behavior changes.

## Safety & Configuration Tips
- Back up input data; both scripts can permanently delete files.
- Clear cache when needed: `--clear-cache` for the duplicate finder.
- Keep model weights in `scripts/models/`; avoid committing large binaries unless required.
