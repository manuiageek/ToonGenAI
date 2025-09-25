# APACHE HOP PROJECT

Projet qui automatise la gestion et la publication d’images IA sur DeviantArt et X/Twitter, avec utilitaires pour Civitai et préparation LORA/LowRank.

## Fonctionnalités

### UPLOAD_SHARING
- Renommage et archivage des images (via pipelines Hop)
- Publication automatique sur DeviantArt et X/Twitter (API, scripts Python)
- Sélection facile des personnages à publier via un fichier Excel

### CIVITAI_UTILITY
- Gestion des téléchargements de LORA depuis Civitai

Exemple pour ce lien : 
https://civitai.com/models/1201834

sauvegarder sous : CF.SkeletonKnight-ILXL_Ariane.1201834.safetensors

<lora_kind>.<anime_show>-<ILXL_or_whatever>_<name_of_char>.<url_number>.safetensor

### LORA_PREPARATION
- Préparation à la création de LORA (LowRank) à base d’anime

## Prérequis
- Apache Hop
- Python 3 avec les librairies : `tweepy`, `requests`, `flask`
- Clés API X/Twitter et DeviantArt


## REQUIREMENTS :

### PYTORCH :

For gpu acceleration, use miniconda : https://docs.anaconda.com/miniconda/
PYTORCH CONDA GUIDE :
conda create -n ptorch_env python=3.9
conda activate ptorch_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy opencv-python ultralytics

Exemple : conda run -n ptorch_env python 4_PT_detect_person_or_delete.py

### TENSORFLOW :

conda create -n tflow_env python=3.10
conda activate tflow_env
conda install cudatoolkit=11.8 cudnn=8.6 -c conda-forge
pip install tensorflow==2.10.0
pip install numpy pillow tensorflow-io deepdanbooru psutil

check TensorFlow GPU :
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### ONNX :
conda create -n onnx_env python=3.10
conda activate onnx_env
conda install -c nvidia/label/cuda-12.6.0 cuda-toolkit
conda install -c conda-forge cudnn=9.3.1
pip install onnxruntime-gpu
pip install --upgrade "onnxruntime-gpu[cuda,cudnn]"

pip install -U "openai>=1.50.0"

Exemple : conda run -n onnx_env --no-capture-output python -u .\anime_autotagger_gpu.py

conda create -n onnx_env python=3.10
conda activate onnx_env
conda install -c nvidia/label/cuda-12.6.0 cuda-toolkit
conda install -c conda-forge cudnn=9.3.1
pip install onnxruntime-gpu
pip install --upgrade "onnxruntime-gpu[cuda,cudnn]"

pip install -U "openai>=1.50.0"

Exemple : conda run -n onnx_env --no-capture-output python -u .\anime_autotagger_gpu.py

### NODEJS :

package.json already exists :
npm install

## DeepDanbooru link :

https://github.com/KichangKim/DeepDanbooru/releases
https://huggingface.co/datasets/nyanko7/danbooru2023

