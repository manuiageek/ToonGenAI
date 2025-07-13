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

---

## Prérequis
- Apache Hop
- Python 3 avec les librairies : `tweepy`, `requests`, `flask`
- Clés API X/Twitter et DeviantArt

---
## Usage rapide
1. Sélectionne les images à publier dans `XLSX/CHARACTER_SELECTOR.xlsx` (colonne TODO)
2. Lance le workflow `_JOB_UPLOAD_XDART.hwf` via Hop

---

> **Note**: Le projet est encore en cours de développement.
