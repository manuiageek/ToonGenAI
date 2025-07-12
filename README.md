** Ce que fait ce repository **
Un projet APACHE HOP PROJECT qui fait : 
- UPLOAD_SHARING : Upload d'images (format png) en mode robot sur X et Deviantart avec des clés 
- CIVITAI_UTILITY : Gestion des téléchargements de LORA depuis civitai
- LORA_PREPARATION : Prépération à la réalisation de LORA (lowRank) à base d'anime

** UPLOAD_SHARING ** 
Automatise le renommage, l’archivage et la publication d’images IA sur DeviantArt et Twitter/X, via Apache Hop et des scripts Python.

*** Fonctionnalités principales *** :
- Renommage et archivage des images (pipelines Hop)
- Publication automatique sur DeviantArt et X/Twitter (API, scripts Python)
- Sélection facile des personnages à publier via un fichier Excel

*** Prérequis *** 
- Apache Hop
- Python 3 avec `tweepy`, `requests`, `flask`
- Clés API X et DeviantArt

*** Usage rapide *** 
1. Sélectionner les images à publier dans `XLSX/CHARACTER_SELECTOR.xlsx` (colonne TODO)
2. Lancer le workflow `_JOB_UPLOAD_XDART.hwf` via Hop


** It's still in progress **