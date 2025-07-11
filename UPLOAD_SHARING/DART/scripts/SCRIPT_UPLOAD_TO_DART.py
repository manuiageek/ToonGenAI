from __future__ import annotations
from flask import Flask, request
import requests, os, sys, shutil, time, json, socket, threading
from pathlib import Path
import webbrowser

app = Flask(__name__)

# -----------------------------------------------------------
# UTILITAIRES DE LOG
# -----------------------------------------------------------
def info(msg: str)  -> None: print(f"[INFO ] {msg}")
def warn(msg: str)  -> None: print(f"[WARN ] {msg}")
def error(msg: str) -> None: print(f"[ERREUR] {msg}")

# -----------------------------------------------------------
# OUTIL D’OUVERTURE DU NAVIGATEUR
# -----------------------------------------------------------
FIREFOX_EXE = Path(r"C:\Program Files\Mozilla Firefox\firefox.exe")

def open_in_firefox(url: str) -> None:
    """
    Ouvre l’URL dans Firefox si disponible ; sinon utilise le navigateur par défaut.
    """
    if FIREFOX_EXE.exists():
        # Enregistrement d’un “controller” webbrowser nommé « firefox »
        webbrowser.register(
            name='firefox',
            klass=webbrowser.BackgroundBrowser,
            instance=webbrowser.BackgroundBrowser(str(FIREFOX_EXE))
        )
        info("Ouverture de l’URL dans Firefox…")
        webbrowser.get('firefox').open(url)
    else:
        warn(f"Firefox introuvable à {FIREFOX_EXE}, ouverture dans le navigateur par défaut.")
        webbrowser.open(url)

# -----------------------------------------------------------
# PARSING CLI
# -----------------------------------------------------------
def get_cli_params() -> dict[str, str]:
    try:
        return {
            "title_info"              : sys.argv[1],
            "gallery_id"              : sys.argv[2],
            "client_id"               : sys.argv[3],
            "client_secret"           : sys.argv[4],
            "redirect_uri"            : sys.argv[5],
            "directory_file_to_upload": os.path.normpath(sys.argv[6]),
        }
    except IndexError:
        error("Paramètres manquants.")
        print("Utilisation : python SCRIPT_UPLOAD_TO_DART.py "
              "<title_info> <gallery_id> <client_id> <client_secret> "
              "<redirect_uri> <directory>")
        sys.exit(1)

cli = get_cli_params()

# -----------------------------------------------------------
# VALIDATIONS PRÉ-LANCEMENT
# -----------------------------------------------------------
def check_internet(host="www.deviantart.com", port=443, timeout=3) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False

def check_directory(path: str) -> tuple[bool, list[str]]:
    if not os.path.isdir(path):
        return False, []
    pngs = [f for f in os.listdir(path) if f.lower().endswith(".png")]
    return bool(pngs), pngs

if not check_internet():
    error("Pas d’accès réseau ou DeviantArt inaccessible.")
    sys.exit(1)

ok_dir, png_list = check_directory(cli["directory_file_to_upload"])
if not ok_dir:
    error(f"Aucune image .png trouvée dans {cli['directory_file_to_upload']}")
    sys.exit(1)
info(f"{len(png_list)} image(s) trouvée(s) à uploader.")

# -----------------------------------------------------------
# FONCTIONS API
# -----------------------------------------------------------
def get_access_token(client_id, client_secret, code, redirect_uri):
    info("Échange du code contre un access_token…")
    url = "https://www.deviantart.com/oauth2/token"
    payload = dict(
        client_id=client_id, client_secret=client_secret,
        grant_type="authorization_code", code=code, redirect_uri=redirect_uri
    )
    try:
        r = requests.post(url, data=payload, timeout=10)
    except requests.RequestException as exc:
        error(f"Requête token échouée : {exc}")
        return None
    if r.ok:
        return r.json().get("access_token")
    error(f"Token non obtenu (status {r.status_code}) : {r.text}")
    return None

def submit_to_stash(access_token, image_path, title, tags):
    url = "https://www.deviantart.com/api/v1/oauth2/stash/submit"
    headers = {"Authorization": f"Bearer {access_token}"}
    with open(image_path, "rb") as fp:
        files = {"image": fp}
        data  = {
            "title": title,
            **{f"tags[{i}]": t for i, t in enumerate(tags[:3])}
        }
        try:
            r = requests.post(url, headers=headers, files=files, data=data, timeout=20)
        except requests.RequestException as exc:
            error(f"submit_to_stash KO : {exc}")
            return None
    if r.ok:
        return r.json().get("itemid")
    warn(f"submit_to_stash NOK ({r.status_code}) : {r.text}")
    return None

def publish_from_stash(access_token, itemid, gallery_ids):
    url = "https://www.deviantart.com/api/v1/oauth2/stash/publish"
    headers = {"Authorization": f"Bearer {access_token}"}
    data = {
        "validate_token": access_token,
        "itemid": itemid,
        **{f"galleryids[{i}]": gid for i, gid in enumerate(gallery_ids)},
        "is_mature": "true",
        "mature_level": "moderate",
        "mature_classification": "sexual",
        "agree_submission": "true",
        "agree_tos": "true",
    }
    try:
        r = requests.post(url, headers=headers, data=data, timeout=20)
    except requests.RequestException as exc:
        error(f"publish_from_stash KO : {exc}")
        return None
    if r.ok:
        return r.json()
    warn(f"publish_from_stash NOK ({r.status_code}) : {r.text}")
    return None

# -----------------------------------------------------------
# FLASK
# -----------------------------------------------------------
def run_flask():
    info("Serveur Flask en écoute sur 5088…")
    app.run(port=5088, debug=False, use_reloader=False)

@app.route("/")
def receive_code():
    code = request.args.get("code")
    info(f"Code OAuth reçu : {code}")
    if not code:
        return "Pas de code ?!"
    access_token = get_access_token(
        cli["client_id"], cli["client_secret"], code, cli["redirect_uri"]
    )
    if not access_token:
        return "Token non obtenu."

    title      = cli["title_info"]
    tags       = [t[1:] for t in title.split() if t.startswith("#")]
    galleries  = ["1551DEF0-436F-4D02-FA1A-9018FB8737C9", cli["gallery_id"]]

    upload_dir = cli["directory_file_to_upload"]
    done_dir   = os.path.join(upload_dir, "done")
    os.makedirs(done_dir, exist_ok=True)

    sent, published = 0, 0
    for png in png_list:
        path = os.path.join(upload_dir, png)
        info(f"→ Upload {png}")
        itemid = submit_to_stash(access_token, path, title, tags)
        if not itemid:
            continue
        sent += 1
        if publish_from_stash(access_token, itemid, galleries):
            published += 1
            shutil.move(path, os.path.join(done_dir, png))

    info(f"Images soumises : {sent} / publiées : {published}")
    threading.Thread(target=shutdown_delayed, daemon=True).start()
    return "Opération terminée, vous pouvez fermer cet onglet."

def shutdown_delayed():
    time.sleep(5)
    info("Arrêt du serveur Flask.")
    os._exit(0)

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    info("Lancement du navigateur pour OAuth…")
    auth_url = (
        "https://www.deviantart.com/oauth2/authorize"
            f"?response_type=code&client_id={cli['client_id']}"
        f"&redirect_uri={cli['redirect_uri']}"
    )
    open_in_firefox(auth_url)      # ← ouverture via Firefox
    run_flask()                    # Flask reste sur le thread principal
