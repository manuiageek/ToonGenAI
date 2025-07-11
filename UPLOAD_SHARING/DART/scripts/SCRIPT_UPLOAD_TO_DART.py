from __future__ import annotations
from flask import Flask, request
import requests, os, sys, shutil, time, json, socket, threading, traceback, textwrap
from pathlib import Path
import webbrowser
from werkzeug.serving import make_server

app = Flask(__name__)

# -----------------------------------------------------------
# UTILITAIRES DE LOG - COMPATIBLE ASCII/CP1252
# -----------------------------------------------------------
def info(msg: str)  -> None: print(f"[INFO ] {msg}")
def warn(msg: str)  -> None: print(f"[WARN ] {msg}")
def error(msg: str) -> None: print(f"[ERREUR] {msg}")

# -----------------------------------------------------------
# OUVERTURE OPTIONNELLE DU NAVIGATEUR (NON UTILISÉ EN MODE HEADLESS)
# -----------------------------------------------------------
FIREFOX_EXE = Path(r"C:\Program Files\Mozilla Firefox\firefox.exe")

def open_in_firefox(url: str) -> None:
    """
    Ouvre l'URL dans Firefox si disponible ; sinon utilise le navigateur par défaut.
    N'EST PLUS APPELÉ par défaut → exécution « headless ».
    """
    try:
        if FIREFOX_EXE.exists():
            webbrowser.register(
                name='firefox',
                klass=webbrowser.BackgroundBrowser,
                instance=webbrowser.BackgroundBrowser(str(FIREFOX_EXE))
            )
            info("Ouverture de l'URL dans Firefox...")
            webbrowser.get('firefox').open(url)
        else:
            warn(f"Firefox introuvable a {FIREFOX_EXE}, ouverture dans le navigateur par defaut.")
            webbrowser.open(url)
    except Exception as e:
        warn(f"Impossible d'ouvrir le navigateur : {e}")

# -----------------------------------------------------------
# PARSING CLI ET VARIABLES D'ENVIRONNEMENT
# -----------------------------------------------------------
def get_cli_params() -> dict[str, str]:
    """
    Récupère les paramètres soit depuis sys.argv soit depuis les variables d'environnement.
    Apache Hop peut passer les variables via l'environnement plutôt que par argv.
    """
    # D'abord, essayer les variables d'environnement (plus fiable avec Hop)
    env_params = {
        "title_info": os.environ.get("DART_TITLE"),
        "gallery_id": os.environ.get("DART_GALLERY_ID"),
        "client_id": os.environ.get("DART_CLIENT_ID"),
        "client_secret": os.environ.get("DART_CLIENT_SECRET"),
        "redirect_uri": os.environ.get("DART_REDIRECT_URI"),
        "directory_file_to_upload": os.environ.get("DART_UPLOAD_DIR"),
    }
    
    # Si toutes les variables d'environnement sont présentes, les utiliser
    if all(env_params.values()):
        info("Parametres recuperes depuis les variables d'environnement")
        env_params["directory_file_to_upload"] = os.path.normpath(env_params["directory_file_to_upload"])
        return env_params
    
    # Sinon essayer sys.argv
    try:
        cli_params = {
            "title_info"              : sys.argv[1],
            "gallery_id"              : sys.argv[2],
            "client_id"               : sys.argv[3],
            "client_secret"           : sys.argv[4],
            "redirect_uri"            : sys.argv[5],
            "directory_file_to_upload": os.path.normpath(sys.argv[6]),
        }
        info("Parametres recuperes depuis sys.argv")
        return cli_params
    except IndexError:
        error("Parametres manquants dans sys.argv et variables d'environnement incompletes.")
        info("Variables d'environnement disponibles :")
        for key, value in env_params.items():
            info(f"  {key}: {value if value else '(manquant)'}")
        info("Utilisation en ligne de commande :")
        print("python SCRIPT_UPLOAD_TO_DART.py "
              "<title_info> <gallery_id> <client_id> <client_secret> "
              "<redirect_uri> <directory>")
        info("Ou definir les variables d'environnement :")
        print("DART_TITLE, DART_GALLERY_ID, DART_CLIENT_ID, DART_CLIENT_SECRET, DART_REDIRECT_URI, DART_UPLOAD_DIR")
        sys.exit(1)

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

# -----------------------------------------------------------
# FONCTIONS API
# -----------------------------------------------------------
def get_access_token(client_id, client_secret, code, redirect_uri):
    info("Echange du code contre un access_token...")
    url = "https://www.deviantart.com/oauth2/token"
    payload = dict(
        client_id=client_id, client_secret=client_secret,
        grant_type="authorization_code", code=code, redirect_uri=redirect_uri
    )
    try:
        r = requests.post(url, data=payload, timeout=10)
    except requests.RequestException as exc:
        error(f"Requete token echouee : {exc}")
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
# VARIABLES GLOBALES POUR PARTAGER LES PARAMÈTRES
# -----------------------------------------------------------
cli = None

# -----------------------------------------------------------
# ROUTE PRINCIPALE
# -----------------------------------------------------------
@app.route("/")
def receive_code():
    global cli
    try:
        code = request.args.get("code")
        info(f"Code OAuth recu : {code}")
        if not code:
            return "<h1>Erreur</h1><p>Pas de code OAuth recu</p>", 400

        if not cli:
            return "<h1>Erreur</h1><p>Configuration non disponible</p>", 500

        access_token = get_access_token(
            cli["client_id"], cli["client_secret"], code, cli["redirect_uri"]
        )
        if not access_token:
            return "<h1>Erreur</h1><p>Token non obtenu</p>", 500

        title      = cli["title_info"]
        tags       = [t[1:] for t in title.split() if t.startswith("#")]
        galleries  = ["1551DEF0-436F-4D02-FA1A-9018FB8737C9", cli["gallery_id"]]

        upload_dir = cli["directory_file_to_upload"]
        
        # Vérifier que le répertoire existe
        if not os.path.isdir(upload_dir):
            return f"<h1>Erreur</h1><p>Repertoire non trouve : {upload_dir}</p>", 500
            
        # Vérifier les images
        ok_dir, png_list = check_directory(upload_dir)
        if not ok_dir:
            return f"<h1>Erreur</h1><p>Aucune image .png trouvee dans {upload_dir}</p>", 500

        done_dir   = os.path.join(upload_dir, "done")
        os.makedirs(done_dir, exist_ok=True)

        sent, published = 0, 0
        results = []
        
        for png in png_list:
            path = os.path.join(upload_dir, png)
            info(f"-> Upload {png}")  # Remplacement de → par ->
            results.append(f"Upload de {png}...")
            itemid = submit_to_stash(access_token, path, title, tags)
            if not itemid:
                results.append(f"  X Echec upload {png}")  # Remplacement de ✗ par X
                continue
            sent += 1
            results.append(f"  √ Upload reussi, publication...")  # Remplacement de ✓ par √
            
            if publish_from_stash(access_token, itemid, galleries):
                published += 1
                shutil.move(path, os.path.join(done_dir, png))
                results.append(f"  √ Publication reussie pour {png}")
            else:
                results.append(f"  X Echec publication {png}")

        info(f"Images soumises : {sent} / publiees : {published}")
        
        # Préparer la réponse HTML
        results_html = "<br>".join(results)
        response_html = f"""
        <html>
        <head>
            <title>Upload termine</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Operation terminee</h1>
            <p><strong>Images soumises :</strong> {sent}</p>
            <p><strong>Images publiees :</strong> {published}</p>
            <h2>Details :</h2>
            <div style="font-family: monospace; background: #f5f5f5; padding: 10px;">
                {results_html}
            </div>
            <p><em>Vous pouvez fermer cet onglet.</em></p>
        </body>
        </html>
        """

        # Arrêt différé du serveur
        shutdown_func = request.environ.get("werkzeug.server.shutdown")

        def delayed_shutdown(delay: int = 5) -> None:
            time.sleep(delay)
            if callable(shutdown_func):
                info("[DEBUG] Arret differe du serveur Flask...")
                shutdown_func()
            else:
                warn("Fonction shutdown introuvable, arret brutal.")
                os._exit(0)

        threading.Thread(target=delayed_shutdown, daemon=True).start()
        return response_html
    except Exception as e:
        error(f"Erreur dans receive_code : {e}")
        traceback.print_exc()
        return f"<h1>Erreur</h1><pre>{traceback.format_exc()}</pre>", 500

@app.route("/stop")
def stop_flask_app() -> str:
    """
    Route d'arrêt : tente d'abord un shutdown propre, sinon exit brutal.
    """
    info("[DEBUG] Route /stop appelee, arret de l'application")
    shutdown = request.environ.get("werkzeug.server.shutdown")
    if callable(shutdown):
        shutdown()
        return "Serveur arrete proprement."
    warn("werkzeug.server.shutdown indisponible, arret brutal.")
    os._exit(0)

@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    """
    Intercepte toute exception Flask non capturée.
    """
    tb = traceback.format_exc()
    error("=== ERREUR NON GEREE ===\n" + tb)
    formatted = tb.replace('\n', '<br>').replace(' ', '&nbsp;')
    return f"""
    <html>
    <head>
        <title>Erreur interne</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>Erreur interne du serveur</h1>
        <h2>Details de l'erreur :</h2>
        <div style="font-family: monospace; background: #ffeeee; padding: 10px; border: 1px solid #ff0000;">
            {formatted}
        </div>
    </body>
    </html>
    """, 500

# -----------------------------------------------------------
# LANCEMENT / ARRÊT DU SERVEUR FLASK
# -----------------------------------------------------------
def run_flask_app(port: int = 5088) -> None:
    """
    Lance l'application Flask sur le port spécifié.
    """
    info(f"[DEBUG] Lancement de l'application Flask sur le port {port}")
    try:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except Exception as e:
        error(f"Erreur lors du lancement de Flask : {e}")
        traceback.print_exc()

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    global cli

    # Récupérer les paramètres
    try:
        cli = get_cli_params()
        info("Configuration recuperee avec succes")
        for key, value in cli.items():
            if 'secret' in key.lower():
                info(f"  {key}: ****")
            else:
                info(f"  {key}: {value}")
    except SystemExit:
        return  # get_cli_params() a déjà affiché les erreurs

    # Validations
    if not check_internet():
        error("Pas d'acces reseau ou DeviantArt inaccessible.")
        return

    ok_dir, png_list = check_directory(cli["directory_file_to_upload"])
    if not ok_dir:
        error(f"Aucune image .png trouvee dans {cli['directory_file_to_upload']}")
        return
    info(f"{len(png_list)} image(s) trouvee(s) a uploader.")

    # Construire l'URL d'authentification
    auth_url = (
        "https://www.deviantart.com/oauth2/authorize"
        f"?response_type=code&client_id={cli['client_id']}"
        f"&redirect_uri={cli['redirect_uri']}"
    )

    # Tentative d'ouverture automatique dans Firefox
    info("Ouverture automatique de l'URL OAuth dans Firefox...")
    try:
        open_in_firefox(auth_url)
    except Exception as exc:
        warn(f"Ouverture automatique impossible ({exc}).")

    # Afficher l'URL pour usage manuel
    print("\n" + "=" * 80)
    print("URL D'AUTHENTIFICATION OAUTH :")
    print(auth_url)
    print("=" * 80 + "\n")

    # Extraire le port de l'URI de redirection
    port = 5088
    if cli["redirect_uri"].endswith(":5088"):
        port = 5088
    elif ":5088/" in cli["redirect_uri"]:
        try:
            port = int(cli["redirect_uri"].split(":")[2].split("/")[0])
        except:
            port = 5088

    # Démarrage du serveur Flask
    info("Demarrage du serveur Flask...")
    try:
        run_flask_app(port)
    except KeyboardInterrupt:
        warn("Interruption manuelle detectee.")
    except Exception as e:
        error(f"Erreur du serveur Flask : {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
