from flask import Flask, request
import requests
import webbrowser
import os
import sys
import threading
import shutil
import time
import json

app = Flask(__name__)

def load_config():
    print("[DEBUG] Entrée dans load_config()")
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    config_path = os.path.join(script_dir, 'config.json')
    print("[DEBUG] Chemin du fichier de config:", config_path)
    if not os.path.exists(config_path):
        print("[ERREUR] Fichier config.json non trouvé à l'emplacement:", config_path)
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    print("[DEBUG] Configuration chargée:", config)
    return config

def get_access_token(client_id, client_secret, code, redirect_uri):
    print("[DEBUG] Entrée dans get_access_token()")
    print("[DEBUG] Paramètres d'auth:", client_id, client_secret, code, redirect_uri)
    url = "https://www.deviantart.com/oauth2/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri
    }
    print("[DEBUG] Requête POST vers:", url)
    print("[DEBUG] Payload:", payload)
    response = requests.post(url, data=payload)
    print("[DEBUG] Réponse get_access_token status_code:", response.status_code)
    print("[DEBUG] Réponse get_access_token texte:", response.text)
    if response.ok:
        token = response.json().get("access_token")
        print("[DEBUG] Token d'accès obtenu:", token)
        return token
    else:
        print("[ERREUR] Impossible d'obtenir le token d'accès")
        return None

def submit_to_stash(access_token, image_path, title, tags):
    print("[DEBUG] Entrée dans submit_to_stash()")
    url = "https://www.deviantart.com/api/v1/oauth2/stash/submit"
    print("[DEBUG] URL:", url)
    headers = {"Authorization": f"Bearer {access_token}"}
    print("[DEBUG] Headers:", headers)
    if not os.path.exists(image_path):
        print("[ERREUR] Le fichier image n'existe pas:", image_path)
    files = {'image': open(image_path, 'rb')}
    data = {
        "title": title,
        "tags[0]": tags[0] if len(tags) > 0 else "",
        "tags[1]": tags[1] if len(tags) > 1 else "",
        "tags[2]": tags[2] if len(tags) > 2 else ""
    }
    print("[DEBUG] Data pour stash submit:", data)
    print("[DEBUG] Envoi de la requête POST à Stash...")
    response = requests.post(url, headers=headers, files=files, data=data)
    print("[DEBUG] Réponse submit_to_stash status_code:", response.status_code)
    print("[DEBUG] Réponse submit_to_stash texte:", response.text)
    if response.ok:
        itemid = response.json().get("itemid")
        print("[DEBUG] Item ID obtenu:", itemid)
        return itemid
    else:
        print("Erreur lors de la soumission à Sta.sh:")
        print("Status code:", response.status_code)
        print("Réponse:", response.text)
        return None

def publish_from_stash(access_token, itemid, gallery_id):
    print("[DEBUG] Entrée dans publish_from_stash()")
    url = "https://www.deviantart.com/api/v1/oauth2/stash/publish"
    print("[DEBUG] URL:", url)
    headers = {"Authorization": f"Bearer {access_token}"}
    print("[DEBUG] Headers:", headers)
    data = {
        "validate_token": access_token,
        "itemid": itemid,
        "galleryids[0]": gallery_id[0] if len(gallery_id) > 0 else "",
        "galleryids[1]": gallery_id[1] if len(gallery_id) > 1 else "",
        "is_mature": "true",
        "mature_level": "moderate",
        "mature_classification": "sexual",
        "agree_submission": "true",
        "agree_tos": "true",
        "feature": "true",
        "allow_comments": "true",
        "display_resolution": "0",
        "sharing": "allow",
        "allow_free_download": "true",
        "add_watermark": "false",
        "mature_content": "true"
    }
    print("[DEBUG] Data pour publish:", data)
    print("[DEBUG] Envoi de la requête POST à publish...")
    response = requests.post(url, headers=headers, data=data)
    print("[DEBUG] Réponse publish_from_stash status_code:", response.status_code)
    print("[DEBUG] Réponse publish_from_stash texte:", response.text)
    try:
        json_resp = response.json()
        print("[DEBUG] Réponse JSON publish_from_stash:", json_resp)
        return json_resp
    except Exception as e:
        print("[ERREUR] Impossible de parser la réponse JSON:", e)
        return None

def run_flask_app():
    print("[DEBUG] Lancement de l'application Flask sur le port 5088")
    # Désactivation du reloader et du debug mode Flask pour éviter l'erreur de thread:
    app.run(port=5088, debug=False, use_reloader=False)

@app.route('/stop')
def stop_flask_app():
    print("[DEBUG] Route /stop appelée, arrêt de l'application")
    os._exit(0)

@app.route('/')
def receive_code():
    print("[DEBUG] Route / appelée")
    code = request.args.get('code')
    print("[DEBUG] Code reçu:", code)
    if code:
        print(f"[DEBUG] Received Authorization Code: {code}")
        
        # Charger la configuration
        print("[DEBUG] Chargement de la configuration...")
        config = load_config()
        client_id = config.get('client_id')
        client_secret = config.get('client_secret')
        redirect_uri = config.get('redirect_uri')

        if not client_id or not client_secret or not redirect_uri:
            print("[ERREUR] Config invalide, client_id/client_secret/redirect_uri manquants")
            return "Configuration invalide."

        # Échanger le code contre un token
        print("[DEBUG] Obtention du token d'accès...")
        access_token = get_access_token(client_id, client_secret, code, redirect_uri)

        # title_info = r"#HasegawaChisato #ShinmaiMaouNoTestament #AIFANART"
        title_info = sys.argv[1]  # Premier argument après le nom du script        
        tags_temp = title_info.split(" ")
        tags_info = [tag[1:] for tag in tags_temp if tag.startswith("#")]
        print("[DEBUG] tags_info:", tags_info)

        # Featured gallery
        gallery_id = ["1551DEF0-436F-4D02-FA1A-9018FB8737C9"]
        # Ajout de la nouvelle ID
        # gallery_id.append("970FF0D7-D86C-BD58-C6B0-3C1C85D3F87F")
        gallery_id.append(sys.argv[2])
        print("[DEBUG] gallery_id:", gallery_id)

        if access_token:
            print("Token d'accès obtenu :", access_token)

            directory_file_to_upload = r"C:\AI_TEMP_UPLOAD\_AI_UPLOAD_DART"
            done_directory = os.path.join(directory_file_to_upload, "done")
            print("[DEBUG] Répertoire d'upload:", directory_file_to_upload)
            print("[DEBUG] Répertoire done:", done_directory)

            if not os.path.exists(directory_file_to_upload):
                print("[ERREUR] Le répertoire d'upload n'existe pas:", directory_file_to_upload)
                return "Répertoire d'upload introuvable."

            if not os.path.exists(done_directory):
                print("[DEBUG] Le répertoire done n'existe pas, création...")
                os.makedirs(done_directory, exist_ok=True)

            print("[DEBUG] Parcours des fichiers dans:", directory_file_to_upload)
            for filename_to_upload in os.listdir(directory_file_to_upload):
                print("[DEBUG] Fichier trouvé:", filename_to_upload)
                if filename_to_upload.endswith(".png"):
                    file_path = os.path.join(directory_file_to_upload, filename_to_upload)
                    print(f"[DEBUG] Traitement de l'image {file_path}...")
                    itemid = submit_to_stash(access_token, file_path, title_info, tags_info)
                    if itemid:
                        print("[DEBUG] Image soumise à Sta.sh, Item ID :", itemid)
                        publish_result = publish_from_stash(access_token, itemid, gallery_id)
                        print(f"[DEBUG] Résultat de la publication:", publish_result)

                        if publish_result:
                            print("[DEBUG] Publication réussie, déplacement du fichier dans done.")
                            shutil.move(file_path, os.path.join(done_directory, filename_to_upload))
                        else:
                            print("[ERREUR] La publication n'a pas retourné de résultat valide.")
                    else:
                        print("[ERREUR] Échec de la soumission à Sta.sh")
                        return "Échec de la soumission à Sta.sh"
                else:
                    print("[DEBUG] Le fichier n'est pas un .png, ignoré:", filename_to_upload)
        else:
            print("[ERREUR] Échec de l'obtention du token d'accès")
            return "Échec de l'obtention du token d'accès"

    print("[DEBUG] Fin du traitement, pause de 8 secondes avant arrêt.")
    time.sleep(8)
    stop_flask_app()
    return "Vous pouvez fermer cette page."

if __name__ == '__main__':
    print("[DEBUG] sys.argv:", sys.argv)
    print("[DEBUG] Démarrage du script principal")
    config = load_config()
    client_id = config.get('client_id')
    redirect_uri = config.get('redirect_uri')
    print("[DEBUG] client_id:", client_id)
    print("[DEBUG] redirect_uri:", redirect_uri)

    if not client_id or not redirect_uri:
        print("[ERREUR] client_id ou redirect_uri manquant dans la config.")
        sys.exit(1)

    auth_url = f"https://www.deviantart.com/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"
    print("[DEBUG] URL d'authentification:", auth_url)
    firefox_path = 'C:/Program Files/Mozilla Firefox/firefox.exe %s'
    print("[DEBUG] Ouverture du navigateur...")
    webbrowser.get(firefox_path).open(auth_url)

    # Lancement du serveur Flask sans debug/reloader dans un thread
    print("[DEBUG] Lancement du thread Flask...")
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()
