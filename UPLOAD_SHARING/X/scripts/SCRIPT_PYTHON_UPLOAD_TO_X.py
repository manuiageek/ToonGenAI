import tweepy
import sys
import shutil
import os

# Lecture des arguments : Texte du tweet et chemins des images
x_consumer_key = sys.argv[1]
x_consumer_secret = sys.argv[2]
x_access_token = sys.argv[3]
x_access_token_secret = sys.argv[4]
x_dest_folder = sys.argv[5]
tweet_text = sys.argv[6]  # Premier argument après le nom du script
image_paths = sys.argv[7:]  # Tous les arguments après le texte du tweet

# Initialisation du client Tweepy avec l'API Twitter v2
client = tweepy.Client(consumer_key=x_consumer_key,
                       consumer_secret=x_consumer_secret,
                       access_token=x_access_token,
                       access_token_secret=x_access_token_secret)

# DEBUG MANUEL (décommenter pour tester manuellement)
# tweet_text = "#Eevin #SkeletonKnight #AIFANART"
# image_paths = [r"C:\AI_TEMP_UPLOAD\_AI_UPLOAD_X\ToonGenAI_00060927.png",
#                r"C:\AI_TEMP_UPLOAD\_AI_UPLOAD_X\ToonGenAI_00060928.png",
#                r"C:\AI_TEMP_UPLOAD\_AI_UPLOAD_X\ToonGenAI_00060929.png"]

# Authentification avec l'API v1 pour l'upload des images
tweepy_auth = tweepy.OAuth1UserHandler(
    consumer_key=x_consumer_key,
    consumer_secret=x_consumer_secret,
    access_token=x_access_token,
    access_token_secret=x_access_token_secret
)
tweepy_api_for_image = tweepy.API(tweepy_auth)

# Chargement et collecte des ID des médias
media_ids = []
for image_path in image_paths:
    media = tweepy_api_for_image.media_upload(image_path)
    media_ids.append(media.media_id_string)
    # Déplacement des fichiers
    shutil.move(image_path, os.path.join(x_dest_folder, os.path.basename(image_path)))

# Création et envoi du tweet avec les images
response = client.create_tweet(text=tweet_text, media_ids=media_ids)

# Affichage de la réponse
print(response)
