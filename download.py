from flask import Flask, send_from_directory
from pyngrok import ngrok
import os

app = Flask(__name__)

DIRECTORY_PATH = "/media/mldrive/tgallard/Analyst/pdf_outputs"

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(DIRECTORY_PATH, filename, as_attachment=False)

if __name__ == '__main__':
    NGROK_AUTH_TOKEN = "2fHb2jK7OZe0XuOIMcS7vWUGRNf_6fhhxJFDxtxBHtc5n4SJw"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    port = 5000

    # Nettoyer les tunnels ngrok existants avant d'en créer un nouveau
    ngrok.kill()

    # Créer un tunnel ngrok au port spécifié et obtenir l'URL publique
    tunnel = ngrok.connect(port)
    public_url = tunnel.public_url  # Obtention de l'URL comme chaîne de caractères
    print("Ngrok Tunnel URL:", public_url)

    # Écrire l'URL dans un fichier en s'assurant que le contenu précédent est supprimé
    with open("ngrok_url.txt", "w") as f:
        f.write(public_url)

    app.run(host='0.0.0.0', port=port)
