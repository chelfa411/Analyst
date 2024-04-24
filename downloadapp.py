import gradio as gr
import os
import subprocess
import time
from scraping import fetch_feeds_and_export_to_jsonl

def launch_flask_app():
    # Lancer le serveur Flask en arrière-plan
    return subprocess.Popen(["python", "download.py"])



def list_files():
    return os.listdir("/media/mldrive/tgallard/Analyst/pdf_outputs")

def create_download_link(filename, base_url):
    # Créer le lien de téléchargement avec l'URL de base
    link = f"{base_url}/download/{filename}"
    return f"<a href='{link}' target='_blank'>Click to download</a>"

def setup_gradio_interface():
    # Lancer Flask et attendre l'URL ngrok
    flask_process = launch_flask_app()
    time.sleep(10)
    with open ("ngrok_url.txt",'r') as f:
        flask_ngrok_url = f.read().strip()

    with gr.Blocks() as demo:
        with gr.Tab("File Downloads"):
            with gr.Row():
                file_selector = gr.Dropdown(label="Select a file", choices=list_files())
                download_button = gr.Button("Download")
            output = gr.HTML()

            def update_output(filename):
                # Utiliser l'URL ngrok mise à jour pour créer les liens de téléchargement
                return create_download_link(filename, flask_ngrok_url)

            download_button.click(update_output, inputs=[file_selector], outputs=[output])

        with gr.Tab("NewsFeed"):
            with gr.Blocks():
                update_news_feed = gr.Button()
                update_status = gr.Textbox()

            update_news_feed.click(fn=fetch_feeds_and_export_to_jsonl, outputs=update_status)


    return demo, flask_process

demo, flask_process = setup_gradio_interface()
demo.launch(share=True)  # Lancement de Gradio avec partage en ligne
