import gradio as gr
from datetime import datetime
import asyncio
import subprocess

def compter_lignes_jsonl(chemin_fichier):
    nombre_de_lignes = 0
    with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
        for ligne in fichier:
            nombre_de_lignes += 1
    return nombre_de_lignes

def get_articles(progress=gr.Progress()):
    progress(0, desc="Starting...")  # Initialize progress
    total_steps= compter_lignes_jsonl('output.jsonl')
    process = subprocess.Popen(['python', '-u', 'f24Scraper.py'], stdout=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())  # Affichage de la sortie en temps réel
            line = output.strip()
            if line.isdigit():  # Vérification que la ligne est numérique
                current_progress = int(line) / total_steps
                progress(current_progress, desc="Processing...", total=total_steps, unit="steps")  # Mise à jour de la progression


    with open ('article_status.txt', 'w') as f:
        f.write('Last update: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M")))
    return 'Last update: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M"))
