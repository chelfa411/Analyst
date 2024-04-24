import gradio as gr
import time
import subprocess


def update_progress(progress=gr.Progress()):
    progress(0, desc="Starting...")  # Initialize progress
    total_steps = 100

    # Créer un processus pour exécuter le script
    process = subprocess.Popen(['python', '-u', 'testscrap.py'], stdout=subprocess.PIPE, text=True)

    # Lire la sortie du script ligne par ligne
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

    # Traitement des dernières lignes de sortie si nécessaire
    output, errors = process.communicate()
    if output:
        print(output.strip())
    if errors:
        print("Errors:", errors.strip())

    # Mise à jour finale de la progression
    progress(1, desc="Completed", total=total_steps, unit="steps")

# Gradio interface setup
with gr.Blocks() as demo:
    with gr.Tab("Progress"):
        start_button = gr.Button('Start')
        progress_bar = gr.Textbox(label="Progress")
    start_button.click(
        fn=update_progress,  # Function to execute
        inputs=[],  # No input widgets
        outputs=[progress_bar]  # Output is the progress bar
    )

demo.launch()
