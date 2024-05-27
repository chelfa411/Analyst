import json
from transformers import pipeline
import gradio as gr

def classify(theme, input_file='articles.jsonl', progress=gr.Progress()):
    # Liste pour stocker les objets JSON mis à jour
    updated_objects = []

    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Création de la pipeline de classification zero-shot
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

    # Traitement de chaque ligne JSON
    for line in progress.tqdm(lines, desc="Sorting articles"):
        obj = json.loads(line)  # Charger chaque ligne comme un objet JSON
        if obj.get('content', '') != "":
            text = obj['content']
            classes_verbalized = [theme]
            hypothesis_template = "This text is about {}"
            # Exécution de la classification zero-shot
            output = zeroshot_classifier(text, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
            # Ajout du thème et du score au label
            score = output['scores'][0] if output['scores'] else 0  # Sécurité en cas de réponse vide
            obj['label'] = f"{theme}"
            obj['score'] = score  # Stocker le score séparément pour le tri
            updated_objects.append(obj)
    
    # Tri des objets par score dans l'ordre décroissant
    updated_objects.sort(key=lambda x: x['score'], reverse=True)

    # Écriture des objets mis à jour dans un nouveau fichier JSONL, triés par score
    output_file = 'articles.jsonl'
    with open(output_file, 'w') as f:
        for obj in updated_objects:
            json.dump(obj, f)
            f.write('\n')  # Écrit chaque objet JSON sur une nouvelle ligne

    with open('theme.txt','w') as f:
        f.write(theme)

    return theme

