from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import os
import gradio as gr
from datetime import datetime

def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    return model, tokenizer, device

def ensure_formatting(summary_text):
    """
    Ajuste le formatage des bullet points et des sauts de ligne. Élimine les introductions non désirées et assure que les bullet points commencent avec '- '.
    Remplace les bullet points incorrects, supprime tous les caractères '*', et réduit les doubles sauts de ligne à un seul.
    """
    # Supprimer les introductions comme "Key points:"
    if "Key points:" in summary_text:
        summary_text = summary_text.split("Key points:")[1]

    lines = summary_text.split('\n')
    corrected_lines = []
    for line in lines:
        # Supprimer tous les '*' du texte
        stripped_line = line.replace('*', '').strip()
        # Assurez que les lignes de bullet points commencent correctement avec '- '
        if stripped_line.startswith(('-', '•', '◦', '>', '→')):
            # Enlever le premier caractère si ce n'est pas déjà un '-'
            if not stripped_line.startswith('-'):
                stripped_line = stripped_line[1:]
            corrected_line = '- ' + stripped_line.lstrip()
        else:
            # Ajouter '- ' si la ligne ne commence par aucun symbole de liste
            corrected_line = '- ' + stripped_line

        if corrected_line.strip() != '-':  # Éviter d'ajouter des lignes qui seraient juste '-'
            corrected_lines.append(corrected_line)

    # Joindre les lignes avec un seul saut de ligne
    return '\n'.join(corrected_lines)





def generate_bullet_points(article_text, model, tokenizer, device):
    messages = [
        {"role": "user", "content": """Please read the following article and provide a concise summary in the form of bullet points. Highlight the key information and main ideas.\nUse this format:\n
         - Element 1\n
         - Element 2\n
         - Element 3\n
         - Element n\n
         
         Be very concise by keeping only essential informations. Stick to the format previously given."""},
        {"role": "assistant", "content": "Sure, I'll summarize the article with concise bullet points using '-' :"},
        {"role": "user", "content": article_text}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(encodeds, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)[0]

    # Trouver la dernière occurrence de [/INST] et tronquer tout avant cela
    last_inst_index = decoded.rfind('[/INST]')
    if last_inst_index != -1:
        response_start = last_inst_index + len('[/INST]')
        summary_text = decoded[response_start:].strip()
    else:
        summary_text = "No summary available."

    # Supprimer la balise '</s>' si elle est présente à la fin du texte généré
    if summary_text.endswith('</s>'):
        summary_text = summary_text[:-4].strip()

    # Assurer que tous les bullet points utilisent '*'
    summary_text = ensure_formatting(summary_text)

    return summary_text


# Exemple d'utilisation
input_filename = 'articles.jsonl'
output_filename = 'articles.jsonl'

def process_articles(sorted=False, max_element=10, input_file='articles.jsonl', output_file='articles.jsonl', progress=gr.Progress()):
    model, tokenizer, device = init_model()
    updated_articles = []

    # Lecture de tous les articles à partir du fichier
    with open(input_file, 'r', encoding='utf-8') as file:
        articles = [json.loads(line) for line in file.readlines()]

    # Traitement des articles pour ajouter les bullet points
    for article in progress.tqdm(articles[:max_element] if sorted else articles, desc="Summarizing Articles"):
        article_text = article.get('content', '')
        if article_text:
            bullet_points = generate_bullet_points(article_text, model, tokenizer, device)
            article['bullet_points'] = bullet_points
            print(bullet_points)
        updated_articles.append(article)

    # Écriture de tous les articles modifiés dans le fichier
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for article in updated_articles:
            json.dump(article, outfile)
            outfile.write('\n')

    with open('sum_status.txt','w') as f:
        f.write('Last update: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M")))




def process_article(article_text):
    model, tokenizer, device = init_model()
    bullet_points = generate_bullet_points(article_text, model, tokenizer, device)
    return bullet_points


def find_object_by_title_and_summarize(title, jsonl_file='articles.jsonl'):
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'title' in data and data['title'] == title:
                bullet_points = process_article(data['content'])
                data['bullet_points'] = bullet_points
                output_file = 'summary.jsonl'
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    json.dump(data, outfile)
                return "Summarized and added to the summary_database!"
    return "Article not found"