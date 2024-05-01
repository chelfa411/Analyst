import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import gradio as gr
from datetime import datetime, timezone
import json
from tqdm import tqdm
import os
import nltk
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz

def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
    return model, tokenizer, device

def ensure_formatting(summary_text):
    # Utilisation de NLTK pour diviser le texte en phrases
    sentences = sent_tokenize(summary_text)
    corrected_lines = []
    for sentence in sentences:
        # Nettoyer l'espace blanc et vérifier si la phrase n'est pas vide
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            # Ajouter un tiret au début de chaque phrase non vide pour créer un bullet point
            corrected_line = '- ' + stripped_sentence
            corrected_lines.append(corrected_line)
    # Joindre les lignes avec un retour à la ligne pour former le texte final
    return '\n'.join(corrected_lines)

def generate_bullet_points(article_text, model, tokenizer, device, preset, use_bullet):
    text_length = len(tokenizer.tokenize(article_text))
    presets = {
        "Basic": {"max_length": int(0.1 * text_length), "min_length": int(0.05 * text_length), "num_beams": 4, "length_penalty": 0.5, "early_stopping": True},
        "Short": {"max_length": int(0.2 * text_length), "min_length": int(0.1 * text_length), "num_beams": 4, "length_penalty": 0.8, "early_stopping": True},
        "Medium": {"max_length": int(0.3 * text_length), "min_length": int(0.2 * text_length), "num_beams": 6, "length_penalty": 1.2, "early_stopping": False},
        "Long": {"max_length": int(0.5 * text_length), "min_length": int(0.3 * text_length), "num_beams": 8, "length_penalty": 1.5, "early_stopping": False},
        "Complete": {"max_length": int(0.8 * text_length), "min_length": int(0.5 * text_length), "num_beams": 10, "length_penalty": 2.0, "early_stopping": False}
    }
    config = presets[preset]
    print(config)
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs,
        max_length=config["max_length"],
        min_length=config["min_length"],
        num_beams=config["num_beams"],
        length_penalty=config["length_penalty"],
        early_stopping=True
    )
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = ensure_formatting(summary_text) if use_bullet else summary_text
    return formatted_summary


def process_article(article_text, preset, use_bullet):
    model, tokenizer, device = init_model()
    bullet_points = generate_bullet_points(article_text, model, tokenizer, device, preset, use_bullet)
    return bullet_points

def process_articles(preset, use_bullet, All=False, max_element=10, input_file='articles.jsonl', output_file='articles.jsonl', progress=gr.Progress()):
    model, tokenizer, device = init_model()
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    if All:
       with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in progress.tqdm(lines, desc="Summarizing Articles"):
            article_obj = json.loads(line)
            article_text = article_obj.get('content', '')
            if article_text:
                bullet_points = generate_bullet_points(article_text, model, tokenizer, device, preset, use_bullet)
                article_obj['bullet_points'] = bullet_points
                json.dump(article_obj, outfile)
                outfile.write('\n') 
    else:
        lines1 = lines[:max_element]
        lines2 = lines[max_element:]
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in progress.tqdm(lines1, desc="Summarizing Articles"):
                article_obj = json.loads(line)
                article_text = article_obj.get('content', '')
                if article_text:
                    bullet_points = generate_bullet_points(article_text, model, tokenizer, device, preset, use_bullet)
                    article_obj['bullet_points'] = bullet_points
                    json.dump(article_obj, outfile)
                    outfile.write('\n')
            for line in lines2:
                article_obj = json.loads(line)
                json.dump(article_obj, outfile)
                outfile.write('\n')

    with open('sum_status.txt', 'w') as f:
        f.write('Last update: ' + str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")) + 'Z')
    
    return 'Last update: ' + str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")) + 'Z'


def find_object_by_title_and_summarize(title, preset, use_bullet, jsonl_file='articles.jsonl'):
    model, tokenizer, device = init_model()
    best_match = None
    best_match_score = 70  # Vous pouvez ajuster ce seuil selon vos besoins

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        articles = [json.loads(line) for line in file]

    for article in articles:
        # Calcul de la correspondance fuzzy entre le titre recherché et le titre de l'article
        match_score = fuzz.partial_ratio(title.lower(), article.get('title', '').lower())
        if match_score > best_match_score:
            best_match_score = match_score
            best_match = article

    if best_match:
        # Si un article correspondant est trouvé avec un score acceptable
        best_match['bullet_points'] = process_article(best_match.get('content', ''), preset, use_bullet)
        # Réécrire tous les articles dans le fichier
        with open(jsonl_file, 'w', encoding='utf-8') as outfile:
            for article in articles:
                json.dump(article, outfile)
                outfile.write('\n')
        return "Article summarized"
    else:
        return "Article not found"

def process_article_list(article_list):
    model, tokenizer, device = init_model()
    response = []
    for article_text in article_list:
        bullet_points = generate_bullet_points(article_text, model, tokenizer, device, preset="Medium", use_bullet=False)
        response.append(bullet_points)

    return response