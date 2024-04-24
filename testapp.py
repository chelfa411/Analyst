import gradio as gr
import time
from scraping import fetch_feeds_and_export_to_jsonl, all_urls
import subprocess
from datetime import datetime, timezone
from test import process_articles, process_article, find_object_by_title_and_summarize
from classify import classify
from analyseAI import process_json, analyse, find_object_by_title_and_analyse
from pdfGenerator import convert_jsonl_to_pdf
from gradioserver import create_search_article_interface
def get_sum_update():
    with open('sum_status.txt','r') as f:
        return f.read().strip()
    
def get_last_update():
    with open('last_update.txt', 'r') as f:
        return f.read().strip()

def get_article_status():
    with open ('article_status.txt', 'r') as f:
        return f.read().strip()
    

def compter_lignes_jsonl(chemin_fichier):
    nombre_de_lignes = 0
    with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
        for ligne in fichier:
            nombre_de_lignes += 1
    return nombre_de_lignes

def get_articles(progress=gr.Progress()):
    progress(0, desc="Scraping articles...")  # Initialize progress
    total_steps= compter_lignes_jsonl('output.jsonl')
    process = subprocess.Popen(['python', '-u', 'bbc_scraper.py'], stdout=subprocess.PIPE, text=True)
    step=0
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()
            if line.isdigit():  # Vérification que la ligne est numérique
                step +=1
                current_progress = step / total_steps
                progress(current_progress, desc="Scraping articles...", total=total_steps, unit="steps")  # Mise à jour de la progression

    process = subprocess.Popen(['python', '-u', 'f24Scraper.py'], stdout=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()
            if line.isdigit():  # Vérification que la ligne est numérique
                step +=1
                current_progress = step / total_steps
                progress(current_progress, desc="Scraping articles...", total=total_steps, unit="steps")  # Mise à jour de la progression
    
    process = subprocess.Popen(['python', '-u', 'theguardianScraper.py'], stdout=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            line = output.strip()
            if line.isdigit():  # Vérification que la ligne est numérique
                step +=1
                current_progress = step / total_steps
                progress(current_progress, desc="Scraping articles...", total=total_steps, unit="steps")  # Mise à jour de la progression
    
    with open ('article_status.txt', 'w') as f:
        f.write('Last update: ' + str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")) + 'Z')
    return 'Last update: ' + str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")) + 'Z'


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    with gr.Tab("Introduction"):
        with gr.Column():
            gr.Markdown("## Welcome to Our NewsFeed Analysis Application")
            gr.Markdown("""
This application allows you to effectively manage and analyze news content from various sources. Below are the key features and their expected durations:

- **Fetching Article Content**: Retrieving content from selected sources can take up to 30 minutes, especially if all sources are selected.
- **Summarizing Articles**: Summarizing all articles can take up to 5 minutes. There is an option to summarize only the articles of interest based on their classification.
- **Article Analysis**: Analyzing a single article takes up to two minutes. To analyze multiple articles, they must be classified and summarized beforehand. Alternatively, you can analyze a single article using either the text of its summary or its title if it is present in the database.
- **Database Status**: The last tab allows you to check the current state of the database at any time. Please use the refresh button to update the display following any changes (analysis, classification, etc.).
- **AI Model**: The AI used for analysis is a basic model and not yet specifically trained to excel at this task.

""")
            gr.Markdown("""
                    ### How to Use the App
                    
                    - **Update and Retrieve Content**: Click **"Update news feed"** followed by **"Get article content"** to refresh the article database.
                    - **Classify Articles**: Input the theme of interest in the 'Theme' textbox to classify articles by their relevance.
                    - **Summarize Articles**: Choose to summarize a specific number of articles by selecting the count or summarize all by checking the **"Summarize all articles"** box. Click **"Summarize articles"** to proceed.
                    - **Analyze Articles**: Generate analyses for multiple articles by clicking **"Launch articles analysis"**.
                    - **Target Specific Articles**: You can target an article by its title or content for summarization and analysis.
                """)

    with gr.Tab("NewsFeed"):
        with gr.Blocks():
            with gr.Column():
                with gr.Row():  # Ajout d'un ID pour cibler avec CSS
                    update_status = gr.Textbox(label="Update status", value=get_last_update(), elem_id="update_textbox", show_label=True, lines=4)
                    feeds_checkbox = gr.components.CheckboxGroup(choices=list(all_urls.keys()), label="Select RSS Feeds", value=list(all_urls.keys()))
                update_news_feed = gr.Button(elem_id="news_feed_button", value='Update news feed')
                

            update_news_feed.click(fn=fetch_feeds_and_export_to_jsonl, inputs=[feeds_checkbox], outputs=update_status, concurrency_limit=1)

            with gr.Row():
                content_status = gr.Textbox(label='Article content status', show_label=True, value=get_article_status(), lines=4)
                get_articles_t = gr.Button(value='Get articles content')
            get_articles_t.click(fn=get_articles, outputs=content_status, concurrency_limit=1)
    with gr.Tab("Classification"):
        with gr.Row():
            theme = gr.Textbox(label="Theme", show_label=True)
            sort_button = gr.Button(value='Sort articles')
        sort_button.click(fn=classify, inputs=[theme], outputs=theme)

    with gr.Tab("Summarization"):
        with gr.Blocks():
            with gr.Row():
                sum_status = gr.Textbox(label='Summarization status', show_label=True, value=get_sum_update())
                box_sort = gr.Checkbox(label="summarize all the articles", show_label=True)
                slide_sort = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Number of articles to summarize")
                get_sum = gr.Button(value='Summarize articles')
            get_sum.click(fn=process_articles, inputs=[box_sort, slide_sort], outputs=sum_status, concurrency_limit=1)
            with gr.Row():
                article_title = gr.Textbox(label="Enter the title of the article for summarizing",show_label=True)
                art_but = gr.Button(value="launch summarizing")
            art_but.click(fn=find_object_by_title_and_summarize, inputs=article_title, outputs=article_title)
            with gr.Column():
                text_entry = gr.Textbox(label='Enter the text of the article to summarize', show_label=True, max_lines=5, lines=5)
                sum = gr.Button(value='Summarize the article')
                sum_out = gr.Markdown()
            sum.click(fn=process_article,inputs=[text_entry], outputs=sum_out, concurrency_limit=1)

    with gr.Tab("Analysis"):
        with gr.Row():
            ana_stat = gr.HTML()
            ana_slider = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Number of articles to analyze (prior classification required)")
            ana_but = gr.Button(value="Launch articles analysis")
        ana_but.click(fn=lambda slide: process_json(slide,demo), inputs=[ana_slider], outputs=ana_stat, concurrency_limit=1)
        with gr.Row():
            art_ana_stat = gr.HTML()
            art_text = gr.Textbox(label="Enter the title of the article to analyse", show_label=True)
            art_ana_but = gr.Button(value="Launch article analysis")
        art_ana_but.click(fn=lambda title: find_object_by_title_and_analyse(demo,title), inputs=art_text, outputs=art_ana_stat)
        with gr.Column():
            text_ana = gr.Textbox(label='Enter the text of the article to analyze', show_label=True, max_lines=6, lines=6)
            link_report = gr.HTML()
            but_ana = gr.Button(value='Launch article analysis')
        but_ana.click(fn=lambda text: analyse(text,demo), inputs=[text_ana], outputs=link_report, concurrency_limit=1)
    with gr.Tab("Read articles'data"):
        create_search_article_interface(demo)

auth = [("Tester1","sadijojqe54572@$#"), ("Tester2","sakdnpowejprhnjdsdf78$@@")]
demo.launch(share=True)


