import json
from weasyprint import HTML
import os
input_jsonl = 'articles.jsonl'
output_folder = '/tmp/gradio/pdf_outputs'
import markdown2
import PyPDF2
def convert_jsonl_to_pdf(input_jsonl=input_jsonl, output_folder=output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_jsonl, 'r', encoding='utf-8') as file:
        for line in file:
            article = json.loads(line)

            title = article.get('title', 'No Title Provided')
            date = article.get('date', 'No Date Provided')
            bullet_points = article.get('bullet_points', '').strip()
            detailed_analysis = article.get('detailed_analysis', '')

            if detailed_analysis !='Unavailable':
                # Préparation du contenu Markdown
                response = f"""
    <div style='font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px; text-align: justify; max-width: 1500px; margin: auto;'>
        <h1 style='color: #333;'>News analysis for: {title}</h1>
        <h2>Date of Publication:</h2>
        <p style='background-color: #f9f9f9; padding: 10px; border-left: 5px solid #007BFF;'>{article["date"]}</p>
        <h2>Link:</h2>
        <p><a href='{article["link"]}' target='_blank' style='color: #007BFF; text-decoration: none;'>Read Article</a></p>
        """
                if "bullet_points" in article:
                    bullet_points_html = markdown2.markdown(article["bullet_points"], extras=["lists", "nl2br"])
                    response += f"<h2>Summary:</h2>{bullet_points_html}"
                if "detailed_analysis" in article:
                    analysis_html = markdown2.markdown(article["detailed_analysis"], extras=["lists", "nl2br"])
                    response += f"<h2>Analysis:</h2>{analysis_html}"
                response += "</div>"

                title_cleaned = title.replace('/', '_').replace(' ', '-').replace("'", "_").replace("|","pipe").replace("?","q")
                output_pdf = f"{output_folder}/detailed_analysis_{title_cleaned}.pdf"
                
                # Conversion en PDF
                HTML(string=response).write_pdf(output_pdf)

def article_to_pdf(text, output_folder=output_folder):
    response = f"""
    <div style='font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px; text-align: justify; max-width: 1500px; margin: auto;'>
    <h1 style='color: #333;'>News analysis</h1>
    """
    analysis_html = markdown2.markdown(text, extras=["lists", "nl2br"])
    response += f"<h2>Analysis:</h2>{analysis_html}"
    response += "</div>"
    output_pdf = f"{output_folder}/detailed_analysis_local.pdf"
    HTML(string=response).write_pdf(output_pdf)

def article_to_pdf_target(article, text=None, output_folder=output_folder):
    title = article.get('title', 'No Title Provided')
    date = article.get('date', 'No Date Provided')
    bullet_points = article.get('bullet_points', '').strip()
    detailed_analysis = article.get('detailed_analysis', '')

    if detailed_analysis !='Unavailable':
    # Préparation du contenu Markdown
        response = f"""
        <div style='font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px; text-align: justify; max-width: 1500px; margin: auto;'>
        <h1 style='color: #333;'>News analysis for: {title}</h1>
        <h2>Date of Publication:</h2>
        <p style='background-color: #f9f9f9; padding: 10px; border-left: 5px solid #007BFF;'>{date}</p>
        <h2>Link:</h2>
        <p><a href='{article["link"]}' target='_blank' style='color: #007BFF; text-decoration: none;'>Read Article</a></p>
        """
        if "bullet_points" in article:
            bullet_points_html = markdown2.markdown(bullet_points, extras=["lists", "nl2br"])
            response += f"<h2>Summary:</h2>{bullet_points_html}"
        if "detailed_analysis" in article:
            analysis_html = markdown2.markdown(detailed_analysis, extras=["lists", "nl2br"])
            response += f"<h2>Analysis:</h2>{analysis_html}"
        response += "</div>"

        title_cleaned = title.replace('/', '_').replace(' ', '-').replace("'", "_").replace("|","pipe").replace("?","q")
        output_pdf = f"{output_folder}/detailed_analysis_{title_cleaned}.pdf"

    # Conversion en PDF
        HTML(string=response).write_pdf(output_pdf)


import markdown2
from weasyprint import HTML, default_url_fetcher

# Définir le custom URL fetcher
def custom_url_fetcher(url):
    if url.startswith('local:'):
        # Convertir 'local:' en chemin absolu sur le système de fichiers
        filepath = url.replace('local:', '/media/mldrive/tgallard/Analyst/')
        try:
            return {
                'file_obj': open(filepath, 'rb'),
                'mime_type': 'image/png'  # Assurez-vous que le MIME type est correct pour votre fichier
            }
        except Exception as e:
            print(f"Erreur lors de l'ouverture du fichier : {e}")
            return default_url_fetcher(url)
    return default_url_fetcher(url)

def global_report_pdf(text, analysis_text, articles, output_folder=output_folder):
    with open('theme.txt', 'r') as f:
        key_word = f.read().strip()
    
    # Préparer le contenu HTML du rapport
    response = f"""
<html>
<head>
    <style>
        body {{
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            font-size: 18px; /* Augmente la taille du texte en général */
            text-align: justify;
            max-width: 1500px;
            margin: auto;
        }}
        .page-break {{
            page-break-after: always;
        }}
        #table-of-contents {{
            font-size: 24px;
        }}
    </style>
</head>
<body>
    <h1 style="color: #333; text-align: center; margin-bottom: 20px;">Global News Report for {key_word}</h1>
    <div class='page-break' style='text-align: center;'>
        <img src='local:News_page.png' alt='AI News Analysis' style='width: 80%; margin-top: 50px;'>
    </div>
    <div id="table-of-contents" class='page-break'>
        <h2>Table of Contents</h2>
        <ul>
            <li>Headlines Summary</li>
            <li>Headlines Analysis</li>
            <li>Articles Analysis</li>
        </ul>
    </div>
    <div class='page-break'>
        <h2>Headlines Summary</h2>
        {markdown2.markdown(text, extras=["lists", "nl2br"])}
    </div>
    <div class='page-break'>
        <h2>Headlines Analysis</h2>
        {markdown2.markdown(analysis_text, extras=["lists", "nl2br"])}
    </div>
    <div class='page-break'>
        <h2>Articles</h2>
        <ul>
    """

    for article in articles:
        response += f"<li><a href='{article['link']}' target='_blank'>{article['title']}</a></li>"
    response += "</ul></div></body></html>"

    # Générer le fichier PDF
    output_pdf = f"{output_folder}/detailed_analysis_global_report_{key_word}.pdf"
    HTML(string=response, url_fetcher=custom_url_fetcher).write_pdf(output_pdf)

    pdf_writer = PyPDF2.PdfWriter()
    with open(output_pdf, 'rb') as file1:
        pdf_reader = PyPDF2.PdfReader(file1)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    for article in articles:
        if article.get('report_path'):
            with open(article['report_path'], 'rb') as file1:
                pdf_reader = PyPDF2.PdfReader(file1)
                for page in pdf_reader.pages:
                    pdf_writer.add_page(page)

    with open(output_pdf, 'wb') as f:
        pdf_writer.write(f)
