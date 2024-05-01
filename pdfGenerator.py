import json
from weasyprint import HTML
import os
input_jsonl = 'articles.jsonl'
output_folder = '/tmp/gradio/pdf_outputs'
import markdown2

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

def article_to_pdf_target(article, text, output_folder=output_folder):
    response = f"""
    <div style='font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 16px; text-align: justify; max-width: 1500px; margin: auto;'>
    <h1 style='color: #333;'>News analysis</h1>
    """
    analysis_html = markdown2.markdown(text, extras=["lists", "nl2br"])
    response += f"<h2>Analysis:</h2>{analysis_html}"
    response += "</div>"
    title = article.get('title', 'No Title Provided')
    title_cleaned = title.replace('/', '_').replace(' ', '-').replace("'", "_").replace("|","pipe").replace("?","q")
    output_pdf = f"{output_folder}/detailed_analysis_{title_cleaned}.pdf"
    HTML(string=response).write_pdf(output_pdf)

    