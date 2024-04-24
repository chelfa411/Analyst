import gradio as gr
import pandas as pd
import json
import markdown2

# Données d'exemple pour les articles
file_choices = {
    "Articles from Feeds": 'output.jsonl',
    'Articles after content extraction': 'articles.jsonl',
    'Articles after classification': 'sorted.jsonl',
    'Articles after summarization': 'summary.jsonl',
    'Articles after analysis': 'analysis.jsonl'
}

def load_articles_from_jsonl(file_path):
    articles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            articles.append(json.loads(line))
    return articles

# Charger les articles au démarrage de l'application
file_path = 'articles.jsonl'
articles = load_articles_from_jsonl(file_path)
current_search_results = articles
current_page = 0

def refresh_articles():
    global articles, current_search_results, current_page
    articles = load_articles_from_jsonl(file_path)
    current_search_results = articles
    current_page = 0
    return get_dataframe(0)

def get_dataframe(page):
    start_index = page * 10
    end_index = min(start_index + 10, len(current_search_results))
    if start_index < len(current_search_results):
        df = pd.DataFrame(current_search_results[start_index:end_index])
        df['index'] = range(start_index, end_index)
        df['theme'] = [article.get('label', 'Not classified yet') for article in current_search_results[start_index:end_index]]
        df['score'] = [article.get('score', 'Not available yet') for article in current_search_results[start_index:end_index]]


        return df[['index', 'title', 'theme', 'score']]
    else:
        df = pd.DataFrame(current_search_results[start_index:end_index])
        df['theme'] = [article.get('theme', 'Not classified yet') for article in current_search_results[start_index:end_index]]
        df['score'] = [article.get('score', 'Not available yet') for article in current_search_results[start_index:end_index]]

        return pd.DataFrame(columns=['index', 'title','theme','score'])

def search_articles(query=""):
    global current_search_results, current_page
    current_page = 0
    if query:
        current_search_results = [article for article in articles if query.lower() in article['title'].lower()]
    else:
        current_search_results = articles
    return get_dataframe(current_page)

def next_page():
    global current_page
    if (current_page + 1) * 10 < len(current_search_results):
        current_page += 1
    return get_dataframe(current_page)

def previous_page():
    global current_page
    if current_page > 0:
        current_page -= 1
    return get_dataframe(current_page)

def confirm_selection(index, demo):
    try:
        float_index = float(index)
        if not float_index.is_integer():
            return "Error: Please enter a whole number without decimals."
        int_index = int(float_index)
        if 0 <= int_index < len(current_search_results):
            article = current_search_results[int_index]
            response = f"""
<div style='font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 18px; text-align: justify; max-width: 1200px; margin: auto;'>
    <h1 style='color: #333;'>Article's Data</h1>
    <h2>Date of Publication:</h2>
    <p style='background-color: #f9f9f9; padding: 10px; border-left: 5px solid #007BFF;'>{article["date"]}</p>
    <h2>Link:</h2>
    <p><a href='{article["link"]}' target='_blank' style='color: #007BFF; text-decoration: none;'>Read Article</a></p>
"""
            if "report_path" in article:
                response += f"""
    <h2>Report:</h2>
    <p><a href='{demo.share_url}/file={article["report_path"]}' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report</a></p>
"""
            if "bullet_points" in article:
                bullet_points_html = markdown2.markdown(article["bullet_points"], extras=["lists", "nl2br"])
                response += f"<h2>Summary:</h2>{bullet_points_html}"
            response += "</div>"
            return response
        else:
            return "Error: Index out of range"
    except ValueError:
        return "Error: Invalid input. Please enter a numeric index."

def create_search_article_interface(demo):
    with gr.Blocks() as app:
        with gr.Row():
            query_input = gr.Textbox(label="Search articles by title:")
            search_button = gr.Button("Search")
            reset_button = gr.Button("Reset")
        with gr.Row():
            refresh_button = gr.Button("Refresh")
            previous_button = gr.Button("Previous")
            next_button = gr.Button("Next")
        df = gr.Dataframe(value=get_dataframe(0), label="Articles")
        index_input = gr.Number(label="Enter the global index of the article:", minimum=0, step=1)
        submit_button = gr.Button("Show Details")
        output = gr.HTML(label="Article Details")
        search_button.click(fn=search_articles, inputs=query_input, outputs=df)
        previous_button.click(fn=previous_page, outputs=df)
        next_button.click(fn=next_page, outputs=df)
        submit_button.click(fn=lambda index: confirm_selection(index, demo), inputs=index_input, outputs=output)
        reset_button.click(fn=search_articles, outputs=df)
        refresh_button.click(fn=refresh_articles, outputs=df)
    return app

