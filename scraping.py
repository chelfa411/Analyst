import feedparser
import json
from tqdm import tqdm
import gradio as gr
from datetime import datetime, timezone
all_urls = {
    'BBC World Asia': 'https://feeds.bbci.co.uk/news/world/asia/rss.xml',
    'BBC World Africa': 'https://feeds.bbci.co.uk/news/world/africa/rss.xml',
    'BBC World US and Canada': 'https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml',
    'BBC World Europe': 'https://feeds.bbci.co.uk/news/world/europe/rss.xml',
    'BBC World Latin America': 'https://feeds.bbci.co.uk/news/world/latin_america/rss.xml',
    'BBC World Middle East': 'https://feeds.bbci.co.uk/news/world/middle_east/rss.xml',
    'BBC News England': 'https://feeds.bbci.co.uk/news/england/rss.xml',
    'BBC News Northern Ireland': 'https://feeds.bbci.co.uk/news/northern_ireland/rss.xml',
    'BBC News Scotland': 'https://feeds.bbci.co.uk/news/scotland/rss.xml',
    'BBC News Wales': 'https://feeds.bbci.co.uk/news/wales/rss.xml',
    'France 24 English': 'https://www.france24.com/en/rss',
    'France 24 Europe': 'https://www.france24.com/en/europe/rss',
    'France 24 France': 'https://www.france24.com/en/france/rss',
    'France 24 Africa': 'https://www.france24.com/en/africa/rss',
    'France 24 Middle-East': 'https://www.france24.com/en/middle-east/rss',
    'France 24 Americas': 'https://www.france24.com/en/americas/rss',
    'France 24 Asia-Pacific': 'https://www.france24.com/en/asia-pacific/rss',
    'Guardian World Africa': 'https://www.theguardian.com/world/africa/rss',
    'Guardian World Americas': 'https://www.theguardian.com/world/americas/rss',
    'Guardian World Asia-Pacific': 'https://www.theguardian.com/world/asia-pacific/rss',
    'Guardian Australia News': 'https://www.theguardian.com/australia-news/rss',
    'Guardian World Europe News': 'https://www.theguardian.com/world/europe-news/rss',
    'Guardian World Middle East': 'https://www.theguardian.com/world/middleeast/rss',
    'Guardian World South and Central Asia': 'https://www.theguardian.com/world/south-and-central-asia/rss',
    'Guardian UK News': 'https://www.theguardian.com/uk-news/rss',
    'Guardian US News': 'https://www.theguardian.com/us-news/rss'
}

# Nom du fichier de sortie
output_file = 'output.jsonl'

def fetch_feeds_and_export_to_jsonl(selected_feeds, output_file=output_file, progress=gr.Progress()):
    # Ouvrir le fichier en mode écriture
    with open(output_file, 'w', encoding='utf-8') as file:
        # Itérer sur chaque URL dans la liste des flux RSS
        for feed_name in progress.tqdm(selected_feeds,desc='Parsing feeds'):
            # Parser le flux RSS à partir de l'URL donnée
            url = all_urls[feed_name]
            feed = feedparser.parse(url)
            
            # Itérer sur chaque entrée du flux RSS
            for entry in feed.entries:
                # Préparer un dictionnaire avec les données souhaitées
                data = {
                    'title': entry.title,
                    'date': entry.published if 'published' in entry else 'No date available',
                    'link': entry.link,
                    'summary': entry.summary if 'summary' in entry else 'No summary available',
                    'author': entry.author if 'author' in entry else 'No author available',
                    'bullet_points':'Unavailable',
                    'detailed_analysis': 'Unavailable',
                    'label': 'Unavaiable',
                    'score': 'Unavailable'
                }
                # Convertir le dictionnaire en JSON et écrire dans le fichier avec un saut de ligne
                file.write(json.dumps(data) + '\n')
     # Préparer la sortie pour la dernière mise à jour et les flux choisis
    last_update_info = 'Last update: ' + str(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")) + 'Z' + '\nFeeds updated:\n'
    feeds_updated = '\n'.join(selected_feeds)  # Joindre les noms des flux dans une seule chaîne

    # Écrire la dernière mise à jour et les flux choisis dans le fichier last_update.txt
    with open('last_update.txt', 'w') as f:
        f.write(last_update_info + feeds_updated)
    
    # Retourner les informations avec les flux mis à jour
    return last_update_info + feeds_updated

# Liste des URLs des flux RSS





