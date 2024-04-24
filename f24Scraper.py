import asyncio
from pyppeteer import launch
import json
from tqdm import tqdm
import gradio as gr 

async def fetch_text_from_url(url):
    """Utiliser Pyppeteer pour récupérer le texte des sections spécifiées, en arrêtant le chargement si nécessaire."""
    try:
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')
        await page.goto(url)
        await page.waitFor(500)
        elements = await page.querySelectorAll('div[class*="t-content__body"] p')
        texts = []
        for element in elements:
            text = await page.evaluate('(element) => element.textContent', element)
            texts.append(text.strip())
        await browser.close()
        return " ".join(texts)
    except Exception as e:
        print(f"Error fetching data from {url}: {str(e)}")
        try:
            # Tenter de récupérer le texte même après une erreur
            elements = await page.querySelectorAll('div[class*="article-body-commercial-selector"] p')
        
            texts = []
            for element in elements:
                text = await page.evaluate('(element) => element.textContent', element)
                texts.append(text.strip())
            await browser.close()
            return " ".join(texts)
        except Exception as e:
            print(f"Failed to fetch text after error: {str(e)}")
            await browser.close()
            return "error"

async def process_jsonl_file(input_file, output_file):
    """Traiter chaque ligne du fichier JSONL pour ajouter le texte extrait."""
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    cpt=0
    # Utilisation de tqdm pour la barre de progression
    for line in lines:
        data = json.loads(line)
        url = data['link']
        if 'france24' in url:
            text_content = await fetch_text_from_url(url)
            data['content'] = text_content
            results.append(json.dumps(data))
            cpt+=1
            print(cpt)

    with open(output_file, 'a', encoding='utf-8') as file:
        for result in results:
            file.write(result + '\n')

# Chemins des fichiers d'entrée et de sortie


# Exécuter le script de traitement

input_file = 'output.jsonl'
output_file = 'articles.jsonl'

asyncio.get_event_loop().run_until_complete(process_jsonl_file(input_file, output_file))
