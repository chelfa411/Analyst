import asyncio
from pyppeteer import launch
import json
from tqdm import tqdm

async def fetch_text_from_url(url):
    """Utiliser Pyppeteer pour récupérer le texte des éléments spécifiés, en arrêtant le chargement si nécessaire."""
    try:
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')
        await page.goto(url)
        await page.waitFor(500)
        elements = await page.querySelectorAll('.wysiwyg.wysiwyg--all-content.css-ibbk12')
        texts = []
        for element in elements:
            text = await page.evaluate('(element) => element.textContent', element)
            texts.append(text.strip())
        await browser.close()
        return " ".join(texts)
    except Exception as e:
        print(f"Error fetching data from {url}: {str(e)}")
        await browser.close()
        return "error"

async def process_jsonl_file(input_file, output_file):
    """Traiter chaque ligne du fichier JSONL pour ajouter le texte extrait."""
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    # Utilisation de tqdm pour la barre de progression
    for line in tqdm(lines, desc="Processing URLs"):
        data = json.loads(line)
        url = data['link']  # Assumer que chaque objet JSON a un champ 'link'
        text_content = await fetch_text_from_url(url)
        data['content'] = text_content
        results.append(json.dumps(data))

    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(result + '\n')

# Chemins des fichiers d'entrée et de sortie
input_file = 'output.jsonl'
output_file = 'updated_output.jsonl'

# Exécuter le script de traitement
asyncio.get_event_loop().run_until_complete(process_jsonl_file(input_file, output_file))
