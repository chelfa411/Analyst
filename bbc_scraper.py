import asyncio
from pyppeteer import launch
import json
import gradio as gr
from tqdm import tqdm
async def fetch_text_from_url(url):
    """Utilize Pyppeteer to fetch text from specified sections, stopping loading if necessary."""
    try:
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')
        await page.goto(url)
        await page.waitFor(2000)
        elements = await page.querySelectorAll('section[data-component="text-block"]')
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
    """Process each line of the JSONL file to add extracted text."""
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []
    cpt=0
    for line in lines:
        data = json.loads(line)
        url = data['link']
        if 'bbc' in url:
            text_content = await fetch_text_from_url(url)
            data['content'] = text_content
            results.append(json.dumps(data))
            cpt+=1
            print(cpt)

    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(result + '\n')

input_file = 'output.jsonl'
output_file = 'articles.jsonl'

asyncio.get_event_loop().run_until_complete(process_jsonl_file(input_file, output_file))
