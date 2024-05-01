from bs4 import BeautifulSoup
import urllib
import requests
import nltk
import torch
from typing import Union
from sentence_transformers import SentenceTransformer, util
from summary2 import process_article_list
from concurrent.futures import ThreadPoolExecutor, as_completed
from searchrequestgenerator import generate_gg_queries
class GoogleSearch:
    def __init__(self, query: str) -> None:
        self.query = query
        escaped_query = urllib.parse.quote_plus(query)
        self.URL = f"https://www.google.com/search?q={escaped_query}"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3538.102 Safari/537.36"
        }
        self.links = self.get_initial_links()
        self.all_page_data = self.all_pages()

    def clean_urls(self, anchors: list[str]) -> list[str]:
        links: list[str] = []
        for a in anchors:
            links.append(
                list(filter(lambda l: l.startswith("url=http"), a["href"].split("&")))
            )

        links = [
            urllib.parse.unquote(link.split("url=")[-1])
            for sublist in links
            for link in sublist
            if len(link) > 0
        ]

        return links

    def read_url_page(self, url: str) -> str:
        response = requests.get(url, headers=self.headers, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(strip=True)

    def get_initial_links(self) -> list[str]:
        print("Searching Google...")
        response = requests.get(self.URL, headers=self.headers, verify=False)
        soup = BeautifulSoup(response.text, "html.parser")
        anchors = soup.find_all("a", href=True)
        return self.clean_urls(anchors)

    def all_pages(self) -> list[tuple[str, str]]:
        data: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(self.read_url_page, url): url for url in self.links}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    output = future.result()
                    data.append((url, output))
                except requests.exceptions.HTTPError as e:
                    print(f"Error fetching {url}: {e}")
        return data

class Document:
    def __init__(self, data: list[tuple[str, str]], min_char_len: int) -> None:
        self.data = data
        self.min_char_len = min_char_len

    def chunk_page(self, page_text: str) -> list[str]:
        min_len_chunks: list[str] = []
        chunk_text = nltk.tokenize.sent_tokenize(page_text)
        sentence: str = ""
        for sent in chunk_text:
            if len(sentence) + len(sent) > self.min_char_len:
                min_len_chunks.append(sentence)
                sentence = sent
            else:
                sentence += " " + sent
        if sentence:
            min_len_chunks.append(sentence)  # Append last remaining sentence if any
        return min_len_chunks

    def doc(self) -> tuple[list[str], list[str]]:
        print("Creating Document...")
        chunked_data: list[str] = []
        urls: list[str] = []
        for url, dataitem in self.data:
            data = self.chunk_page(dataitem)
            chunked_data.extend(data)  # Using extend to flatten the list
            urls.extend([url] * len(data))
        return chunked_data, urls

class SemanticSearch:
    def __init__(self, doc_chunks: tuple[list, list], model_path: str, device: str) -> None:
        self.doc_chunks, self.urls = doc_chunks
        self.st = SentenceTransformer(model_path, device=device)

    def semantic_search(self, query: str, k: int = 10):
        print("Searching Top k in document...")
        query_embedding = self.get_embedding(query)
        doc_embeddings = self.get_embedding(self.doc_chunks)
        scores = util.dot_score(query_embedding, doc_embeddings)[0]
        top_k = torch.topk(scores, k=k).indices.cpu().tolist()
        return [(self.doc_chunks[i], self.urls[i]) for i in top_k]

    def get_embedding(self, text: Union[list[str], str]):
        return self.st.encode(text)



def make_researches(query_list):
    list_of_researches = []
    for query in query_list:
        gs = GoogleSearch(query)
        document_processor = Document(gs.all_page_data, min_char_len=512)
        doc_chunks, urls = document_processor.doc()
        searcher = SemanticSearch((doc_chunks, urls), model_path="sentence-transformers/all-mpnet-base-v2", device="cuda")
        results = [(t[0]+"\nSource: "+t[1]) for t in searcher.semantic_search(query, k=16)]
        list_of_researches.append(results)
    return list_of_researches

def make_researches_from_article(article):
    query_list = generate_gg_queries(article)
    print(query_list)
    outlist = make_researches(query_list)
    answer_1 = []
    for list in outlist:
        answer_1 += list
    return answer_1 #process_article_list(answer_1) 


