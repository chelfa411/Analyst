import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def generate_wiki_query(prompt):
    torch.random.manual_seed(0)
    device = "cuda"  # Assurez-vous que CUDA est disponible ou changez à "cpu"

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # Définir les messages de l'utilisateur
    messages = [
        {"role": "user", "content": """You are an assistant returning text queries to search Wikipedia articles containing relevant information about the prompt. Write the queries and nothing else. Example: [prompt] Tell me about the heatwave in Europe in summer 2023 [query] heatwave, weather, temperatures, europe, summer 2023."""},
        {"role": "user", "content": f"[prompt] {prompt} [query]"},
    ]

    # Créer un pipeline de génération de texte
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Configurer les paramètres de génération
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": False,
    }

    # Générer la requête
    output = pipe(messages, **generation_args)
    query = output[0]['generated_text'].strip()
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(query)
    return query

def chat_with_wikipedia(prompt, lang='en', load_max_docs=5):
    query = generate_wiki_query(prompt)
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


def split_text_into_chunks(data, chunk_size=1024):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks



import chromadb
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 


def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        # persist_directory=default_persist_directory
    )
    return vectordb


def generate_db_query(prompt):
    torch.random.manual_seed(0)
    device = "cuda"  # Assurez-vous que CUDA est disponible ou changez à "cpu"

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # Définir les messages de l'utilisateur
    messages = [
        {"role": "user", "content": """You are an assistant returning text queries to search data with similarity research in vectorise database. Write the queries and nothing else. Example: [prompt] Tell me about the heatwave in Europe in summer 2023 [query] heatwave, weather, temperatures, europe, summer 2023."""},
        {"role": "user", "content": f"[prompt] {prompt} [query]"},
    ]

    # Créer un pipeline de génération de texte
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Configurer les paramètres de génération
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": False,
    }

    # Générer la requête
    output = pipe(messages, **generation_args)
    query = output[0]['generated_text'].strip()
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(query)
    return query



def wiki_search(prompt, doc_num, doc_load):
    data = split_text_into_chunks(chat_with_wikipedia(prompt, load_max_docs=doc_load))
    collection_name = "temp_wiki_articles"
    vectordb = create_db(data, collection_name)
    v_query = generate_db_query(prompt)
    print(v_query)
    result = vectordb.similarity_search(v_query, doc_num)
    return result







def generate_prompt(question, pages):
    prompt = f"You have retrieved the following extracts from the following Wikipedia pages:\n"
    for i, page in enumerate(pages, start=1):
        title, content = page.metadata['title'], page.page_content
        prompt += f"Page {i}: {title}\n{content}\n\n"
    prompt+= "You are expected to give truthful answers based on the previous extracts. If it doesn't include relevant information for the request just say so and don't make up false information.\n"
    prompt += f"Question:\n{question}"
    return prompt


def answer_based_wiki(prompt, doc_num, doc_load):
    result = wiki_search(prompt, doc_num, doc_load)
    final_prompt = generate_prompt(prompt, result)
    torch.random.manual_seed(0)
    device = "cuda"  # Assurez-vous que CUDA est disponible ou changez à "cpu"

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # Définir les messages de l'utilisateur
    messages = [
        {"role": "user", "content": final_prompt},
    ]

    # Créer un pipeline de génération de texte
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Configurer les paramètres de génération
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": False,
    }

    # Générer la requête
    output = pipe(messages, **generation_args)
    query = output[0]['generated_text'].strip()
    return query

print(answer_based_wiki("Casualties ukraine from 2021 to 2024",10,20))