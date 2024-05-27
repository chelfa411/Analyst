import gradio as gr
import os
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import jsonlines
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple, Optional
import threading
from upload_sys import setup_interface_upload_sys
from langchain.retrievers import ContextualCompressionRetriever
stop_generation_event = threading.Event()




global database
database = None



def stop_generation():
    stop_generation_event.set()  # Signal pour arrêter la génération



def load_articles_database(chunk_size,chunk_overlap):
        # Paramètres pour l'affichage des DataFrame
    pd.set_option("display.max_colwidth", None)

    # Conversion de JSONL à CSV
    def jsonl_to_csv(input_path, output_path):
        data = []
        with jsonlines.open(input_path) as reader:
            for obj in reader:
                data.append(obj)
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    jsonl_to_csv('articles.jsonl', 'articles.csv')

    # Chargement des articles
    loader = CSVLoader(file_path='articles.csv', source_column='link', metadata_columns=['title', 'author', 'label', 'score', 'bullet_points', 'date', 'summary', 'report_path', 'link','detailed_analysis'])
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=chunk_overlap,  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    )
    docs_processed = text_splitter.split_documents(data)
    #Indexation avec FAISS
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    print("DB article ok")
    return FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )


def load_pdfs(chunk_size,chunk_overlap):
    loader = PyPDFDirectoryLoader(path='./pdf_db')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=chunk_overlap,  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    )
    doc_processed = text_splitter.split_documents(data)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    print("pdf database ok")
    return FAISS.from_documents(doc_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)


def load_database(choice,chunk_size,chunk_overlap):
    print(choice)
    global database
    if choice == 'articles':
        database = load_articles_database(chunk_size,chunk_overlap)
    elif choice == 'pdf':
        database = load_pdfs(chunk_size,chunk_overlap)
    elif choice == 'both':
        database = load_articles_database(chunk_size,chunk_overlap)
        database.merge_from(load_pdfs(chunk_size,chunk_overlap))
    

def unload_db():
    global database
    database = None

def expand_query(query,context):
    messages = [
    {"role": "system", "content": """You are a query generator for helping retrieval augmented generation. You will need to provide an expanded query. You will be given user queries and the whole discussion before it. Answer only directly with the query. The query should be only keywords."""},
    {"role": "user", "content": f"Query:{query}\nContext{context}"},
]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    print(answer)
    return answer

def rag(
    question: str,
    knowledge_index = database,
    reranker: Optional[AutoModelForCausalLM] = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0"),
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:

    # Gather documents with retriever
    print("=> Retrieving documents...")
    retriever = knowledge_index.as_retriever(
    search_kwargs={"k": num_retrieved_docs}
)
    relevant_docs = retriever.invoke(question)
    cop_rev = relevant_docs.copy()
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        from ragatouille import RAGPretrainedModel
        print("=> Reranking documents...")
        compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker.as_langchain_document_compressor(), base_retriever=retriever)

        relevant_docs = compression_retriever.invoke(question)[:num_docs_final]
    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {i}:::\n{doc.page_content}\n" for i, doc in enumerate(relevant_docs)])

    return context,relevant_docs


css = """
h1 {
text-align: center;
display: block;
}
#duplicate-button {
margin: auto;
color: white;
background: #1565c0;
border-radius: 100vh;
}
"""
global tokenizer, model, terminators
tokenizer, model, terminators = None, None, None
def load_model():
    global tokenizer, model, terminators
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
        terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
    return 

def unload_model():
    global tokenizer, model, terminators
    if model is not None:
        del tokenizer, model, terminators
        tokenizer, model, terminators = None, None, None
    torch.cuda.empty_cache()
    return "Modèle déchargé!"

def chat_llama3_8b(history, num_retrieved_docs, num_docs_final):
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    global model, tokenizer, terminators, database
    if model is None:
        return "Model not loaded"
    conversation = [{"role": "system", "content": """
Depending on the user's message:
- If the message is a greeting or a general statement without a specific question or information need, respond in a friendly and engaging manner appropriate for casual conversation.
- If the message contains a specific question or request for information, use your knowledge and any provided context to give a comprehensive answer. Be sure to assess whether the context enhances the understanding of the question and integrate it thoughtfully if it does. Provide the source document number when relevant.
- Always aim to respond directly and appropriately to the content of the user's message, adjusting the depth and detail of your response to match the user's inquiry.
"""

}]
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    message = conversation[-2]['content']

    rag_m = conversation[:-2]

    query = expand_query(message,history)
    context, docs = rag(query,num_retrieved_docs=num_retrieved_docs, num_docs_final=num_docs_final, knowledge_index=database)


    rag_m.append( {"role": "user", "content": f"""Context:
    {context}
    ---
    Now here is the message from the user.

    Message: {message}"""})
    input_ids = tokenizer.apply_chat_template(rag_m, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.6,
        eos_token_id=terminators,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.             
    # if temperature == 0:
    #     generate_kwargs['do_sample'] = False
    stop_generation_event.clear()     
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    source_0 = "Source:\n\n"+docs[0].metadata.get('source')+"\n\n"+docs[0].page_content.strip()
    source_1 = "Source:\n\n"+docs[1].metadata.get('source')+"\n\n"+docs[1].page_content.strip()
    source_2 = "Source:\n\n"+docs[2].metadata.get('source')+"\n\n"+docs[2].page_content.strip()

    history[-1][1] = ""
    for text in streamer:
        if stop_generation_event.is_set():
                print("Génération arrêtée")
                break
        history[-1][1] += text
        #print(outputs)
        yield history,source_0,source_1,source_2
            


def user(user_message, history):
    return "", history + [[user_message, None]]

def undo_action(chat_history):
    if len(chat_history) >= 1:
        chat_history.pop()  # Retire le dernier message
    return chat_history

def retry_action(chat_history):
    print("debug: chat history before:",chat_history)
    if chat_history:
    # Réinitialiser la dernière réponse de l'assistant
        if len(chat_history[-1]) == 2:  # Assure que l'historique est correct
            chat_history[-1][1] = None  # Efface la dernière réponse
        else:
    # Si l'élément n'est pas conforme, on le réinitialise correctement
            chat_history[-1] = [chat_history[-1][0], None]
            print("debug: chat after",chat_history)

    return chat_history


def setup_chat_bot():
    with gr.Blocks() as app:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        with gr.Row():
            clear = gr.Button("Clear 🧹")
            undo = gr.Button("Undo ↩️")   # Emoji de flèche pour "annuler"
            retry = gr.Button("Retry 🔁")
            stop = gr.Button("Stop 🛑")

        clear.click(fn=stop_generation).then(lambda: None, None, chatbot, queue=False)
        stop.click(fn=stop_generation)
        with gr.Accordion("Parameters", open=False):
            with gr.Row():
                load = gr.Button('load model')
                unload = gr.Button('unload model')
                num_retrieved_doc = gr.Slider(minimum=1,maximum=100,value=30,step=1, label="First number of documents ofr RAG",show_label=True)
                num_final_doc = gr.Slider(minimum=1,maximum=100,step=1,value=5,label="Final number of documents for RAG",show_label=True)
        with gr.Accordion("Database", open=False):
            with gr.Row():
                choice = gr.Dropdown(choices=['articles','pdf','both'],label='Select a database',value='articles')
                loadb = gr.Button("Load database")
                unloadb = gr.Button("Unload database")
            with gr.Row():
                chunk_size = gr.Slider(minimum=200,maximum=4000,value=1000,step=1,label="chunk_size",show_label=True)
                chunk_overlap = gr.Slider(minimum=50,maximum=1000,value=200,step=1,label="chunk_overlap",show_label=True)
        with gr.Accordion('Sources',open=False):
            with gr.Row():
                doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
            with gr.Row():
                doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
            with gr.Row():
                doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
        with gr.Accordion('Upload pdf', open=False):
            setup_interface_upload_sys()
        loadb.click(fn=load_database, inputs=[choice,chunk_size,chunk_overlap],outputs=msg)
        unloadb.click(fn=unload_db)
        load.click(fn=load_model, outputs=msg)
        unload.click(fn=unload_model)
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            chat_llama3_8b, [chatbot,num_retrieved_doc,num_final_doc], [chatbot,doc_source1,doc_source2,doc_source3]
        )
        undo.click(fn=stop_generation).then(fn=undo_action,inputs=chatbot,outputs=chatbot)
        retry.click(fn=stop_generation).then(fn=retry_action,inputs=[chatbot],outputs=chatbot).then(fn=chat_llama3_8b, inputs=[chatbot,num_retrieved_doc,num_final_doc],outputs=[chatbot, doc_source1, doc_source2,doc_source3])
    return app

if __name__ == "__main__":
    demo = setup_chat_bot()
    demo.launch()