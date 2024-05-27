import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
import gradio as gr
import re
from pdfGenerator import article_to_pdf, convert_jsonl_to_pdf, article_to_pdf_target, global_report_pdf

def generate_detailed_analysis(model, tokenizer, device, bullet_points_text, model_name, use_search = False, article=None):
    prompt = """
    Based on the provided article, please conduct a detailed and flexible analysis. 
    Evaluate and decide which of the following dimensions are relevant based on the information available. 
    For each relevant dimension, provide a clear justification with specific examples, all formatted as bullet points.
    Cite any sources or data that support your analysis. If any aspect is not relevant, omit it. 
    Additionally, if you identify new impacts or categories that are pertinent but not listed, 
    please create and elaborate on these, ensuring to use bullet points and cite sources for your reasoning:

    * **Geopolitical Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]


    * **Military Strength and Strategy Analysis**:
    * *Short-term*: [Discuss the immediate military strategies and deployments relevant to the situation, provide specific examples, and cite sources.]
    * *Long-term*: [Explore potential shifts in military alliances, defense capabilities, and strategic postures over the long term, provide specific examples, and cite sources.]
    

    * **Economic Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Social Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Technological Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Environmental Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Health Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Legal/Regulatory Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Security Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Educational Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    

    * **Public Policy Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    
    * **Other category [Replace by the neame of the category you want to add]**
    * *Short-term*: [Discuss if relevant, provide specific examples, and cite sources.]
    * *Long-term*: [Discuss if relevant, provide specific examples, and cite sources.]

    * **References**: [Cite all the references used in the analysis.]

    Please ensure that each category you discuss is clearly justified with evidence from the article summary and integrates historical and geographical contexts where applicable. 
    Each point should be backed by citations from credible sources to validate the analysis. Please start directly in the format provided. Do not add any introductive phrase. Answer in the exact format given to you. Put the references section after the additionnal category.
    """

    if use_search:
        from search import make_researches_from_article
        research_results = make_researches_from_article(article)
        context = "\n".join(research_results)
        messages = [
            {"role": "user", "content": f"""
Hello! Before we proceed, I want to provide you with some context based on information I've gathered from the internet. This context is crucial for the task you will be handling. Please review these details carefully:
{context}
Remember, your role right now is not to take any immediate action, but to integrate this context into the guidance I will provide next. Additionally, feel free to utilize your own knowledge base to enrich the task where appropriate. Ensure that any use of this information is accurate and verify the facts if necessary before proceeding.
"""},
            {"role": "assistant", "content": "Thank you for providing the information. I've reviewed the details you shared. Please go ahead and let me know what specific task you'd like me to perform, and I’ll use the information accordingly to assist you."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
            {"role": "user", "content": bullet_points_text}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
            {"role": "user", "content": bullet_points_text}           
        ]

    if model_name == 'Mistral-7B-Instruct-v0.2':
        detailed_analysis = generate_mistral(model=model, tokenizer=tokenizer, messages=messages, device=device)

    elif model_name == 'Meta-Llama-3-8B-Instruct':
        detailed_analysis = generate_llama_7b(model=model, tokenizer=tokenizer, messages=messages, device=device)
    return detailed_analysis

input_filename = 'articles.jsonl'
output_filename = 'articles.jsonl'


def process_json(t_number, demo, model_name, use_search = False, input_file=input_filename, output_file=output_filename, progress=gr.Progress()):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.cuda.empty_cache()
    # Configuration pour utiliser le GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(model_name)

    with open(input_file, 'r', encoding='utf-8') as file:
        articles = [json.loads(line) for line in file]

    results = []
    response = ""
    for article in progress.tqdm(articles[:t_number], desc="Analyzing Articles"):
        bullet_points = article.get('content', '')
        analysis = generate_detailed_analysis(model, tokenizer, device, bullet_points, model_name=model_name,use_search=use_search, article=article)
        article['detailed_analysis'] = analysis
        title_cleaned = ((article.get('title').replace('/', '_')).replace(' ', '-')).replace("'", "_").replace("|","pipe").replace("?","q")
        fp = f"/tmp/gradio/pdf_outputs/detailed_analysis_{title_cleaned}.pdf"
        article['report_path'] = fp
        response += f"<p><a href='{demo.share_url}/file={fp}' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report: {title_cleaned}</a></p>"
        results.append(article)
    for article in articles[t_number:]:
        results.append(article)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write('\n')

    convert_jsonl_to_pdf()

    return response



def analyse(article_text,demo, model_name, use_search=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # Configuration pour utiliser le GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    article = {'content': article_text, 'date':'not available'}
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model(model_name)


    article_to_pdf(generate_detailed_analysis(model, tokenizer, device, article_text, model_name=model_name,article=article, use_search= use_search))

    return f"<p><a href='{demo.share_url}/file=/tmp/gradio/pdf_outputs/detailed_analysis_local.pdf' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report</a></p>"



from fuzzywuzzy import fuzz
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_object_by_title_and_analyse(demo, title, model_name, use_search=False, jsonl_file='articles.jsonl'):
    articles = []
    found = False
    best_match_score = 70  # Définir un seuil de correspondance, par exemple 70%
    best_match = None
    fp = None  # Initialisation de fp pour éviter des erreurs si non défini

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        articles = [json.loads(line) for line in file]

    print(f"Debug: Starting search for title '{title}'.")
    for data in articles:
        # Utilisation de fuzz.partial_ratio pour comparer les titres de manière insensible à la casse
        match_score = fuzz.partial_ratio(title.lower(), data.get('title', '').lower())
        if match_score > best_match_score:
            best_match_score = match_score
            best_match = data
    if best_match:
        found = True
        print(f"Debug: Best match confirmed: {best_match.get('title')}")
        bullet_points = best_match.get('content')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, tokenizer = load_model(model_name)

        analysis = generate_detailed_analysis(model, tokenizer, device, bullet_points, model_name=model_name,use_search=use_search, article=best_match)
        best_match['detailed_analysis'] = analysis
        title_cleaned = ((best_match.get('title').replace('/', '_')).replace(' ', '-')).replace("'", "_").replace("|","pipe").replace("?","q")
        fp = f"/tmp/gradio/pdf_outputs/detailed_analysis_{title_cleaned}.pdf"
        best_match['report_path'] = fp
        article_to_pdf_target(best_match, analysis)

    if found:
        with open(jsonl_file, 'w', encoding='utf-8') as outfile:
            for article in articles:
                json.dump(article, outfile)
                outfile.write('\n')
        return f"<a href='{demo.share_url}/file={fp}' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report</a>"
    else:
        print("Debug: No matching articles found.")
        return "Article not found"



def generate_mistral(model, tokenizer, messages, device):
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(encodeds, max_new_tokens=8192, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)[0]

    # Trouver la dernière occurrence de [/INST] et tronquer tout avant cela
    last_inst_index = decoded.rfind('[/INST]') #<|assistant|>  [/INST]
    if last_inst_index != -1:
        response_start = last_inst_index + len('[/INST]')
        detailed_analysis = decoded[response_start:].strip()
    else:
        detailed_analysis = "No detailed analysis available."

    # Supprimer la balise '</s>' si elle est présente à la fin du texte généré
    if detailed_analysis.endswith('</s>'):
        detailed_analysis = detailed_analysis[:-4].strip()

    return detailed_analysis

def generate_llama_7b(model, tokenizer, messages, device):
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
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    detailed_analysis = tokenizer.decode(response, skip_special_tokens=True)
    if "here" in detailed_analysis.split("\n")[0].lower():
        detailed_analysis = '\n'.join(detailed_analysis.split('\n')[1:]).strip()

    return detailed_analysis


def load_model(model_name):
    if model_name == 'Mistral-7B-Instruct-v0.2':
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", 
            device_map="auto")   
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    elif model_name == 'Meta-Llama-3-8B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        )
    return model, tokenizer

def gen(model_name, messages):
    model, tokenizer = load_model(model_name)
    device = model.device
    if model_name == 'Mistral-7B-Instruct-v0.2':
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        generated_ids = model.generate(encodeds, max_new_tokens=8192, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)[0]

        # Trouver la dernière occurrence de [/INST] et tronquer tout avant cela
        last_inst_index = decoded.rfind('[/INST]') #<|assistant|>  [/INST]
        if last_inst_index != -1:
            response_start = last_inst_index + len('[/INST]')
            answer = decoded[response_start:].strip()
        else:
            answer = "Debug: Error global generation mistral, end token not found"

        # Supprimer la balise '</s>' si elle est présente à la fin du texte généré
        if answer.endswith('</s>'):
            answer = answer[:-4].strip()
    
    elif model_name == 'Meta-Llama-3-8B-Instruct':
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
            max_new_tokens=8192,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        answer = tokenizer.decode(response, skip_special_tokens=True)

    return answer






def gen_global_report(model_name, score=0.70):
    articles = []

    with open('articles.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            article = json.loads(line)
            if article.get('score') > score:
                articles.append(article)

    prompt = """Hello, I am providing you with summaries of articles along with their titles. Could you combine them into a single summary article that integrates the key information from each summary while maintaining coherence and a journalistic style? Do not add introductive phrases.Here are the summaries:\n"""

    for article in articles:
        title = article['title']
        bullet = article['bullet_points']
        prompt += title
        prompt += '\n\n'
        prompt += bullet
        prompt += '\n\n'
    messages = [
    {"role": "user", "content": prompt},      
    ]

    answer = gen(model_name, messages)
    return answer, articles









def generate_detailed_analysis_global(model, tokenizer, device, bullet_points_text, model_name, use_search = False, article=None):
    with open('theme.txt','r') as f:
        keyword = f.read().strip()
    prompt = f"""
    Based on the provided summary of articles related to the subject {keyword}, please conduct a detailed and flexible analysis. 
    Evaluate and decide which of the following dimensions are relevant based on the information available. 
    For each relevant dimension, provide a clear justification with specific examples, all formatted as bullet points.
    Cite any sources or data that support your analysis. If any aspect is not relevant, omit it. 
    Additionally, if you identify new impacts or categories that are pertinent but not listed, 
    please create and elaborate on these:

    * **Geopolitical Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]


    * **Military Strength and Strategy Analysis**:
    * *Short-term*: [Discuss the immediate military strategies and deployments relevant to the situation, provide specific examples.]
    * *Long-term*: [Explore potential shifts in military alliances, defense capabilities, and strategic postures over the long term, provide specific examples.]

    * **Economic Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Social Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Technological Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Environmental Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples]
    

    * **Health Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Legal/Regulatory Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Security Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Educational Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    

    * **Public Policy Impact**:
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]
    
    * **Other category [Replace by the neame of the category you want to add]**
    * *Short-term*: [Discuss if relevant, provide specific examples.]
    * *Long-term*: [Discuss if relevant, provide specific examples.]

    Please ensure that each category you discuss integrates historical and geographical contexts where applicable. 
    Please start directly in the format provided. Do not add any introductive phrase. 
    """

    if use_search:
        from search import make_researches_from_article
        research_results = make_researches_from_article(article)
        context = "\n".join(research_results)
        messages = [
            {"role": "user", "content": f"""
Hello! Before we proceed, I want to provide you with some context based on information I've gathered from the internet. This context is crucial for the task you will be handling. Please review these details carefully:
{context}
Remember, your role right now is not to take any immediate action, but to integrate this context into the guidance I will provide next. Additionally, feel free to utilize your own knowledge base to enrich the task where appropriate. Ensure that any use of this information is accurate and verify the facts if necessary before proceeding.
"""},
            {"role": "assistant", "content": "Thank you for providing the information. I've reviewed the details you shared. Please go ahead and let me know what specific task you'd like me to perform, and I’ll use the information accordingly to assist you."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
            {"role": "user", "content": bullet_points_text}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
            {"role": "user", "content": bullet_points_text}           
        ]

    if model_name == 'Mistral-7B-Instruct-v0.2':
        detailed_analysis = generate_mistral(model=model, tokenizer=tokenizer, messages=messages, device=device)

    elif model_name == 'Meta-Llama-3-8B-Instruct':
        detailed_analysis = generate_llama_7b(model=model, tokenizer=tokenizer, messages=messages, device=device)
    return detailed_analysis+f"\n\n **Context Used:** \n {context}" if use_search else detailed_analysis














def global_report(demo, model_name, score=0.70, use_search=False, individual=False, progress = gr.Progress()):
    from datetime import datetime
    sum, articles = gen_global_report(model_name,score)
    model, tokenizer = load_model(model_name=model_name)
    device = model.device
    if individual:
        articles3=[]
        for article in progress.tqdm(articles, desc=f"Generating Individual Analyse reports"):
            if not article.get('report_path'):
                detailed_analysis = generate_detailed_analysis(model, tokenizer, device, article.get('content'),model_name, use_search, article)
                article['detailed_analysis'] = detailed_analysis
                title_cleaned = ((article.get('title').replace('/', '_')).replace(' ', '-')).replace("'", "_").replace("|","pipe").replace("?","q")
                fp = f"/tmp/gradio/pdf_outputs/detailed_analysis_{title_cleaned}.pdf"
                article['report_path'] = fp
                article_to_pdf_target(article)
        with open('articles.jsonl', 'r', encoding='utf-8') as file:
            articles2 = [json.loads(line) for line in file]
        articles3 = articles+articles2[len(articles):]
        with open('articles.jsonl', 'w', encoding='utf-8') as outfile:
                for article in articles3:
                    json.dump(article, outfile)
                    outfile.write('\n')

    article = {'content':sum, 'date':str(datetime.now().date())}
    analysis = generate_detailed_analysis_global(model,tokenizer,device,sum,model_name, use_search,article)
    global_report_pdf(sum, analysis, articles)
    with open('theme.txt','r') as f:
        key_word = f.read().strip()
    fp = f"/tmp/gradio/pdf_outputs/detailed_analysis_global_report_{key_word}.pdf"

    return f"<a href='{demo.share_url}/file={fp}' target='_blank' style='color: #007BFF; text-decoration: none;'>View Global Report</a>"


