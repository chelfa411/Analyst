import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
import gradio as gr
from pdfGenerator import article_to_pdf, convert_jsonl_to_pdf, article_to_pdf_target

def generate_detailed_analysis(model, tokenizer, device, bullet_points_text, use_search = False, article=None):
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

    * **References**: [Cite all the references usd in the analysis.]

    Please ensure that each category you discuss is clearly justified with evidence from the article summary and integrates historical and geographical contexts where applicable. 
    Each point should be backed by citations from credible sources to validate the analysis. Please start directly in the format provided. Do not add any introductive phrase. Answer in the exact format given to you. Put the references section after the additionnal category.
    """

    if use_search:
        from search import make_researches_from_article
        research_results = make_researches_from_article(article)
        context = "\n".join(research_results)
        messages = [
            {"role": "user", "content": f"""Hello! Before we proceed, I want to share some information that I've gathered from the internet to help you understand the context better.
            This information is important for the task you will be handling. Please take a moment to review these details carefully.
            Remember, you don't need to take any action right now; just make sure you consider this information when I give you further instructions.
            Context:
            {context}"""},
            {"role": "assistant", "content": "Thank you for providing the information. I've reviewed the details you shared. Please go ahead and let me know what specific task you'd like me to perform, and I’ll use the information accordingly to assist you."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
            {"role": "user", "content": bullet_points_text}
        ]
        print(f"Debug: Using google search : {context}")
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
            {"role": "user", "content": bullet_points_text}           
        ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(encodeds, max_new_tokens=4096, do_sample=True)
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

input_filename = 'articles.jsonl'
output_filename = 'articles.jsonl'
def process_json(t_number, demo, use_search = False, input_file=input_filename, output_file=output_filename, progress=gr.Progress()):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    torch.cuda.empty_cache()
    # Configuration pour utiliser le GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", 
        device_map="auto", 
)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    with open(input_file, 'r', encoding='utf-8') as file:
        articles = [json.loads(line) for line in file]

    results = []
    response = ""
    for article in progress.tqdm(articles[:t_number], desc="Analyzing Articles"):
        bullet_points = article.get('content', '')
        analysis = generate_detailed_analysis(model, tokenizer, device, bullet_points, use_search=use_search, article=article)
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



def analyse(article_text,demo, use_search):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # Configuration pour utiliser le GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    article = {'content': article_text, 'date':'not available'}
    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", 
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    article_to_pdf(generate_detailed_analysis(model, tokenizer, device, article_text, article=article, use_search= use_search))

    return f"<p><a href='{demo.share_url}/file=/tmp/gradio/pdf_outputs/detailed_analysis_local.pdf' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report</a></p>"



from fuzzywuzzy import fuzz
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_object_by_title_and_analyse(demo, title, use_search, jsonl_file='articles.jsonl'):
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
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", 
            device_map="auto")   
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        analysis = generate_detailed_analysis(model, tokenizer, device, bullet_points, use_search=use_search, article=best_match)
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


