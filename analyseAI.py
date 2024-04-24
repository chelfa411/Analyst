import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
import gradio as gr
from pdfGenerator import article_to_pdf, convert_jsonl_to_pdf

def generate_detailed_analysis(model, tokenizer, device, bullet_points_text):
    prompt = """
    Based on the summarized bullet points provided from the article, please conduct a detailed and flexible analysis. 
    Evaluate and decide which of the following dimensions are relevant based on the information available in the summary. 
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



    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I will analyze the points and provide a comprehensive insight:"},
        {"role": "user", "content": bullet_points_text}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(encodeds, max_new_tokens=4096, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)[0]

    # Trouver la dernière occurrence de [/INST] et tronquer tout avant cela
    last_inst_index = decoded.rfind('[/INST]')
    if last_inst_index != -1:
        response_start = last_inst_index + len('[/INST]')
        detailed_analysis = decoded[response_start:].strip()
    else:
        detailed_analysis = "No detailed analysis available."

    # Supprimer la balise '</s>' si elle est présente à la fin du texte généré
    if detailed_analysis.endswith('</s>'):
        detailed_analysis = detailed_analysis[:-4].strip()

    return detailed_analysis

input_filename = 'summary.jsonl'
output_filename = 'analysis.jsonl'
def process_json(t_number, demo, input_file=input_filename, output_file=output_filename, progress=gr.Progress()):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    # Configuration pour utiliser le GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    with open(input_file, 'r', encoding='utf-8') as file:
        articles = [json.loads(line) for line in file]
        articles = articles[:t_number]

    results = []
    response = ""
    for article in progress.tqdm(articles, desc="Analyzing Articles"):
        bullet_points = article.get('bullet_points', '')
        analysis = generate_detailed_analysis(model, tokenizer, device, bullet_points)
        article['detailed_analysis'] = analysis
        fp = '/tmp/gradio/pdf_outputs'+f"/detailed_analysis_{article.get('title')}.pdf"
        article['report_path'] = fp
        response += f"<p><a href='{demo.share_url}/file={fp}' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report</a></p>"
        results.append(article)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write('\n')

    convert_jsonl_to_pdf()

    return response



def analyse(article,demo):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    # Configuration pour utiliser le GPU si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    article_to_pdf(generate_detailed_analysis(model, tokenizer, device, article))

    return f"<p><a href='{demo.share_url}/file=/tmp/gradio/pdf_outputs/detailed_analysis_local.pdf' target='_blank' style='color: #007BFF; text-decoration: none;'>View Report</a></p>"

