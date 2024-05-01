from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

def generate_gg_queries(article):
      
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    messages = [
        {"role": "user", "content": f"""You will later need to generate an in-depth analysis of this article: "{article.get('content')}" published in {article.get('date')}. You have the opportunity to 3 searches for additional information to enhance your analysis. 

    Please specify the information you require in the format of google researches.

    - Google Searches:  
        1. [Insert search term or phrase related to the question here]
        2. [Insert search term or phrase related to the question here]
        3. [Insert search term or phrase related to the question here]
        """},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)[0]

    # Trouver la dernière occurrence de [/INST] et tronquer tout avant cela
    last_inst_index = decoded.rfind('[/INST]') #<|assistant|>  [/INST]
    if last_inst_index != -1:
        response_start = last_inst_index + len('[/INST]')
        searches = decoded[response_start:].strip()

    if searches.endswith('</s>'):
            searches = searches[:-4].strip()
    print(searches)
    # Regex pour extraire le contenu entre guillemets
    pattern = r'\d+\.\s(.*?)(?=\s\d+\.\s|$)'

    # Utilisation de re.findall pour récupérer tous les matches
    phrases = re.findall(pattern, searches)

    # Affichage des phrases récupérées
    return phrases