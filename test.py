#!pip install transformers[sentencepiece]
from transformers import pipeline
text = "Angela Merkel is a politician in Germany and leader of the CDU. The global wrming is about to ruin the world"
hypothesis_template = "This text is about {}"
classes_verbalized = ["politics", "economy", "entertainment", "global warming"]
zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")  # change the model identifier here
output = zeroshot_classifier(text, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=True)
print(output)
