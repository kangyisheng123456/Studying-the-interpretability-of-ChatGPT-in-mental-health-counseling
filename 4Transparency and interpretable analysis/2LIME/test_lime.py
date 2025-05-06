import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

data_path = "test_ai_human.csv"
df = pd.read_csv(data_path)
texts = df['text'].tolist()
true_labels = df['label'].tolist()

model_path = "./local_pretrained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)

def predict_proba(text_list):
    results = pipeline(text_list)
    prob_matrix = []
    for result in results:
        if result[0]['label'] == 'LABEL_0':
            prob_matrix.append([result[0]['score'], result[1]['score']])
        else:
            prob_matrix.append([result[1]['score'], result[0]['score']])
    return np.array(prob_matrix)

class_names = ['HUMAN', 'AI']
explainer = LimeTextExplainer(class_names=class_names)

idx = 10
text_instance = texts[idx]
true_label = true_labels[idx]

print(text_instance)
print("\nTrue Label:", class_names[true_label])
print("Predicted Probabilities:", predict_proba([text_instance])[0])

exp = explainer.explain_instance(text_instance, predict_proba, num_features=10)

print("\nTop words contributing to classification:")
for word, weight in exp.as_list():
    print(f"{word}: {weight:.4f}")

fig = exp.as_pyplot_figure()
plt.title("LIME Explanation for HUMAN vs AI")
plt.tight_layout()
plt.savefig("lime_explanation.png", dpi=300)

exp.save_to_file("lime_explanation.html")
exp.show_in_notebook(text=True)
