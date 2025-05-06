import os
import csv
import torch
import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from sklearn.metrics import precision_score, recall_score, f1_score

def csv_to_tuple(file_document, encoding='utf-8-sig'):
    with open(file_document, 'r', encoding=encoding, newline='') as f_r:
        reader_obj = csv.reader(f_r)
        data_list = []
        for i in reader_obj:
            label = i[0]
            content = i[1]
            tuple_1 = (label, content)
            data_list.append(tuple_1)
        return data_list

def process_data(file_path):
    data = csv_to_tuple(file_path)
    labels = []
    content = []
    for row in data:
        if row[0] == 'ai':
            labels.append(0)
        elif row[0] == 'human':
            labels.append(1)
        content.append(row[1])
    return labels, content

def evaluate_model(pipeline, content, labels):
    preds = []
    for text in content:
        pred = pipeline(text)
        score_ai = pred[0][0]['score']
        score_human = pred[0][1]['score']
        preds.append(0 if score_ai > score_human else 1)
    accuracy = np.mean(np.array(preds) == np.array(labels))
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1

def lime_explain_and_save_html(pipeline, X_test, index_to_explain, html_save_path):
    explainer = LimeTextExplainer(class_names=['ai', 'human'])
    text_to_explain = X_test[index_to_explain]
    def predict_proba(texts):
        probs = []
        for text in texts:
            pred = pipeline(text)[0]
            probs.append([pred[0]['score'], pred[1]['score']] if pred[0]['label'] == 'LABEL_0' else [pred[1]['score'], pred[0]['score']])
        return np.array(probs)
    exp = explainer.explain_instance(text_to_explain, classifier_fn=predict_proba, num_features=10)
    exp.save_to_file(html_save_path)
    print(f'Explanation for index {index_to_explain} saved as {html_save_path}')

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

if __name__ == '__main__':
    model_path = './local_pretrained_model'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)

    test_labels, test_content = process_data('human.csv')

    test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(pipeline, test_content, test_labels)
    print(f'Test Results: Accuracy: {test_accuracy:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1 Score: {test_f1:.4f}')

    explanation_directory = 'huaman_lime_bert_results'
    create_directory(explanation_directory)

    for index in range(len(test_content)):
        html_file_name = f'{explanation_directory}/lime_explanation_{index}.html'
        print('*' * 50)
        print(f'Explaining sample {index}:')
        print(test_content[index])
        lime_explain_and_save_html(pipeline, test_content, index_to_explain=index, html_save_path=html_file_name)

    print('All explanations saved successfully.')
