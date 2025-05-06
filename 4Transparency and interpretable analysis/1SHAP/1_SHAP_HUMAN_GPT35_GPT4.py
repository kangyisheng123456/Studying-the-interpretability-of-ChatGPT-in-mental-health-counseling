import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import csv
import shap


tokenizer = BertTokenizer.from_pretrained('./local_pretrained_model')
model1 = BertModel.from_pretrained('./local_pretrained_model')

# 创建模型
class BERTMLPClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTMLPClassifier, self).__init__()
        self.bert_model = bert_model
        self.mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):

        with torch.cuda.amp.autocast():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            logits = self.mlp(cls_embeddings)
        return logits

# 初始化模型和优化器
model = BERTMLPClassifier(model1, num_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 测试函数
def test_text_classifier(texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 对文本进行编码和处理
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)

    # 获取模型输出
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_masks)
        probabilities = torch.softmax(logits, dim=1)

    # 将结果转化为指定格式
    results = []
    for i in range(len(texts)):
        result = [
            {"label": "NEGATIVE", "score": probabilities[i][0].item()},
            {"label": "POSITIVE", "score": probabilities[i][1].item()}
        ]
        results.append(result)

    return results


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

# 处理数据，提取标签和内容
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

# 数据编码和处理
def preprocess_texts(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels


if __name__ == '__main__':


    new_test_labels, new_test_content = process_data('train.csv')
    test_data = new_test_content[:20]

    results = test_text_classifier(test_data)
    print(len(results))


    predicted_labels = [result[0]['label'] for result in results]


    def classifier(texts):
        results = test_text_classifier(texts)
        predicted_labels = [result[0]['label'] for result in results]
        return predicted_labels


    explainer = shap.Explainer(classifier)

    shap_values = explainer(test_data)

    shap.plots.bar(shap_values[0])