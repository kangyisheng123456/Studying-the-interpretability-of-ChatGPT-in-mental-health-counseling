import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import csv
import os


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


if __name__ == '__main__':

    labels, content = process_data(r'test.csv')
    content = [v[:70] for v in content]
    train_texts, test_texts, train_labels, test_labels = train_test_split(content, labels, test_size=0.2, random_state=42)

    # 加载预训练的BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 输出维度为2，表示两个类别


    batch_size = 8


    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = 6
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

    model.eval()
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            total_correct += torch.sum(predictions == labels).item()
            total_samples += len(labels)

        accuracy = total_correct / total_samples
        print(f'Accuracy: {accuracy * 100:.2f}%')



    output_dir = "./model/"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    config = model.config
    config.save_pretrained(output_dir)

    print(f"Model and tokenizer files saved in {output_dir}")
