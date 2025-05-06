from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

import os

import pandas as pd
import string


def topic(train, path, docs):
    if train == True:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        topic_model = BERTopic(embedding_model=model, language="english")
        print('training')
        topics, probs = topic_model.fit_transform(docs)
        topic_model.save(path + 'model')
    else:
        print('load')
        topic_model = BERTopic.load(path + 'model')
    # print(topic_model.get_topic_info())
    topic_model.get_topic_freq().to_csv(path + 'topic_freq.csv')
    topic_model.visualize_topics().write_html(path + 'v_topics' + '.html')
    topic_model.visualize_barchart(n_words=9,top_n_topics=7).write_html(path + 'v_bar' + '.html')
    topic_model.visualize_hierarchy(top_n_topics=8).write_html(path + 'v_hierarchy' + '.html')
    topic_model.visualize_heatmap(top_n_topics=8).write_html(path + 'v_heatmap' + '.html')
    topic_model.visualize_term_rank().write_html(path + 'v_term_rand' + '.html')
    topic_model.get_document_info(docs).to_csv(path + 'result' + '.csv')
    return topic_model.get_document_info(docs)

# 加载停用词列表
def load_stopwords(stopwords_file):
    stopwords = set()
    if os.path.exists(stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip().lower())
    return stopwords

# 删除停用词
def remove_stopwords(text_list, stopwords):
    filtered_text_list = []
    for text in text_list:
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        filtered_text = ' '.join(filtered_words)
        filtered_text_list.append(filtered_text)
    return filtered_text_list


'''
CHATGPT4
'''
path = f'ChatGPT4/'
path_read= '../1DATASET/Human_GPT3.5_GPT4.CSV'
print(path_read)
docs = list(pd.read_csv(path_read,header=None).iloc[:,-1])
stopwords_file = load_stopwords('stopwords.txt')
docs=remove_stopwords(docs,stopwords_file)
print(len(docs))
print(type(docs))
print(docs[0])
for num,i in enumerate(docs):
    if i=='':
        print(num,i)
has_missing = None in docs
print(has_missing)
print(f'the doc number is {len(docs)}')
if len(docs)<1000:
    docs = docs*(1000//len(docs))
topic(False, path, docs)