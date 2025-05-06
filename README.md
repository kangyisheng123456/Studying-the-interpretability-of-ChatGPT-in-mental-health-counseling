# Studying-the-interpretability-of-ChatGPT-in-mental-health-counseling
Reproducible Code: Studying the Interpretability of ChatGPT in Mental Health Counseling
#This project aims to explore the interpretability of ChatGPT in the domain of mental health counseling through:

This project aims to explore the interpretability of ChatGPT in the domain of mental health counseling through:

1 Dataset Construction
2 Topic Analysis
3 Recognizing AIGC and UGC 
4 Transparency and interpretable analysis
Model Interpretability using SHAP and LIME
![image](https://github.com/user-attachments/assets/70a45159-5ec8-401d-b9e9-9f5fdaec7158)
Fig. 1 Dataset construction and research design framework
Figure 1 shows the dataset construction process and research design. Study the interpretability of ChatGPT in mental health counseling. The specific steps are shown in the figure above.
The dataset construction has only one part. The research design is divided into three parts.
![image](https://github.com/user-attachments/assets/e73265bb-0263-4a1f-bed7-d3def7fd6d72)
Figure 2 Flowchart of the construction of the mental health consultation question and answer dataset
![image](https://github.com/user-attachments/assets/11358e6d-3ece-457c-a3f7-f528ed7b101e)
Figure 3 Schematic diagram of the prompt word design for generating AIGC psychological consultation responses

1 Dataset Construction

The detailed process of dataset construction is shown in Figure 2. We selected the data in the public dataset counsel-chat as the response data of human psychological counselors, and then divided it into 12 categories of psychological problems. We input the questions and answers of human psychological counselors into ChatGPT in sequence according to the steps above Figure 1, and obtained the responses of ChatGPT3.5 and ChatGPT4.0 respectively. The prompt words used are shown in Figure 3. The obtained datasets are UGC-AIGC3.5.CSV and UGC-AIGC4.0.CSV. In order to facilitate subsequent processing, we merged these datasets into one and named it Human_GPT3.5_GPT4.CSV



2 Topic Analysis

The detailed process of dataset construction is shown in Figure 2. We selected the data in the public dataset counsel-chat as the response data of human psychological counselors, and then divided it into 12 categories of psychological problems. We input the questions and answers of human psychological counselors into ChatGPT in sequence according to the steps above Figure 1, and obtained the responses of ChatGPT3.5 and ChatGPT4.0 respectively. The prompt words used are shown in Figure 3. The obtained datasets are UGC-AIGC3.5.CSV and UGC-AIGC4.0.CSV. In order to facilitate subsequent processing, we merged these datasets into one and named it Human_GPT3.5_GPT4.CSV


Run the scripts and programs in 2Analysis of differences in topic perspectives, such as topic words, topic distribution diagrams, and topic hierarchical clustering diagrams. The environment configuration is as follows:

bertopic==0.13.0  
Python==3.8  
sentence-transformers==2.2.2  
pandas==2.0.0  
nltk==3.8.1  
scikit-learn==1.2.2

3 Recognizing AIGC and UGC 

We first divide the training set, validation set and test set
Then use the code in 2Recognizing AIGC and UGC to train and test the results of deep learning and machine learning models. At the same time, we conduct recognition experiments in gptzero. Website: https://gptzero.me/. In the machine learning part, the divided training data set is placed under example.
In deep learning training, pre-trained models are required
The bert model is placed in the bert_pretain directory, and the ERNIE model is placed in the ERNIE_pretrain directory. There are three files in each directory. The names of bert's pre-trained models are as follows:
pytorch_model.bin
bert_config.json
vocab.txt

Then run the following code to train and test:
# Train and test with BERT
python run.py --model bert

# Train and test with ERNIE
python run.py --model ERNIE

The dependent libraries are as follows:
pytextclassifier  
loguru  
jieba  
scikit-learn  
pandas  
numpy  
transformers


4 Transparency and interpretable analysis

We use SHAP and LIME to perform interpretability analysis on the best model. First, we use 1model in 3Transparency and interpretable analysis to save the model to be explained for shap interpretability analysis, and then interpret human, chatgpt3.5 and chatgppt4.0 respectively. Then use the LIME algorithm to perform interpretability analysis on the model. The code can be found in 1SHAP and 2LIME in 3Transparency and interpretable analysis.
The libraries that SHAP interpretability analysis depends on include:
shap  
transformers  
datasets  
matplotlib

The libraries that LIME interpretability analysis relies on include:
scikit-learn  
lime  
pandas  
torch  
joblib


