import transformers
import datasets
import shap
import numpy as np
import csv
import matplotlib.pyplot as plt

# 设置tight_layout为默认
plt.rcParams['figure.autolayout'] = True
def csv_to_tuple(file_document, encoding='utf-8-sig'):
    with open(file_document, 'r', encoding=encoding, newline='') as f_r:
        reader_obj = csv.reader(f_r)
        data_list = []
        for i in reader_obj:
            # label = i[0]
            # content = i[1]
            # tuple_1 = (label, content)
            data_list.append(i)
        return data_list


if __name__ == '__main__':
    all_list = csv_to_tuple('GPT4.csv')
    print(all_list[0])
    line_begin = 3000
    w_line=3200

    line_1= line_begin-1
    line=w_line-1

    print('*'*100)

    new_test_content = []
    for i in all_list:
        new_test_content.append(i[-1])
    print(len(new_test_content))

    '''
    '''
    short_data = [v[:70] for v in new_test_content]


    model_path = "../"
    # 加载本地模型
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = transformers.pipeline('text-classification', model=model_path)
    explainer = shap.Explainer(classifier)
    shap_values = explainer(short_data[line_1:line])



    for i_ai in range(line-line_1):
        print(i_ai)
        # 创建新的图形对象
        plt.figure()
        shap.plots.bar(shap_values[i_ai,:,0],show=False)
        num = line_begin+i_ai
        out_path_ai=r'./_CHATGPT4_'+str(num)+'.png'
        plt.savefig(out_path_ai)













    # plt.tight_layout()
    # plt.show()
    # shap_toxic.plots.bar(shap_values[0,:])#human






