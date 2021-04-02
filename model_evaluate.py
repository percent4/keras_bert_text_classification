# -*- coding: utf-8 -*-
# @Time : 2020/12/23 15:28
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Yangpu, Shanghai
# 模型评估脚本
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_bert import get_custom_objects
from sklearn.metrics import classification_report

from model_train import token_dict, OurTokenizer

maxlen = 300

# 加载训练好的模型
model = load_model("cls_sougou_mini.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())


# 对单句话进行预测
def predict_single_text(text):
    # 利用BERT进行tokenize
    text = text[:maxlen]
    X1, X2 = tokenizer.encode(first=text, max_len=maxlen)

    # 模型预测并输出预测结果
    predicted = model.predict([[X1], [X2]])
    y = np.argmax(predicted[0])
    return label_dict[str(y)]


# 模型评估
def evaluate():
    test_df = pd.read_csv("data/cnews/cnews_test.csv").fillna(value="")
    true_y_list, pred_y_list = [], []
    for i in range(test_df.shape[0]):
        print("predict %d samples" % (i+1))
        true_y, content = test_df.iloc[i, :]
        pred_y = predict_single_text(content)
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)

    return classification_report(true_y_list, pred_y_list, digits=4)


output_data = evaluate()
print("model evaluate result:\n")
print(output_data)