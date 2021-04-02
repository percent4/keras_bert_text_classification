# -*- coding: utf-8 -*-
# @Time : 2021/4/2 12:09
# @Author : Jclian91
# @File : model_server.py
# @Place : Yangpu, Shanghai
import json
import traceback
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model
from keras_bert import get_custom_objects
from flask import Flask, request

from model_train import token_dict, OurTokenizer

maxlen = 300
tokenizer = OurTokenizer(token_dict)

with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

app = Flask(__name__)


@app.route("/model/cls", methods=["GET", "POST"])
def get_geo():
    return_result = {"code": 200, "message": "success", "data": []}
    try:
        text = request.get_json()["text"]

        # 模型预测
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            # 利用BERT进行tokenize
            X1, X2 = tokenizer.encode(first=text, max_len=maxlen)
            # 模型预测并输出预测结果
            predicted = model.predict([[X1], [X2]])
            y = np.argmax(predicted[0])

        return_result["data"] = {"text": text, "label": label_dict[str(y)]}

    except Exception:
        return_result["code"] = 400
        return_result["message"] = traceback.format_exc()

    return json.dumps(return_result, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    # 加载训练好的模型
    custom_objects = get_custom_objects()
    model = load_model("cls_sougou.h5", custom_objects=get_custom_objects())
    app.run(host="0.0.0.0", port=15000)
