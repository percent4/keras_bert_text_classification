本项目采用Keras和Keras-bert实现文本多分类任务。

### 维护者

- jclian91

### 数据集

#### sougou小分类数据集

sougou小分类数据集，共有5个类别，分别为体育、健康、军事、教育、汽车。

划分为训练集和测试集，其中训练集每个分类800条样本，测试集每个分类100条样本。

#### THUCNews数据集

使用THUCNews数据集进行训练与测试，10个分类，每个分类6500条数据。
类别如下：
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
数据集划分如下：
训练集: 5000 * 10
测试集: 1000 * 10

### 代码结构

```
.
├── chinese_L-12_H-768_A-12（BERT中文预训练模型）
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── data（数据集）
│   └── sougou_mini
│       ├── test.csv
│       └── train.csv
├── label.json（类别词典，生成文件）
├── model_evaluate.py（模型评估脚本）
├── model_predict.py（模型预测脚本）
├── model_train.py（模型训练脚本）
└── requirements.txt
```

## 模型效果

#### sougou数据集

模型参数: batch_size = 8, maxlen = 256, epoch=10

评估结果:

```
                  precision    recall  f1-score   support

          体育     0.9802    1.0000    0.9900        99
          健康     0.9495    0.9495    0.9495        99
          军事     1.0000    1.0000    1.0000        99
          教育     0.9307    0.9495    0.9400        99
          汽车     0.9895    0.9495    0.9691        99

    accuracy                         0.9697       495
   macro avg     0.9700    0.9697    0.9697       495
weighted avg     0.9700    0.9697    0.9697       495
```

#### THUCNews数据集

模型参数: batch_size = 8, maxlen = 300, epoch=3

评估结果:

```
                precision    recall  f1-score   support

          体育     0.9970    0.9990    0.9980      1000
          娱乐     0.9890    0.9890    0.9890      1000
          家居     0.9949    0.7820    0.8757      1000
          房产     0.8006    0.8710    0.8343      1000
          教育     0.9753    0.9480    0.9615      1000
          时尚     0.9708    0.9980    0.9842      1000
          时政     0.9318    0.9560    0.9437      1000
          游戏     0.9851    0.9950    0.9900      1000
          科技     0.9689    0.9970    0.9828      1000
          财经     0.9377    0.9930    0.9645      1000

    accuracy                         0.9528     10000
   macro avg     0.9551    0.9528    0.9524     10000
weighted avg     0.9551    0.9528    0.9524     10000
```

### 项目启动

1. 将BERT中文预训练模型chinese_L-12_H-768_A-12放在chinese_L-12_H-768_A-12文件夹下
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/sougou_mini的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行评估