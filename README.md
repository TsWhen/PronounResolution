# Deep Learning final project

主要环境：Python3.6.8、Keras 2.2.4、TensorFlow-GPU 1.13.1 、Cuda10.0

代码文件及主要函数功能说明：

- 工程中bert文件夹以及顶层目录下tokenization.py、extract_features.py、modeling.py皆为Google BERT源码文件无需关注。
- BERTembedding.py主要功能为利用BERT预训练模型在GAP数据集上进行微调，并获取相关层token的embedding，输出的embedding文件存在data/vector/文件夹下
- MLPBaseline.py是本次工程的baseline模型
- NLIModel.py在baseline的基础上增加了QA特征

### 超参数

所有超参数都在model代码中，MLP_model、NLI_model的输入参数。

bert_layer_num : 抽取bert相应层中的特征即embedding，默认输入19，不建议调整

dense_layer_size : mlp网络每层的神经元个数，默认32，优先级较高，建议32 - 128之间进行调参

embedding_size : embedding的大小，根据bert而定，无需调整

dropout_rate ： dropout层的值，调参优先级较低，建议范围0.5-0.8之间

lr : 学习率，调参优先级较低，建议范围：根据loss和acc曲线进行响应调整

epoch_num :  训练轮数，无需调参

patience ： loss比上一轮没有下降，则继续执行patience个轮次后，提前终止训练，优先级较低

lambd ：l2正则化参数，优先级较低

### 训练与预测

保证对应的文件结构只需创建相应模型对象后调用train()即可训练

预测前保证该模型至少训练过一次，即model文件夹下有对应的NLI\*/MLP\*.pt 文件,运行prediction()函数时需要传入预测文件路径以及模型路径

