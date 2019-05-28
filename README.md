# Deep Learning final project

主要环境：Python3.6.8、Keras 2.2.4、TensorFlow-GPU 1.13.1 、Cuda10.0

代码文件及主要函数功能说明：

- 工程中bert文件夹以及顶层目录下tokenization.py、extract_features.py、modeling.py皆为Google BERT源码文件无需关注。
- BERTembedding.py主要功能为利用BERT预训练模型在GAP数据集上进行微调，并获取相关层token的embedding，输出的embedding文件存在data/vector/文件夹下
- MLPBaseline.py是本次工程的baseline模型
- NLIModel.py在baseline的基础上增加了QA特征

