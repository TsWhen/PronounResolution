
from keras import models,layers,initializers,regularizers,constraints,optimizers
from keras import callbacks
from keras import optimizers

from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd



import os



# 解析embedding json文件 构造mlp模型的所用的数据
def parse_json(embedding_df):

    embedding_df.sort_index(inplace=True)

    X_data = np.zeros((len(embedding_df),3*1024))
    Y_data = np.zeros((len(embedding_df),3))

    # 构建特征数据
    for i in range(len(embedding_df)):

        A_embedding = np.array(embedding_df.loc[i,"A_embedding"])
        B_embedding = np.array(embedding_df.loc[i, "B_embedding"])
        P_embedding = np.array(embedding_df.loc[i, "P_embedding"])
        X_data[i] = np.concatenate((A_embedding,B_embedding,P_embedding))

    # 构建标签数据
    for i in range(len(embedding_df)):

        label = embedding_df.loc[i,"label"]

        if label == "A":
            Y_data[i,0] = 1
        elif label == "B":
            Y_data[i,1] = 1
        else:
            Y_data[i,2] = 1

    return X_data,Y_data

# 获取训练数据
def get_train_data(layer_num):

    model_version = "bert-large-uncased-seq300-"
    model_layer = model_version + str(layer_num)

    develop_data = pd.read_json("./data/vector/{}contextual_embedding_gap_develop.json".format(model_layer))
    X_develop,Y_develop = parse_json(develop_data)

    val_data = pd.read_json("./data/vector/{}contextual_embedding_gap_val.json".format(model_layer))
    X_val, Y_val = parse_json(val_data)

    test_data = pd.read_json("./data/vector/{}contextual_embedding_gap_test.json".format(model_layer))
    X_test, Y_test = parse_json(test_data)

    # 存在少量句子长度大于bert的最大句子长度导致的NaN值，将其删去
    remove_develop = [row for row in range(len(X_develop)) if np.sum(np.isnan(X_develop[row]))]
    X_develop[remove_develop] = np.zeros(3*1024)

    remove_val = [row for row in range(len(X_val)) if np.sum(np.isnan(X_val[row]))]
    X_val = np.delete(X_val,remove_val,0)
    Y_val = np.delete(Y_val, remove_val, 0)

    remove_test = [row for row in range(len(X_test)) if np.sum(np.isnan(X_test[row]))]
    X_test = np.delete(X_test, remove_test, 0)
    Y_test = np.delete(Y_test, remove_test, 0)

    # 构造训练集、测试集
    X_train = np.concatenate((X_test,X_val),axis=0)
    Y_train = np.concatenate((Y_test,Y_val),axis=0)

    return X_train,Y_train,X_develop,Y_develop

# 模型结构搭建
def gen_MLP_net(input_size,dropout_rate=0.6,dense_layer_size=37,lambd=0.1):

    input = layers.Input(input_size)

    out = layers.Dropout(dropout_rate,seed=7)(input)
    out = layers.Dense(dense_layer_size,name='dense0')(out)
    out = layers.BatchNormalization(name='bn0')(out)
    out = layers.Activation('relu')(out)
    out = layers.Dropout(dropout_rate,seed=7)(out)

    out = layers.Dense(3,name='output',kernel_regularizer=regularizers.l2(lambd))(out)
    out = layers.Activation('softmax')(out)

    return models.Model(input=input,output=out,name='pronoun_class_model')

class MLP_model():

    def __init__(self,bert_layer_num=19,lr=0.001,dense_layer_size=32,dropout_rate=0.6,n_fold=7,batch_size=32,epoch_num=1000,patience=20,lambd=0.1):

        self.X_train, self.Y_train, self.X_pred, self.Y_pred = get_train_data(bert_layer_num)
        self.bert_layer_num = bert_layer_num
        self.dense_layer_size = dense_layer_size
        self.lr = lr
        self.n_fold = n_fold
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.patience = patience
        self.lambd = lambd
        self.dropout_rate = dropout_rate

        self.mlp_model = gen_MLP_net(input_size=[self.X_train.shape[-1]], dropout_rate=self.dropout_rate,
                                     dense_layer_size=self.dense_layer_size, lambd=self.lambd)

        # 编译模型
        self.mlp_model.compile(optimizer=optimizers.Adam(lr=self.lr),loss="categorical_crossentropy")

    def train(self):

        folds = KFold(n_splits=self.n_fold,shuffle=True,random_state=2)

        tmp_val_pred = np.zeros_like(self.Y_train)
        val_score_lsit = []
        pred_result = np.zeros((len(self.X_pred),3))

        for fold_num,(train_index,valid_index) in enumerate(folds.split(self.X_train)):

            # 划分训练集、验证集
            train_X,val_X = self.X_train[train_index],self.X_train[valid_index]
            train_Y,val_Y = self.Y_train[train_index],self.Y_train[valid_index]


            callback = [callbacks.EarlyStopping(monitor='val_loss',patience=self.patience,restore_best_weights=False),
                        callbacks.ModelCheckpoint('./model/MLP_bert-large-uncased-seq300-'+str(self.bert_layer_num)+'-'+str(self.n_fold)+'.pt',
                                                  monitor='val_loss',verbose=0,save_best_only=True,mode='min')]

            self.mlp_model.fit(x=train_X,y=train_Y,epochs=self.epoch_num,batch_size=self.batch_size,
                                   callbacks=callback,validation_data=(val_X,val_Y),verbose=0)

            pred_valid = self.mlp_model.predict(x=val_X,verbose=0)
            tmp_val_pred[valid_index] = pred_valid

            pred_test = self.mlp_model.predict(x=self.X_pred,verbose=0)

            val_score_lsit.append(log_loss(val_Y,pred_valid))

            pred_result += pred_test

        pred_result /= self.n_fold

        print("laye " + str(self.bert_layer_num) + " :")
        print("CV mean score:{0:.4f},std: {1:.4f}".format(np.mean(val_score_lsit),np.std(val_score_lsit)))
        print(val_score_lsit)
        print("pred score:", log_loss(self.Y_pred,pred_result))

    def prediction(self,pred_file_path,model_path):

        naive_data = pd.read_json(pred_file_path)
        X_pred,Y_pred = parse_json(naive_data)

        pred_result = np.zeros((len(X_pred),3))

        # for fold_num in range(self.n_fold):

        self.mlp_model = gen_MLP_net(input_size=[X_pred.shape[-1]], dropout_rate=self.dropout_rate,
                                     dense_layer_size=self.dense_layer_size, lambd=self.lambd)
        self.mlp_model.load_weights(model_path)
        print("--------------predict----------------------------------")
        pred_data = self.mlp_model.predict(x=X_pred, verbose=0)
        pred_result += pred_data

        # pred_result /= self.n_fold

        submission_data = pd.read_csv("./data/sample_submission_stage_2.csv", index_col="ID")
        submission_data["A"] = pred_result[:, 0]
        submission_data["B"] = pred_result[:, 1]
        submission_data["NEITHER"] = pred_result[:, 2]
        submission_data.to_csv("./data/MLP_pred_submission.csv")

if __name__ == '__main__':

    model = MLP_model()
    print('----------------------')
    model.train()
    model.prediction()
    print('----------------------')
