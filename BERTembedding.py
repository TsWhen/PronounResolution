import tensorflow as tf

import numpy as np
import pandas as pd
import tokenization,extract_features,modeling

import sys,os,time

# 获取非空格偏移量
def get_offset_no_spaces(gap_text,offset):

    count = 0
    for pos in range(offset):

        if gap_text[pos] != " ":
            count += 1

    return count

# 获取单词字符长度(不含特殊字符)
def get_chars_length_no_special(gap_text):

    count = 0
    special_char_list = ["#"]
    for pos in range(len(gap_text)):

        if gap_text[pos] not in special_char_list:

            count += 1

    return count

# 获取文本长度(不含特殊字符以及空格)
def get_count_lenght_no_special(gap_text):

    count = 0;

    special_char_list = ["#"," "]
    for pos in range(len(gap_text)):

        if gap_text[pos] not in special_char_list:
            count += 1

    return count

# 利用bert获取GAP数据集上的word embedding
def train_bert_embedding(data,output,embedding_size=1024,layer_num=-1):

    gap_text = data["Text"]
    gap_text.to_csv("input.csv",index=False,header=False)

    # 执行extract_features.py获取对应层输出
    os.system("python3 extract_features.py --input_file=input.csv --output_file=output.json \
                        --vocab_file=./data/uncased_L-24_H-1024_A-16/vocab.txt \
                        --bert_config_file=./data/uncased_L-24_H-1024_A-16/bert_config.json \
                        --init_checkpoint=./data/uncased_L-24_H-1024_A-16/bert_model.ckpt \
                        --layers={} --max_seq_length=300 --batch_size=1".format(layer_num)
                       )

    bert_output = pd.read_json('output.json',lines=True)

    # 删除中间文件
    # os.system("rm output.json")
    # os.system("rm input.csv")

    data_index = data.index
    column_list = ["A_embedding","B_embedding","P_embedding","label"]
    embedding_df = pd.DataFrame(index=data_index,columns=column_list)
    embedding_df.index.name = "ID"


    for i in range(data.shape[0]):

        # 单词大小写不敏感处理
        vocab_P = data.loc[i,"Pronoun"].lower()
        vocab_A = data.loc[i, "A"].lower()
        vocab_B= data.loc[i, "B"].lower()

        # 获取偏移量
        P_offset = get_offset_no_spaces(data.loc[i,"Text"],data.loc[i,"Pronoun-offset"])
        A_offset = get_offset_no_spaces(data.loc[i,"Text"],data.loc[i,"A-offset"])
        B_offset = get_offset_no_spaces(data.loc[i,"Text"],data.loc[i,"B-offset"])

        # A、B名词字符长度
        A_length = get_chars_length_no_special(vocab_A)
        B_length = get_chars_length_no_special(vocab_B)

        A_embedding = np.zeros(embedding_size)
        B_embedding = np.zeros(embedding_size)
        P_embedding = np.zeros(embedding_size)

        # bert embedding获取
        features_df = pd.DataFrame(bert_output.loc[i,"features"])

        count_num = 0
        count_A = 0
        count_B = 0
        count_P = 0

        # 取三个名词/代词 向量
        for j in range(2,len(features_df)):

            token = features_df.loc[j,"token"]

            if count_num == P_offset:

                P_embedding += np.array(features_df.loc[j,"layers"][0]["values"])
                count_P += 1
            # 对于名词可能存在多个单词表达一个人名，而bert的分词则可能将一个词分为几个部分
            if count_num in range(A_offset,A_offset+A_length):
                A_embedding += np.array(features_df.loc[j,"layers"][0]["values"])
                count_A += 1

            if count_num in range(B_offset,B_offset+B_length):
                B_embedding += np.array(features_df.loc[j,"layers"][0]["values"])
                count_B += 1

            count_num += get_count_lenght_no_special(token)

        # 将多个部分求均值来表达A、B人名
        A_embedding /= count_A
        B_embedding /= count_B

        label = ""
        if data.loc[i,"A-coref"] == True:
            label = "A"
        elif data.loc[i,"B-coref"] == True:
            label = "B"
        else:
            label = "Neither"

        embedding_df.iloc[i] = [A_embedding,B_embedding,P_embedding,label]

    return embedding_df

# 生成GAP上的embedding
def gen_embedding(start_layer,end_layer):

    for i in range(start_layer,end_layer):

        model_version = "aug_bert-large-uncased-seq300-"
        embedding_size = 1024
        layer = i
        model_layer = model_version + str(i)
        
        
        # gap_test_data = pd.read_csv("./data/gap-test.tsv",sep='\t')
        test_naive_data = pd.read_csv("./data/gap-test.tsv",sep='\t')
        test_aug_1_data = pd.read_csv("./data/test_augment_data_1.tsv",sep='\t')
        test_aug_2_data = pd.read_csv("./data/test_augment_data_2.tsv", sep='\t')
        test_aug_3_data = pd.read_csv("./data/test_augment_data_3.tsv", sep='\t')
        test_aug_4_data = pd.read_csv("./data/test_augment_data_4.tsv", sep='\t')

        gap_test_data = test_naive_data.append([test_aug_1_data,test_aug_2_data,test_aug_3_data,test_aug_4_data])
        print(gap_test_data)

        test_embedding_df = train_bert_embedding(gap_test_data,output="{}contextual_embedding_gap_test.json".format(model_layer),layer_num=layer)
        test_embedding_df.to_json("./data/vector/{}contextual_embedding_gap_test.json".format(model_layer),orient="columns")

        # gap_val_data = pd.read_csv("./data/gap-validation.tsv", sep='\t')
        val_naive_data = pd.read_csv("./data/gap-validation.tsv", sep='\t')
        val_aug_1_data = pd.read_csv("./data/val_augment_data_1.tsv", sep='\t')
        val_aug_2_data = pd.read_csv("./data/val_augment_data_2.tsv", sep='\t')
        val_aug_3_data = pd.read_csv("./data/val_augment_data_3.tsv", sep='\t')
        val_aug_4_data = pd.read_csv("./data/val_augment_data_4.tsv", sep='\t')

        gap_val_data = val_naive_data.append([val_aug_1_data, val_aug_2_data, val_aug_3_data, val_aug_4_data])
        print(gap_val_data)
        val_embedding_df = train_bert_embedding(gap_val_data,
                                           output="{}contextual_embedding_gap_val.json".format(model_layer),
                                           layer_num=layer)
        val_embedding_df.to_json("./data/vector/{}contextual_embedding_gap_val.json".format(model_layer),
                                  orient="columns")

        # gap_develop_data = pd.read_csv("./data/gap-development.tsv", sep='\t')

        develop_naive_data = pd.read_csv("./data/gap-development.tsv", sep='\t')
        develop_aug_1_data = pd.read_csv("./data/develop_augment_data_1.tsv", sep='\t')
        develop_aug_2_data = pd.read_csv("./data/develop_augment_data_2.tsv", sep='\t')
        develop_aug_3_data = pd.read_csv("./data/develop_augment_data_3.tsv", sep='\t')
        develop_aug_4_data = pd.read_csv("./data/develop_augment_data_4.tsv", sep='\t')

        gap_develop_data = develop_naive_data.append([develop_aug_1_data, develop_aug_2_data, develop_aug_3_data, develop_aug_4_data])
        print(gap_develop_data)

        develop_embedding_df = train_bert_embedding(gap_develop_data,
                                           output="{}contextual_embedding_gap_develop.json".format(model_layer),
                                           layer_num=layer)
        develop_embedding_df.to_json("./data/vector/{}contextual_embedding_gap_develop.json".format(model_layer),
                                  orient="columns")
        
        

#
if __name__ == '__main__':

    print("开始时间:",time.ctime())
    gen_embedding(18,21)