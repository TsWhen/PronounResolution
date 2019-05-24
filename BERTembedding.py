import tensorflow as tf

import numpy as np
import pandas as pd
from bert import tokenization,extract_features,modeling

# 获取非空格偏移量
def get_offset_no_spaces(gap_text,offset):

    count = 0
    for pos in range(offset):

        if gap_text[pos] != " ":
            count += 1

    return count

# 获取文本字符数量(不含特殊字符)
def get_count_chars_no_special(gap_text):

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

def bert_embedding(data,output,embedding_size=1024,layer_num=-1):

    gap_text = data["Text"]
    gap_text.to_csv("input.csv",index=False,header=False)




if __name__ == '__main__':

    pass