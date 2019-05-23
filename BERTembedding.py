import tensorflow as tf

import numpy as np
import pandas as pd
from bert import tokenization,extract_features,modeling


def get_offset_no_spaces(gap_text,offset):

    count = 0
    for pos in range(offset):

        if gap_text[pos] != " ":
            count += 1

    return count

def get_count_chars_no_special(gap_text):

    count = 0
    special_char_list = ["#"]
    for pos in range(len(gap_text)):

        if gap_text[pos] not in special_char_list:

            count += 1

    return count


def get_count_lenght_no_special(gap_text):

    count = 0;

    special_char_list = ["#"," "]
    for pos in range(len(gap_text)):

        if gap_text[pos] not in special_char_list:
            count += 1

    return count

