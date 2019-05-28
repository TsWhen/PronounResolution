import numpy as np
import pandas as pd
from tqdm import tqdm

# 数据增强

def data_augment(data_file_path,output_tag):

    replace_name_list = [{'male':['John','Michael'],"female":['Alice','Kate']},
                         {"male":['James','Henry'],"female":['Elizabeth','Mary']},
                         {"male":['Michael','James'],"female":['Kate','Elizabeth']},
                         {"male":['Henry','John'],"female":['Mary','Alice']}]

    aug_count = 1

    for name_dict in replace_name_list:

        data = pd.read_csv(data_file_path,sep='\t')

        for i in tqdm(range(data.shape[0])):

            is_replace_A = True
            is_replace_B = True

            gap_text = data.loc[i,"Text"]
            A = data.loc[i,"A"]
            B = data.loc[i, "B"]
            P = data.loc[i, "Pronoun"]

            # AB之间包含关系
            if (A in B) or (B in A):
                continue

            if data.loc[i,"Pronoun"].lower() in ['she','her']:
                sexy = 'female'
            else:
                sexy = "male"

            # 姓名在文本中出现过
            if name_dict[sexy][0] in gap_text:
                continue
            elif name_dict[sexy][1] in gap_text:
                continue

            # 姓名为替换名子串
            if (A in name_dict[sexy][0]) or (A in name_dict[sexy][1]):
                continue
            elif (B in name_dict[sexy][0]) or (B in name_dict[sexy][1]):
                continue

            # 姓名超过两个单词
            if len(A.split(" ")) > 2:
                is_replace_A = False
            if len(B.split(" ")) > 2:
                is_replace_B = False


            # 姓名全称和简称全都在文中出现
            if len(A.split(" ")) == 2:
                if gap_text.count(A.split(" ")[0]) > gap_text.count(A):
                    is_replace_A = False
                elif gap_text.count(A.split(" ")[1]) > gap_text.count(A):
                    is_replace_A = False

            if len(B.split(" ")) == 2:
                if gap_text.count(B.split(" ")[0]) > gap_text.count(B):
                    is_replace_B = False
                elif gap_text.count(B.split(" ")[1]) > gap_text.count(B):
                    is_replace_B = False

            if (not is_replace_A) and not (is_replace_B):
                continue

            # 替换姓名及重置偏移
            A_offset = data.loc[i,"A-offset"]
            B_offset = data.loc[i, "B-offset"]
            P_offset = data.loc[i, "Pronoun-offset"]

            if is_replace_A:

                while (A in gap_text):
                    A_pos = gap_text.index(A)
                    gap_text = gap_text.replace(A,name_dict[sexy][0],1)
                    if A_pos < A_offset: A_offset += len(name_dict[sexy][0]) - len(A)
                    if A_pos < B_offset: B_offset += len(name_dict[sexy][0]) - len(A)
                    if A_pos < P_offset: P_offset += len(name_dict[sexy][0]) - len(A)
                data.loc[i, 'A'] = name_dict[sexy][0]

            if is_replace_B:

                while (B in gap_text):
                    B_pos = gap_text.index(B)
                    gap_text = gap_text.replace(B,name_dict[sexy][1],1)
                    if B_pos < A_offset: A_offset += len(name_dict[sexy][1]) - len(B)
                    if B_pos < B_offset: B_offset += len(name_dict[sexy][1]) - len(B)
                    if B_pos < P_offset: P_offset += len(name_dict[sexy][1]) - len(B)
                data.loc[i, 'B'] = name_dict[sexy][1]

            data.loc[i,"A-offset"] = A_offset
            data.loc[i,"B-offset"] = B_offset
            data.loc[i,"Pronoun-offset"] = P_offset
            data.loc[i,"Text"] = gap_text

        # 检查
        for i in tqdm(range(data.shape[0])):

            gap_text = data.loc[i, "Text"]
            A = data.loc[i, "A"]
            B = data.loc[i, "B"]
            P = data.loc[i, "Pronoun"]
            A_offset = data.loc[i, "A-offset"]
            B_offset = data.loc[i, "B-offset"]
            P_offset = data.loc[i, "Pronoun-offset"]
            assert gap_text[A_offset:(A_offset+len(A))] == A
            assert gap_text[B_offset:(B_offset + len(B))] == B
            assert gap_text[P_offset:(P_offset + len(P))] == P

        data.to_csv("./data/"+output_tag+"_augment_data_"+str(aug_count)+".tsv",sep='\t',index=False)
        aug_count += 1

if __name__ == "__main__":

    data_augment("./data/gap-development.tsv","develop")
    data_augment("./data/gap-test.tsv", "test")
    data_augment("./data/gap-validation.tsv", "val")

