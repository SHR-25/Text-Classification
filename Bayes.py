import pickle
from collections import Counter

import numpy as np
import pandas as pd

class_list = {'财经': 'Economics', '房产': 'House', '股票': 'Stock', '家居': 'Household', '教育': 'Education', '科技': 'Technology',
              '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}
word_df = pd.DataFrame()


def train_Bays():
    """
    构造关键词词频矩阵
    :return: 无
    """
    global word_df
    with open('data_train/key_words.txt', 'r', encoding='utf-8') as f:
        key_words = [i.strip() for i in f.readlines()]
    word_df = pd.DataFrame(np.zeros((2410, 10)), columns=class_list.values(),
                           index=key_words)

    for CLASS_NAME_EN in class_list.values():
        with open('pkls/' + CLASS_NAME_EN + '/TF.pkl', 'rb') as f:
            TF_dic = pickle.load(f)
            for tup in word_df.itertuples():
                if TF_dic.get(tup[0]) is None:
                    continue
                else:
                    word_df.at[tup[0], CLASS_NAME_EN] = TF_dic.get(tup[0])
    # word_df.to_csv('TF_Matrix.csv')
    with open('pkls/Bays.pkl', 'wb') as file:
        pickle.dump(word_df, file)


def Bays(text: str):
    global word_df
    v_NB = {}
    words = [k for k in text.split() if k in word_df.index.to_list()]

    for CLASS_NAME_EN in class_list.values():
        v_NB[CLASS_NAME_EN] = 0
        for w in words:
            # print(np.array(df).sum())
            # m-估计
            v_NB[CLASS_NAME_EN] += (
                    (word_df.at[w, CLASS_NAME_EN] + 1) / (word_df[CLASS_NAME_EN].sum() + np.array(word_df).sum()))

    # print(v_NB)
    return Counter(v_NB).most_common(1)[0][0]


if __name__ == '__main__':
    # print(train_Bays().index.to_list())
    train_Bays()
    confusion_matrix = pd.DataFrame(np.zeros((10, 10)), columns=class_list.values(),
                                    index=class_list.values())
    for class_name_en in class_list.values():
        for i in range(5000):
            with open('data_test/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
                content = f.read()
            s = Bays(content)
            # 横向为预测
            # 纵向为真实值
            confusion_matrix.at[class_name_en, s] += 1
            print(s + ':' + str(i))
    print(confusion_matrix)
    with open('pkls/Confusion_Matrix.pkl', 'wb') as file:
        pickle.dump(confusion_matrix, file)
