"""
5.贝叶斯分类器
"""
import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

word_df = pd.DataFrame()
bayes_df = pd.DataFrame()
df_sum = 1
# csc_test = load_npz('coo_test_dic.npz').tocsr()
csc_test = load_npz('coo_test.npz').tocsr()
value = csc_test.data
column_index = csc_test.indices
row_pointers = csc_test.indptr
with open('data_train/key_words.txt', 'r', encoding='utf-8') as f:
    key_words = [key_word.strip() for key_word in f.readlines()]


# with open('data_train/dict.txt', 'r', encoding='utf-8') as f:
#     key_words = [key_word.strip() for key_word in f.readlines()]


def train_Bays():
    global word_df, key_words, df_sum, bayes_df

    # 3685
    word_df = pd.DataFrame(np.zeros((3685, 10)), columns=class_list.values(),
                           index=key_words)
    bayes_df = pd.DataFrame(np.zeros((3685, 10)), columns=class_list.values(),
                            index=key_words)
    # word_df = pd.DataFrame(np.zeros((5000, 10)), columns=class_list.values(),
    #                        index=key_words)
    # bayes_df = pd.DataFrame(np.zeros((5000, 10)), columns=class_list.values(),
    #                         index=key_words)

    # 构造关键词词频矩阵
    for CLASS_NAME_EN in class_list.values():
        with open('pkls/' + CLASS_NAME_EN + '/TF.pkl', 'rb') as F:
            TF_dic = pickle.load(F)
            for tup in word_df.itertuples():
                if TF_dic.get(tup[0]) is None:
                    continue
                else:
                    word_df.at[tup[0], CLASS_NAME_EN] = TF_dic.get(tup[0])
    df_sum = np.array(word_df).sum()
    word_df.to_csv('TF_Matrix.csv')
    with open('pkls/TF_Matrix.pkl', 'wb') as F:
        pickle.dump(word_df, F)

    # 构造条件概率矩阵
    for tup in bayes_df.itertuples():
        for CLASS_NAME_EN in class_list.values():
            # m-估计
            bayes_df.at[tup[0], CLASS_NAME_EN] = (word_df.at[tup[0], CLASS_NAME_EN] + 1) / (
                    word_df[CLASS_NAME_EN].sum() + df_sum)
    bayes_df.to_csv('Bayes.csv')
    with open('pkls/Bayes.pkl', 'wb') as F:
        pickle.dump(word_df, F)


def Bays(text_pos: int):
    global word_df, bayes_df
    v_NB = {}

    for CLASS_NAME_EN in class_list.values():
        v_NB[CLASS_NAME_EN] = 1
        for v in column_index[row_pointers[text_pos]:row_pointers[text_pos + 1]]:
            # print(np.array(df).sum())
            w = key_words[v]
            v_NB[CLASS_NAME_EN] *= bayes_df.at[w, CLASS_NAME_EN]

    # print(v_NB)
    res = sorted(v_NB.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return res[0][0]


if __name__ == '__main__':
    start = time.time()
    train_Bays()
    end = time.time()
    train_time = end - start

    confusion_matrix = pd.DataFrame(np.zeros((10, 10)), columns=class_list.values(),
                                    index=class_list.values())

    start = time.time()
    class_index = 0
    for class_name_en in class_list.values():
        for i in range(5000):
            # with open('data_test/' + class_name_en + '/' + str(i) + '.txt', 'r', encoding='utf-8') as f:
            #     content = f.read()
            s = Bays(class_index * 5000 + i)
            print('class:' + class_name_en + ' pre:' + s + ' id:' + str(i))
            # 横向为预测
            # 纵向为真实值
            confusion_matrix.at[class_name_en, s] += 1
        class_index += 1
    end = time.time()
    print('\n\nTrain time: %s Seconds' % train_time)
    print('Test time: %s Seconds' % (end - start))
    print(confusion_matrix)
    confusion_matrix.to_csv('Confusion_Matrix.csv')

    with open('pkls/Confusion_Matrix.pkl', 'wb') as file:
        pickle.dump(confusion_matrix, file)
