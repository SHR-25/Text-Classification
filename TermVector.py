"""
生成词向量，用于SVM
"""
import pickle

import numpy as np
from scipy.sparse import coo_matrix, save_npz

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

with open('pkls/key_words.pkl', 'rb') as f:
    key_words_dic = pickle.load(f)
key_words = list(key_words_dic.keys())
test_arr = np.zeros(shape=(50000, 3685))
train_arr = np.zeros(shape=(50000, 3685))
test_index = 0
train_index = 0
for class_name_en in class_list.values():
    with open('data_test/' + class_name_en + '/all.txt', 'r', encoding='utf-8') as f:
        content = f.readlines()
        for text in content:
            print(class_name_en + ':' + str(test_index))
            for w in text.split():
                if w not in key_words:
                    continue
                else:
                    index = key_words.index(w)
                    test_arr[test_index][index] += 1
            test_index += 1
    with open('data_train/' + class_name_en + '/all.txt', 'r', encoding='utf-8') as f:
        content = f.readlines()
        for text in content:
            print(class_name_en + ':' + str(train_index))
            for w in text.split():
                if w not in key_words:
                    continue
                else:
                    index = key_words.index(w)
                    train_arr[train_index][index] += 1
            train_index += 1

coo_test = coo_matrix(test_arr)
# print(coo_test)
save_npz('coo_test.npz', coo_test)
coo_train = coo_matrix(train_arr)
# print(coo_train)
save_npz('coo_train.npz', coo_train)

# df_0 = pd.DataFrame(test_arr)
# df_1 = pd.DataFrame(train_arr)
#
# with open('pkls/Word_Vector_Test.pkl', 'wb') as file:
#     pickle.dump(df_0, file)
# with open('pkls/Word_Vector_Train.pkl', 'wb') as file:
#     pickle.dump(df_1, file)

# df_0.to_csv('Word_Vector_Test.csv')
# df_1.to_csv('Word_Vector_Train.csv')
