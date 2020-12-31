"""
SVM建模与评价
"""

import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn import metrics
from sklearn.svm import SVC

class_list = {'财经': 'Economics', '房产': 'House', '社会': 'Society', '时尚': 'Fashion', '教育': 'Education',
              '科技': 'Technology', '时政': 'Politics', '体育': 'PE', '游戏': 'Game', '娱乐': 'Entertainment'}

coo_test = load_npz('coo_test.npz')
# print(coo_test)
coo_train = load_npz('coo_train.npz')
# print(coo_train)
class_arr = np.array([int(i / 5000) for i in range(50000)])

model = SVC(kernel='rbf', C=6, gamma=0.001)
start = time.time()
model.fit(coo_train.tocsr(), class_arr)
end = time.time()
print('Train time: %s Seconds' % (end - start))
start = time.time()
pre = model.predict(coo_test.tocsr())
end = time.time()
print('Test time: %s Seconds' % (end - start))
print(pre)
with open('pkls/svm_pre.pkl', 'wb') as f:
    pickle.dump(pre, f)

# with open('pkls/svm_pre.pkl', 'rb') as f:
#     pre = pickle.load(f)

# 混淆矩阵
C = metrics.confusion_matrix(class_arr, pre)
confusion_matrix = pd.DataFrame(C, columns=class_list.values(),
                                index=class_list.values())
confusion_matrix.to_csv('Confusion_Matrix_SVM.csv')
with open('pkls/confusion_matrix_svm.pkl', 'wb') as f:
    pickle.dump(C, f)
print("混淆矩阵为：\n", C)
# 计算准确率（accuracy）
accuracy = metrics.accuracy_score(class_arr, pre)
print("准确率为：\n", accuracy)
# 计算精确率（precision）
precision = metrics.precision_score(class_arr, pre, average=None)
print("精确率为：\n", precision)
print('均值{:.4f}\n'.format(sum(precision) / 10))
# 计算召回率（recall）
recall = metrics.recall_score(class_arr, pre, average=None)
print("召回率为：\n", recall)
print('均值{:.4f}\n'.format(sum(recall) / 10))
# 计算F1-score（F1-score）
F1_score = metrics.f1_score(class_arr, pre, average=None)
print("F1值为：\n", F1_score)

cp = metrics.classification_report(class_arr, pre)
print("---------------分类报告---------------\n", cp)
