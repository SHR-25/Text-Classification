"""
逻辑回归模型
"""
import pickle
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

coo_test = load_npz('coo_test.npz')
# print(coo_test)
coo_train = load_npz('coo_train.npz')
# print(coo_train)
class_arr = np.array([int(i / 5000) for i in range(50000)])

model = LogisticRegression(C=0.01, max_iter=3000)
start = time.time()
model.fit(coo_train, class_arr)
end = time.time()
print('Train time: %s Seconds' % (end - start))
start = time.time()
pre = model.predict(coo_test)
end = time.time()
print('Test time: %s Seconds' % (end - start))
with open('pkls/LR_pre.pkl', 'wb') as f:
    pickle.dump(pre, f)

# 混淆矩阵
C = metrics.confusion_matrix(class_arr, pre)
with open('pkls/confusion_matrix_LR.pkl', 'wb') as f:
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
