import numpy as np
from scipy.sparse import coo_matrix
from sklearn.svm import SVC

x = np.array([[2, 0], [0, 2], [0, 0], [3, 0], [0, 3], [3, 3]])
y = np.array([1, 1, 1, 2, 2, 2])
model = SVC(kernel='linear', probability=True)  # probability=False时，没办法调用 model.predict_proba()函数
model.fit(coo_matrix(x).tocsr(), y)
C = [[-1, -1], [4, 4]]
pre = model.predict_proba(coo_matrix(C).tocsr())
print(pre)
pre1 = model.predict(C)
print(pre1)
