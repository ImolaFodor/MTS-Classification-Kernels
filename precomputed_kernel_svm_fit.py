from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
from pandas import read_csv
from scipy.io import savemat
import os
from numpy import genfromtxt


# X_train, X_test = X[:1000, :], X[100:, :]
# y_train, y_test = y[:1000], y[100:]

K = genfromtxt('data/gram_matrix.csv', delimiter=',')
Y = genfromtxt('data/labels.csv', delimiter=',')
print(Y)
print(K)

svc = SVC(kernel='precomputed')
svm = svc.fit(K, Y)
print(svm.support_)

# define output path
path = './'

# extract params from fitted svm
mat = {'SVIdx': svm.support_.astype(np.float32),
       'DualCoeff': svm.dual_coef_[0].astype(np.float32),
       'Bias': svm.intercept_.astype(np.float32)}

# for every params store it as a bin file
for k, v in mat.items():
    v.tofile(os.path.join(path, f'{k}.bin'))






