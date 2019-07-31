#  多标签 多分类问题
# 本例子尝试先用朴素贝叶斯来进行分类
import pandas as pd
import numpy as np
import scipy
from scipy.io import arff

# data, meta = scipy.io.arff.loadarff('E:\\study\\ML-aura\\dataset\\yeast\\yeast-train.arff')
# df = pd.DataFrame(data)
# print(df)
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
# this will generate a random multi-label dataset  sparse代表是稀疏矩阵
# 调试时发现 X是format是csr  延申 ---> https://www.cnblogs.com/xbinworld/p/4273506.html  稀疏矩阵的存储
#  csr格式的稀疏矩阵 转化为 dataframe ---> https://stackoverflow.com/questions/36967666/transform-scipy-sparse-csr-to-pandas
#  df = pd.DataFrame(X.toarray())

X, y = make_multilabel_classification(sparse=True, n_labels=20, return_indicator='sparse', allow_unlabeled=False)
# 取出了样本数据之后 进行训练集的拆分
# X_df = pd.DataFrame(X.toarray())
# y_df = pd.DataFrame(y.toarray())
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)



# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
##  用朴素贝叶斯来做分类 延申--->  https://zhuanlan.zhihu.com/p/26262151
#  注意 样本多个特征 他们之间的相关性较小时，朴素贝叶斯性能最为良好 因为朴素贝叶斯的满足条件是 特征互相独立
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(X_train, y_train)
# predict
y_predict = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
naive_bayes_score = accuracy_score(y_test, y_predict)
print(naive_bayes_score)
### 0.72 准确率还ok
