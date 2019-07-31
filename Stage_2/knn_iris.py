#  https://www.cnblogs.com/ybjourney/p/4702562.html
#  上篇链接是 knn算法的原理介绍

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
#coding: UTF-8
#Definiation of COLs:
#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
#5. class:
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica
#Missing Attribute Values: None

from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

iris = load_iris()
X = iris.data
Y = iris.target

# the degree-3 polynomial features
poly = PolynomialFeatures(3)
X_Poly = poly.fit_transform(X)
#  用衍生出来的数据X_Poly来进行特征工程，
#  这个函数train_test_split是 分割出来 样本内，样本外 train是训练数据集 也就是样本内 test是测试数据集  也就是样本外
X_train, X_test, Y_train, Y_test = train_test_split(X_Poly, Y, random_state=4)

# 初始化模型 取k为6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
Y_predict = knn.predict(X_test)
# 下一步是看 这个预测值与 实际的样本外的目标值的差异，计算准确率
# print(metrics.accuracy_score(Y_test, Y_predict))
# print(Y_predict)
# print(Y_test)

# n_neighbors 的值也会影响score的得分
#  这个score的底层调用的也是accuracy_score函数
#  accuracy_score(y, self.predict(X), sample_weight=sample_weight)
# 所以下面的函数调用和 metrics.accuracy_score的返回值是一样的
print(knn.score(X_test, Y_test))

# 交叉验证
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
##    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    scores = cross_val_score(knn, X_Poly, Y, cv=10, scoring='accuracy') # for classification
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.savefig('./knn.jpg')
plt.show()




