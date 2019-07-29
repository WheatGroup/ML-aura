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

print(Y_predict)
###### 策略的持久化
# method 2: joblib
from sklearn.externals import joblib
# Save
joblib.dump(knn, '../save/knn.pkl')
# restore
knn2 = joblib.load('../save/knn.pkl')
print(knn2.predict(X_test))
