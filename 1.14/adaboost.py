import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from util import plot_learning_curve, print_dts
train = np.load("../data/train.npz")
test = np.load("../data/test.npz")

# 手写数字集合有5万多的数据 这边取了6000个作为训练集
vect_t = np.array([[itr] for itr in range(10)])
X_train = train["images"][:6000]
# train['labels']相当于做了独热分布  这边做一下点乘 相当于转成了数值
y_train = np.dot(train["labels"][:6000], vect_t).ravel().astype('int')
X_test = test["images"][:6000]
y_test = np.dot(test["labels"][:6000], vect_t).ravel().astype("int")
#  此处一定要注意  决策树和adaboost都区分回归还是分类 DecisionTreeRegressor  分类 DecisionTreeClassifier
DT_model = DecisionTreeClassifier(max_depth=4)

# n_estimators代表 最大的弱学习器个数 过大容易过拟合 过小欠拟合 默认100
AdaBoost_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, min_samples_split=5, min_samples_leaf=5), n_estimators=200, learning_rate=0.05, algorithm='SAMME.R')
print('AdaBoost')
print_dts(AdaBoost_model, X_train, y_train, X_test, y_test)

print('GBDT')
original_params = {'n_estimators': 100, 'learning_rate': 0.01}
GBDT_model = GradientBoostingClassifier(**original_params)
print_dts(GBDT_model, X_train, y_train, X_test, y_test)

# plot_learning_curve(AdaBoost_model, 'adaboost', X_train, y_train)

# DT_model.fit(X_train, y_train)
# AdaBoost_model.fit(X_train, y_train)



