import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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


DT_model = DecisionTreeRegressor(max_depth=4)

AdaBoost_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300)

print_dts(DT_model, X_train, y_train, X_test, y_test)

# DT_model.fit(X_train, y_train)
# AdaBoost_model.fit(X_train, y_train)



