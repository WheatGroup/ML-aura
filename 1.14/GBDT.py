import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
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

import matplotlib.pyplot as plt
original_params = {'n_estimators': 100}
plt.figure()

for label, color, setting in [('learning_rate= 0.01', 'orange',
                               {'learning_rate': 0.01}),
                              ('learning_rate= 0.05', 'turquoise',
                               {'learning_rate': 0.05})]:
    params = dict(original_params)
    params.update(setting)

    clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = clf.loss_(y_test, y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)

plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')
plt.show()
