# 特征排序：通过递归特征消除，获取鸢尾花数据集特征排名

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()
# 探索数据
print(type(iris))
print(dir(iris))
print(iris.data.shape)
print(iris.target.shape)
print(iris.feature_names)


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

NBM = [MultinomialNB(alpha=0.01), BernoulliNB(alpha=0.01)]
NAME = ["多项式","伯努利"]

from sklearn import tree
DT = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10)
NBM.append(DT)
NAME.append("决策树")
#
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(criterion='gini', max_depth=2, n_estimators=5, oob_score=True)
NBM.append(RF)
NAME.append("随机森林")

from sklearn.model_selection import train_test_split

# Train test split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.20, random_state=0)
# print(X_train.shape, X_test.shape)
import numpy as np
for itr, itrname in zip(NBM, NAME):
    print(itr, itrname)
    itr.fit(X_train, y_train)
    pred = itr.predict(X_test)
    dts = len(np.where(pred == y_test)[0]) / len(y_test)
    print("{} 精度:{:.3f}".format(itrname, dts * 100))

