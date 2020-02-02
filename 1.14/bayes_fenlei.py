from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from util import print_dts, fit_model
import numpy as np

#获取数据
newsgroups_train = fetch_20newsgroups(data_home="data", subset='train')
newsgroups_test = fetch_20newsgroups(data_home="data", subset = 'test')

print(type(newsgroups_train))
print(type(newsgroups_test))

#单词向量化, 单词向量化的模型不需要label
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)


NBM = [MultinomialNB(alpha=0.01), BernoulliNB(alpha=0.01)]
NAME = ["多项式","伯努利"]
# 假如决策树模型

from sklearn import tree
DT = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10)
NBM.append(DT)
NAME.append("决策树")
#
RF = RandomForestClassifier(criterion='gini', max_depth=2, n_estimators=5, oob_score=True)
NBM.append(RF)
NAME.append("随机森林")

# print_dts(RF, vectors, newsgroups_train.target, vectors_test, newsgroups_test.target)
# print_dts(DT, vectors, newsgroups_train.target, vectors_test, newsgroups_test.target)
best_estimator_ = fit_model(vectors, newsgroups_train.target)
print(best_estimator_)

'''
for itr, itrname in zip(NBM, NAME):
    print(itr, itrname)
    itr.fit(vectors, newsgroups_train.target)
    pred = itr.predict(vectors_test)
    dts = len(np.where(pred == newsgroups_test.target)[0]) / len(newsgroups_test.target)
    print("{} 精度:{:.3f}".format(itrname, dts * 100))


'''