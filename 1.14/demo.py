from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from util import print_dts



def naive_bayes_classifier(train_x, train_y):  # 朴素贝叶斯
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


def knn_classifier(train_x, train_y):  # knn
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


def logistic_regression_classifier(train_x, train_y):  # 逻辑回归树
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


def random_forest_classifier(train_x, train_y):  # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(n_estimators=8)
    model = RandomForestClassifier(criterion='gini', max_depth=2, n_estimators=8, oob_score=True)
    model.fit(train_x, train_y)
    return model


def decision_tree_classifier(train_x, train_y):  # 决策树
    from sklearn import tree
    model = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10)
    model.fit(train_x, train_y)

    return model


def svm_classifier(train_x, train_y):  # svm
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


train = fetch_20newsgroups(data_home="data", subset='train')
test = fetch_20newsgroups(data_home="data", subset = 'test')

vectorizer = TfidfVectorizer()  # 词频逆文本频率，把一段话转化为向量
train_v = vectorizer.fit_transform(train.data)
test_v = vectorizer.transform(test.data)
model = random_forest_classifier(train_v, train.target)  # 多次调用不同函数获取结果
y_predict = model.predict(test_v)
s = model.score(test_v, test.target)
print(s)

print_dts(model, train_v, train.target, test_v, test.target)

