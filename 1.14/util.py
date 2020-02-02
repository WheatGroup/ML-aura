# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.model_selection import learning_curve
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd #数据分析
import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 30), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"测试集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def one_hot_encoding(data_train):
    # 因为逻辑回归建模时，需要输入的特征都是数值型特征
    # 我们先对类目型的特征离散/因子化
    # 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
    # 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
    # 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
    # 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示
    # 处理categorical feature：一般就是通过dummy variable的方式解决，也叫one hot encode，可以通过pandas.get_dummies()或者 
    # sklearn中preprocessing.OneHotEncoder(), 本例子选用pandas的get_dummies()

    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def print_dts(estimator, X_train, y_train, X_test, y_test):
    '''
    打印模型的精度
    :param estimator: 模型
    '''
    print('开始训练')
    start = time.time()
    estimator.fit(X_train, y_train)
    print('训练完成')
    fit_end = time.time()
    print(fit_end - start)
    test_type = type(y_train[0])
    print('开始预测')
    pdt_start = time.time()
    test_y_pdt = estimator.predict(X_test).astype(test_type)
    print('测试集预测完成')
    pdt_end = time.time()
    print(pdt_end - pdt_start)
    train_y_pdt = estimator.predict(X_train).astype(test_type)
    test_dts = len(np.where(test_y_pdt == y_test)[0]) / len(y_test)
    train_dts = len(np.where(train_y_pdt == y_train)[0]) / len(y_train)

    # print(accuracy_score(y_test, estimator.predict(X_test).astype(test_type)))
    # print(accuracy_score(y_train, estimator.predict(X_train).astype(test_type)))

    print("{} 训练集精度:{:.3f}".format("Line", train_dts * 100))
    print("{} 测试集精度:{:.3f}".format("Line", test_dts * 100))


def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""

    score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)

    return score




# 提示: 导入 'KFold' 、决策树模型、 'make_scorer'、 'GridSearchCV'
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def fit_model(X, y):
    """ 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""

    cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)

    Classifier = DecisionTreeClassifier()

    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

    scoring_fnc = make_scorer(performance_metric)
    print('开始网格搜索')

    grid = GridSearchCV(estimator=Classifier, param_grid=params, scoring=scoring_fnc, cv=cross_validator)

    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)

    # 返回网格搜索后的最优模型
    return grid.best_estimator_