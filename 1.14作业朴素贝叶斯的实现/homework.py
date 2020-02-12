# 本作业是 完成朴素贝叶斯的内部实现

from sklearn.datasets import make_moons, make_circles, make_classification
#获取数据
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

# 取完数据之后 需要看一下数据的样式 开始画图

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#调整图片风格
# mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 应该把数据的维度都拼到一起喂给模型
# Z = method.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()

#  先尝试用现成的朴素贝叶斯进行分类
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#训练
gnb.fit(X, y)
# 模型训练完毕之后 开始预测 预测出数据属于某类的概率
Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
print(Z.shape)
Z1 = Z[:, 1]
Z2 = Z1.reshape(xx.shape)
# 画出数据的等高线
plt.contourf(xx, yy, Z2, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()

# 开始手写朴素贝叶斯
# -*- coding: utf-8 -*-

import numpy as np
# 朴素贝叶斯算法
class NaiveBayes():
    def N(self, x, mu, std):
        """
        标准正态分布
        """
        # 求高斯分布的概率
        par = 1/(np.sqrt(2*np.pi)*std)
        return par*np.exp(-(x-mu)**2/2/std**2)

    def logN(self, x, class_type):
        """
        标准正态分布对数
        """
        if class_type==0:
            # 此处为什么要取一个log呢
            return np.log(self.N(x, self.mu0, self.std0))
        else:
            return np.log(self.N(x, self.mu1, self.std1))

    def fit(self, X, y):
        """
        训练过程为对于数据的统计
        """
        X0 = X[y==0]
        X1 = X[y==1]
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)
        self.std0 = np.std(X0, axis=0)
        self.std1 = np.std(X1, axis=0)

    def predict_proba(self, xx):
        """
        预测过程
        """
        # predict_proba是求出 在已知xx的条件下 xx分别属于y1...yk的概率
        prb = []
        for x in xx:
            # 这里有无问题？
            prb0_log = np.sum(self.logN(x, 0))
            prb1_log = np.sum(self.logN(x, 1))
            prb0 = np.exp(prb0_log)
            prb1 = np.exp(prb1_log)
            #这里有无编程问题？
            prb0 = prb0 / (prb0 + prb1)
            prb1 = prb1 / (prb0 + prb1)
            prb.append([prb0, prb1])
        return np.array(prb)


method = NaiveBayes()
method.fit(X, y)

#调整图片风格
mpl.style.use('fivethirtyeight')
#定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
#预测可能性
Z = method.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
#绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("GaussianNaiveBayes")
plt.axis("equal")
plt.show()