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

import matplotlib.pyplot as plt
# 画出各个特征在类别上的分布情况
splt = plt.subplot(221)
# 数据类型  横坐标是数据的分布 纵坐标是数据的种类
splt.scatter(iris.data[:, 0], iris.target, c=iris.target)
splt = plt.subplot(222)
splt.scatter(iris.data[:, 1], iris.target, c=iris.target)
splt = plt.subplot(223)
splt.scatter(iris.data[:, 2], iris.target, c=iris.target)
splt = plt.subplot(224)
splt.scatter(iris.data[:, 3], iris.target, c=iris.target)

plt.show()

import matplotlib.pyplot as plt
# scatter(x, y) 能看出来x, y的相关性
# 在矩阵的对称轴上的相关性肯定是1  所以说 散点图有相关性的情况下 画出来的散点图是直线
index = 1
for i in range(4):
    for j in range(4):
        splt = plt.subplot(4, 4, index)
        splt.scatter(iris.data[:, i], iris.data[:, j], c=iris.target)
        index += 1
plt.show()


from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
iris = load_iris()
data = iris.data
# 添加max_iter = 1000,以消除警告信息
selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2).fit(iris.data, iris.target)
data = selector.transform(iris.data)
print(data[0:5])
print(selector.ranking_)