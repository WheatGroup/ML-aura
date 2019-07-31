# 多项式回归
import itertools
import matplotlib.pyplot as plt
import numpy as np
from abupy import ABuSymbolPd
from sklearn import metrics
from abupy import ABuSymbolPd

tsla_close = ABuSymbolPd.make_kl_df('usTSLA').close
# x序列: 0，1，2, ...len(tsla_close)
x = np.arange(0, tsla_close.shape[0])
# 收盘价格序列
y = tsla_close.values


_, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
axs_list = list(itertools.chain.from_iterable(axs))

poly = np.arange(1, 10, 1)
##  poly即为多项式的意思 p_count=1 也就是将x, y按照一阶多项式线性回归 p_count=2 二阶（0,x,y,xy,x^2, y^2）
# 三阶（多项式由0,x,y,x^3,y^3,x*y^2,x^2*y组成 ，拟合出来系数）
for p_count, ax in zip(poly, axs_list):
    p = np.polynomial.Chebyshev.fit(x, y, p_count)
    y_fit = p(x)
    mse = metrics.mean_squared_error(y, y_fit)
    ax.set_title('{} poly MSE={}'.format(p_count, mse))
    ax.plot(x, y, '', x, y_fit, 'r.')
plt.show()

