# 多项式回归
import itertools
import matplotlib.pyplot as plt
import numpy as np
from abupy import ABuSymbolPd
from sklearn import metrics

tsla_close = ABuSymbolPd.make_kl_df('usTSLA').close
# x序列: 0，1，2, ...len(tsla_close)
x = np.arange(0, tsla_close.shape[0])
# 收盘价格序列
y = tsla_close.values


_, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
axs_list = list(itertools.chain.from_iterable(axs))

poly = np.arange(1, 10, 1)
for p_count, ax in zip(poly, axs_list):
    p = np.polynomial.Chebyshev.fit(x, y, p_count)
    y_fit = p(x)
    mse = metrics.mean_squared_error(y, y_fit)
    print(mse)

