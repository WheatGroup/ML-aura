#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""

@version: 0.1
@author:  admin
@email:   wangrui0810@gmail.com
@file:    jiangwei.py
@time:    2020/1/20 14:23
"""

'''
此处先涉及PCA 降维
'''
import numpy as np
x = np.mat([[ 0.9, 2.4, 1.2, 0.5, 0.3, 1.8, 0.5, 0.3, 2.5, 1.3],
            [ 1, 2.6, 1.7, 0.7, 0.7, 1.4, 0.6, 0.6, 2.6, 1.1]])
x = x.T
T = x - x.mean(axis=0)
# C是矩阵x的协方差矩阵
C = np.cov(x.T)

# x矩阵的主成分是 其协方差矩阵的特征向量按照对应的特征值大小排序得到的
# 求矩阵x的特征向量v
w, v = np.linalg.eig(C)
v_ = np.mat(v[:,0])     #每个特征值对应的是特征矩阵的每个列向量

# v中的n个列向量就是矩阵的特征向量 按照特征值大小排列  第一个特征向量就是第一主成分
# x.dot(v[0])所得到的结果就是按照第一主成分映射的结果 完成了降维
v_ = v_.T       #默认以行向量保存，转换成公式中的列向量形式
y = T * v_
print(y)