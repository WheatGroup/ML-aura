##  作业 总结
---
2020.2.8
*  画图时 matplotlib.pyplot.contourf 函数的作用：
    是来绘制等高线的，contour和contourf都是画三维等高线图的，
    不同点在于contour() 是绘制轮廓线，contourf()会填充轮廓
    注意：
    但在这个函数中输入的参数是 <u>x,y对应的网格数据 以及此网格对应的高度值</u>
* 对make_moons的数据画法是固定的 如作业中

* 关于<u>生成模型</u>和 <u>判别模型
    - 这篇解读特别好 https://www.zhihu.com/question/20446337
--------
##  作业 疑问点

* predict_proba这个函数是做什么用的
    -  解决了 predict_proba是返回样本属于每个类别的概率
* 高斯分布的实现中 为甚庅要对样本的概率密度取log  也就是logN