import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston

train = np.load("../data/train.npz")
test = np.load("../data/test.npz")

# 手写数字集合有5万多的数据 这边取了6000个作为训练集
vect_t = np.array([[itr] for itr in range(10)])
X_train = train["images"][:6000]
# train['labels']相当于做了独热分布  这边做一下点乘 相当于转成了数值
y_train = np.dot(train["labels"][:6000], vect_t).ravel().astype('int')
X_test = test["images"][:6000]
y_test = np.dot(test["labels"][:6000], vect_t).ravel().astype("int")
#  此处一定要注意  决策树和adaboost都区分回归还是分类 DecisionTreeRegressor  分类 DecisionTreeClassifier


xgb_model = xgb.XGBRegressor()
# 网格搜索完成 xgboost的手写数字识别分类的调参问题
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2, 4, 6],
                    'n_estimators': [50, 100, 200]}, verbose=1)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)
'''
0.8119864189891217
{'max_depth': 6, 'n_estimators': 200}
'''
