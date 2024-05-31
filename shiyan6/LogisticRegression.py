import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 数据读入
from sklearn.datasets import load_breast_cancer

breast_canser = load_breast_cancer()

# 显示数据字典的键
print(breast_canser.keys())

# 数据集的完整描述
print((breast_canser.DESCR))

# 数据库的特征名称
print(breast_canser.feature_names)

# 数据形状
print(breast_canser.data.shape)

# 将data与target转化为数据框类型
X = pd.DataFrame(breast_canser.data, columns=breast_canser.feature_names)
y = pd.DataFrame(breast_canser.target, columns=['class'])

# 合并数据框
df = pd.concat([X, y], axis=1)
df

# 查看数据基本信息
df.info()

# 查看描述性统计信息
df.describe()

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test

# 调整y_train的形状
y_train = y_train.values.ravel()

# 模型训练
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 默认参数模型预测结果
y_pred = model.predict(X_test)
y_pred

# 混淆矩阵
matrix = metrics.confusion_matrix(y_test, y_pred)
matrix

# 准确度
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# 精度
print("Precision: ", metrics.precision_score(y_test, y_pred))

# 以C、penalty参数和值设置字典列表param_grid。设置cv参数值为5
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 20, 50, 100], 'penalty': ['l2']}
n_folds = 5

# 调用GridSearchCV函数，进行5折交叉验证，对估计器LogisticRegression()的指定参数param_grid进行详尽搜索，最终得到最终的最优模型参数
from sklearn.model_selection import GridSearchCV

estimator = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=n_folds)
estimator.fit(X_train, y_train)

# 搜索选择的最高分
estimator.best_estimator_

# 调参后的模型训练
model1 = LogisticRegression(max_iter=10000, C=50, penalty='l2')
model1.fit(X_train, y_train)

# 调参后的预测结果
y_pred1 = model1.predict(X_test)
y_pred1

# 调参后的模型混淆矩阵结果
matrix1 = metrics.confusion_matrix(y_test, y_pred1)
matrix1

# 调参后的准确度、精度
print("Accuracy1: ", metrics.accuracy_score(y_test, y_pred1))
print("Precision1: ", metrics.precision_score(y_test, y_pred1))
