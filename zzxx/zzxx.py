import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from pylab import mpl

df_boston = pd.read_csv('housing.csv', header=0)  # 数据读入
df_boston.head()

df_boston.shape  # 探索性分析

df_boston.isnull().any()  # 判断哪些列包含缺失值

X = df_boston.drop(columns=['MEDV'])
y = df_boston['MEDV']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train_shape:", X_train.shape)
print("X_test_shape:", X_test.shape)
print("y_train_shape:", y_train.shape)
print("y_test_shape:", y_test.shape)

# 用随机森林解决分类和回归问题
rf = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

# 回归模型评价指标
y_pred = rf.predict(X_test)
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
mape = np.mean(np.abs(y_test - y_pred) / y_test)
print('Mean Absolute Percentage Error(MAPE):', round(mape * 100, 2))
print('score:', rf.score(X_test, y_test))

# GridSearchCV网格调参
param_grid = {'bootstrap': [True], 'n_estimators': [5, 10, 20, 50, 100, 150, 200], 'max_depth': [3, 5, 7],
              'max_features': [0.6, 0.7, 0.8, 1], 'min_samples_split': [2, 3, 4], 'min_samples_leaf': [1, 2, ]}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid=param_grid, cv=3)
grid.fit(X_train, y_train)

grid.best_params_  # 通过best_params_查看回归效果最好的模型参数

# 基于以上参数建立随机森林回归模型
rf_reg = RandomForestRegressor(bootstrap=True, n_estimators=50, max_depth=7, max_features=0.6, min_samples_leaf=1,
                               min_samples_split=3, random_state=42)
rf_reg.fit(X_train, y_train)

# 为评价调参后的模型效果，重新输出模型回归评价指标结果
y_pred = rf_reg.predict(X_test)
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
mape = np.mean(np.abs(y_test - y_pred) / y_test)
print('Mean Absolute Percentage Error(MAPE):', round(mape * 100, 2))
print('score:', rf_reg.score(X_test, y_test))

'''
与调参前的模型相比，该输出结果显示了调参后的模型平均绝对误差MAE、均方误差MSE、均方根误差RMSE和平均绝对误差百分比MAPE的值均降低了，
回归系数R^2(score）的值上升，表明回归效果显著提升。
'''

# 进行特征重要度分析并以可视化的方式直观展现各变量在房价预测中的重要性
rf_reg.feature_importances_

X.columns  # 查看各个得分下对应特征名称

# 按照重要度递增对特征进行排序
print('特征排序:')
feature_names = X.columns
feature_importances = rf_reg.feature_importances_
indices = np.argsort(feature_importances)

for index in indices:
    print('feature %s (%f)' % (feature_names[index], feature_importances[index]))

# 将特征重要度分析结果以条形图的方式展现
plt.figure(figsize=(7, 5))
plt.title('基于随机森林模型的波士顿房价回归中特征重要度')

plt.bar(range(len(feature_importances)), feature_importances[indices], color='gray')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], color='black', rotation=45)
plt.show()

# 将预测结果和实际结果组合对比，并通过可视化散点图查看预测效果


mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Songti SC']
prediction = pd.DataFrame(rf_reg.predict(X_test), columns=['prediction'])
MEDV = pd.DataFrame(y_test, columns=['MEDV']).reset_index()
result = pd.concat([prediction, MEDV], axis=1).drop('index', axis=1)

result['MEDV'].plot(style='k.', figsize=(15, 5))
result['prediction'].plot(style='r.')
plt.legend(fontsize='15', markerscale=3)  # 设置图例字号
plt.tick_params(labelsize=15)  # 设置坐标数字大小
plt.grid()
