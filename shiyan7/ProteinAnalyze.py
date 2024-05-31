import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sympy.physics.control.control_plots import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用中文宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

protein = pd.read_table('protein.txt', sep='\t')
protein.head()

protein.describe()

# 删除名为Country的数据
sprotein = protein.drop(['Country'], axis=1)
sprotein.head()

# Z-Score标准化方法
from sklearn import preprocessing

sprotein_scaled = preprocessing.scale(sprotein)
sprotein_scaled

# K值选择
from sklearn.cluster import KMeans

NumberOfClusters = range(1, 20)
kmeans = [KMeans(n_clusters=i, n_init=10) for i in NumberOfClusters]
score = [kmeans[i].fit(sprotein_scaled).score(sprotein_scaled) for i in range(len(kmeans))]
score

# 绘制ROC曲线
%matplotlib inline
plt.plot(NumberOfClusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# 随机设定聚类的数量未5，并以此为基础在数据矩阵上执行均值聚类
myKmeans = KMeans(algorithm="auto", n_clusters=5, n_init=10, max_iter=200, verbose=1)
myKmeans.fit(sprotein_scaled)
y_kmeans = myKmeans.predict(sprotein)
print(y_kmeans)

protein["所隶属的类簇"] = y_kmeans
protein


# 计算K=5时的轮廓系数
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(sprotein_scaled, y_kmeans)
# silhouette_avg
print("当K=", myKmeans.n_clusters, "时，轮廓系数为：", silhouette_avg)

# 计算K=2到20时的轮廓系数
# 定义可能的K值范围
k_range = range(2, 21)

# 用于存储每个K值对应的轮廓系数
silhouette_avgs = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)   # 设定聚类模型的参数
    cluster_labels = kmeans.fit_predict(sprotein_scaled)    # 获取聚类结果
    silhouette_avg = silhouette_score(sprotein_scaled, cluster_labels)
    silhouette_avgs.append(silhouette_avg)

# 绘制K值和对应的轮廓系数
plt.figure(figsize=(8, 6))
plt.rcParams["font.family"] = "simHei"  # 汉字显示
plt.plot(k_range, silhouette_avgs, "o-")
plt.xlabel("K值")
plt.ylabel("K值及其轮廓系数")
plt.show()


# from Bio.Cluster import kcluster
# from sklearn.metrics import silhouette_score
# clusters, error, nfound = kcluster(sprotein_scaled, nclusters=5, dist='u', npass=100)
# silhouette_avg = silhouette_score(sprotein_scaled, clusters, metric='cosine')


# # 设置聚类数量范围
# NumberOfClusters = range(2, 21)
#
# # 初始化轮廓系数列表
# silhouette_scores = []
#
# # 计算每个聚类数量的轮廓系数
# for n_clusters in NumberOfClusters:
#     clusters, _, _ = kcluster(sprotein_scaled, nclusters=n_clusters, dist='u', npass=100)
#     silhouette_avg = silhouette_score(sprotein_scaled, clusters, metric='cosine')
#     silhouette_scores.append(silhouette_avg)
#
# # 绘制图像
# plt.plot(NumberOfClusters, silhouette_scores, marker='o')
# plt.xlabel('K值')
# plt.ylabel('轮廓系数')
# plt.title('K值及轮廓系数')
# plt.grid(True)
# plt.show()

estimator = KMeans(algorithm="auto", n_clusters=4, n_init=10, max_iter=200, verbose=0)
estimator.fit(sprotein_scaled)

# 绘制聚类结果的散点图
for i in range(4):
    plt.scatter(sprotein_scaled[estimator.labels_ == i, 0],
                sprotein_scaled[estimator.labels_ == i, 1],
                label=f'Cluster {i+1}')

plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1],  c='red')
plt.title('Clustered Data')
plt.legend()
plt.show()
