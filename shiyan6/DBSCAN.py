import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 指定中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用中文宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据点及其名称
data = np.array([[2, 10, 'A'],
                 [2, 5, 'B'],
                 [8, 4, 'C'],
                 [5, 8, 'D'],
                 [7, 5, 'E'],
                 [6, 4, 'F'],
                 [1, 2, 'G'],
                 [4, 9, 'H']])

# 提取坐标数据
coordinates = data[:, :2].astype(float)

# 创建DBSCAN模型
dbscan = DBSCAN(eps=2, min_samples=2)

# 运行DBSCAN算法
clusters = dbscan.fit_predict(coordinates)

# 绘制原始数据和聚类结果
plt.figure(figsize=(10, 8))

# 绘制聚类结果及点的名称
for i, cluster_num in enumerate(np.unique(clusters)):
    if cluster_num == -1:
        plt.scatter(coordinates[clusters == cluster_num][:, 0], coordinates[clusters == cluster_num][:, 1],
                    c='gray', marker='o', s=100, label='噪声点')
    else:
        plt.scatter(coordinates[clusters == cluster_num][:, 0], coordinates[clusters == cluster_num][:, 1],
                    marker='o', s=100, label=f'聚类簇 {cluster_num + 1}')

# 标注点的名称
for i, txt in enumerate(data[:, 2]):
    plt.annotate(txt, (coordinates[i, 0], coordinates[i, 1]), fontsize=12, ha='right', va='bottom')

plt.title('DBSCAN，eps=2, MinPts=2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# 更新ɛ值为√10≈3.16
dbscan = DBSCAN(eps=np.sqrt(10), min_samples=2)
clusters = dbscan.fit_predict(coordinates)

# 绘制新的聚类结果
plt.figure(figsize=(10, 8))

# 绘制聚类结果及点的名称
for i, cluster_num in enumerate(np.unique(clusters)):
    if cluster_num == -1:
        plt.scatter(coordinates[clusters == cluster_num][:, 0], coordinates[clusters == cluster_num][:, 1],
                    c='gray', marker='o', s=100, label='噪声点')
    else:
        plt.scatter(coordinates[clusters == cluster_num][:, 0], coordinates[clusters == cluster_num][:, 1],
                    marker='o', s=100, label=f'聚类簇 {cluster_num + 1}')

# 标注点的名称
for i, txt in enumerate(data[:, 2]):
    plt.annotate(txt, (coordinates[i, 0], coordinates[i, 1]), fontsize=12, ha='right', va='bottom')

plt.title('DBSCAN，eps=√10, MinPts=2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
