import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# 加载MNIST数据集
digits = load_digits()
X = digits.data
y = digits.target

# 数据归一化处理
X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))

# 使用PCA降维到2维方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

# 使用K-means进行聚类，分成三个类
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X_norm)

# 定义颜色和标签
colors = ['red', 'green', 'blue']
labels = ['High Protection', 'Medium Protection', 'Low Protection']

plt.figure(figsize=(10, 8))  # 调整图形尺寸
for i, color in enumerate(colors):
    # 每个聚类的数据
    pca_data = X_pca[y_pred == i]
    plt.scatter(pca_data[:, 0], pca_data[:, 1], color=color, label=labels[i])

plt.title("K-means Clustering of MNIST Data with Different Protection Levels")
plt.xlabel("The variation direction mainly affected by the degree of protection 1")
plt.ylabel("The variation direction mainly affected by the degree of protection 2")
plt.legend()  # 显示图例
plt.show()
