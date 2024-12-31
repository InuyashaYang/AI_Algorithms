# algorithms/KNN/knn.py
import numpy as np
from collections import Counter

class KNN:
    """
    K-近邻分类算法实现。
    """

    def __init__(self, n_neighbors=5, distance_metric='euclidean'):
        """
        初始化KNN模型。

        Args:
            n_neighbors: 最近邻的数量。
            distance_metric: 距离度量方法，支持 'euclidean' 和 'manhattan'。
        """
        self.n_neighbors = n_neighbors
        if distance_metric not in ['euclidean', 'manhattan']:
            raise ValueError("distance_metric 必须是 'euclidean' 或 'manhattan'")
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        保存训练数据。

        Args:
            X: 训练数据的特征矩阵，形状为 (n_samples, n_features)。
            y: 训练数据的目标向量，形状为 (n_samples,)。
        """
        self.X_train = X
        self.y_train = y
        return self

    def _compute_distance(self, x):
        """
        计算输入样本x与所有训练样本之间的距离。

        Args:
            x: 单个样本的特征向量，形状为 (n_features,)。

        Returns:
            距离向量，形状为 (n_samples,)。
        """
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        return distances

    def predict(self, X):
        """
        对输入数据进行预测。

        Args:
            X: 待预测数据的特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            预测标签，形状为 (n_samples,)。
        """
        predictions = []
        for x in X:
            distances = self._compute_distance(x)
            neighbors_idx = np.argsort(distances)[:self.n_neighbors]
            neighbors_labels = self.y_train[neighbors_idx]
            most_common = Counter(neighbors_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
