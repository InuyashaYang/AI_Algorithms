# algorithms/Linear_Regression/linear_regression.py
import numpy as np

class LinearRegression:
    """
    使用梯度下降法实现的线性回归模型。
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化线性回归模型。

        Args:
            learning_rate: 学习率，控制梯度下降的步长。
            n_iterations: 迭代次数，控制梯度下降的迭代次数。
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None  # 模型参数
        self.costs = []  # 记录每次迭代的损失值

    def fit(self, X, y):
        """
        使用训练数据拟合模型。

        Args:
            X: 训练数据的特征矩阵，形状为 (n_samples, n_features)。
            y: 训练数据的目标向量，形状为 (n_samples,)。
        """
        # 添加偏置项 (x0 = 1)
        X = np.c_[np.ones(X.shape[0]), X]

        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)  # 初始化参数

        for _ in range(self.n_iterations):
            # 预测值
            y_pred = np.dot(X, self.theta)

            # 计算误差
            error = y_pred - y

            # 计算梯度
            gradient = (1 / n_samples) * np.dot(X.T, error)

            # 更新参数
            self.theta -= self.learning_rate * gradient

            # 计算损失并记录
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.costs.append(cost)

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        Args:
            X: 待预测数据的特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            预测值，形状为 (n_samples,)。
        """
        # 添加偏置项 (x0 = 1)
        X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.theta)
