# algorithms/Logistic_Regression/logistic_regression.py
import numpy as np

class LogisticRegression:
    """
    使用梯度下降法实现的逻辑回归模型。
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True, verbose=False):
        """
        初始化逻辑回归模型。

        Args:
            learning_rate: 学习率，控制梯度下降的步长。
            n_iterations: 迭代次数，控制梯度下降的迭代次数。
            fit_intercept: 是否添加截距项（偏置项）。
            verbose: 是否在训练过程中输出损失值。
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None  # 模型参数
        self.costs = []    # 记录每次迭代的损失值

    def _add_intercept(self, X):
        """
        添加截距项（偏置项）到特征矩阵中。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            添加了偏置项后的特征矩阵，形状为 (n_samples, n_features + 1)。
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z):
        """
        计算 Sigmoid 函数。

        Args:
            z: 输入值，可以是标量、向量或矩阵。

        Returns:
            Sigmoid 函数的输出。
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, h, y):
        """
        计算对数损失函数（Log Loss）。

        Args:
            h: 预测概率，形状为 (n_samples,)。
            y: 真实标签，形状为 (n_samples,)。

        Returns:
            对数损失的值。
        """
        m = y.shape[0]
        epsilon = 1e-5  # 为了避免 log(0) 的情况
        cost = -(1/m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
        return cost

    def fit(self, X, y):
        """
        使用训练数据拟合模型。

        Args:
            X: 训练数据的特征矩阵，形状为 (n_samples, n_features)。
            y: 训练数据的目标向量，形状为 (n_samples,)。

        Returns:
            self
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)  # 初始化参数为0

        for i in range(self.n_iterations):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / n_samples
            self.theta -= self.learning_rate * gradient
            cost = self._cost_function(h, y)
            self.costs.append(cost)

            if self.verbose and i % 100 == 0:
                print(f'迭代 {i}: 损失 = {cost}')

        return self

    def predict_prob(self, X):
        """
        预测给定输入的概率。

        Args:
            X: 待预测数据的特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            预测概率，形状为 (n_samples,)。
        """
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        """
        预测给定输入的类别标签。

        Args:
            X: 待预测数据的特征矩阵，形状为 (n_samples, n_features)。
            threshold: 判定阈值，大于等于该阈值预测为1，否则为0。

        Returns:
            预测标签，形状为 (n_samples,)。
        """
        return self.predict_prob(X) >= threshold
