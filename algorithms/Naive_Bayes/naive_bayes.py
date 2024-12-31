# algorithms/Naive_Bayes/naive_bayes.py
import numpy as np

class GaussianNaiveBayes:
    """
    高斯朴素贝叶斯分类器实现。
    """

    def __init__(self):
        """
        初始化模型参数。
        """
        self.classes = None
        self.priors = {}
        self.mean = {}
        self.var = {}
        self.fitted = False  # 标记模型是否已经被拟合

    def fit(self, X, y):
        """
        训练模型，计算先验概率、均值和方差。

        Args:
            X: 训练数据的特征矩阵，形状为 (n_samples, n_features)。
            y: 训练数据的目标向量，形状为 (n_samples,)。
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("训练数据 X 和 y 不能为空。")
        if X.shape[0] != y.shape[0]:
            raise ValueError("训练数据 X 和 y 的样本数量不匹配。")

        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.priors[cls] = X_c.shape[0] / X.shape[0]
            self.mean[cls] = np.mean(X_c, axis=0)
            self.var[cls] = np.var(X_c, axis=0) + 1e-9  # 加上一个小常数防止除零
        self.fitted = True
        return self

    def _gaussian_prob(self, cls, x):
        """
        计算每个特征在给定类别下的概率密度。

        Args:
            cls: 类别标签。
            x: 单个样本的特征向量，形状为 (n_features,)。

        Returns:
            概率密度数组，形状为 (n_features,)。
        """
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        """
        对输入数据进行预测。

        Args:
            X: 待预测数据的特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            预测标签，形状为 (n_samples,)。
        """
        if not self.fitted:
            raise ValueError("模型尚未被拟合，请先调用 fit 方法。")
        if X.shape[1] != len(next(iter(self.mean.values()))):
            raise ValueError(f"输入特征维度 {X.shape[1]} 与训练时的维度 {len(next(iter(self.mean.values())))} 不匹配。")

        y_pred = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                conditional = np.sum(np.log(self._gaussian_prob(cls, x)))
                posterior = prior + conditional
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

    def predict_proba(self, X):
        """
        计算每个类别的预测概率。

        Args:
            X: 待预测数据的特征矩阵，形状为 (n_samples, n_features)。

        Returns:
            预测概率矩阵，形状为 (n_samples, n_classes)。
        """
        if not self.fitted:
            raise ValueError("模型尚未被拟合，请先调用 fit 方法。")
        if X.shape[1] != len(next(iter(self.mean.values()))):
            raise ValueError(f"输入特征维度 {X.shape[1]} 与训练时的维度 {len(next(iter(self.mean.values())))} 不匹配。")

        proba = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                conditional = np.sum(np.log(self._gaussian_prob(cls, x)))
                posterior = prior + conditional
                posteriors.append(posterior)
            # 转换为概率
            max_log = np.max(posteriors)
            exps = np.exp(posteriors - max_log)  # 为了数值稳定性
            proba_cls = exps / np.sum(exps)
            proba.append(proba_cls)
        return np.array(proba)
