# test/Naive_Bayes/test_naive_bayes.py
import unittest
import numpy as np
from algorithms.Naive_Bayes.naive_bayes import GaussianNaiveBayes

class TestGaussianNaiveBayes(unittest.TestCase):
    def setUp(self):
        """
        设置测试数据。
        """
        # 简单的二分类数据集
        self.X_train = np.array([
            [1.0, 2.1],
            [1.5, 1.6],
            [2.0, 1.1],
            [3.0, 3.1],
            [3.5, 2.9],
            [4.0, 3.0],
            [5.0, 5.1],
            [5.5, 4.8],
            [6.0, 5.0],
            [6.5, 5.2]
        ])
        self.y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

        # 测试数据
        self.X_test = np.array([
            [1.2, 1.9],
            [3.2, 3.0],
            [5.1, 5.0],
            [2.5, 1.5]
        ])
        self.y_test = np.array([0, 1, 1, 0])

    def test_fit(self):
        """
        测试模型的拟合功能。
        """
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        
        # 检查类别
        self.assertListEqual(list(model.classes), [0, 1])
        
        # 检查先验概率
        self.assertAlmostEqual(model.priors[0], 3/10)
        self.assertAlmostEqual(model.priors[1], 7/10)
        
        # 检查均值和方差的形状
        self.assertEqual(model.mean[0].shape[0], self.X_train.shape[1])
        self.assertEqual(model.var[1].shape[0], self.X_train.shape[1])

    def test_predict(self):
        """
        测试模型的预测功能。
        """
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test)

    def test_predict_proba(self):
        """
        测试模型的预测概率计算功能。
        """
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_test)
        # 检查概率矩阵的形状
        self.assertEqual(proba.shape, (self.X_test.shape[0], len(model.classes)))
        # 检查概率之和为1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(self.X_test.shape[0]))

    def test_single_sample(self):
        """
        测试单个样本的预测。
        """
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        x_single = np.array([[2.5, 1.5]])
        y_pred = model.predict(x_single)
        self.assertEqual(y_pred[0], 0)

    def test_mismatched_feature_dimensions(self):
        """
        测试输入特征维度不匹配时是否抛出错误。
        """
        model = GaussianNaiveBayes()
        model.fit(self.X_train, self.y_train)
        x_invalid = np.array([[1.0]])  # 错误的特征维度
        with self.assertRaises(ValueError):
            model.predict(x_invalid)

    def test_empty_training_data(self):
        """
        测试在空训练数据上拟合是否抛出错误。
        """
        model = GaussianNaiveBayes()
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])
        with self.assertRaises(ValueError):
            model.fit(X_empty, y_empty)

if __name__ == '__main__':
    unittest.main()
