# test/KNN/test_knn.py
import unittest
import numpy as np
from algorithms.K_Nearest_Neighbors.k_nearest_neighbors import KNN

class TestKNN(unittest.TestCase):
    def setUp(self):
        """
        设置测试数据。
        """
        # 简单的二分类数据集
        self.X_train = np.array([
            [1, 2],
            [2, 3],
            [3, 3],
            [6, 5],
            [7, 7],
            [8, 6]
        ])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])

        # 测试数据
        self.X_test = np.array([
            [1.5, 2.5],
            [5, 5],
            [7, 6]
        ])
        self.y_test = np.array([0, 1, 1])

    def test_predict_euclidean(self):
        """
        测试使用欧氏距离的预测功能。
        """
        model = KNN(n_neighbors=3, distance_metric='euclidean')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test)

    def test_predict_manhattan(self):
        """
        测试使用曼哈顿距离的预测功能。
        """
        # 调整期望结果以适应曼哈顿距离
        y_test_manhattan = np.array([0, 1, 1])
        model = KNN(n_neighbors=3, distance_metric='manhattan')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, y_test_manhattan)

    def test_different_k(self):
        """
        测试不同k值对预测结果的影响。
        """
        # 使用k=1
        model_k1 = KNN(n_neighbors=1, distance_metric='euclidean')
        model_k1.fit(self.X_train, self.y_train)
        y_pred_k1 = model_k1.predict(self.X_test)
        expected_k1 = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred_k1, expected_k1)

        # 使用k=5
        model_k5 = KNN(n_neighbors=5, distance_metric='euclidean')
        model_k5.fit(self.X_train, self.y_train)
        y_pred_k5 = model_k5.predict(self.X_test)
        expected_k5 = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred_k5, expected_k5)

    def test_invalid_distance_metric(self):
        """
        测试无效的距离度量方法是否抛出错误。
        """
        with self.assertRaises(ValueError):
            KNN(n_neighbors=3, distance_metric='invalid_metric')

if __name__ == '__main__':
    unittest.main()
