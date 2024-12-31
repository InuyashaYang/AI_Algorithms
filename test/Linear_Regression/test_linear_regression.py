# test/Linear_Regression/test_linear_regression.py
import unittest
import numpy as np
from algorithms.Linear_Regression.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def test_fit_and_predict(self):
        # 创建一些简单的训练数据
        X_train = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
        y_train = np.array([3, 6, 9, 12])

        # 创建并训练模型
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X_train, y_train)

        # 创建一些测试数据
        X_test = np.array([[5, 10], [6, 12]])
        y_test = np.array([15, 18])

        # 进行预测
        y_pred = model.predict(X_test)

        # 断言预测值与真实值的差的绝对值小于一个小的阈值
        self.assertTrue(np.allclose(y_pred, y_test, atol=0.1))

    def test_cost_decreasing(self):
        # 创建一些简单的训练数据
        X_train = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
        y_train = np.array([3, 6, 9, 12])

        # 创建并训练模型
        model = LinearRegression(learning_rate=0.01, n_iterations=100)
        model.fit(X_train, y_train)

        # 检查损失是否在递减
        for i in range(1, len(model.costs)):
            self.assertLess(model.costs[i], model.costs[i - 1])

if __name__ == '__main__':
    unittest.main()
