# test/Logistic_Regression/test_logistic_regression.py
import unittest
import numpy as np
from algorithms.Logistic_Regression.logistic_regression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def test_fit_and_predict(self):
        """
        测试逻辑回归的拟合和预测功能。
        """
        # 创建一个简单的二分类数据集
        X_train = np.array([
            [0.5, 1.5],
            [1.0, 1.8],
            [1.5, 2.0],
            [3.0, 3.5],
            [3.5, 3.8],
            [4.0, 4.0]
        ])
        y_train = np.array([0, 0, 0, 1, 1, 1])

        X_test = np.array([
            [1.2, 1.9],
            [3.2, 3.6]
        ])
        y_test = np.array([0, 1])

        # 创建并训练模型
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X_train, y_train)

        # 进行预测
        y_pred = model.predict(X_test)

        # 断言预测结果与真实标签相同
        np.testing.assert_array_equal(y_pred, y_test)

    def test_cost_decreasing(self):
        """
        测试损失函数是否在迭代过程中递减。
        """
        # 创建一个简单的二分类数据集
        X_train = np.array([
            [0.5, 1.5],
            [1.0, 1.8],
            [1.5, 2.0],
            [3.0, 3.5],
            [3.5, 3.8],
            [4.0, 4.0]
        ])
        y_train = np.array([0, 0, 0, 1, 1, 1])

        # 创建并训练模型
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X_train, y_train)

        # 检查损失是否在递减
        for i in range(1, len(model.costs)):
            self.assertLessEqual(model.costs[i], model.costs[i - 1], 
                                 msg=f"损失函数在迭代 {i} 时未递减")

if __name__ == '__main__':
    unittest.main()
