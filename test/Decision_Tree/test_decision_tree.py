# test/Decision_Tree/test_decision_tree.py
import unittest
import numpy as np
from algorithms.Decision_Tree.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        """
        设置测试数据。
        """
        # 简单的二分类数据集
        self.X_train = np.array([
            [2.771244718, 1.784783929],
            [1.728571309, 1.169761413],
            [3.678319846, 2.81281357],
            [3.961043357, 2.61995032],
            [2.999208922, 2.209014212],
            [7.497545867, 3.162953546],
            [9.00220326,  3.339047188],
            [7.444542326, 0.476683375],
            [10.12493903, 3.234550982],
            [6.642287351, 3.319983761]
        ])
        self.y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        # 测试数据
        self.X_test = np.array([
            [1.5, 2.0],
            [8.0, 3.0],
            [5.0, 1.5]
        ])
        self.y_test = np.array([0, 1, 1])

    def test_fit_predict_gini(self):
        """
        测试使用基尼不纯度的拟合和预测功能。
        """
        model = DecisionTree(max_depth=3, min_samples_split=2, criterion='gini')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test)

    def test_fit_predict_entropy(self):
        """
        测试使用信息熵的拟合和预测功能。
        """
        model = DecisionTree(max_depth=3, min_samples_split=2, criterion='entropy')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test)

    def test_max_depth(self):
        """
        测试设置不同最大深度对模型的影响。
        """
        # 最大深度为1
        model_depth1 = DecisionTree(max_depth=1, min_samples_split=2, criterion='gini')
        model_depth1.fit(self.X_train, self.y_train)
        y_pred_depth1 = model_depth1.predict(self.X_test)
        # 由于树非常浅，所有预测可能相同
        expected_depth1 = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred_depth1, expected_depth1)

    def test_min_samples_split(self):
        """
        测试设置不同最小样本分裂数对模型的影响。
        """
        # 设置较高的最小样本分裂数
        model_min_split = DecisionTree(max_depth=None, min_samples_split=6, criterion='gini')
        model_min_split.fit(self.X_train, self.y_train)
        y_pred_min_split = model_min_split.predict(self.X_test)
        expected_min_split = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred_min_split, expected_min_split)

    def test_invalid_criterion(self):
        """
        测试无效的分裂标准是否抛出错误。
        """
        with self.assertRaises(ValueError):
            DecisionTree(criterion='invalid_criterion')

    def test_single_feature(self):
        """
        测试只有一个特征时模型的拟合和预测功能。
        """
        X_train_single = self.X_train[:, [0]]  # 只使用第一个特征
        X_test_single = self.X_test[:, [0]]
        model_single = DecisionTree(max_depth=2, min_samples_split=2, criterion='gini')
        model_single.fit(X_train_single, self.y_train)
        y_pred_single = model_single.predict(X_test_single)
        expected_single = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred_single, expected_single)

    def test_empty_training_data(self):
        """
        测试在空训练数据上拟合是否抛出错误。
        """
        model = DecisionTree()
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])
        with self.assertRaises(IndexError):
            model.fit(X_empty, y_empty)

    def test_no_split_possible(self):
        """
        测试当无法进一步分裂时模型的行为。
        """
        X_uniform = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1]
        ])
        y_uniform = np.array([0, 0, 0, 0])
        model = DecisionTree()
        model.fit(X_uniform, y_uniform)
        y_pred = model.predict(X_uniform)
        expected = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(y_pred, expected)

    def test_print_tree(self):
        """
        测试打印决策树结构是否正常运行。
        """
        model = DecisionTree(max_depth=2, min_samples_split=2, criterion='gini')
        model.fit(self.X_train, self.y_train)
        try:
            model.print_tree()
        except Exception as e:
            self.fail(f"print_tree 方法抛出异常: {e}")

if __name__ == '__main__':
    unittest.main()
