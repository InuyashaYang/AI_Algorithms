import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DecisionTree:
    """
    决策树分类器实现
    支持 'gini' 和 'entropy' 两种分裂标准
    """

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_index = feature_index  # 分裂特征索引
            self.threshold = threshold          # 分裂阈值
            self.left = left                    # 左子节点
            self.right = right                  # 右子节点
            self.value = value                  # 叶节点的类别

    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        """
        初始化决策树

        参数:
        - max_depth: 树的最大深度
        - min_samples_split: 分裂节点所需的最小样本数
        - criterion: 分裂标准，支持 'gini' 和 'entropy'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if criterion not in ['gini', 'entropy']:
            raise ValueError("criterion 必须是 'gini' 或 'entropy'")
        self.criterion = criterion
        self.root = None
        self.le = None  # LabelEncoder 对象

    def fit(self, X, y):
        """
        训练决策树

        参数:
        - X: 特征矩阵，形状为 (n_samples, n_features)
        - y: 标签向量，形状为 (n_samples,)
        """
        # 转换为 NumPy 数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 检查是否为空
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise IndexError("Empty training data!")

        # 标签编码
        self.le = LabelEncoder()
        y = self.le.fit_transform(y)
        y = y.astype(int)

        # 构建决策树
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) \
           or num_labels == 1 \
           or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # 选择最佳分裂
        best_feat, best_thresh, best_gain = self._best_split(X, y, num_features)
        if best_gain is None or best_gain <= 0:
            # 无法进一步分裂
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # 分裂数据集
        left_indices = X[:, best_feat] <= best_thresh
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # 递归构建子树
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return self.Node(feature_index=best_feat, threshold=best_thresh,
                         left=left_child, right=right_child)

    def _best_split(self, X, y, num_features):
        best_gain = -float('inf')
        split_idx, split_threshold = None, None

        for feature_index in range(num_features):
            X_col = X[:, feature_index]
            unique_vals = np.unique(X_col)
            
            # 对每个特征值都尝试作为分割点
            for val in unique_vals:
                left_indices = X_col <= val
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                    
                gain = self._information_gain(y, left_indices, right_indices)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = val

        return split_idx, split_threshold, best_gain


    def _information_gain(self, y, left_indices, right_indices):
        """
        计算信息增益
        """
        if self.criterion == 'gini':
            parent_imp = self._gini(y)
            left_imp = self._gini(y[left_indices])
            right_imp = self._gini(y[right_indices])
        else:
            parent_imp = self._entropy(y)
            left_imp = self._entropy(y[left_indices])
            right_imp = self._entropy(y[right_indices])

        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        child_imp = (n_left / n) * left_imp + (n_right / n) * right_imp
        return parent_imp - child_imp

    def _gini(self, y):
        """
        计算基尼不纯度
        """
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _entropy(self, y):
        """
        计算熵
        """
        counts = np.bincount(y)
        probs = counts / len(y)
        ent = 0
        for p in probs:
            if p > 0:
                ent -= p * np.log2(p)
        return ent

    def _most_common_label(self, y):
        """
        返回最常见的标签
        """
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        """
        预测输入样本的类别

        参数:
        - X: 特征矩阵，形状为 (n_samples, n_features)

        返回:
        - y_pred: 预测的标签向量，形状为 (n_samples,)
        """
        # 转换为 NumPy 数组
        if isinstance(X, pd.DataFrame):
            X = X.values

        preds = np.array([self._traverse_tree(x, self.root) for x in X])
        return self.le.inverse_transform(preds)

    def _traverse_tree(self, x, node, path=None):
        """
        遍历决策树以进行预测

        参数:
        - x: 单个样本的特征向量
        - node: 当前节点
        - path: 决策路径（可选，用于调试）

        返回:
        - 预测的类别
        """
        if path is None:
            path = []
        
        if node.value is not None:
            path.append(node.value)
            # 调试信息，可以根据需要启用
            # print(f"Sample {x} reached leaf node with value: {self.le.inverse_transform([node.value])[0]}")
            return node.value

        if x[node.feature_index] <= node.threshold:
            path.append(f"Feature {node.feature_index} <= {node.threshold}")
            return self._traverse_tree(x, node.left, path)
        else:
            path.append(f"Feature {node.feature_index} > {node.threshold}")
            return self._traverse_tree(x, node.right, path)

    def print_tree(self, node=None, depth=0):
        """
        打印决策树结构

        参数:
        - node: 当前节点（默认从根开始）
        - depth: 当前深度（用于格式化）
        """
        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{'|   ' * depth}叶节点: 类别 = {self.le.inverse_transform([node.value])[0]}")
            return

        print(f"{'|   ' * depth}分裂: 特征 {node.feature_index} <= {node.threshold:.4f}")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)
