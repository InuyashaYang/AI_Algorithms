# Machine Learning Algorithms From Scratch | 机器学习算法从零实现

A systematic implementation of classical machine learning algorithms using basic Python packages.
使用基础Python包系统性地实现经典机器学习算法。

## Description | 项目描述

This repository provides clear, well-documented implementations of fundamental machine learning algorithms using only basic Python packages (primarily NumPy). Each algorithm is built from the ground up to deepen understanding of the underlying mathematical principles and algorithmic mechanisms.

本仓库使用基础Python包（主要是NumPy）实现基础机器学习算法，提供清晰的文档说明。每个算法都从零开始构建，旨在加深对底层数学原理和算法机制的理解。

## Key Features | 主要特点
- Pure Python/NumPy implementations without relying on high-level ML libraries
  纯Python/NumPy实现，不依赖高级机器学习库
- Detailed mathematical explanations and derivations
  详细的数学解释和推导
- Step-by-step code walkthrough with comprehensive comments
  包含完整注释的逐步代码讲解
- Practical examples and visualizations
  实用的示例和可视化
- Comparative analysis with scikit-learn implementations
  与scikit-learn实现的对比分析

## Algorithms Included | 算法列表
- Linear Regression (OLS, Ridge, Lasso) | 线性回归（普通最小二乘、岭回归、Lasso）
- Logistic Regression | 逻辑回归
- K-Nearest Neighbors (KNN) | K近邻
- Decision Trees | 决策树
- Support Vector Machines (SVM) | 支持向量机
- K-Means Clustering | K均值聚类
- Principal Component Analysis (PCA) | 主成分分析

## Target Audience | 目标受众
- ML beginners seeking deep understanding | 寻求深入理解的机器学习初学者
- Students learning ML fundamentals | 学习机器学习基础的学生
- Developers wanting to implement ML from scratch | 想要从零实现机器学习的开发者
- Anyone interested in ML mathematical foundations | 对机器学习数学基础感兴趣的人群

## Dependencies | 项目依赖
- Python 3.x
- NumPy
- Matplotlib (for visualizations | 用于可视化)
- Pandas (for data handling | 用于数据处理)


## 项目架构
```
Machine_Learning_Algorithms_From_Scratch/
│
├── documents/
│   ├── README.md
│   ├── installation.md
│   ├── usage.md
│   └── algorithms_overview.md
│
├── test/
│   ├── Linear_Regression/
│   │   ├── __init__.py
│   │   └── test_linear_regression.py
│   │
│   ├── Logistic_Regression/
│   │   ├── __init__.py
│   │   └── test_logistic_regression.py
│   │
│   ├── K_Nearest_Neighbors/
│   │   ├── __init__.py
│   │   └── test_k_nearest_neighbors.py
│   │
│   ├── Decision_Tree/
│   │   ├── __init__.py
│   │   └── test_decision_tree.py
│   │
│   ├── Naive_Bayes/
│   │   ├── __init__.py
│   │   └── test_naive_bayes.py
│   │
│   └── Transformer/
│       ├── __init__.py
│       ├── test_decoder.py
│       ├── test_decoder_layer.py
│       ├── test_embeddings.py
│       ├── test_encoder.py
│       ├── test_encoder_layer.py
│       ├── test_feed_forward.py
│       ├── test_layer_norm.py
│       ├── test_multi_head_attention.py
│       ├── test_positional_encoding.py
│       └── test_transformer.py
│
├── algorithms/
│   ├── Linear_Regression/
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   └── utils.py
│   │
│   ├── Logistic_Regression/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   └── utils.py
│   │
│   ├── K_Nearest_Neighbors/
│   │   ├── __init__.py
│   │   ├── k_nearest_neighbors.py
│   │   └── utils.py
│   │
│   ├── Decision_Tree/
│   │   ├── __init__.py
│   │   ├── decision_tree.py
│   │   └── utils.py
│   │
│   ├── Naive_Bayes/
│   │   ├── __init__.py
│   │   ├── naive_bayes.py
│   │   └── utils.py
│   │
│   └── Transformer/
│       ├── __init__.py
│       ├── decoder.py
│       ├── decoder_layer.py
│       ├── embeddings.py
│       ├── encoder.py
│       ├── encoder_layer.py
│       ├── feed_forward.py
│       ├── layer_norm.py
│       ├── multi_head_attention.py
│       ├── positional_encoding.py
│       ├── transformer.py
│       └── utils.py
│
├── README.md
└── requirements.txt
```