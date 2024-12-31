# utils.py
import numpy as np

def softmax(x, axis=-1):
    """
    计算 softmax 函数。
    
    参数:
    - x: 输入的 Numpy 数组
    - axis: 计算 softmax 的轴
    
    返回:
    - softmax 结果，形状与输入相同
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def relu(x):
    """
    计算 ReLU 激活函数。
    
    参数:
    - x: 输入的 Numpy 数组
    
    返回:
    - ReLU 结果，形状与输入相同
    """
    return np.maximum(0, x)
