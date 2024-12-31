# feed_forward.py
import numpy as np
from .utils import relu

class FeedForward:
    def __init__(self, d_model, d_ff):
        """
        初始化前馈全连接网络。
        
        参数:
        - d_model: 模型的维度
        - d_ff: 前馈网络的隐藏层维度
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 使用 Xavier 初始化权重
        limit1 = np.sqrt(6 / (d_model + d_ff))
        limit2 = np.sqrt(6 / (d_ff + d_model))
        
        self.W1 = np.random.uniform(-limit1, limit1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.uniform(-limit2, limit2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """
        前向传播计算前馈网络的输出。
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - 输出张量，形状为 (batch_size, seq_length, d_model)
        """
        self.input = x  # 保存输入以备反向传播
        self.z1 = x @ self.W1 + self.b1  # (batch_size, seq_length, d_ff)
        self.a1 = relu(self.z1)          # (batch_size, seq_length, d_ff)
        self.z2 = self.a1 @ self.W2 + self.b2  # (batch_size, seq_length, d_model)
        return self.z2
    
    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - grad_input: 输入的梯度，形状为 (batch_size, seq_length, d_model)
        """
        # 实现反向传播逻辑
        pass
