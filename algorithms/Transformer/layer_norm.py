# layer_norm.py
import numpy as np

class LayerNorm:
    def __init__(self, d_model, epsilon=1e-6):
        """
        初始化层归一化。
        
        参数:
        - d_model: 模型的维度
        - epsilon: 防止除零的小常数
        """
        self.d_model = d_model
        self.epsilon = epsilon
        # 初始化可训练参数 gamma 和 beta
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        """
        前向传播计算层归一化的输出。
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - 输出张量，形状与输入相同
        """
        self.mean = np.mean(x, axis=-1, keepdims=True)  # (batch_size, seq_length, 1)
        self.var = np.var(x, axis=-1, keepdims=True)    # (batch_size, seq_length, 1)
        self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)  # (batch_size, seq_length, d_model)
        return self.gamma * self.x_normalized + self.beta
    
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
