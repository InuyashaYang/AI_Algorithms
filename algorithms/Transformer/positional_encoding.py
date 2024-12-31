# positional_encoding.py
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码。
        
        参数:
        - d_model: 模型的维度
        - max_len: 序列的最大长度
        """
        self.d_model = d_model
        
        # 预先计算位置编码
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe  # (max_len, d_model)
    
    def forward(self, x):
        """
        将位置编码添加到输入张量中。
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - 添加了位置编码的张量，形状与输入相同
        """
        seq_length = x.shape[1]
        if seq_length > self.pe.shape[0]:
            raise ValueError(f"序列长度 {seq_length} 超过了位置编码的最大长度 {self.pe.shape[0]}")
        return x + self.pe[:seq_length]
    
    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - grad_input: 输入的梯度，形状为 (batch_size, seq_length, d_model)
        """
        # 位置编码部分没有可训练参数，所以梯度直接传递
        return grad_output
