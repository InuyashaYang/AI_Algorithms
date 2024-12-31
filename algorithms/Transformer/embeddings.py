# embeddings.py
import numpy as np

class Embeddings:
    def __init__(self, vocab_size, d_model):
        """
        初始化词嵌入层。
        
        参数:
        - vocab_size: 词汇表大小
        - d_model: 模型的维度
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 使用 Xavier 初始化嵌入矩阵
        limit = np.sqrt(6 / (vocab_size + d_model))
        self.embedding = np.random.uniform(-limit, limit, (vocab_size, d_model))
    
    def forward(self, x):
        """
        将词索引转换为嵌入向量。
        
        参数:
        - x: 输入词索引，形状为 (batch_size, seq_length)
        
        返回:
        - 嵌入向量，形状为 (batch_size, seq_length, d_model)
        """
        return self.embedding[x] * np.sqrt(self.d_model)
    
    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - grad_input: 输入的梯度，形状为 (batch_size, seq_length)
        """
        # 实现反向传播逻辑
        pass
