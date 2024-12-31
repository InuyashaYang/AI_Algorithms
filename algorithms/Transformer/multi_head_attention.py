# multi_head_attention.py
import numpy as np
from .utils import softmax

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力机制。
        
        参数:
        - d_model: 模型的维度
        - num_heads: 注意力头的数量
        """
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.depth = d_model // num_heads

        # 初始化权重矩阵，使用 Xavier 初始化
        limit = np.sqrt(6 / (d_model + d_model))
        self.Wq = np.random.uniform(-limit, limit, (d_model, d_model))
        self.Wk = np.random.uniform(-limit, limit, (d_model, d_model))
        self.Wv = np.random.uniform(-limit, limit, (d_model, d_model))
        self.Wo = np.random.uniform(-limit, limit, (d_model, d_model))

    def split_heads(self, x):
        """
        将输入分割成多个头。
        
        参数:
        - x: 形状为 (batch_size, seq_length, d_model) 的输入张量
        
        返回:
        - 形状为 (batch_size, num_heads, seq_length, depth) 的张量
        """
        batch_size, seq_length, d_model = x.shape
        # 重新调整形状以分割头
        x = x.reshape(batch_size, seq_length, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, depth)

    def forward(self, query, key, value, mask=None):
        """
        前向传播计算多头注意力。
        
        参数:
        - query: 查询张量，形状为 (batch_size, seq_length, d_model)
        - key: 键张量，形状为 (batch_size, seq_length, d_model)
        - value: 值张量，形状为 (batch_size, seq_length, d_model)
        - mask: 掩码张量，形状为 (batch_size, 1, 1, seq_length) 或适当形状
        
        返回:
        - 注意力输出，形状为 (batch_size, seq_length, d_model)
        """
        # 线性变换
        Q = query @ self.Wq  # (batch_size, seq_length, d_model)
        K = key @ self.Wk
        V = value @ self.Wv

        # 分头
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_length, depth)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled Dot-Product Attention
        matmul_qk = np.matmul(Q, K.transpose(0, 1, 3, 2))  # (batch_size, num_heads, seq_length, seq_length)
        dk = self.depth
        scaled_attention_logits = matmul_qk / np.sqrt(dk)  # 缩放

        if mask is not None:
            # 通过将掩码位置设置为 -1e9 来遮蔽
            scaled_attention_logits += (mask * -1e9)

        # 计算注意力权重
        attention_weights = softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # 应用注意力权重到 V
        output = np.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, depth)

        # 合并头
        output = output.transpose(0, 2, 1, 3).reshape(query.shape[0], -1, self.d_model)  # (batch_size, seq_length, d_model)

        # 最后一层线性变换
        output = output @ self.Wo  # (batch_size, seq_length, d_model)

        return output

    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, seq_length, d_model)
        
        返回:
        - grad_query, grad_key, grad_value: 查询、键、值的梯度
        """
        # 实现反向传播逻辑
        pass
