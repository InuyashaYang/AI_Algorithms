# encoder.py
import numpy as np
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器层。
        
        参数:
        - d_model: 模型的维度
        - num_heads: 注意力头的数量
        - d_ff: 前馈网络的隐藏层维度
        - dropout: Dropout 概率（当前未实现）
        """
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # Dropout 可在未来实现
    
    def forward(self, x, mask=None):
        """
        前向传播通过编码器层。
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, d_model)
        - mask: 掩码张量，形状为 (batch_size, 1, 1, seq_length) 或适当形状
        
        返回:
        - 输出张量，形状为 (batch_size, seq_length, d_model)
        """
        # 自注意力子层
        attn_output = self.mha.forward(x, x, x, mask)  # (batch_size, seq_length, d_model)
        out1 = self.norm1.forward(x + attn_output)     # 残差连接 + 层归一化
        
        # 前馈网络子层
        ffn_output = self.ffn.forward(out1)          # (batch_size, seq_length, d_model)
        out2 = self.norm2.forward(out1 + ffn_output) # 残差连接 + 层归一化
        
        return out2
    
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
