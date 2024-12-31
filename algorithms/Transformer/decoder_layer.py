# decoder.py
import numpy as np
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器层。
        
        参数:
        - d_model: 模型的维度
        - num_heads: 注意力头的数量
        - d_ff: 前馈网络的隐藏层维度
        - dropout: Dropout 概率（当前未实现）
        """
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # 自注意力
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # 编码器-解码器注意力
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        # Dropout 可在未来实现
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播通过解码器层。
        
        参数:
        - x: 输入张量，形状为 (batch_size, tgt_seq_length, d_model)
        - enc_output: 编码器的输出张量，形状为 (batch_size, src_seq_length, d_model)
        - src_mask: 源端掩码，形状为 (batch_size, 1, 1, src_seq_length)
        - tgt_mask: 目标端掩码，形状为 (batch_size, 1, tgt_seq_length, tgt_seq_length)
        
        返回:
        - 输出张量，形状为 (batch_size, tgt_seq_length, d_model)
        """
        # 自注意力子层
        attn1 = self.mha1.forward(x, x, x, tgt_mask)  # (batch_size, tgt_seq_length, d_model)
        out1 = self.norm1.forward(x + attn1)          # 残差连接 + 层归一化
        
        # 编码器-解码器注意力子层
        attn2 = self.mha2.forward(out1, enc_output, enc_output, src_mask)  # (batch_size, tgt_seq_length, d_model)
        out2 = self.norm2.forward(out1 + attn2)  # 残差连接 + 层归一化
        
        # 前馈网络子层
        ffn_output = self.ffn.forward(out2)          # (batch_size, tgt_seq_length, d_model)
        out3 = self.norm3.forward(out2 + ffn_output) # 残差连接 + 层归一化
        
        return out3
    
    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, tgt_seq_length, d_model)
        
        返回:
        - grad_input: 输入的梯度，形状为 (batch_size, tgt_seq_length, d_model)
        """
        # 实现反向传播逻辑
        pass
