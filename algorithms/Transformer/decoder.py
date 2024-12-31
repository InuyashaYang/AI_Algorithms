# decoder.py
import numpy as np
from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding
from .layer_norm import LayerNorm
from .embeddings import Embeddings

class Decoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=5000):
        """
        初始化解码器。
        
        参数:
        - num_layers: 解码器层的数量
        - d_model: 模型的维度
        - num_heads: 注意力头的数量
        - d_ff: 前馈网络的隐藏层维度
        - vocab_size: 词汇表大小
        - max_len: 序列的最大长度
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embeddings(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播通过解码器。
        
        参数:
        - x: 输入词索引，形状为 (batch_size, tgt_seq_length)
        - enc_output: 编码器的输出，形状为 (batch_size, src_seq_length, d_model)
        - src_mask: 源端掩码，形状为 (batch_size, 1, 1, src_seq_length)
        - tgt_mask: 目标端掩码，形状为 (batch_size, 1, tgt_seq_length, tgt_seq_length)
        
        返回:
        - 解码器输出，形状为 (batch_size, tgt_seq_length, d_model)
        """
        x = self.embedding.forward(x)  # (batch_size, tgt_seq_length, d_model)
        x = self.pos_encoding.forward(x)  # 添加位置编码
        for layer in self.layers:
            x = layer.forward(x, enc_output, src_mask, tgt_mask)
        return self.norm.forward(x)
    
    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, tgt_seq_length, d_model)
        
        返回:
        - grad_input: 输入的梯度，形状为 (batch_size, tgt_seq_length)
        """
        # 实现反向传播逻辑
        pass
