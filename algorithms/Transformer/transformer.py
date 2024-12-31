# transformer.py
import numpy as np
from .encoder import Encoder
from .decoder import Decoder
from .embeddings import Embeddings
from .utils import softmax

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, src_vocab_size, tgt_vocab_size, max_len=5000):
        """
        初始化完整的 Transformer 模型。
        
        参数:
        - num_layers: 编码器和解码器层的数量
        - d_model: 模型的维度
        - num_heads: 注意力头的数量
        - d_ff: 前馈网络的隐藏层维度
        - src_vocab_size: 源语言词汇表大小
        - tgt_vocab_size: 目标语言词汇表大小
        - max_len: 序列的最大长度
        """
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=src_vocab_size, max_len=max_len)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=tgt_vocab_size, max_len=max_len)
        # 输出层
        self.fc_out = np.random.randn(d_model, tgt_vocab_size) / np.sqrt(d_model)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播计算 Transformer 的输出。
        
        参数:
        - src: 源语言输入词索引，形状为 (batch_size, src_seq_length)
        - tgt: 目标语言输入词索引，形状为 (batch_size, tgt_seq_length)
        - src_mask: 源端掩码，形状为 (batch_size, 1, 1, src_seq_length)
        - tgt_mask: 目标端掩码，形状为 (batch_size, 1, tgt_seq_length, tgt_seq_length)
        
        返回:
        - 输出概率分布，形状为 (batch_size, tgt_seq_length, tgt_vocab_size)
        """
        enc_output = self.encoder.forward(src, src_mask)  # (batch_size, src_seq_length, d_model)
        dec_output = self.decoder.forward(tgt, enc_output, src_mask, tgt_mask)  # (batch_size, tgt_seq_length, d_model)
        logits = dec_output @ self.fc_out  # (batch_size, tgt_seq_length, tgt_vocab_size)
        probs = softmax(logits, axis=-1)   # (batch_size, tgt_seq_length, tgt_vocab_size)
        return probs
    
    # 预留反向传播接口
    def backward(self, grad_output):
        """
        反向传播计算（待实现）。
        
        参数:
        - grad_output: 输出的梯度，形状为 (batch_size, tgt_seq_length, tgt_vocab_size)
        
        返回:
        - grad_input: 输入的梯度，形状为 (batch_size, src_seq_length)
        """
        # 实现反向传播逻辑
        pass
