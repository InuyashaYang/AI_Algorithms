# test_transformer.py
import numpy as np
from algorithms.Transformer.transformer import Transformer

def test_transformer():
    # 参数设置
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 8
    num_layers = 2
    d_model = 16
    num_heads = 4
    d_ff = 64
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    max_len = 50

    # 初始化 Transformer
    transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                              src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, max_len=max_len)

    # 随机输入词索引
    src = np.random.randint(0, src_vocab_size, (batch_size, src_seq_length))
    tgt = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

    # 前向传播
    output = transformer.forward(src, tgt, src_mask=None, tgt_mask=None)

    # 输出形状验证
    assert output.shape == (batch_size, tgt_seq_length, tgt_vocab_size), f"Expected output shape {(batch_size, tgt_seq_length, tgt_vocab_size)}, but got {output.shape}"

    print("Transformer 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_transformer()
