# test_encoder.py
import numpy as np
from algorithms.Transformer.encoder import Encoder
from algorithms.Transformer.decoder import Decoder

def test_encoder():
    # 参数设置
    batch_size = 2
    seq_length = 10
    num_layers = 2
    d_model = 16
    num_heads = 4
    d_ff = 64
    vocab_size = 1000
    max_len = 50

    # 初始化 Encoder
    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=vocab_size, max_len=max_len)

    # 随机输入词索引
    x = np.random.randint(0, vocab_size, (batch_size, seq_length))
    mask = None  # 可根据需要定义掩码

    # 前向传播
    output = encoder.forward(x, mask)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    print("Encoder 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_encoder()
