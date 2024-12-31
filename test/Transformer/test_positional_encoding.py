# test_positional_encoding.py
import numpy as np
from algorithms.Transformer.positional_encoding import PositionalEncoding

def test_positional_encoding():
    # 参数设置
    batch_size = 2
    seq_length = 10
    d_model = 16
    max_len = 20

    # 初始化 PositionalEncoding
    pe = PositionalEncoding(d_model=d_model, max_len=max_len)

    # 随机输入
    x = np.random.rand(batch_size, seq_length, d_model)

    # 前向传播
    output = pe.forward(x)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    print("PositionalEncoding 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_positional_encoding()
