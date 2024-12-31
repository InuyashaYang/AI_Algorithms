# test_encoder_layer.py
import numpy as np
from algorithms.Transformer.encoder_layer import EncoderLayer

def test_encoder_layer():
    # 参数设置
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2
    d_ff = 16

    # 初始化 EncoderLayer
    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # 随机输入
    x = np.random.rand(batch_size, seq_length, d_model)
    mask = None  # 可根据需要定义掩码

    # 前向传播
    output = encoder_layer.forward(x, mask)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    print("EncoderLayer 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_encoder_layer()
