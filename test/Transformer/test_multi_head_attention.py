# test_multi_head_attention.py
import numpy as np
from algorithms.Transformer.multi_head_attention import MultiHeadAttention

def test_multi_head_attention():
    # 参数设置
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2

    # 初始化 MultiHeadAttention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 随机输入
    query = np.random.rand(batch_size, seq_length, d_model)
    key = np.random.rand(batch_size, seq_length, d_model)
    value = np.random.rand(batch_size, seq_length, d_model)
    mask = None  # 可根据需要定义掩码，例如 np.zeros((batch_size, 1, 1, seq_length))

    # 前向传播
    output = mha.forward(query, key, value, mask)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    print("MultiHeadAttention 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_multi_head_attention()
