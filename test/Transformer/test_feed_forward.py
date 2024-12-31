# test_feed_forward.py
import numpy as np
from algorithms.Transformer.feed_forward import FeedForward

def test_feed_forward():
    # 参数设置
    batch_size = 2
    seq_length = 4
    d_model = 8
    d_ff = 16

    # 初始化 FeedForward
    ff = FeedForward(d_model=d_model, d_ff=d_ff)

    # 随机输入
    x = np.random.rand(batch_size, seq_length, d_model)

    # 前向传播
    output = ff.forward(x)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    print("FeedForward 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_feed_forward()
