# test_layer_norm.py
import numpy as np
from algorithms.Transformer.layer_norm import LayerNorm

def test_layer_norm():
    # 参数设置
    batch_size = 2
    seq_length = 4
    d_model = 8

    # 初始化 LayerNorm
    ln = LayerNorm(d_model=d_model)

    # 随机输入
    x = np.random.rand(batch_size, seq_length, d_model)

    # 前向传播
    output = ln.forward(x)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    # 验证输出的均值接近 0，方差接近 1
    mean = np.mean(output, axis=-1)
    var = np.var(output, axis=-1)
    assert np.allclose(mean, np.mean(ln.gamma * ln.x_normalized + ln.beta, axis=-1), atol=1e-5), "Mean normalization failed"
    print("LayerNorm 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
        test_layer_norm()
