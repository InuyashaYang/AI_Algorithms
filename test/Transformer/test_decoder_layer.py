# test_decoder_layer.py
import numpy as np
from algorithms.Transformer.decoder_layer import DecoderLayer

def test_decoder_layer():
    # 参数设置
    batch_size = 2
    src_seq_length = 5
    tgt_seq_length = 4
    d_model = 8
    num_heads = 2
    d_ff = 16

    # 初始化 DecoderLayer
    decoder_layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    # 随机输入
    x = np.random.rand(batch_size, tgt_seq_length, d_model)
    enc_output = np.random.rand(batch_size, src_seq_length, d_model)
    src_mask = None  # 可根据需要定义源端掩码
    tgt_mask = None  # 可根据需要定义目标端掩码

    # 前向传播
    output = decoder_layer.forward(x, enc_output, src_mask, tgt_mask)

    # 输出形状验证
    assert output.shape == (batch_size, tgt_seq_length, d_model), f"Expected output shape {(batch_size, tgt_seq_length, d_model)}, but got {output.shape}"

    print("DecoderLayer 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_decoder_layer()
