# test_decoder.py
import numpy as np
from algorithms.Transformer.encoder import Encoder
from algorithms.Transformer.decoder import Decoder

def test_decoder():
    # 参数设置
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 8
    num_layers = 2
    d_model = 16
    num_heads = 4
    d_ff = 64
    vocab_size = 1000
    max_len = 50

    # 初始化 Encoder 和 Decoder
    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=vocab_size, max_len=max_len)
    decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=vocab_size, max_len=max_len)

    # 随机输入词索引
    src = np.random.randint(0, vocab_size, (batch_size, src_seq_length))
    tgt = np.random.randint(0, vocab_size, (batch_size, tgt_seq_length))

    # 前向传播通过编码器
    enc_output = encoder.forward(src, mask=None)

    # 前向传播通过解码器
    dec_output = decoder.forward(tgt, enc_output, src_mask=None, tgt_mask=None)

    # 输出形状验证
    assert dec_output.shape == (batch_size, tgt_seq_length, d_model), f"Expected output shape {(batch_size, tgt_seq_length, d_model)}, but got {dec_output.shape}"

    print("Decoder 前向传播测试通过！")
    print("输出形状:", dec_output.shape)
    print("输出值:", dec_output)

if __name__ == "__main__":
    test_decoder()
