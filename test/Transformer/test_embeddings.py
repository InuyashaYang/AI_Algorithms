# test_embeddings.py
import numpy as np
from algorithms.Transformer.embeddings import Embeddings

def test_embeddings():
    # 参数设置
    vocab_size = 100
    d_model = 16
    batch_size = 2
    seq_length = 5

    # 初始化 Embeddings
    embed = Embeddings(vocab_size=vocab_size, d_model=d_model)

    # 随机输入词索引
    x = np.random.randint(0, vocab_size, (batch_size, seq_length))

    # 前向传播
    output = embed.forward(x)

    # 输出形状验证
    assert output.shape == (batch_size, seq_length, d_model), f"Expected output shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    print("Embeddings 前向传播测试通过！")
    print("输出形状:", output.shape)
    print("输出值:", output)

if __name__ == "__main__":
    test_embeddings()
