"""测试用的假 jev:把字符哈希成固定维度向量,只用来验证 JevEncoder 的接入方式。"""
import numpy as np


def encode(texts):
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for a, b in zip(t, t[1:]):
            out[i, hash(a + b) % 256] += 1.0
    return out + 1e-6
