"""8x8 DCT transform helpers for TinyVid."""

import numpy as np
from scipy.fft import dctn, idctn


def dct2(block):
    return dctn(block.astype(np.float32), type=2, norm="ortho")


def idct2(block):
    return idctn(block, type=2, norm="ortho")


def block_dct(frame, bs=8):
    h, w = frame.shape
    out = np.zeros_like(frame, dtype=np.float32)
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            out[y:y+bs, x:x+bs] = dct2(frame[y:y+bs, x:x+bs])
    # handle remainder by zero-pad? keep aligned; assume divisible
    return out


def block_idct(frame, bs=8):
    h, w = frame.shape
    out = np.zeros_like(frame, dtype=np.float32)
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            out[y:y+bs, x:x+bs] = idct2(frame[y:y+bs, x:x+bs])
    return out