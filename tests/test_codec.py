"""Tests for TinyVid codec."""

import numpy as np
import cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyvid.codec import encode_video, decode_video


def make_frames(n=8, w=160, h=120):
    frames = []
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cx = int(w / 2 + (w // 4) * np.cos(i * 0.4))
        cy = int(h / 2 + (h // 4) * np.sin(i * 0.4))
        cv2.circle(img, (cx, cy), 20, (0, 200, 100), -1)
        cv2.rectangle(img, (10 + i * 3, 10), (60 + i * 3, 60), (200, 50, 50), -1)
        frames.append(img)
    return frames


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return 10 * np.log10(255.0 * 255.0 / mse)


def test_roundtrip_shape():
    frames = make_frames()
    data = encode_video(frames, qp=20, gop=4, fps=30.0)
    dec, (w, h, fps) = decode_video(data)
    assert len(dec) == len(frames)
    assert dec[0].shape == frames[0].shape


def test_roundtrip_quality():
    frames = make_frames()
    data = encode_video(frames, qp=15, gop=4, fps=30.0)
    dec, _ = decode_video(data)
    p = np.mean([psnr(a, b) for a, b in zip(frames, dec)])
    assert p > 27.0, f"PSNR too low: {p}"


def test_i_frame_only():
    frames = make_frames(n=3)
    data = encode_video(frames, qp=10, gop=999, fps=30.0)
    dec, _ = decode_video(data)
    p = np.mean([psnr(a, b) for a, b in zip(frames, dec)])
    assert p > 29.0, f"PSNR too low: {p}"


def test_compression():
    frames = make_frames(n=30)
    raw = sum(f.size for f in frames) * 1  # bytes if raw
    data = encode_video(frames, qp=25, gop=15, fps=30.0)
    assert len(data) < raw, "no compression"


if __name__ == "__main__":
    test_roundtrip_shape()
    test_roundtrip_quality()
    test_i_frame_only()
    test_compression()
    print("all tests passed")
