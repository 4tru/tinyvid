# TinyVid

A **tiny, surprisingly good** video codec written in pure Python.

TinyVid is a compact experimental video codec that combines the classic
building blocks of modern video compression into a small, readable codebase:

- **YCbCr 4:2:0** color subsampling
- **8×8 DCT** (via scipy) with zigzag-friendly block layout
- **Quantization** with JPEG-style matrices scaled by a single QP
- **Block motion compensation** (16×16, integer-pel search) for P-frames
- **Skip blocks** for static regions
- **zlib** entropy coding (pragmatic, robust, small)
- A simple **TVID container** format

It is small enough to read in an afternoon and good enough to beat naive
raw storage by ~100–600× on typical content.

## Install

```bash
pip install -e .
```

Requires Python 3.9+, numpy, opencv-python, scipy.

## Usage

```bash
# Encode
tinyvid encode input.mp4 output.tvid --qp 20 --gop 30

# Decode
tinyvid decode output.tvid decoded.mp4

# Inspect
tinyvid info output.tvid
```

Or from Python:

```python
from tinyvid.codec import encode_video, decode_video
import cv2

cap = cv2.VideoCapture("input.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()

data = encode_video(frames, qp=20, gop=30, fps=30.0)
open("out.tvid", "wb").write(data)

dec, (w, h, fps) = decode_video(data)
```

## Tuning

- `--qp` — quantization parameter (0..51). Lower = better quality, bigger file. Default 20.
- `--gop` — I-frame interval. Default 30.

## Limitations

- Integer-pel motion only (no half-pel) — a known shortcut; upgrade to sub-pixel for +1–2 dB.
- No B-frames, no deblocking, no CABAC — entropy is zlib.
- Pure Python; encoding is slow. Fine for small clips and research.

## Tests

```bash
python tests/test_codec.py
```

## License

MIT
