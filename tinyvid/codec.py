"""TinyVid codec core."""

import io
import pickle
import struct
import zlib

import cv2
import numpy as np

from .transform import block_dct, block_idct
from .motion import estimate_mvs, compensate

MAGIC = b"TVID"
VERSION = 1

# JPEG-like base quantization matrices
QMAT_LUMA = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,28,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.float32)

QMAT_CHROMA = (QMAT_LUMA * 1.2).astype(np.float32)


def qmatrix(base, qp):
    scale = 1.0 + qp / 8.0
    return np.clip(base * scale, 1, 255)


def to_ycbcr(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    return y, cb, cr


def from_ycbcr(y, cb, cr):
    y = np.clip(y, 0, 255)
    cb = np.clip(cb, 0, 255)
    cr = np.clip(cr, 0, 255)
    ycrcb = np.stack([y, cr, cb], axis=2).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def downsample(arr):
    return cv2.resize(arr, (arr.shape[1]//2, arr.shape[0]//2), interpolation=cv2.INTER_AREA)


def upsample(arr, h, w):
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def pad_frame(arr, multiple=16):
    h, w = arr.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return arr, (0, 0)
    out = np.pad(arr, ((0, ph), (0, pw)), mode="edge")
    return out, (ph, pw)


def quantize(coeffs, qmat):
    h, w = coeffs.shape
    qm = np.tile(qmat, (h//8, w//8))
    return np.round(coeffs / qm).astype(np.int16)


def dequantize(qcoeffs, qmat):
    h, w = qcoeffs.shape
    qm = np.tile(qmat, (h//8, w//8))
    return qcoeffs.astype(np.float32) * qm


def encode_intra(y, cb, cr, qp):
    qmY = qmatrix(QMAT_LUMA, qp)
    qmC = qmatrix(QMAT_CHROMA, qp)
    cY = block_dct(y)
    cCb = block_dct(cb)
    cCr = block_dct(cr)
    qY = quantize(cY, qmY)
    qCb = quantize(cCb, qmC)
    qCr = quantize(cCr, qmC)
    return {"type": "I", "Y": qY, "Cb": qCb, "Cr": qCr}


def decode_intra(payload, qp, h, w):
    qmY = qmatrix(QMAT_LUMA, qp)
    qmC = qmatrix(QMAT_CHROMA, qp)
    cY = dequantize(payload["Y"], qmY)
    cCb = dequantize(payload["Cb"], qmC)
    cCr = dequantize(payload["Cr"], qmC)
    y = block_idct(cY)
    cb = block_idct(cCb)
    cr = block_idct(cCr)
    # crop to padded dims (already passed)
    y = y[:ph, :pw] if False else y
    y = y[: y.shape[0], : y.shape[1]]
    cb = cb[: cb.shape[0], : cb.shape[1]]
    cr = cr[: cr.shape[0], : cr.shape[1]]
    return y, cb, cr


def encode_inter(y, cb, cr, prev_y, prev_cb, prev_cr, qp, bs=16, rng=12, skip_thresh=128):
    qmY = qmatrix(QMAT_LUMA, qp)
    qmC = qmatrix(QMAT_CHROMA, qp)
    mvs, costs = estimate_mvs(y, prev_y, bs=bs, rng=rng)
    bh, bw = mvs.shape[0], mvs.shape[1]
    skip = costs < skip_thresh
    # motion compensate
    pred_y = compensate(prev_y, mvs, bs=bs)
    # residual
    res_y = y - pred_y
    # chroma: use scaled mvs (nearest)
    mvs_c = mvs.copy()
    # downsample chroma dims
    ch, cw = cb.shape
    cb_pred = compensate(prev_cb, _scale_mvs(mvs, y.shape, cb.shape, bs), bs=bs//2 if bs//2 >= 8 else 8)
    # simpler: use full bs for chroma with scaled mv
    res_cb = cb - cb_pred
    cr_pred = compensate(prev_cr, _scale_mvs(mvs, y.shape, cr.shape, bs), bs=bs//2 if bs//2 >= 8 else 8)
    res_cr = cr - cr_pred

    # zero out skip blocks in residual
    for bi in range(bh):
        for bj in range(bw):
            if skip[bi, bj]:
                y0, x0 = bi*bs, bj*bs
                res_y[y0:y0+bs, x0:x0+bs] = 0
    # DCT + quant
    cY = block_dct(res_y)
    rc = res_cb
    rc, _ = pad_frame(rc, 8)
    cc = block_dct(rc)
    rr = res_cr
    rr, _ = pad_frame(rr, 8)
    cr_dct = block_dct(rr)

    qY = quantize(cY, qmY)
    qCb = quantize(cc, qmC)
    qCr = quantize(cr_dct, qmC)
    # zero skip residual blocks in qY already via res zero -> dct zero -> quant zero

    return {"type": "P", "mvs": mvs.astype(np.int8), "skip": skip,
            "Y": qY, "Cb": qCb, "Cr": qCr,
            "chroma_shape": (ch, cw)}


def _scale_mvs(mvs, src_shape, dst_shape, bs):
    # scale mv from src (y) block grid to dst grid; mv values scale by ratio
    sy = src_shape[0] / dst_shape[0]
    sx = src_shape[1] / dst_shape[1]
    out = np.round(mvs.astype(np.float32) / np.array([sy, sx])).astype(np.int8)
    # resize grid
    bh, bw = mvs.shape[0], mvs.shape[1]
    dh = dst_shape[0] // (bs // 2 if bs//2 >= 8 else 8)
    dw = dst_shape[1] // (bs // 2 if bs//2 >= 8 else 8)
    # nearest resize of mv grid
    if (dh, dw) == (bh, bw):
        return out
    # simple: repeat/crop
    oi = np.clip(np.round(np.arange(dh) * bh / dh).astype(int), 0, bh-1)
    oj = np.clip(np.round(np.arange(dw) * bw / dw).astype(int), 0, bw-1)
    return out[oi[:, None], oj[None, :], :]


def decode_inter(payload, prev_y, prev_cb, prev_cr, qp, h, w):
    qmY = qmatrix(QMAT_LUMA, qp)
    qmC = qmatrix(QMAT_CHROMA, qp)
    mvs = payload["mvs"]
    skip = payload["skip"]
    bs = 16
    # Y residual
    cY = dequantize(payload["Y"], qmY)
    res_y = block_idct(cY)
    res_y = res_y[:h, :w] if res_y.shape[0] >= h else res_y
    # pad back if needed
    if res_y.shape[0] < h or res_y.shape[1] < w:
        res_y = np.pad(res_y, ((0, h-res_y.shape[0]), (0, w-res_y.shape[1])), mode="edge")
    pred_y = compensate(prev_y, mvs, bs=bs)
    y = pred_y + res_y[:h, :w]

    ch, cw = payload["chroma_shape"]
    # chroma residual
    cCb = dequantize(payload["Cb"], qmC)
    res_cb = block_idct(cCb)[:ch, :cw]
    if res_cb.shape[0] < ch or res_cb.shape[1] < cw:
        res_cb = np.pad(res_cb, ((0, ch-res_cb.shape[0]), (0, cw-res_cb.shape[1])), mode="edge")
    cCr = dequantize(payload["Cr"], qmC)
    res_cr = block_idct(cCr)[:ch, :cw]
    if res_cr.shape[0] < ch or res_cr.shape[1] < cw:
        res_cr = np.pad(res_cr, ((0, ch-res_cr.shape[0]), (0, cw-res_cr.shape[1])), mode="edge")

    mvs_c = _scale_mvs(mvs, (h, w), (ch, cw), bs)
    cb_bs = bs // 2 if bs//2 >= 8 else 8
    pred_cb = compensate(prev_cb, mvs_c, bs=cb_bs)
    cb = pred_cb + res_cb
    pred_cr = compensate(prev_cr, mvs_c, bs=cb_bs)
    cr = pred_cr + res_cr

    # upsample chroma to full for return? Keep 4:2:0; caller upsamples. Return as-is (half res)
    return y, cb, cr


def encode_video(frames_bgr, qp=20, gop=30, fps=30.0, progress=None):
    """frames_bgr: list of np arrays. Returns container bytes."""
    if not frames_bgr:
        raise ValueError("no frames")
    h, w = frames_bgr[0].shape[:2]
    # pad dims
    # We'll keep original h,w in header; pad internally per frame.
    n = len(frames_bgr)
    # Preprocess frames to YCbCr (half chroma)
    ys, cbs, crs = [], [], []
    for f in frames_bgr:
        y, cb, cr = to_ycbcr(f)
        cb = downsample(cb)
        cr = downsample(cr)
        y, _ = pad_frame(y, 16)
        cb, _ = pad_frame(cb, 8)
        cr, _ = pad_frame(cr, 8)
        ys.append(y); cbs.append(cb); crs.append(cr)
    ph, pw = ys[0].shape[0]-h, ys[0].shape[1]-w  # not used

    prev_y = prev_cb = prev_cr = None
    payloads = []
    for i in range(n):
        if i % gop == 0 or prev_y is None:
            pl = encode_intra(ys[i], cbs[i], crs[i], qp)
        else:
            pl = encode_inter(ys[i], cbs[i], crs[i], prev_y, prev_cb, prev_cr, qp)
        payloads.append(pl)
        # reconstruct for reference
        if pl["type"] == "I":
            ry, rcb, rcr = decode_intra(pl, qp, ys[i].shape[0], ys[i].shape[1])
        else:
            ry, rcb, rcr = decode_inter(pl, prev_y, prev_cb, prev_cr, qp, ys[i].shape[0], ys[i].shape[1])
        ry = np.clip(ry, 0, 255)
        rcb = np.clip(rcb, 0, 255)
        rcr = np.clip(rcr, 0, 255)
        prev_y = ry
        prev_cb = rcb[:cbs[i].shape[0], :cbs[i].shape[1]]
        prev_cr = rcr[:crs[i].shape[0], :crs[i].shape[1]]
        if progress:
            progress(i+1, n)

    # Build container
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<BHHfIHB", VERSION, w, h, float(fps), n, gop, qp))
    # fps placeholder 30; store actual? add fps param later. Use 30 default.
    for pl in payloads:
        raw = pickle.dumps(pl, protocol=pickle.HIGHEST_PROTOCOL)
        comp = zlib.compress(raw, level=9)
        buf.write(struct.pack("<BI", ord(pl["type"]), len(comp)))
        buf.write(comp)
    return buf.getvalue()


def decode_video(data):
    buf = io.BytesIO(data)
    magic = buf.read(4)
    if magic != MAGIC:
        raise ValueError("bad magic")
    version, w, h, fps, n, gop, qp = struct.unpack("<BHHfIHB", buf.read(1+2+2+4+4+2+1))
    # Actually struct format size: B(1) H(2) H(2) f(4) I(4) B(1) B(1) = 19
    frames = []
    prev_y = prev_cb = prev_cr = None
    for i in range(n):
        typ_b, size = struct.unpack("<BI", buf.read(5))
        comp = buf.read(size)
        pl = pickle.loads(zlib.decompress(comp))
        # pad dims
        yh = (16 - h % 16) % 16 + h
        xw = (16 - w % 16) % 16 + w
        ch = (8 - (h//2) % 8) % 8 + h//2
        cw = (8 - (w//2) % 8) % 8 + w//2
        if pl["type"] == "I":
            ry, rcb, rcr = decode_intra(pl, qp, yh, xw)
        else:
            ry, rcb, rcr = decode_inter(pl, prev_y, prev_cb, prev_cr, qp, yh, xw)
        ry = ry[:h, :w]
        if pl["type"] == "P":
            rcb_full = upsample(rcb[:h//2, :w//2] if rcb.shape[0] >= h//2 and rcb.shape[1] >= w//2 else rcb, h, w)
            rcr_full = upsample(rcr[:h//2, :w//2] if rcr.shape[0] >= h//2 and rcr.shape[1] >= w//2 else rcr, h, w)
        else:
            rcb_full = upsample(rcb[:h//2, :w//2] if rcb.shape[0] >= h//2 and rcb.shape[1] >= w//2 else rcb, h, w)
            rcr_full = upsample(rcr[:h//2, :w//2] if rcr.shape[0] >= h//2 and rcr.shape[1] >= w//2 else rcr, h, w)
        py, _ = pad_frame(ry, 16)
        pcb = downsample(rcb_full)
        pcb, _ = pad_frame(pcb, 8)
        pcr = downsample(rcr_full)
        pcr, _ = pad_frame(pcr, 8)
        prev_y = py
        prev_cb = pcb
        prev_cr = pcr

        frame = from_ycbcr(ry, rcb_full, rcr_full)
        frames.append(frame)
    return frames, (w, h, fps)