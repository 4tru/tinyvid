"""Block-based motion estimation for TinyVid."""

import numpy as np


def sad(a, b):
    return int(np.sum(np.abs(a.astype(np.int32) - b.astype(np.int32))))


def estimate_mvs(cur, ref, bs=16, rng=12):
    """Return mv array shape (bh, bw, 2) of (dy,dx) and cost array."""
    h, w = cur.shape
    bh, bw = h // bs, w // bs
    mvs = np.zeros((bh, bw, 2), dtype=np.int8)
    costs = np.zeros((bh, bw), dtype=np.int32)
    for bi in range(bh):
        for bj in range(bw):
            y, x = bi * bs, bj * bs
            best = (0, 0)
            bestcost = sad(cur[y:y+bs, x:x+bs], ref[y:y+bs, x:x+bs])
            for dy in range(-rng, rng + 1):
                for dx in range(-rng, rng + 1):
                    ry, rx = y + dy, x + dx
                    if ry < 0 or rx < 0 or ry + bs > h or rx + bs > w:
                        continue
                    c = sad(cur[y:y+bs, x:x+bs], ref[ry:ry+bs, rx:rx+bs])
                    if c < bestcost:
                        bestcost = c
                        best = (dy, dx)
            mvs[bi, bj] = best
            costs[bi, bj] = bestcost
    return mvs, costs


def compensate(ref, mvs, bs=16):
    h, w = ref.shape
    bh, bw = mvs.shape[0], mvs.shape[1]
    out = np.zeros((h, w), dtype=np.float32)
    for bi in range(bh):
        for bj in range(bw):
            y, x = bi * bs, bj * bs
            dy, dx = int(mvs[bi, bj, 0]), int(mvs[bi, bj, 1])
            ry, rx = y + dy, x + dx
            ry = min(max(ry, 0), h - bs)
            rx = min(max(rx, 0), w - bs)
            out[y:y+bs, x:x+bs] = ref[ry:ry+bs, rx:rx+bs]
    return out