"""Command-line interface for TinyVid."""

import argparse
import os
import sys

import cv2
import numpy as np

from .codec import encode_video, decode_video


def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_video(path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def cmd_encode(args):
    frames, fps = read_video(args.input)
    print(f"read {len(frames)} frames @ {fps:.2f} fps")
    data = encode_video(frames, qp=args.qp, gop=args.gop, fps=fps,
                        progress=lambda i, n: print(f"\rencoding {i}/{n}", end=""))
    print()
    with open(args.output, "wb") as f:
        f.write(data)
    print(f"wrote {args.output} ({len(data)} bytes) "
          f"ratio {sum(f.size for f in frames)*3/len(data):.1f}x")


def cmd_decode(args):
    with open(args.input, "rb") as f:
        data = f.read()
    frames, (w, h, fps) = decode_video(data)
    write_video(args.output, frames, fps)
    print(f"decoded {len(frames)} frames -> {args.output}")


def cmd_info(args):
    with open(args.input, "rb") as f:
        data = f.read()
    import struct
    magic = data[:4]
    ver, w, h, fps, n, gop, qp = struct.unpack("<BHHfIHB", data[4:4+16])
    print(f"magic={magic} version={ver} {w}x{h} fps={fps:.2f} frames={n} gop={gop} qp={qp} size={len(data)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tinyvid")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("input"); e.add_argument("output"); e.add_argument("--qp", type=int, default=20); e.add_argument("--gop", type=int, default=30)
    e.set_defaults(func=cmd_encode)
    d = sub.add_parser("decode")
    d.add_argument("input"); d.add_argument("output")
    d.set_defaults(func=cmd_decode)
    i = sub.add_parser("info")
    i.add_argument("input")
    i.set_defaults(func=cmd_info)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()