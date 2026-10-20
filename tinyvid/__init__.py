"""TinyVid: a tiny, surprisingly good video codec."""

from .codec import encode_video, decode_video

__version__ = "0.1.0"
__all__ = ["encode_video", "decode_video"]