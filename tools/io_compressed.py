"""
tools/io_compressed.py

Save/load torch objects with optional compression.

Why:
- torch.save() itself is not a strong compressor for large dense tensors
- for 5-10M sims you want:
  * torch tensors
  * float16 where possible
  * strong on-disk compression (lzma) or faster compression (gzip)

Usage
-----
from tools.io_compressed import torch_save_compressed, torch_load_compressed

torch_save_compressed(obj, "dataset.pt.xz", codec="lzma", preset=6)
obj = torch_load_compressed("dataset.pt.xz", codec="lzma", map_location="cpu")
"""

from __future__ import annotations

import gzip
import io
import lzma
from pathlib import Path
from typing import Any, Literal, Optional

import torch

Codec = Literal["lzma", "gzip", "none"]


def torch_save_compressed(
    obj: Any,
    path: str | Path,
    *,
    codec: Codec = "lzma",
    preset: int = 6,
) -> None:
    """
    Save a Python object (typically dict of tensors) with torch.save + optional compression.

    codec:
      - "lzma": best compression, slower
      - "gzip": decent compression, faster
      - "none": plain torch.save

    preset:
      - lzma: 0..9 (higher = more compression, slower)
      - gzip: 1..9 (higher = more compression, slower)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if codec == "none":
        torch.save(obj, path)
        return

    buffer = io.BytesIO()
    torch.save(obj, buffer)
    data = buffer.getvalue()

    if codec == "lzma":
        comp = lzma.compress(data, preset=preset)
        path.write_bytes(comp)
    elif codec == "gzip":
        level = preset if 1 <= preset <= 9 else 6
        with gzip.open(path, "wb", compresslevel=level) as f:
            f.write(data)
    else:
        raise ValueError(f"Unknown codec: {codec}")


def torch_load_compressed(
    path: str | Path,
    *,
    codec: Codec = "lzma",
    map_location: Optional[str] = "cpu",
) -> Any:
    """
    Load an object saved with torch_save_compressed().
    """
    path = Path(path)

    if codec == "none":
        return torch.load(path, map_location=map_location)

    raw = path.read_bytes()
    if codec == "lzma":
        data = lzma.decompress(raw)
    elif codec == "gzip":
        data = gzip.decompress(raw)
    else:
        raise ValueError(f"Unknown codec: {codec}")

    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location=map_location)