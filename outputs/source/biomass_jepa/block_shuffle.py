"""Deterministic, pixel-conserving permutations of equal blocks within ONE field tile.

Apply to a square PIL tile before any resize/normalization. Do NOT mix images.
"""
import hashlib
import random
from PIL import Image


def permutation_seed(image_sha256: str, grid: int, variant: int, view: int) -> int:
    payload = f'blockshuffle-v1|{image_sha256}|{grid}|{variant}|{view}'
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'little')


def shuffle_square_tile(tile: Image.Image, grid: int, seed: int) -> Image.Image:
    width, height = tile.size
    if width != height or grid not in (4, 10) or width % grid:
        raise ValueError(f'Expected square tile divisible by grid=4/10, got {tile.size}, grid {grid}')
    block = width // grid
    order = list(range(grid*grid))
    random.Random(seed).shuffle(order)
    if order == list(range(grid*grid)):
        order = order[1:] + order[:1]
    out = Image.new(tile.mode, tile.size)
    for dest, src in enumerate(order):
        sx, sy = (src % grid)*block, (src//grid)*block
        dx, dy = (dest % grid)*block, (dest//grid)*block
        out.paste(tile.crop((sx,sy,sx+block,sy+block)), (dx,dy))
    return out
