"""RLE-кодек формата submission (CLAUDE.md §5).

Пиксели нумеруются с 1, row-major (построчно слева направо, сверху вниз).
Строка — пары `start length` через пробел по возрастанию `start`; серии не
пересекаются и не соприкасаются; пустая маска → пустая строка.
"""
from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """Бинарная маска (H, W) (bool / 0-1) → RLE-строка. Любое ненулевое значение = 1."""
    flat = np.asarray(mask).reshape(-1) != 0
    if not flat.any():
        return ""
    padded = np.concatenate(([False], flat, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])  # чётные — начала, нечётные — концы (0-based)
    starts = edges[0::2] + 1
    lengths = edges[1::2] - edges[0::2]
    pairs = np.empty(starts.size * 2, dtype=np.int64)
    pairs[0::2] = starts
    pairs[1::2] = lengths
    return " ".join(map(str, pairs.tolist()))


def _parse(rle: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(rle, str):
        raise ValueError(f"RLE должен быть строкой, получено {type(rle).__name__}: {rle!r}")
    tokens = rle.split()
    if len(tokens) % 2:
        raise ValueError(f"нечётное число чисел в RLE ({len(tokens)}): ожидаются пары 'start length'")
    try:
        values = np.array([int(t) for t in tokens], dtype=np.int64)
    except ValueError as e:
        raise ValueError(f"нецелое значение в RLE: {e}") from None
    return values[0::2], values[1::2]


def rle_pixel_count(rle: str) -> int:
    """Число пикселей маски по RLE-строке без декодирования (сумма длин серий)."""
    return int(_parse(rle)[1].sum())


def validate_rle(rle: str, shape: tuple[int, int]) -> None:
    """Кидает ValueError, если RLE нарушает формат ТЗ. Пустая строка валидна."""
    starts, lengths = _parse(rle)
    if starts.size == 0:
        return
    total = int(shape[0]) * int(shape[1])
    if (starts < 1).any():
        raise ValueError(f"start < 1 (нумерация пикселей с 1): {int(starts.min())}")
    if (lengths < 1).any():
        raise ValueError(f"length < 1: {int(lengths.min())}")
    ends = starts + lengths - 1
    if (ends > total).any():
        raise ValueError(f"серия выходит за пределы H*W={total}: конец {int(ends.max())}")
    gaps = starts[1:] - ends[:-1]
    if (starts[1:] <= starts[:-1]).any():
        i = int(np.flatnonzero(starts[1:] <= starts[:-1])[0])
        raise ValueError(f"start не возрастает: {int(starts[i])} → {int(starts[i + 1])}")
    if (gaps <= 0).any():
        i = int(np.flatnonzero(gaps <= 0)[0])
        raise ValueError(f"серии пересекаются: [{int(starts[i])}..{int(ends[i])}] и start={int(starts[i + 1])}")
    if (gaps == 1).any():
        i = int(np.flatnonzero(gaps == 1)[0])
        raise ValueError(f"серии соприкасаются (нужно слить): конец {int(ends[i])} и start={int(starts[i + 1])}")


def rle_decode(rle: str, shape: tuple[int, int], validate: bool = True) -> np.ndarray:
    """RLE-строка → маска uint8 (H, W)."""
    if validate:
        validate_rle(rle, shape)
    starts, lengths = _parse(rle)
    total = int(shape[0]) * int(shape[1])
    # разностное кодирование: +1 в начале серии, −1 сразу после конца → cumsum
    delta = np.zeros(total + 1, dtype=np.int32)
    np.add.at(delta, starts - 1, 1)
    np.add.at(delta, np.minimum(starts - 1 + lengths, total), -1)
    return (np.cumsum(delta[:total]) > 0).astype(np.uint8).reshape(shape)
