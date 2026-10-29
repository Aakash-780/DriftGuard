"""Deterministic, identity-based train/holdout partitioning of buffered rows.

Rows are assigned by a hash of their *identity* (row id), not by position or by random draw.
The same underlying observation therefore always lands on the same side, even if it is
re-sampled into several batches. This prevents the same observation from appearing in both the
candidate's training data and its validation holdout (data leakage).
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


def _bucket(row_id: str, salt: int) -> float:
    digest = hashlib.blake2b(f"{salt}:{row_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def holdout_mask(row_ids: Iterable, salt: int, fraction: float) -> np.ndarray:
    """True where the row belongs to the validation holdout."""
    return np.array([_bucket(str(r), salt) < fraction for r in row_ids], dtype=bool)


def content_row_ids(X: pd.DataFrame) -> np.ndarray:
    """Row identities for batches that arrive without ids: a hash of each row's values."""
    arr = np.ascontiguousarray(X.to_numpy(dtype="float64"))
    return np.array(
        [hashlib.blake2b(arr[i].tobytes(), digest_size=8).hexdigest() for i in range(len(arr))],
        dtype=object,
    )
