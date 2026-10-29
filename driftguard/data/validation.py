"""Schema and value validation for incoming batches."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "n_rows": self.n_rows,
        }


@dataclass(frozen=True)
class FeatureSchema:
    feature_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @classmethod
    def from_reference(cls, X_ref: pd.DataFrame) -> "FeatureSchema":
        return cls(
            feature_names=tuple(X_ref.columns),
            lower=tuple(float(v) for v in X_ref.min().to_numpy()),
            upper=tuple(float(v) for v in X_ref.max().to_numpy()),
        )


def validate_features(
    X: pd.DataFrame,
    schema: FeatureSchema,
    *,
    min_rows: int = 1,
    out_of_range_warn_share: float = 0.20,
) -> ValidationReport:
    """Validate feature columns. Missing/non-finite values are errors (no silent imputation)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(X, pd.DataFrame):
        return ValidationReport(False, ["features must be a pandas DataFrame"], [], 0)
    n = len(X)
    if n < min_rows:
        errors.append(f"batch has {n} rows; at least {min_rows} required")
    expected = list(schema.feature_names)
    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]
    if missing:
        errors.append(f"missing feature columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    if missing or extra:
        return ValidationReport(False, errors, warnings, n)

    for col in expected:
        if not pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
            errors.append(f"column {col!r} is not numeric (dtype {X[col].dtype})")
    if errors:
        return ValidationReport(False, errors, warnings, n)

    arr = X[expected].to_numpy(dtype="float64")
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan:
        errors.append(f"{n_nan} missing (NaN) values")
    if n_inf:
        errors.append(f"{n_inf} infinite values")
    if not errors and n > 0:
        lo = np.array(schema.lower)
        hi = np.array(schema.upper)
        outside = ((arr < lo) | (arr > hi)).mean()
        if outside > out_of_range_warn_share:
            warnings.append(
                f"{outside:.0%} of values fall outside the reference min/max range "
                "(possible drift or a unit/scale error)"
            )
    return ValidationReport(not errors, errors, warnings, n)


def validate_labels(y, n_rows: int) -> list[str]:
    errors: list[str] = []
    if y is None:
        return errors
    y = pd.Series(y)
    if len(y) != n_rows:
        errors.append(f"labels length {len(y)} does not match batch rows {n_rows}")
        return errors
    if y.isna().any():
        errors.append("labels contain missing values")
        return errors
    bad = sorted(set(y.unique()) - {0, 1})
    if bad:
        errors.append(f"labels must be 0/1; found {bad[:5]}")
    return errors


def validate_batch(
    X: pd.DataFrame, y, schema: FeatureSchema, *, min_rows: int = 1
) -> ValidationReport:
    report = validate_features(X, schema, min_rows=min_rows)
    label_errors = validate_labels(y, report.n_rows)
    if label_errors:
        report.errors.extend(label_errors)
        report.valid = False
    return report


def align_columns(X: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Return a float64 copy with columns in schema order and a clean RangeIndex."""
    return X[list(schema.feature_names)].astype("float64").reset_index(drop=True)
