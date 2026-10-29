import numpy as np
import pandas as pd

from driftguard.data.validation import FeatureSchema, align_columns, validate_batch, validate_features


def _schema(splits):
    return FeatureSchema.from_reference(splits.X_ref)


def test_valid_batch_passes(splits):
    rep = validate_batch(splits.X_test, splits.y_test, _schema(splits), min_rows=20)
    assert rep.valid and not rep.errors


def test_missing_column_is_error(splits):
    rep = validate_features(splits.X_test.drop(columns=["mean radius"]), _schema(splits))
    assert not rep.valid and "missing feature columns" in rep.errors[0]


def test_unexpected_column_is_error(splits):
    X = splits.X_test.copy()
    X["bogus"] = 1.0
    assert not validate_features(X, _schema(splits)).valid


def test_nan_and_inf_rejected(splits):
    X = splits.X_test.copy()
    X.iloc[0, 0] = np.nan
    X.iloc[1, 1] = np.inf
    rep = validate_features(X, _schema(splits))
    assert not rep.valid
    assert any("NaN" in e for e in rep.errors) and any("infinite" in e for e in rep.errors)


def test_non_numeric_rejected(splits):
    X = splits.X_test.copy().astype(object)
    X["mean radius"] = "abc"
    assert not validate_features(X, _schema(splits)).valid


def test_too_few_rows(splits):
    assert not validate_features(splits.X_test.head(5), _schema(splits), min_rows=20).valid


def test_bad_labels(splits):
    X = splits.X_test.head(30)
    assert not validate_batch(X, pd.Series([0, 1, 2] * 10), _schema(splits)).valid
    assert not validate_batch(X, pd.Series([0, 1] * 5), _schema(splits)).valid  # wrong length


def test_out_of_range_is_warning_not_error(splits):
    rep = validate_features(splits.X_test * 100, _schema(splits))
    assert rep.valid and rep.warnings


def test_align_columns_reorders(splits):
    X = splits.X_test.head(3)[list(reversed(splits.feature_names))]
    assert list(align_columns(X, _schema(splits)).columns) == list(splits.feature_names)
