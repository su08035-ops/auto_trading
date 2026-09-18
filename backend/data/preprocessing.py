"""Preprocessing helpers for cleaning OHLCV market data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame indexed by trading date.

    The loader should already return this shape, but this function makes the
    downstream feature code tolerant of CSV/API sources with minor differences.
    """

    cleaned = df.copy()
    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"])
        cleaned = cleaned.set_index("date")
    elif not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index)

    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in cleaned]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    for column in REQUIRED_OHLCV_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=REQUIRED_OHLCV_COLUMNS)

    valid_price = (cleaned[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_volume = cleaned["volume"] >= 0
    valid_range = (cleaned["high"] >= cleaned[["open", "close", "low"]].max(axis=1)) & (
        cleaned["low"] <= cleaned[["open", "close"]].min(axis=1)
    )
    cleaned = cleaned[valid_price & valid_volume & valid_range & ~cleaned.index.isna()]

    return cleaned[list(REQUIRED_OHLCV_COLUMNS)]


def split_by_date(
    df: pd.DataFrame,
    split_date: str | date | datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sections by date."""

    split_timestamp = pd.Timestamp(split_date)
    cleaned = df.sort_index()
    train_df = cleaned[cleaned.index < split_timestamp]
    test_df = cleaned[cleaned.index >= split_timestamp]
    return train_df, test_df


def train_validation_test_split(
    df: pd.DataFrame,
    validation_start: str | date | datetime,
    test_start: str | date | datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sections by date."""

    validation_timestamp = pd.Timestamp(validation_start)
    test_timestamp = pd.Timestamp(test_start)
    if validation_timestamp >= test_timestamp:
        raise ValueError("validation_start must be earlier than test_start.")
    cleaned = df.sort_index()

    train_df = cleaned[cleaned.index < validation_timestamp]
    validation_df = cleaned[
        (cleaned.index >= validation_timestamp) & (cleaned.index < test_timestamp)
    ]
    test_df = cleaned[cleaned.index >= test_timestamp]
    return train_df, validation_df, test_df


def prepare_splits(
    feature_df: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Keep trailing context, but begin validation/test execution at their boundary.

    With next-open execution, W preceding rows provide the initial state. Only
    the rows after this context generate evaluation returns or trades.
    """
    window = config["state"]["window"]
    validation_date = pd.Timestamp(config["data"]["validation_start"])
    test_date = pd.Timestamp(config["data"]["split_date"])
    if not feature_df.index.is_monotonic_increasing or not feature_df.index.is_unique:
        raise ValueError("Feature dates must be sorted and unique.")
    if validation_date >= test_date:
        raise ValueError("validation_start must precede split_date.")
    validation_index = int(feature_df.index.searchsorted(validation_date))
    test_index = int(feature_df.index.searchsorted(test_date))
    if validation_index < window + 1 or not validation_index < test_index < len(
        feature_df
    ):
        raise ValueError(
            "Need nonempty train, validation and test periods with enough context."
        )
    return {
        "train": feature_df.iloc[:validation_index].copy(),
        "validation": feature_df.iloc[validation_index - window : test_index].copy(),
        "test": feature_df.iloc[test_index - window :].copy(),
    }


def validate_feature_frame(
    df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """Raise a clear error when expected feature columns are unavailable."""

    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def to_numpy_features(
    df: pd.DataFrame, feature_names: list[str]
) -> np.ndarray[Any, Any]:
    """Convert selected feature columns to a float32 NumPy array."""

    validate_feature_frame(df, feature_names)
    return df[feature_names].astype("float32").to_numpy()
