"""HELOC feature-engineering entry points."""

from __future__ import annotations

import numpy as np
import pandas as pd


HELOC_TARGET = "bad_flag"

HELOC_ENGINEERED_FEATURES = [
    {
        "feature": "delinquency_intensity",
        "formula": (
            "NumTrades60Ever2DerogPubRec + 2*NumTrades90Ever2DerogPubRec "
            "+ max(0, 100 - PercentTradesNeverDelq) / 100"
        ),
        "source_columns": (
            "NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, "
            "PercentTradesNeverDelq"
        ),
        "concept": "delinquency_frequency_and_severity",
    },
    {
        "feature": "trade_activity_ratio",
        "formula": "NumTradesOpeninLast12M / (NumTotalTrades + 1)",
        "source_columns": "NumTradesOpeninLast12M, NumTotalTrades",
        "concept": "recent_trade_activity_relative_to_total_history",
    },
    {
        "feature": "recent_inquiry_pressure",
        "formula": "NumInqLast6Mexcl7days / (MSinceMostRecentInqexcl7days + 1)",
        "source_columns": "NumInqLast6Mexcl7days, MSinceMostRecentInqexcl7days",
        "concept": "recent_credit_inquiry_pressure",
    },
    {
        "feature": "revolving_burden_proxy",
        "formula": "NetFractionRevolvingBurden",
        "source_columns": "NetFractionRevolvingBurden",
        "concept": "revolving_credit_burden",
    },
    {
        "feature": "installment_burden_proxy",
        "formula": "NetFractionInstallBurden",
        "source_columns": "NetFractionInstallBurden",
        "concept": "installment_credit_burden",
    },
    {
        "feature": "credit_history_length_proxy",
        "formula": "row-wise max(MSinceOldestTradeOpen, AverageMInFile)",
        "source_columns": "MSinceOldestTradeOpen, AverageMInFile",
        "concept": "credit_history_length",
    },
    {
        "feature": "negative_trade_signal",
        "formula": (
            "1 if NumTrades60Ever2DerogPubRec > 0 or NumTrades90Ever2DerogPubRec > 0 "
            "or PercentTradesNeverDelq < 100 else 0"
        ),
        "source_columns": (
            "NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, "
            "PercentTradesNeverDelq"
        ),
        "concept": "negative_trade_history_indicator",
    },
    {
        "feature": "high_utilization_signal",
        "formula": "1 if NetFractionRevolvingBurden >= 80 or NumBank2NatlTradesWHighUtilization > 0 else 0",
        "source_columns": "NetFractionRevolvingBurden, NumBank2NatlTradesWHighUtilization",
        "concept": "high_credit_utilization_indicator",
    },
]


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric source column or an all-NaN placeholder."""

    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def add_heloc_aligned_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-wise, target-free HELOC features aligned with credit-risk concepts."""

    engineered = df.copy()

    trades_60 = _series(engineered, "NumTrades60Ever2DerogPubRec")
    trades_90 = _series(engineered, "NumTrades90Ever2DerogPubRec")
    percent_never_delq = _series(engineered, "PercentTradesNeverDelq")
    total_trades = _series(engineered, "NumTotalTrades")
    trades_open_12m = _series(engineered, "NumTradesOpeninLast12M")
    recent_inquiries = _series(engineered, "NumInqLast6Mexcl7days")
    months_since_inquiry = _series(engineered, "MSinceMostRecentInqexcl7days")
    revolving_burden = _series(engineered, "NetFractionRevolvingBurden")
    installment_burden = _series(engineered, "NetFractionInstallBurden")
    oldest_trade = _series(engineered, "MSinceOldestTradeOpen")
    average_months = _series(engineered, "AverageMInFile")
    high_util_trade_count = _series(engineered, "NumBank2NatlTradesWHighUtilization")

    non_delq_gap = (100 - percent_never_delq).clip(lower=0) / 100
    engineered["delinquency_intensity"] = trades_60.fillna(0) + (2 * trades_90.fillna(0)) + non_delq_gap.fillna(0)
    engineered["trade_activity_ratio"] = trades_open_12m / (total_trades + 1)
    engineered["recent_inquiry_pressure"] = recent_inquiries / (months_since_inquiry + 1)
    engineered["revolving_burden_proxy"] = revolving_burden
    engineered["installment_burden_proxy"] = installment_burden
    engineered["credit_history_length_proxy"] = pd.concat(
        [oldest_trade, average_months],
        axis=1,
    ).max(axis=1)

    negative_sources_available = trades_60.notna() | trades_90.notna() | percent_never_delq.notna()
    negative_signal = (
        (trades_60.fillna(0) > 0)
        | (trades_90.fillna(0) > 0)
        | (percent_never_delq < 100)
    )
    engineered["negative_trade_signal"] = negative_signal.astype(float).where(
        negative_sources_available,
        np.nan,
    )

    utilization_sources_available = revolving_burden.notna() | high_util_trade_count.notna()
    high_utilization = (revolving_burden >= 80) | (high_util_trade_count.fillna(0) > 0)
    engineered["high_utilization_signal"] = high_utilization.astype(float).where(
        utilization_sources_available,
        np.nan,
    )

    return engineered


def make_heloc_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return HELOC model-ready data with row-wise aligned features.

    Missing values are intentionally preserved and handled inside model
    pipelines after splitting.
    """

    model_ready = add_heloc_aligned_features(df)
    if HELOC_TARGET not in model_ready.columns:
        raise KeyError(f"Target column {HELOC_TARGET!r} is missing.")
    return model_ready


def build_heloc_feature_dictionary() -> pd.DataFrame:
    """Return the HELOC engineered-feature dictionary."""

    dictionary = pd.DataFrame(HELOC_ENGINEERED_FEATURES)
    dictionary["dataset"] = "heloc"
    dictionary["feature_type"] = "engineered_row_wise"
    dictionary["uses_target"] = False
    dictionary["uses_dataset_level_statistics"] = False
    dictionary["fit_scope"] = "pre_split_row_wise_allowed"
    return dictionary[
        [
            "dataset",
            "feature",
            "feature_type",
            "concept",
            "formula",
            "source_columns",
            "uses_target",
            "uses_dataset_level_statistics",
            "fit_scope",
        ]
    ]


__all__ = [
    "add_heloc_aligned_features",
    "build_heloc_feature_dictionary",
    "make_heloc_model_ready_dataset",
]
