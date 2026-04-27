"""Feature engineering for payment behavior and credit utilization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import INTERIM_DATA_PATH, PROJECT_ROOT, TARGET_COLUMN


PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "taiwan_model_ready.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FEATURE_DICTIONARY_PATH = TABLES_DIR / "feature_dictionary_taiwan.csv"
LEGACY_FEATURE_LIST_PATH = TABLES_DIR / "feature_engineering_list.csv"

PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

TAIWAN_ENGINEERED_FEATURES = [
    {
        "feature": "delay_count",
        "formula": "count(PAY_0..PAY_6 > 0 after clipping negative pay-status codes to 0)",
        "source_columns": ", ".join(PAY_STATUS_COLUMNS),
        "concept": "payment_delay_frequency",
    },
    {
        "feature": "severe_delay_count",
        "formula": "count(PAY_0..PAY_6 >= 2 after clipping negative pay-status codes to 0)",
        "source_columns": ", ".join(PAY_STATUS_COLUMNS),
        "concept": "severe_payment_delay_frequency",
    },
    {
        "feature": "max_delay",
        "formula": "row-wise max(PAY_0..PAY_6 after clipping negative pay-status codes to 0)",
        "source_columns": ", ".join(PAY_STATUS_COLUMNS),
        "concept": "maximum_delay_severity",
    },
    {
        "feature": "avg_delay",
        "formula": "row-wise mean(PAY_0..PAY_6 after clipping negative pay-status codes to 0)",
        "source_columns": ", ".join(PAY_STATUS_COLUMNS),
        "concept": "average_delay_severity",
    },
    {
        "feature": "recent_delay",
        "formula": "1 if PAY_0 > 0 or PAY_2 > 0 after clipping negative pay-status codes to 0 else 0",
        "source_columns": "PAY_0, PAY_2",
        "concept": "recent_payment_delay",
    },
    {
        "feature": "total_bill_amt",
        "formula": "sum(BILL_AMT1..BILL_AMT6)",
        "source_columns": ", ".join(BILL_COLUMNS),
        "concept": "six_month_total_billed_amount",
    },
    {
        "feature": "avg_bill_amt",
        "formula": "mean(BILL_AMT1..BILL_AMT6)",
        "source_columns": ", ".join(BILL_COLUMNS),
        "concept": "average_bill_amount",
    },
    {
        "feature": "max_bill_amt",
        "formula": "max(BILL_AMT1..BILL_AMT6)",
        "source_columns": ", ".join(BILL_COLUMNS),
        "concept": "maximum_bill_amount",
    },
    {
        "feature": "bill_trend",
        "formula": "BILL_AMT1 - BILL_AMT6",
        "source_columns": "BILL_AMT1, BILL_AMT6",
        "concept": "recent_change_in_bill_balance",
    },
    {
        "feature": "total_pay_amt",
        "formula": "sum(PAY_AMT1..PAY_AMT6)",
        "source_columns": ", ".join(PAY_AMOUNT_COLUMNS),
        "concept": "six_month_total_payment_amount",
    },
    {
        "feature": "avg_pay_amt",
        "formula": "mean(PAY_AMT1..PAY_AMT6)",
        "source_columns": ", ".join(PAY_AMOUNT_COLUMNS),
        "concept": "average_payment_amount",
    },
    {
        "feature": "max_pay_amt",
        "formula": "max(PAY_AMT1..PAY_AMT6)",
        "source_columns": ", ".join(PAY_AMOUNT_COLUMNS),
        "concept": "maximum_payment_amount",
    },
    {
        "feature": "payment_to_bill_ratio",
        "formula": "total_pay_amt / (abs(total_bill_amt) + 1)",
        "source_columns": "total_pay_amt, total_bill_amt",
        "concept": "repayment_capacity_relative_to_balance",
    },
    {
        "feature": "utilization_proxy",
        "formula": "avg_bill_amt / (LIMIT_BAL + 1)",
        "source_columns": "avg_bill_amt, LIMIT_BAL",
        "concept": "credit_limit_utilization_proxy",
    },
    {
        "feature": "recent_payment_intensity",
        "formula": "(PAY_AMT1 + PAY_AMT2) / (abs(BILL_AMT1) + abs(BILL_AMT2) + 1)",
        "source_columns": "PAY_AMT1, PAY_AMT2, BILL_AMT1, BILL_AMT2",
        "concept": "recent_repayment_intensity",
    },
    {
        "feature": "bill_volatility",
        "formula": "row-wise population std(BILL_AMT1..BILL_AMT6)",
        "source_columns": ", ".join(BILL_COLUMNS),
        "concept": "bill_balance_variability",
    },
    {
        "feature": "payment_volatility",
        "formula": "row-wise population std(PAY_AMT1..PAY_AMT6)",
        "source_columns": ", ".join(PAY_AMOUNT_COLUMNS),
        "concept": "payment_amount_variability",
    },
]


def add_payment_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create non-leaky features from the six-month payment history."""

    engineered = df.copy()

    pay_status = engineered[PAY_STATUS_COLUMNS].clip(lower=0)
    engineered["delay_count"] = (pay_status > 0).sum(axis=1)
    engineered["max_delay"] = pay_status.max(axis=1)
    engineered["avg_delay"] = pay_status.mean(axis=1)
    engineered["recent_delay"] = (
        (pay_status["PAY_0"] > 0) | (pay_status["PAY_2"] > 0)
    ).astype(int)
    engineered["severe_delay_count"] = (pay_status >= 2).sum(axis=1)

    bills = engineered[BILL_COLUMNS]
    engineered["total_bill_amt"] = bills.sum(axis=1)
    engineered["avg_bill_amt"] = bills.mean(axis=1)
    engineered["max_bill_amt"] = bills.max(axis=1)
    engineered["bill_trend"] = engineered["BILL_AMT1"] - engineered["BILL_AMT6"]

    payments = engineered[PAY_AMOUNT_COLUMNS]
    engineered["total_pay_amt"] = payments.sum(axis=1)
    engineered["avg_pay_amt"] = payments.mean(axis=1)
    engineered["max_pay_amt"] = payments.max(axis=1)

    # Use an absolute denominator so credit-balance rows do not create unstable ratios.
    engineered["payment_to_bill_ratio"] = engineered["total_pay_amt"] / (
        np.abs(engineered["total_bill_amt"]) + 1
    )
    engineered["utilization_proxy"] = engineered["avg_bill_amt"] / (
        engineered["LIMIT_BAL"] + 1
    )
    engineered["recent_payment_intensity"] = (
        engineered["PAY_AMT1"] + engineered["PAY_AMT2"]
    ) / (np.abs(engineered["BILL_AMT1"]) + np.abs(engineered["BILL_AMT2"]) + 1)
    engineered["bill_volatility"] = bills.std(axis=1, ddof=0)
    engineered["payment_volatility"] = payments.std(axis=1, ddof=0)

    return engineered


def build_feature_dictionary() -> pd.DataFrame:
    """Return the Taiwan engineered-feature dictionary."""

    dictionary = pd.DataFrame(TAIWAN_ENGINEERED_FEATURES)
    dictionary["dataset"] = "taiwan"
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


def make_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready table with engineered features and no ID column."""

    model_ready = add_payment_behavior_features(df)
    if "ID" in model_ready.columns:
        model_ready = model_ready.drop(columns=["ID"])

    if TARGET_COLUMN not in model_ready.columns:
        raise KeyError(f"Target column {TARGET_COLUMN!r} is missing.")

    return model_ready


def main(input_path: str | Path = INTERIM_DATA_PATH) -> None:
    """CLI entry point for feature engineering."""

    cleaned = pd.read_csv(input_path)
    model_ready = make_model_ready_dataset(cleaned)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(PROCESSED_DATA_PATH, index=False)
    feature_dictionary = build_feature_dictionary()
    feature_dictionary.to_csv(FEATURE_DICTIONARY_PATH, index=False)
    feature_dictionary.to_csv(LEGACY_FEATURE_LIST_PATH, index=False)

    print(f"Saved model-ready data to {PROCESSED_DATA_PATH}")
    print(f"Saved Taiwan feature dictionary to {FEATURE_DICTIONARY_PATH}")


if __name__ == "__main__":
    main()
