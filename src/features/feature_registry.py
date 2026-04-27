"""Feature group registry."""

TAIWAN_PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TAIWAN_BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
TAIWAN_PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
TAIWAN_CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]
TAIWAN_ENGINEERED_FEATURES = [
    "delay_count",
    "severe_delay_count",
    "max_delay",
    "avg_delay",
    "recent_delay",
    "total_bill_amt",
    "avg_bill_amt",
    "max_bill_amt",
    "bill_trend",
    "total_pay_amt",
    "avg_pay_amt",
    "max_pay_amt",
    "payment_to_bill_ratio",
    "utilization_proxy",
    "recent_payment_intensity",
    "bill_volatility",
    "payment_volatility",
]

HELOC_TARGET = "bad_flag"
HELOC_CATEGORICAL_COLUMNS = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]
HELOC_ENGINEERED_FEATURES = [
    "delinquency_intensity",
    "trade_activity_ratio",
    "recent_inquiry_pressure",
    "revolving_burden_proxy",
    "installment_burden_proxy",
    "credit_history_length_proxy",
    "negative_trade_signal",
    "high_utilization_signal",
]

TAIWAN_MONOTONIC_PRIORS = {
    "LIMIT_BAL": 0,
    "SEX": 0,
    "EDUCATION": 0,
    "MARRIAGE": 0,
    "AGE": 0,
    "PAY_0": 1,
    "PAY_2": 1,
    "PAY_3": 1,
    "PAY_4": 1,
    "PAY_5": 1,
    "PAY_6": 1,
    "delay_count": 1,
    "max_delay": 1,
    "avg_delay": 1,
    "recent_delay": 1,
    "severe_delay_count": 1,
    "payment_to_bill_ratio": -1,
    "utilization_proxy": 1,
    "recent_payment_intensity": -1,
    "bill_volatility": 0,
    "payment_volatility": 0,
}

HELOC_MONOTONIC_PRIORS = {
    "ExternalRiskEstimate": -1,
    "NumTrades60Ever2DerogPubRec": 1,
    "NumTrades90Ever2DerogPubRec": 1,
    "PercentTradesNeverDelq": -1,
    "MSinceMostRecentDelq": -1,
    "NumInqLast6M": 1,
    "NumInqLast6Mexcl7days": 1,
    "NetFractionRevolvingBurden": 1,
    "NetFractionInstallBurden": 1,
    "NumBank2NatlTradesWHighUtilization": 1,
    "delinquency_intensity": 1,
    "trade_activity_ratio": 1,
    "recent_inquiry_pressure": 1,
    "revolving_burden_proxy": 1,
    "installment_burden_proxy": 1,
    "credit_history_length_proxy": -1,
    "negative_trade_signal": 1,
    "high_utilization_signal": 1,
}


def check_feature_leakage(
    feature_columns: list[str],
    target_column: str,
    id_columns: list[str] | None = None,
    future_terms: list[str] | None = None,
    target_derived_terms: list[str] | None = None,
) -> list[str]:
    """Return feature names that violate the project leakage policy."""

    id_columns = id_columns or ["ID"]
    future_terms = future_terms or ["future", "next_month", "after_default"]
    target_derived_terms = target_derived_terms or [
        target_column,
        "target",
        "label",
        "default_next_month",
        "bad_flag",
    ]

    violations: list[str] = []
    target_lower = target_column.lower()
    for column in feature_columns:
        lower = column.lower()
        if column in id_columns:
            violations.append(column)
        elif lower == target_lower:
            violations.append(column)
        elif any(term.lower() in lower for term in future_terms):
            violations.append(column)
        elif any(term.lower() in lower for term in target_derived_terms if term.lower() != target_lower):
            violations.append(column)
    return sorted(set(violations))


def assert_no_feature_leakage(
    feature_columns: list[str],
    target_column: str,
    id_columns: list[str] | None = None,
) -> None:
    """Raise when ID, target, future, or target-derived columns are used as features."""

    violations = check_feature_leakage(
        feature_columns=feature_columns,
        target_column=target_column,
        id_columns=id_columns,
    )
    if violations:
        raise AssertionError(f"Potential feature leakage columns found: {violations}")
