"""Data loading and cleaning utilities for the Taiwan credit-card default data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "taiwan_default" / "default_credit_card_clients.xls"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "taiwan_default" / "default_credit_card_clients.csv"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "taiwan_cleaned.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TAIWAN_AUDIT_PATH = TABLES_DIR / "taiwan_data_audit.csv"
TAIWAN_TARGET_FIGURE_PATH = FIGURES_DIR / "taiwan_target_distribution.png"

TARGET_COLUMN = "default_next_month"
ID_COLUMN = "ID"

COLUMN_RENAME = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
    "Y": TARGET_COLUMN,
    "default.payment.next.month": TARGET_COLUMN,
    "default payment next month": TARGET_COLUMN,
}

EDUCATION_VALID = {1, 2, 3, 4}
MARRIAGE_VALID = {1, 2, 3}
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
REQUIRED_COLUMNS = [
    ID_COLUMN,
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *BILL_COLUMNS,
    *[f"PAY_AMT{i}" for i in range(1, 7)],
    TARGET_COLUMN,
]


def load_raw_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the UCI Excel file and normalize its header row.

    The original UCI workbook is commonly read with ``header=1``. This loader
    also tries ``header=0`` so it is robust to CSV-converted versions of the
    same data.
    """

    path = Path(path)
    errors: list[str] = []

    if path.suffix.lower() == ".csv":
        return _load_raw_csv(path)

    for header in (1, 0):
        try:
            df = pd.read_excel(path, header=header)
            df = normalize_column_names(df)
            if {"LIMIT_BAL", TARGET_COLUMN}.issubset(df.columns):
                return df.dropna(how="all").reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - useful diagnostic path
            errors.append(f"header={header}: {exc}")

    csv_fallback = path.with_suffix(".csv")
    if csv_fallback.exists():
        try:
            return _load_raw_csv(csv_fallback)
        except Exception as exc:  # pragma: no cover
            errors.append(f"csv_fallback={csv_fallback}: {exc}")

    raise ValueError(
        "Could not load the raw dataset with a recognized schema. "
        f"Tried {path}. Errors: {' | '.join(errors)}"
    )


def _load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV conversion of the official UCI workbook."""

    for header in (1, 0):
        df = pd.read_csv(path, header=header)
        df = normalize_column_names(df)
        if {"LIMIT_BAL", TARGET_COLUMN}.issubset(df.columns):
            return df.dropna(how="all").reset_index(drop=True)

    raise ValueError(f"CSV file does not match the expected UCI schema: {path}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip columns and rename UCI X1..Y names to descriptive project names."""

    renamed = df.copy()
    renamed.columns = [str(col).strip() for col in renamed.columns]
    renamed = renamed.rename(columns=COLUMN_RENAME)

    for column in list(renamed.columns):
        normalized = column.strip().lower().replace(".", "_").replace(" ", "_")
        if normalized in {"default_payment_next_month", "y"}:
            renamed = renamed.rename(columns={column: TARGET_COLUMN})

    return renamed


def clean_dataset(df: pd.DataFrame, drop_duplicate_rows: bool = False) -> pd.DataFrame:
    """Apply project cleaning rules without discarding rows by default."""

    cleaned = normalize_column_names(df)

    for column in cleaned.columns:
        try:
            cleaned[column] = pd.to_numeric(cleaned[column])
        except (TypeError, ValueError):
            pass

    if "EDUCATION" in cleaned.columns:
        cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    if "MARRIAGE" in cleaned.columns:
        cleaned["MARRIAGE"] = cleaned["MARRIAGE"].replace({0: 3})

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    if drop_duplicate_rows:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    validate_taiwan_cleaned(cleaned)
    return cleaned


def validate_taiwan_cleaned(df: pd.DataFrame) -> None:
    """Validate the cleaned Taiwan table and raise on contract violations."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required Taiwan columns after cleaning: {missing_columns}")

    target_values = set(pd.Series(df[TARGET_COLUMN]).dropna().astype(int).unique())
    if target_values - {0, 1}:
        raise ValueError(f"Target must be binary 0/1; observed {sorted(target_values)}")
    assert target_values.issubset({0, 1}), "Taiwan target must be binary 0/1."

    education_values = set(pd.Series(df["EDUCATION"]).dropna().astype(int).unique())
    invalid_education = sorted(education_values - EDUCATION_VALID)
    if invalid_education:
        raise ValueError(f"Invalid EDUCATION values after cleaning: {invalid_education}")

    marriage_values = set(pd.Series(df["MARRIAGE"]).dropna().astype(int).unique())
    invalid_marriage = sorted(marriage_values - MARRIAGE_VALID)
    if invalid_marriage:
        raise ValueError(f"Invalid MARRIAGE values after cleaning: {invalid_marriage}")


def data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Return quality checks used in the report and EDA notebooks."""

    report: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            column: int(value) for column, value in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if TARGET_COLUMN in df.columns:
        report["target_distribution"] = {
            str(label): int(count)
            for label, count in df[TARGET_COLUMN].value_counts().sort_index().items()
        }

    if "EDUCATION" in df.columns:
        observed = set(pd.Series(df["EDUCATION"]).dropna().astype(int).unique())
        report["invalid_education_values"] = [
            int(value) for value in sorted(observed - EDUCATION_VALID)
        ]

    if "MARRIAGE" in df.columns:
        observed = set(pd.Series(df["MARRIAGE"]).dropna().astype(int).unique())
        report["invalid_marriage_values"] = [
            int(value) for value in sorted(observed - MARRIAGE_VALID)
        ]

    negative_bill_counts = {
        column: int((df[column] < 0).sum())
        for column in BILL_COLUMNS
        if column in df.columns
    }
    report["negative_bill_counts"] = negative_bill_counts

    return report


def build_data_audit_table(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create a flat audit table for the Taiwan data pipeline."""

    missing_total = int(cleaned.isna().sum().sum())
    duplicate_full = int(cleaned.duplicated().sum())
    duplicate_without_id = (
        int(cleaned.drop(columns=[ID_COLUMN]).duplicated().sum())
        if ID_COLUMN in cleaned.columns
        else duplicate_full
    )
    target_counts = cleaned[TARGET_COLUMN].value_counts().sort_index()
    target_rates = cleaned[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    invalid_education = sorted(set(cleaned["EDUCATION"].dropna().astype(int)) - EDUCATION_VALID)
    invalid_marriage = sorted(set(cleaned["MARRIAGE"].dropna().astype(int)) - MARRIAGE_VALID)
    negative_bill_counts = {
        column: int((cleaned[column] < 0).sum())
        for column in BILL_COLUMNS
        if column in cleaned.columns
    }
    negative_bill_rates = {
        column: float((cleaned[column] < 0).mean())
        for column in BILL_COLUMNS
        if column in cleaned.columns
    }

    rows = [
        {
            "check": "raw_xls_read",
            "status": "pass",
            "value": str(RAW_DATA_PATH),
            "notes": "Raw Taiwan workbook was read with a robust header strategy.",
        },
        {
            "check": "header_problem_fixed",
            "status": "pass",
            "value": "header=1/header=0 fallback",
            "notes": "UCI X1..Y style columns and target aliases are normalized.",
        },
        {
            "check": "raw_shape",
            "status": "pass",
            "value": f"{raw.shape[0]} rows x {raw.shape[1]} columns",
            "notes": "Shape after dropping fully empty rows.",
        },
        {
            "check": "cleaned_shape",
            "status": "pass",
            "value": f"{cleaned.shape[0]} rows x {cleaned.shape[1]} columns",
            "notes": "Cleaned interim dataset before feature engineering.",
        },
        {
            "check": "column_names_normalized",
            "status": "pass",
            "value": ", ".join(cleaned.columns),
            "notes": "Project uses descriptive column names and default_next_month target.",
        },
        {
            "check": "id_model_exclusion_policy",
            "status": "pass",
            "value": "ID retained in interim; dropped in model_ready dataset",
            "notes": "ID is kept for auditability but excluded before modelling.",
        },
        {
            "check": "education_anomaly_mapping",
            "status": "pass" if not invalid_education else "fail",
            "value": f"invalid_after_cleaning={invalid_education}",
            "notes": "EDUCATION 0, 5, and 6 are mapped to 4 = Others.",
        },
        {
            "check": "marriage_anomaly_mapping",
            "status": "pass" if not invalid_marriage else "fail",
            "value": f"invalid_after_cleaning={invalid_marriage}",
            "notes": "MARRIAGE 0 is mapped to 3 = Others.",
        },
        {
            "check": "missing_values",
            "status": "pass" if missing_total == 0 else "review",
            "value": str(missing_total),
            "notes": "Missing values are reported before modelling.",
        },
        {
            "check": "duplicate_full_rows",
            "status": "pass" if duplicate_full == 0 else "review",
            "value": str(duplicate_full),
            "notes": "Full-row duplicates are reported and not dropped automatically.",
        },
        {
            "check": "duplicate_rows_excluding_id",
            "status": "pass" if duplicate_without_id == 0 else "review",
            "value": str(duplicate_without_id),
            "notes": "Duplicate customer profiles excluding ID are reported separately.",
        },
        {
            "check": "target_distribution_non_default",
            "status": "pass",
            "value": f"{int(target_counts.get(0, 0))} ({target_rates.get(0, 0.0):.4f})",
            "notes": "Class 0 = non-default.",
        },
        {
            "check": "target_distribution_default",
            "status": "pass",
            "value": f"{int(target_counts.get(1, 0))} ({target_rates.get(1, 0.0):.4f})",
            "notes": "Class 1 = default next month.",
        },
        {
            "check": "negative_bill_amounts",
            "status": "review" if any(negative_bill_counts.values()) else "pass",
            "value": json.dumps(negative_bill_counts, sort_keys=True),
            "notes": "Negative BILL_AMT values are plausible credit balances and are not dropped automatically.",
        },
    ]

    for column in BILL_COLUMNS:
        if column in cleaned.columns:
            rows.append(
                {
                    "check": f"{column}_negative_count",
                    "status": "review" if negative_bill_counts.get(column, 0) else "pass",
                    "value": str(negative_bill_counts.get(column, 0)),
                    "notes": "Reported for audit only; no automatic deletion.",
                }
            )
            rows.append(
                {
                    "check": f"{column}_negative_rate",
                    "status": "review" if negative_bill_counts.get(column, 0) else "pass",
                    "value": f"{negative_bill_rates.get(column, 0.0):.6f}",
                    "notes": "Share of rows with negative bill amount; values are preserved.",
                }
            )

    return pd.DataFrame(rows)


def save_target_distribution_plot(df: pd.DataFrame, path: str | Path = TAIWAN_TARGET_FIGURE_PATH) -> None:
    """Save the Taiwan target class distribution figure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Non-default", "Default"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#2f6f73", "#b84a39"])
    ax.set_title("Taiwan Target Class Distribution")
    ax.set_ylabel("Number of customers")
    ax.set_xlabel("Target class")
    ax.spines[["top", "right"]].set_visible(False)

    total = max(int(counts.sum()), 1)
    for bar in bars:
        height = int(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,}\n{height / total:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Save dictionaries as UTF-8 JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entry point for the first cleaning pass."""

    raw = load_raw_dataset()
    cleaned = clean_dataset(raw, drop_duplicate_rows=False)

    INTERIM_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(INTERIM_DATA_PATH, index=False)
    build_data_audit_table(raw, cleaned).to_csv(TAIWAN_AUDIT_PATH, index=False)
    save_target_distribution_plot(cleaned, TAIWAN_TARGET_FIGURE_PATH)
    save_json(
        {
            "before_cleaning": data_quality_report(raw),
            "after_cleaning": data_quality_report(cleaned),
            "cleaning_decisions": [
                "EDUCATION values 0, 5, and 6 were merged into 4 = Others.",
                "MARRIAGE value 0 was merged into 3 = Others.",
                "Duplicate rows are reported but not dropped automatically.",
                "Negative BILL_AMT values are reported and preserved for domain review.",
            ],
        },
        TABLES_DIR / "data_quality_report.json",
    )

    print(f"Saved cleaned data to {INTERIM_DATA_PATH}")
    print(f"Saved Taiwan data audit to {TAIWAN_AUDIT_PATH}")
    print(f"Saved target distribution figure to {TAIWAN_TARGET_FIGURE_PATH}")
    print(f"Saved quality report to {TABLES_DIR / 'data_quality_report.json'}")


if __name__ == "__main__":
    main()
