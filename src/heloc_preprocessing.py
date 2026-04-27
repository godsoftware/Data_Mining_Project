"""Loading and preprocessing for the FICO HELOC external validation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preprocessing import PROJECT_ROOT


HELOC_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "heloc"
HELOC_RAW_CSV_PATH = HELOC_RAW_DIR / "heloc_dataset.csv"
HELOC_INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "heloc_cleaned.csv"
HELOC_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "heloc_model_ready.csv"
HELOC_QUALITY_REPORT_PATH = PROJECT_ROOT / "outputs" / "tables" / "heloc_data_quality_report.json"
HELOC_AUDIT_PATH = PROJECT_ROOT / "outputs" / "tables" / "heloc_data_audit.csv"
HELOC_DATA_DICTIONARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "heloc_data_dictionary.csv"
HELOC_FEATURE_DICTIONARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "feature_dictionary_heloc.csv"
HELOC_CLEANING_DECISIONS_PATH = PROJECT_ROOT / "outputs" / "tables" / "heloc_cleaning_decisions.csv"
HELOC_TARGET_FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "heloc_target_distribution.png"

HELOC_TARGET = "bad_flag"
HELOC_ORIGINAL_TARGET = "RiskPerformance"
HELOC_SPECIAL_CODES = [-9, -8, -7]
HELOC_SPECIAL_CODE_MEANINGS = {
    -9: "No bureau record or no investigation",
    -8: "No usable / valid trades or inquiries",
    -7: "Condition not met for this attribute",
}
HELOC_CATEGORICAL_COLUMNS = ["MaxDelq2PublicRecLast12M", "MaxDelqEver"]

HELOC_DESCRIPTIONS = {
    "RiskPerformance": "Original binary outcome: Good or Bad account performance.",
    HELOC_TARGET: "Binary target used by this project; 1 = Bad, 0 = Good.",
    "ExternalRiskEstimate": "External bureau risk estimate score.",
    "MSinceOldestTradeOpen": "Months since oldest trade was opened.",
    "MSinceMostRecentTradeOpen": "Months since most recent trade was opened.",
    "AverageMInFile": "Average months that trades have been in file.",
    "NumSatisfactoryTrades": "Number of satisfactory trades.",
    "NumTrades60Ever2DerogPubRec": "Number of trades ever 60+ days delinquent or derogatory public records.",
    "NumTrades90Ever2DerogPubRec": "Number of trades ever 90+ days delinquent or derogatory public records.",
    "PercentTradesNeverDelq": "Percentage of trades that were never delinquent.",
    "MSinceMostRecentDelq": "Months since most recent delinquency.",
    "MaxDelq2PublicRecLast12M": "Maximum delinquency or public record severity in the last 12 months.",
    "MaxDelqEver": "Maximum delinquency severity ever observed.",
    "NumTotalTrades": "Total number of trades.",
    "NumTradesOpeninLast12M": "Number of trades opened in the last 12 months.",
    "PercentInstallTrades": "Percentage of installment trades.",
    "MSinceMostRecentInqexcl7days": "Months since most recent inquiry excluding the last 7 days.",
    "NumInqLast6M": "Number of inquiries in the last 6 months.",
    "NumInqLast6Mexcl7days": "Number of inquiries in the last 6 months excluding the last 7 days.",
    "NetFractionRevolvingBurden": "Net fraction of revolving credit burden.",
    "NetFractionInstallBurden": "Net fraction of installment credit burden.",
    "NumRevolvingTradesWBalance": "Number of revolving trades with balance.",
    "NumInstallTradesWBalance": "Number of installment trades with balance.",
    "NumBank2NatlTradesWHighUtilization": "Number of bankcard or national trades with high utilization.",
    "PercentTradesWBalance": "Percentage of trades with balance.",
}

from features.heloc_features import build_heloc_feature_dictionary, make_heloc_model_ready_dataset


def load_heloc_dataset(path: str | Path = HELOC_RAW_CSV_PATH) -> pd.DataFrame:
    """Load the cleaned OpenML mirror of the FICO HELOC dataset."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"HELOC file not found: {path}. Download the OpenML 45554 ARFF/CSV first."
        )
    return pd.read_csv(path)


def clean_heloc_dataset(
    df: pd.DataFrame,
    target_column: str = HELOC_ORIGINAL_TARGET,
    special_code_strategy: str = "nan",
) -> pd.DataFrame:
    """Normalize target, numeric columns, and configurable special-code handling."""

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    if target_column not in cleaned.columns:
        raise KeyError(f"Expected target column {target_column!r}.")
    if special_code_strategy not in {"nan", "indicator", "category"}:
        raise ValueError("special_code_strategy must be 'nan', 'indicator', or 'category'.")

    target_map = {"Bad": 1, "Good": 0, "bad": 1, "good": 0, 1: 1, 0: 0}
    cleaned[HELOC_TARGET] = cleaned[target_column].map(target_map)
    if cleaned[HELOC_TARGET].isna().any():
        unknown = cleaned.loc[cleaned[HELOC_TARGET].isna(), target_column].unique()
        raise ValueError(f"Unknown HELOC target values: {unknown}")

    cleaned = cleaned.drop(columns=[target_column])

    for column in cleaned.columns:
        if column == HELOC_TARGET:
            cleaned[column] = cleaned[column].astype(int)
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if special_code_strategy == "indicator":
            for code in HELOC_SPECIAL_CODES:
                cleaned[f"{column}_special_code_{abs(code)}"] = (cleaned[column] == code).astype(int)
            cleaned[column] = cleaned[column].replace(HELOC_SPECIAL_CODES, np.nan)
        elif special_code_strategy == "nan":
            cleaned[column] = cleaned[column].replace(HELOC_SPECIAL_CODES, np.nan)
        else:
            cleaned[column] = cleaned[column].replace({code: float(code) for code in HELOC_SPECIAL_CODES})

    return cleaned


def special_code_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count HELOC special missing-value codes in raw feature columns."""

    counts: dict[str, dict[str, int]] = {}
    for column in df.columns:
        if column == HELOC_ORIGINAL_TARGET:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        column_counts = {
            str(code): int((numeric == code).sum())
            for code in HELOC_SPECIAL_CODES
        }
        counts[column] = column_counts
    return counts


def _total_special_code_counts(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    """Aggregate special-code counts across all features."""

    return {
        str(code): int(sum(column_counts.get(str(code), 0) for column_counts in counts.values()))
        for code in HELOC_SPECIAL_CODES
    }


def build_heloc_data_audit_table(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create a flat HELOC audit table compatible with the Taiwan audit output."""

    raw_special_counts = special_code_counts(raw)
    total_special_counts = _total_special_code_counts(raw_special_counts)
    target_counts = cleaned[HELOC_TARGET].value_counts().sort_index()
    target_rates = cleaned[HELOC_TARGET].value_counts(normalize=True).sort_index()
    missing_total = int(cleaned.isna().sum().sum())
    duplicate_rows = int(cleaned.duplicated().sum())
    feature_columns = [column for column in cleaned.columns if column != HELOC_TARGET]
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    target_values = sorted(cleaned[HELOC_TARGET].dropna().unique().astype(int).tolist())

    rows = [
        {
            "check": "raw_csv_location",
            "status": "pass" if HELOC_RAW_CSV_PATH.exists() else "fail",
            "value": str(HELOC_RAW_CSV_PATH),
            "notes": "HELOC raw CSV is stored separately from the Taiwan dataset.",
        },
        {
            "check": "raw_shape",
            "status": "pass",
            "value": f"{raw.shape[0]} rows x {raw.shape[1]} columns",
            "notes": "Raw external-validation dataset before target normalization and missing-code handling.",
        },
        {
            "check": "cleaned_shape",
            "status": "pass",
            "value": f"{cleaned.shape[0]} rows x {cleaned.shape[1]} columns",
            "notes": "Cleaned HELOC table after dropping original target and adding bad_flag.",
        },
        {
            "check": "target_binary_mapping",
            "status": "pass" if target_values == [0, 1] else "fail",
            "value": "Good -> 0, Bad -> 1",
            "notes": "Positive class is bad_flag = 1 for the common binary metric interface.",
        },
        {
            "check": "original_target_removed",
            "status": "pass" if HELOC_ORIGINAL_TARGET not in cleaned.columns else "fail",
            "value": str(HELOC_ORIGINAL_TARGET not in cleaned.columns),
            "notes": "Original string target is removed after binary conversion.",
        },
        {
            "check": "special_missing_codes_detected",
            "status": "review" if any(total_special_counts.values()) else "pass",
            "value": json.dumps(total_special_counts, sort_keys=True),
            "notes": "Counts are measured on raw data before cleaning. Some mirrors may already expose these as NaN.",
        },
        {
            "check": "special_missing_code_strategy",
            "status": "pass",
            "value": json.dumps({str(k): v for k, v in HELOC_SPECIAL_CODE_MEANINGS.items()}, sort_keys=True),
            "notes": "Codes -9, -8, and -7 are converted to NaN; model pipelines impute within train folds.",
        },
        {
            "check": "missing_values_after_cleaning",
            "status": "review" if missing_total else "pass",
            "value": str(missing_total),
            "notes": "Missing values are intentionally preserved for train-fold imputation.",
        },
        {
            "check": "duplicate_rows",
            "status": "pass" if duplicate_rows == 0 else "review",
            "value": str(duplicate_rows),
            "notes": "Duplicates are reported and not dropped automatically.",
        },
        {
            "check": "target_distribution_good",
            "status": "pass",
            "value": f"{int(target_counts.get(0, 0))} ({target_rates.get(0, 0.0):.4f})",
            "notes": "Class 0 = Good / non-bad account performance.",
        },
        {
            "check": "target_distribution_bad",
            "status": "pass",
            "value": f"{int(target_counts.get(1, 0))} ({target_rates.get(1, 0.0):.4f})",
            "notes": "Class 1 = Bad account performance.",
        },
        {
            "check": "numeric_feature_count",
            "status": "pass",
            "value": str(len(numeric_columns)),
            "notes": "Numeric features are audited below with missing/min/max/mean/std.",
        },
        {
            "check": "categorical_feature_interface",
            "status": "pass",
            "value": ", ".join(HELOC_CATEGORICAL_COLUMNS),
            "notes": "These ordinal delinquency variables are passed as categorical columns where supported.",
        },
        {
            "check": "metric_interface",
            "status": "pass",
            "value": f"target={HELOC_TARGET}; positive_class=1; feature_count={len(feature_columns)}",
            "notes": "HELOC is aligned with the same binary metric functions used for Taiwan.",
        },
    ]

    for column in numeric_columns:
        series = cleaned[column]
        raw_counts = raw_special_counts.get(column, {})
        summary = {
            "missing_after_cleaning": int(series.isna().sum()),
            "special_code_counts_raw": raw_counts,
            "min": None if series.dropna().empty else float(series.min()),
            "max": None if series.dropna().empty else float(series.max()),
            "mean": None if series.dropna().empty else float(series.mean()),
            "std": None if series.dropna().empty else float(series.std()),
        }
        rows.append(
            {
                "check": f"numeric_audit_{column}",
                "status": "review" if summary["missing_after_cleaning"] else "pass",
                "value": json.dumps(summary, sort_keys=True),
                "notes": HELOC_DESCRIPTIONS.get(column, "HELOC numeric feature."),
            }
        )

    return pd.DataFrame(rows)


def build_heloc_cleaning_decisions(
    target_column: str = HELOC_ORIGINAL_TARGET,
    special_code_strategy: str = "nan",
) -> pd.DataFrame:
    """Document the HELOC cleaning decisions applied by the pipeline."""

    rows = [
        {
            "decision": "target_column",
            "setting": target_column,
            "rationale": "Target is supplied explicitly instead of inferred automatically.",
            "output_effect": f"{target_column} is mapped to {HELOC_TARGET} with Bad=1 and Good=0.",
        },
        {
            "decision": "drop_original_target",
            "setting": "true",
            "rationale": "Avoid duplicate target leakage after binary mapping.",
            "output_effect": f"{target_column} is removed from cleaned and model-ready features.",
        },
        {
            "decision": "special_missing_codes",
            "setting": ", ".join(str(code) for code in HELOC_SPECIAL_CODES),
            "rationale": "FICO HELOC negative sentinel codes encode semantic missingness.",
            "output_effect": "Codes are audited before cleaning.",
        },
        {
            "decision": "special_code_strategy",
            "setting": special_code_strategy,
            "rationale": "The default keeps missingness for train-fold imputation.",
            "output_effect": "nan converts sentinels to NaN; indicator adds flags; category keeps code values.",
        },
        {
            "decision": "duplicate_rows",
            "setting": "report_only",
            "rationale": "Duplicates are data-quality evidence and are not dropped without a dataset-specific rule.",
            "output_effect": "Duplicate count is written to heloc_data_audit.csv.",
        },
    ]
    return pd.DataFrame(rows)


def build_heloc_data_dictionary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build a lightweight data dictionary for the HELOC external dataset."""

    raw_special_counts = special_code_counts(raw)
    rows = []

    for original_column in raw.columns:
        if original_column == HELOC_ORIGINAL_TARGET:
            cleaned_column = HELOC_TARGET
            role = "target"
            cleaned_missing = int(cleaned[HELOC_TARGET].isna().sum())
            unique_values = "0=Good, 1=Bad"
            min_value = max_value = None
        else:
            cleaned_column = original_column
            role = "categorical_feature" if original_column in HELOC_CATEGORICAL_COLUMNS else "numeric_feature"
            cleaned_missing = int(cleaned[cleaned_column].isna().sum())
            unique_values = ""
            non_missing = cleaned[cleaned_column].dropna()
            min_value = None if non_missing.empty else float(non_missing.min())
            max_value = None if non_missing.empty else float(non_missing.max())

        rows.append(
            {
                "original_column": original_column,
                "cleaned_column": cleaned_column,
                "role": role,
                "raw_dtype": str(raw[original_column].dtype),
                "cleaned_dtype": str(cleaned[cleaned_column].dtype),
                "raw_missing_count": int(raw[original_column].isna().sum()),
                "cleaned_missing_count": cleaned_missing,
                "special_code_counts_raw": json.dumps(
                    raw_special_counts.get(original_column, {}),
                    sort_keys=True,
                ),
                "min_cleaned": min_value,
                "max_cleaned": max_value,
                "unique_values": unique_values,
                "description": HELOC_DESCRIPTIONS.get(
                    original_column,
                    HELOC_DESCRIPTIONS.get(cleaned_column, "HELOC feature."),
                ),
            }
        )

    return pd.DataFrame(rows)


def save_heloc_target_distribution_plot(
    df: pd.DataFrame,
    path: str | Path = HELOC_TARGET_FIGURE_PATH,
) -> None:
    """Save the HELOC binary target distribution figure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = df[HELOC_TARGET].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(["Good", "Bad"], [counts.get(0, 0), counts.get(1, 0)], color=["#2f6f73", "#b84a39"])
    ax.set_title("HELOC Target Class Distribution")
    ax.set_ylabel("Number of accounts")
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


def heloc_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic quality checks for the external HELOC dataset."""

    report: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {
            str(label): int(count)
            for label, count in df[HELOC_TARGET].value_counts().sort_index().items()
        }
        if HELOC_TARGET in df.columns
        else {},
        "external_validation_note": (
            "HELOC has a different feature schema from Taiwan. It is used for "
            "methodological robustness, not direct cross-feature transfer."
        ),
    }
    return report


def main() -> None:
    """Create the model-ready HELOC table and quality report."""

    raw = load_heloc_dataset()
    cleaned = clean_heloc_dataset(raw)
    model_ready = make_heloc_model_ready_dataset(cleaned)

    HELOC_INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELOC_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELOC_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELOC_CLEANING_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELOC_TARGET_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(HELOC_INTERIM_PATH, index=False)
    model_ready.to_csv(HELOC_PROCESSED_PATH, index=False)
    build_heloc_data_audit_table(raw, cleaned).to_csv(HELOC_AUDIT_PATH, index=False)
    build_heloc_data_dictionary(raw, cleaned).to_csv(HELOC_DATA_DICTIONARY_PATH, index=False)
    build_heloc_feature_dictionary().to_csv(HELOC_FEATURE_DICTIONARY_PATH, index=False)
    build_heloc_cleaning_decisions().to_csv(HELOC_CLEANING_DECISIONS_PATH, index=False)
    save_heloc_target_distribution_plot(cleaned, HELOC_TARGET_FIGURE_PATH)
    HELOC_QUALITY_REPORT_PATH.write_text(
        json.dumps(heloc_quality_report(cleaned), indent=2),
        encoding="utf-8",
    )
    print(f"Saved HELOC cleaned data to {HELOC_INTERIM_PATH}")
    print(f"Saved HELOC model-ready data to {HELOC_PROCESSED_PATH}")
    print(f"Saved HELOC data audit to {HELOC_AUDIT_PATH}")
    print(f"Saved HELOC data dictionary to {HELOC_DATA_DICTIONARY_PATH}")
    print(f"Saved HELOC feature dictionary to {HELOC_FEATURE_DICTIONARY_PATH}")
    print(f"Saved HELOC cleaning decisions to {HELOC_CLEANING_DECISIONS_PATH}")
    print(f"Saved HELOC target distribution figure to {HELOC_TARGET_FIGURE_PATH}")
    print(f"Saved HELOC quality report to {HELOC_QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
