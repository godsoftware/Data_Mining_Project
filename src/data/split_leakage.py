"""Split summaries and leakage controls for the SCRE-Credit datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_SEED, TEST_SIZE, TRAIN_SIZE, VALIDATION_SIZE


@dataclass
class TrainValidationTestSplit:
    """Container for stratified train/validation/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def stratified_train_validation_test_split(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = TRAIN_SIZE,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> TrainValidationTestSplit:
    """Create a 60/20/20 stratified split without fitting transforms."""

    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split sizes must sum to 1.0, got {total}.")
    if target_column not in df.columns:
        raise KeyError(f"Target column is missing: {target_column}")

    train_validation, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )
    validation_fraction_of_train_validation = validation_size / (train_size + validation_size)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_of_train_validation,
        stratify=train_validation[target_column],
        random_state=random_state,
    )
    return TrainValidationTestSplit(
        train=train.copy(),
        validation=validation.copy(),
        test=test.copy(),
    )


def build_split_summary(
    dataset: str,
    df: pd.DataFrame,
    target_column: str,
    train_size: float = TRAIN_SIZE,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Summarize split sizes and class ratios."""

    split = stratified_train_validation_test_split(
        df,
        target_column=target_column,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    total_rows = len(df)
    rows = []
    for split_name, split_df in [
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ]:
        counts = split_df[target_column].value_counts().sort_index()
        rows.append(
            {
                "dataset": dataset,
                "split": split_name,
                "rows": int(len(split_df)),
                "row_fraction": len(split_df) / total_rows,
                "target_0_count": int(counts.get(0, 0)),
                "target_1_count": int(counts.get(1, 0)),
                "target_1_rate": float(split_df[target_column].mean()),
                "target_column": target_column,
                "random_seed": random_state,
                "split_policy": f"{train_size:.2f}/{validation_size:.2f}/{test_size:.2f}",
                "stratified": True,
            }
        )
    return pd.DataFrame(rows)


def build_leakage_checklist() -> pd.DataFrame:
    """Return the project leakage-control checklist."""

    rows = [
        {
            "check": "stratified_train_validation_test_split",
            "status": "pass",
            "evidence": "Phase 5 split summaries use a 60/20/20 stratified split with the central random seed.",
        },
        {
            "check": "test_set_reserved",
            "status": "pass",
            "evidence": "Test split is used only for final metric calculation; SCRE threshold and review-band search use internal validation probabilities.",
        },
        {
            "check": "row_wise_feature_engineering_only_before_split",
            "status": "pass",
            "evidence": "Feature dictionaries mark uses_target=False and uses_dataset_level_statistics=False.",
        },
        {
            "check": "no_global_mean_or_target_encoding",
            "status": "pass",
            "evidence": "No target encoding or global mean encoding is implemented; categorical variables use one-hot encoding inside pipelines.",
        },
        {
            "check": "scaling_fit_only_on_train_fold",
            "status": "pass",
            "evidence": "StandardScaler is inside sklearn Pipeline/ColumnTransformer and is fit after splitting or inside CV folds.",
        },
        {
            "check": "encoding_fit_only_on_train_fold",
            "status": "pass",
            "evidence": "OneHotEncoder is inside sklearn Pipeline/ColumnTransformer and is fit after splitting or inside CV folds.",
        },
        {
            "check": "missing_imputation_fit_only_on_train_fold",
            "status": "pass",
            "evidence": "SCRE-Credit preprocessors use SimpleImputer inside model pipelines; HELOC missing values are preserved until pipeline fit.",
        },
        {
            "check": "resampling_only_inside_train_fold",
            "status": "pass",
            "evidence": "SMOTE/SMOTENC/SMOTEENN are wrapped in imbalanced-learn pipelines and are not applied to full data or test data.",
        },
        {
            "check": "taiwan_id_excluded_from_model_features",
            "status": "pass",
            "evidence": "ID is retained only in interim data and removed from data/processed/taiwan_model_ready.csv.",
        },
        {
            "check": "heloc_original_target_removed",
            "status": "pass",
            "evidence": "RiskPerformance is mapped to bad_flag and removed from cleaned/model-ready HELOC tables.",
        },
        {
            "check": "calibration_policy",
            "status": "pass",
            "evidence": "CV-based calibration is allowed on training data; test predictions are not used to fit calibrators.",
        },
        {
            "check": "dataset_level_statistics_policy",
            "status": "pass",
            "evidence": "Any future dataset-level statistic must be implemented as a train-fitted transformer, not precomputed on full data.",
        },
    ]
    return pd.DataFrame(rows)
