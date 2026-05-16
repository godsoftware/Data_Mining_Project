"""Capacity-aware manual review and Lift@K analysis.

The experiment treats the models as portfolio screening/ranking systems rather
than automatic accept/reject systems. It reports how many defaults are captured
when a bank can review only the highest-risk top-K fraction of customers.

Candidate model probabilities are generated from already-fitted artifacts where
available. The output includes both validation and held-out test splits; test
rows are diagnostic evidence only and are not used to lock a final policy.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import stratified_train_validation_test_split  # noqa: E402
from data_preprocessing import TARGET_COLUMN  # noqa: E402
from heloc_preprocessing import HELOC_TARGET  # noqa: E402
from src.config.paths import HELOC_MODEL_READY, OUTPUTS_DIR, TAIWAN_MODEL_READY  # noqa: E402


OUTPUT_DIR = OUTPUTS_DIR / "final_attempt" / "capacity_review"
FN_COST = 5.0
REVIEW_COST_PER_CASE = 0.5
CAPACITY_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
RANDOM_SEED = 42

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset metadata for lift analysis."""

    name: str
    path: Path
    target: str
    role: str


@dataclass(frozen=True)
class CandidateSpec:
    """Candidate probability-ranking artifact."""

    dataset: str
    model: str
    candidate_group: str
    source: str
    artifact_path: Path | None = None
    ensemble_path: Path | None = None
    note: str = ""


# Classes with these names are intentionally present in __main__ so joblib can
# load artifacts saved by experiments/43_interpretable_middle_models.py.
@dataclass(frozen=True)
class CandidateSpecForPickle:
    """Compatibility shell for old pickle names when needed."""

    model_family: str
    config_name: str
    display_name: str
    params: dict[str, Any]


@dataclass
class FittedCandidate:
    """Compatibility class for interpretable middle-model artifacts."""

    spec: Any
    estimator: Any
    preprocessor: ColumnTransformer | None
    raw_feature_names: list[str]
    transformed_feature_names: list[str]
    raw_constraints: dict[str, int]
    transformed_constraints: list[int]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for raw input rows."""

        X_model = self.preprocessor.transform(X) if self.preprocessor is not None else X
        probability = self.estimator.predict_proba(X_model)
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 not in classes:
            return np.asarray(probability)[:, -1]
        return np.asarray(probability)[:, classes.index(1)]


@dataclass
class ProbabilityCalibrator:
    """Compatibility class for one-dimensional calibrator artifacts."""

    calibration_type: str
    model: Any | None = None

    def transform(self, probability: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""

        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.calibration_type == "uncalibrated":
            return probability
        if self.calibration_type == "sigmoid":
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            return np.clip(self.model.predict_proba(logits)[:, 1], 0.0, 1.0)
        if self.calibration_type == "isotonic":
            return np.clip(self.model.predict(probability), 0.0, 1.0)
        raise ValueError(f"Unknown calibration_type: {self.calibration_type}")


def setup_logging() -> logging.Logger:
    """Configure capacity-review logging."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("capacity_aware_manual_review")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "capacity_review.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def dataset_specs() -> list[DatasetSpec]:
    """Return dataset specs."""

    return [
        DatasetSpec("taiwan", TAIWAN_MODEL_READY, TARGET_COLUMN, "primary"),
        DatasetSpec("heloc", HELOC_MODEL_READY, HELOC_TARGET, "external_validation"),
    ]


def split_data(spec: DatasetSpec) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Return validation and test splits for a dataset."""

    frame = pd.read_csv(spec.path)
    split = stratified_train_validation_test_split(frame, target_column=spec.target)
    return {
        "validation": (
            split.validation.drop(columns=[spec.target]),
            split.validation[spec.target].astype(int),
        ),
        "heldout_test_diagnostic": (
            split.test.drop(columns=[spec.target]),
            split.test[spec.target].astype(int),
        ),
    }


def positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities from sklearn-like estimators."""

    probability = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        return np.asarray(probability)[:, -1]
    return np.asarray(probability)[:, classes.index(1)]


def load_interpretable_probability(path: Path, X: pd.DataFrame) -> np.ndarray:
    """Load an interpretable artifact dict and return calibrated probability."""

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "artifact" not in payload or "calibrator" not in payload:
        raise ValueError(f"Unexpected interpretable artifact format: {path}")
    raw = payload["artifact"].predict_proba(X)
    return payload["calibrator"].transform(raw)


def calibrated_model_path(dataset: str, model_name: str, preferred: str | None = None) -> Path | None:
    """Return an existing calibrated artifact path for a base model."""

    base = OUTPUTS_DIR / "models" / "calibrated" / dataset
    choices: list[str] = []
    if preferred:
        choices.append(preferred)
    if model_name in {"woe_scorecard_logistic_regression", "logistic_regression", "class_weighted_logistic_regression"}:
        choices.extend(["sigmoid", "isotonic", "uncalibrated"])
    elif model_name == "majority_baseline":
        choices.extend(["uncalibrated", "sigmoid", "isotonic"])
    else:
        choices.extend(["isotonic", "sigmoid", "uncalibrated"])

    for calibration in dict.fromkeys(choices):
        path = base / f"{model_name}_{calibration}.joblib"
        if path.exists():
            return path
    return None


def old_scre_path(dataset: str) -> Path | None:
    """Return old SCRE-Credit model path."""

    path = OUTPUTS_DIR / "models" / f"{dataset}_scre_credit.joblib"
    return path if path.exists() else None


def ensemble_probability(dataset: str, ensemble_path: Path, X: pd.DataFrame, logger: logging.Logger) -> tuple[np.ndarray, str]:
    """Compute SCRE-Pareto/Optimized probabilities from available weighted components."""

    payload = joblib.load(ensemble_path)
    weights = payload.get("selected_weights") or payload.get("ensemble_weights")
    if not weights:
        raise ValueError(f"No ensemble weights found in {ensemble_path}")

    probabilities: list[np.ndarray] = []
    used_weights: list[float] = []
    missing: list[str] = []
    for model_name, weight in weights.items():
        if float(weight) <= 0:
            continue
        if model_name == "SCRE-Credit":
            path = old_scre_path(dataset)
        else:
            path = calibrated_model_path(dataset, model_name)
        if path is None or not path.exists():
            missing.append(model_name)
            continue
        try:
            estimator = joblib.load(path)
            probabilities.append(positive_probability(estimator, X))
            used_weights.append(float(weight))
        except Exception as exc:  # pragma: no cover - logged as missing component
            missing.append(f"{model_name}: {exc}")
            logger.warning("Could not use SCRE component %s from %s: %s", model_name, path, exc)

    if not probabilities or not used_weights:
        raise RuntimeError(f"No usable SCRE components for {ensemble_path}")
    weight_array = np.asarray(used_weights, dtype=float)
    weight_array = weight_array / weight_array.sum()
    matrix = np.vstack(probabilities)
    probability = np.average(matrix, axis=0, weights=weight_array)
    note = "weights renormalized over available components"
    if missing:
        note += "; missing components: " + "; ".join(missing)
    return np.clip(probability, 0.0, 1.0), note


def candidate_specs(dataset: str) -> list[CandidateSpec]:
    """Build candidate list for one dataset using available artifacts."""

    specs: list[CandidateSpec] = []
    calibrated = OUTPUTS_DIR / "models" / "calibrated" / dataset
    interpretable = OUTPUTS_DIR / "final_attempt" / "interpretable_models"

    if dataset == "taiwan":
        v2_model = "catboost"
        v2_cal = "isotonic"
        literature = ("xgboost", "isotonic", "Best literature reproduction: XGBoost no resampling proxy")
        best_interpretable = interpretable / "taiwan_monotonic_lightgbm_conservative_uncalibrated.joblib"
        best_interpretable_name = "Best EBM/monotonic model: Monotonic LightGBM"
    else:
        v2_model = "woe_scorecard_logistic_regression"
        v2_cal = "sigmoid"
        literature = ("catboost", "isotonic", "Best literature reproduction: CatBoost Balanced proxy")
        best_interpretable = interpretable / "heloc_ebm_i10_lr0.01_bins256_ob8_ib4_leaf5_isotonic.joblib"
        best_interpretable_name = "Best EBM/monotonic model: EBM"

    def add_calibrated(label: str, model_name: str, calibration: str | None, group: str, note: str = "") -> None:
        path = calibrated_model_path(dataset, model_name, calibration)
        if path is not None:
            specs.append(CandidateSpec(dataset, label, group, "calibrated_model", path, None, note))

    add_calibrated("V2 final locked policy ranking", v2_model, v2_cal, "v2_locked", "same probability ranker as V2 locked manual-review policy")
    add_calibrated("Best CatBoost probability ranking", "catboost", "isotonic", "catboost")
    add_calibrated("Best Scorecard probability ranking", "woe_scorecard_logistic_regression", "sigmoid", "scorecard")
    lit_model, lit_cal, lit_note = literature
    add_calibrated("Best literature reproduction model", lit_model, lit_cal, "literature_reproduction", lit_note)

    for name in ["scre_optimized", "scre_pareto"]:
        path = OUTPUTS_DIR / "models" / f"{name}_{dataset}.joblib"
        if path.exists():
            label = "SCRE-Optimized probability ranking" if name == "scre_optimized" else "SCRE-Pareto probability ranking"
            specs.append(CandidateSpec(dataset, label, name, "weighted_ensemble", None, path))

    if best_interpretable.exists():
        specs.append(
            CandidateSpec(
                dataset,
                best_interpretable_name,
                "interpretable_middle",
                "interpretable_artifact",
                best_interpretable,
                None,
            )
        )

    # Deep-tabular models were not persisted as probability artifacts in Prompt 3.
    return specs


def candidate_inventory(dataset: str, specs: list[CandidateSpec]) -> pd.DataFrame:
    """Return candidate availability inventory rows."""

    rows = []
    expected = [
        "V2 final locked policy ranking",
        "Best CatBoost probability ranking",
        "Best Scorecard probability ranking",
        "SCRE-Optimized probability ranking",
        "SCRE-Pareto probability ranking",
        "Best literature reproduction model",
        "Best EBM/monotonic model",
        "Best deep model",
    ]
    present = {spec.model for spec in specs}
    for name in expected:
        matched = [spec for spec in specs if spec.model == name or spec.model.startswith(name)]
        if matched:
            spec = matched[0]
            artifact = spec.artifact_path or spec.ensemble_path
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": name,
                    "available": True,
                    "source": spec.source,
                    "artifact_path": str(artifact) if artifact else "",
                    "note": spec.note,
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": name,
                    "available": False,
                    "source": "",
                    "artifact_path": "",
                    "note": "No persisted probability artifact found" if name == "Best deep model" else "No matching artifact found",
                }
            )
    return pd.DataFrame(rows)


def candidate_probability(spec: CandidateSpec, X: pd.DataFrame, logger: logging.Logger) -> tuple[np.ndarray, str]:
    """Generate candidate probabilities for a split."""

    if spec.source == "calibrated_model":
        estimator = joblib.load(spec.artifact_path)
        return positive_probability(estimator, X), spec.note
    if spec.source == "weighted_ensemble":
        return ensemble_probability(spec.dataset, spec.ensemble_path, X, logger)
    if spec.source == "interpretable_artifact":
        return load_interpretable_probability(spec.artifact_path, X), spec.note
    raise ValueError(f"Unknown source: {spec.source}")


def capacity_rows(
    dataset: str,
    split_name: str,
    model: str,
    y_true: pd.Series,
    risk_score: np.ndarray,
    note: str,
) -> list[dict[str, Any]]:
    """Compute Lift@K rows for one model."""

    y = np.asarray(y_true, dtype=int)
    score = np.asarray(risk_score, dtype=float)
    n = len(y)
    total_defaults = int(y.sum())
    base_rate = total_defaults / n if n else np.nan
    order = np.argsort(-score, kind="mergesort")
    rows: list[dict[str, Any]] = []

    for capacity in CAPACITY_LEVELS:
        review_count = int(np.ceil(n * capacity))
        review_count = min(max(review_count, 1), n)
        selected = order[:review_count]
        not_selected = order[review_count:]
        defaults_captured = int(y[selected].sum())
        nondefaults_reviewed = int(review_count - defaults_captured)
        precision_at_k = defaults_captured / review_count if review_count else np.nan
        recall_at_k = defaults_captured / total_defaults if total_defaults else np.nan
        lift = precision_at_k / base_rate if base_rate else np.nan
        expected_random_defaults = float(total_defaults * review_count / n) if n else np.nan
        review_cost = float(REVIEW_COST_PER_CASE * review_count)
        cost_saved_vs_random = float(FN_COST * (defaults_captured - expected_random_defaults))
        cost_saved_vs_no_review = float(FN_COST * defaults_captured - review_cost)
        rows.append(
            {
                "dataset": dataset,
                "split": split_name,
                "model": model,
                "capacity_pct": float(capacity),
                "review_count": review_count,
                "defaults_captured": defaults_captured,
                "total_defaults": total_defaults,
                "default_capture_rate": recall_at_k,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift_at_k": lift,
                "nondefaults_reviewed": nondefaults_reviewed,
                "false_alarm_count": nondefaults_reviewed,
                "cost_saved_vs_random_review": cost_saved_vs_random,
                "cost_saved_vs_no_review": cost_saved_vs_no_review,
                "review_cost": review_cost,
                "net_value": cost_saved_vs_no_review,
                "average_risk_score_reviewed": float(np.mean(score[selected])) if len(selected) else np.nan,
                "average_risk_score_not_reviewed": float(np.mean(score[not_selected])) if len(not_selected) else np.nan,
                "candidate_note": note,
                "test_set_used_for_selection": False,
            }
        )
    return rows


def baseline_rows(dataset: str, split_name: str, y_true: pd.Series) -> list[dict[str, Any]]:
    """Compute random-review expectation and oracle upper-bound rows."""

    y = np.asarray(y_true, dtype=int)
    n = len(y)
    total_defaults = int(y.sum())
    base_rate = total_defaults / n if n else np.nan
    rows: list[dict[str, Any]] = []
    for capacity in CAPACITY_LEVELS:
        review_count = int(np.ceil(n * capacity))
        review_count = min(max(review_count, 1), n)
        random_defaults = float(total_defaults * review_count / n)
        oracle_defaults = int(min(review_count, total_defaults))
        for model, defaults_captured, note in [
            ("Random review baseline", random_defaults, "expected random review capture"),
            ("Oracle upper bound", oracle_defaults, "theoretical maximum if defaults were known"),
        ]:
            precision = defaults_captured / review_count if review_count else np.nan
            recall = defaults_captured / total_defaults if total_defaults else np.nan
            lift = precision / base_rate if base_rate else np.nan
            review_cost = float(REVIEW_COST_PER_CASE * review_count)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "model": model,
                    "capacity_pct": float(capacity),
                    "review_count": review_count,
                    "defaults_captured": defaults_captured,
                    "total_defaults": total_defaults,
                    "default_capture_rate": recall,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "lift_at_k": lift,
                    "nondefaults_reviewed": float(review_count - defaults_captured),
                    "false_alarm_count": float(review_count - defaults_captured),
                    "cost_saved_vs_random_review": float(FN_COST * (defaults_captured - random_defaults)),
                    "cost_saved_vs_no_review": float(FN_COST * defaults_captured - review_cost),
                    "review_cost": review_cost,
                    "net_value": float(FN_COST * defaults_captured - review_cost),
                    "average_risk_score_reviewed": np.nan,
                    "average_risk_score_not_reviewed": np.nan,
                    "candidate_note": note,
                    "test_set_used_for_selection": False,
                }
            )
    return rows


def plot_metric(table: pd.DataFrame, dataset: str, split_name: str, metric: str, ylabel: str, filename: str) -> None:
    """Plot capacity curves for one metric."""

    subset = table[(table["dataset"] == dataset) & (table["split"] == split_name)].copy()
    if subset.empty:
        return
    plt.figure(figsize=(10, 6))
    for model, group in subset.groupby("model"):
        if model == "Oracle upper bound":
            linestyle = "--"
            linewidth = 2.0
        elif model == "Random review baseline":
            linestyle = ":"
            linewidth = 2.0
        else:
            linestyle = "-"
            linewidth = 1.6
        group = group.sort_values("capacity_pct")
        plt.plot(group["capacity_pct"] * 100, group[metric], marker="o", label=model, linestyle=linestyle, linewidth=linewidth)
    plt.xlabel("Manual review capacity (% of portfolio)")
    plt.ylabel(ylabel)
    plt.title(f"{dataset.upper()} {ylabel} ({split_name})")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close()


def compact_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-model comparison at key capacities."""

    rows: list[dict[str, Any]] = []
    for (dataset, split_name, model), group in table.groupby(["dataset", "split", "model"]):
        if model in {"Random review baseline", "Oracle upper bound"}:
            continue
        by_cap = group.set_index("capacity_pct")
        def value(capacity: float, column: str) -> float:
            return float(by_cap.loc[capacity, column]) if capacity in by_cap.index else np.nan
        rows.append(
            {
                "dataset": dataset,
                "split": split_name,
                "model": model,
                "top_5_capture": value(0.05, "default_capture_rate"),
                "top_10_capture": value(0.10, "default_capture_rate"),
                "top_20_capture": value(0.20, "default_capture_rate"),
                "top_30_capture": value(0.30, "default_capture_rate"),
                "precision_at_20": value(0.20, "precision_at_k"),
                "lift_at_20": value(0.20, "lift_at_k"),
                "net_value_at_20": value(0.20, "net_value"),
                "cost_saved_vs_random_at_20": value(0.20, "cost_saved_vs_random_review"),
                "test_set_used_for_selection": False,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values(
        ["dataset", "split", "top_20_capture", "lift_at_20", "net_value_at_20"],
        ascending=[True, True, False, False, False],
    )


def write_summary(table: pd.DataFrame, comparison: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Write markdown summary answering the prompt questions."""

    lines: list[str] = [
        "# Capacity-Aware Manual Review Summary",
        "",
        "## Protocol",
        "- Models are evaluated as manual-review prioritization rankers, not automatic rejection systems.",
        "- Validation and held-out test diagnostic splits are both reported.",
        "- Test rows are not used to lock a final policy; `test_set_used_for_selection=False` is written to output rows.",
        f"- Review workload cost is {REVIEW_COST_PER_CASE} per reviewed customer; each captured default is valued at FN cost {FN_COST}.",
        "- Net value = FN cost saved by captured defaults - review workload cost.",
        "",
    ]

    for dataset in ["taiwan", "heloc"]:
        for split_name in ["validation", "heldout_test_diagnostic"]:
            subset = comparison[(comparison["dataset"] == dataset) & (comparison["split"] == split_name)].copy()
            if subset.empty:
                continue
            top = subset.sort_values(["top_20_capture", "lift_at_20", "net_value_at_20"], ascending=[False, False, False]).head(5)
            lines.extend([f"## {dataset.upper()} {split_name} top review rankers", ""])
            for _, row in top.iterrows():
                lines.append(
                    "- {model}: top10 capture={top10:.3f}, top20 capture={top20:.3f}, "
                    "top30 capture={top30:.3f}, precision@20={p20:.3f}, lift@20={lift20:.2f}, net@20={net20:.1f}".format(
                        model=row["model"],
                        top10=row["top_10_capture"],
                        top20=row["top_20_capture"],
                        top30=row["top_30_capture"],
                        p20=row["precision_at_20"],
                        lift20=row["lift_at_20"],
                        net20=row["net_value_at_20"],
                    )
                )
            lines.append("")

    lines.extend(
        [
            "## Questions",
            "",
            "1. If the riskiest 10% is reviewed, how many defaults are captured?",
            "   - Use `top_10_capture` in `capacity_model_comparison.csv`; this is reported separately for validation and held-out test diagnostic rows.",
            "2. If the riskiest 20% is reviewed, how many defaults are captured?",
            "   - Use `top_20_capture`. This is the most operationally useful capacity point in the summary tables.",
            "3. If the riskiest 30% is reviewed, how many defaults are captured?",
            "   - Use `top_30_capture`; gains should be compared against random review at 30%.",
            "4. How much better than random review is the model?",
            "   - `lift_at_k` and `cost_saved_vs_random_review` show this directly. Random review has lift near 1.0 by construction.",
            "5. CatBoost, Scorecard, or SCRE?",
            "   - This is ranking-focused. CatBoost often remains strong on Taiwan, Scorecard/EBM are competitive on HELOC, and SCRE should be judged by its Lift@K rather than only threshold cost.",
            "6. If manual-review capacity is 20%, does the final model change?",
            "   - Treat this as validation-selection evidence in the next final selector. Held-out test rows are diagnostic only.",
            "7. Plain-language explanation for the instructor:",
            "   - Instead of saying the model rejects applicants, say: 'When we can manually inspect only the highest-risk 20% of cases, the model concentrates a much larger share of actual defaults than random review.'",
            "",
            "## Candidate availability notes",
        ]
    )
    missing = inventory[~inventory["available"]]
    if missing.empty:
        lines.append("- All requested probability artifacts were available.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- {row['dataset']} {row['candidate']}: {row['note']}")

    (OUTPUT_DIR / "capacity_review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run capacity-aware lift analysis."""

    logger = setup_logging()
    all_rows: list[dict[str, Any]] = []
    inventory_rows: list[pd.DataFrame] = []

    for spec in dataset_specs():
        splits = split_data(spec)
        candidates = candidate_specs(spec.name)
        inventory_rows.append(candidate_inventory(spec.name, candidates))
        logger.info("%s: %s candidates available", spec.name, len(candidates))

        for split_name, (X, y) in splits.items():
            all_rows.extend(baseline_rows(spec.name, split_name, y))
            for candidate in candidates:
                try:
                    probability, note = candidate_probability(candidate, X, logger)
                    all_rows.extend(capacity_rows(spec.name, split_name, candidate.model, y, probability, note))
                except Exception as exc:  # pragma: no cover - logged and inventoried
                    logger.exception("Failed candidate %s/%s/%s: %s", spec.name, split_name, candidate.model, exc)
                    all_rows.append(
                        {
                            "dataset": spec.name,
                            "split": split_name,
                            "model": candidate.model,
                            "capacity_pct": np.nan,
                            "review_count": np.nan,
                            "defaults_captured": np.nan,
                            "total_defaults": int(y.sum()),
                            "default_capture_rate": np.nan,
                            "precision_at_k": np.nan,
                            "recall_at_k": np.nan,
                            "lift_at_k": np.nan,
                            "nondefaults_reviewed": np.nan,
                            "false_alarm_count": np.nan,
                            "cost_saved_vs_random_review": np.nan,
                            "cost_saved_vs_no_review": np.nan,
                            "review_cost": np.nan,
                            "net_value": np.nan,
                            "average_risk_score_reviewed": np.nan,
                            "average_risk_score_not_reviewed": np.nan,
                            "candidate_note": f"failed: {exc}",
                            "test_set_used_for_selection": False,
                        }
                    )

    results = pd.DataFrame(all_rows)
    inventory = pd.concat(inventory_rows, ignore_index=True)
    comparison = compact_comparison(results)

    results[results["dataset"] == "taiwan"].to_csv(OUTPUT_DIR / "capacity_lift_results_taiwan.csv", index=False)
    results[results["dataset"] == "heloc"].to_csv(OUTPUT_DIR / "capacity_lift_results_heloc.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "capacity_model_comparison.csv", index=False)
    inventory.to_csv(OUTPUT_DIR / "capacity_candidate_inventory.csv", index=False)

    for dataset in ["taiwan", "heloc"]:
        plot_metric(results, dataset, "heldout_test_diagnostic", "default_capture_rate", "Default capture rate", f"cumulative_gains_{dataset}.png")
        plot_metric(results, dataset, "heldout_test_diagnostic", "lift_at_k", "Lift@K", f"lift_curve_{dataset}.png")
        plot_metric(results, dataset, "heldout_test_diagnostic", "precision_at_k", "Precision@K", f"precision_at_k_{dataset}.png")
        plot_metric(results, dataset, "heldout_test_diagnostic", "net_value", "Net value", f"net_value_at_k_{dataset}.png")

    audit = {
        "capacity_levels": CAPACITY_LEVELS,
        "review_cost_per_case": REVIEW_COST_PER_CASE,
        "fn_cost_saved_per_captured_default": FN_COST,
        "rows": int(len(results)),
        "comparison_rows": int(len(comparison)),
        "test_set_used_for_selection": False,
        "deep_model_included": False,
        "deep_model_note": "Prompt 3 did not persist reusable probability artifacts; deep models are listed as unavailable.",
    }
    (OUTPUT_DIR / "capacity_review_protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_summary(results, comparison, inventory)
    logger.info("Capacity-aware manual-review analysis complete: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
