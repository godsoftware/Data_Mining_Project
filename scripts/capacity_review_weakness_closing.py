"""Capacity-aware review analysis for the weakness-closing phase.

No models are trained and no features are created here. Existing fitted
artifacts are used only to regenerate probability scores. Candidate model and
capacity selection is performed on validation rows; held-out test rows are
written only for validation-locked policies.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT_ROOT / "experiments/44_capacity_aware_manual_review.py"
OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/capacity_review"
CAPACITY_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COST = 0.5
LOCKED_CAPACITIES = [0.20, 0.25]
RANDOM_SEED = 42
V2_BANDS = {
    "taiwan": {"model": "V2 final locked policy ranking", "t_low": 0.14, "t_high": 0.28},
    "heloc": {"model": "V2 final locked policy ranking", "t_low": 0.16, "t_high": 0.39},
}


@dataclass(frozen=True)
class ScoreBundle:
    """Probability scores for one candidate on validation and test."""

    dataset: str
    model: str
    note: str
    validation_y: np.ndarray
    validation_score: np.ndarray
    test_y: np.ndarray
    test_score: np.ndarray


def _load_helper_module() -> Any:
    """Load final-attempt capacity helper functions without executing main."""

    spec = importlib.util.spec_from_file_location("capacity_review_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name in ["FittedCandidate", "ProbabilityCalibrator", "CandidateSpecForPickle", "CandidateSpec"]:
            if hasattr(module, name):
                setattr(main_module, name, getattr(module, name))
    return module


def _logger() -> logging.Logger:
    """Create a local logger for candidate probability loading."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("capacity_review_weakness_closing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(OUT_DIR / "capacity_review_weakness_closing.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely, returning 0 for zero denominators."""

    return float(numerator / denominator) if denominator else 0.0


def _load_scores() -> tuple[list[ScoreBundle], pd.DataFrame]:
    """Load validation/test scores for all available candidates."""

    helper = _load_helper_module()
    logger = _logger()
    bundles: list[ScoreBundle] = []
    inventory_rows: list[dict[str, Any]] = []

    for dataset_spec in helper.dataset_specs():
        splits = helper.split_data(dataset_spec)
        candidates = helper.candidate_specs(dataset_spec.name)
        validation_x, validation_y = splits["validation"]
        test_x, test_y = splits["heldout_test_diagnostic"]
        for candidate in candidates:
            try:
                validation_score, validation_note = helper.candidate_probability(candidate, validation_x, logger)
                test_score, test_note = helper.candidate_probability(candidate, test_x, logger)
                bundles.append(
                    ScoreBundle(
                        dataset=dataset_spec.name,
                        model=candidate.model,
                        note="; ".join(sorted(set([str(validation_note), str(test_note)]))).strip("; "),
                        validation_y=np.asarray(validation_y, dtype=int),
                        validation_score=np.asarray(validation_score, dtype=float),
                        test_y=np.asarray(test_y, dtype=int),
                        test_score=np.asarray(test_score, dtype=float),
                    )
                )
                inventory_rows.append(
                    {
                        "dataset": dataset_spec.name,
                        "model": candidate.model,
                        "available": True,
                        "source": candidate.source,
                        "artifact_path": str(candidate.artifact_path or candidate.ensemble_path),
                        "note": validation_note,
                    }
                )
            except Exception as exc:
                logger.exception("Failed to score %s/%s", dataset_spec.name, candidate.model)
                inventory_rows.append(
                    {
                        "dataset": dataset_spec.name,
                        "model": candidate.model,
                        "available": False,
                        "source": candidate.source,
                        "artifact_path": str(candidate.artifact_path or candidate.ensemble_path),
                        "note": f"failed: {exc}",
                    }
                )
    return bundles, pd.DataFrame(inventory_rows)


def _random_cost(y: np.ndarray, review_count: int) -> float:
    """Expected top-K review cost under random review."""

    total_defaults = int(np.sum(y))
    expected_captured = total_defaults * review_count / len(y)
    low_defaults = total_defaults - expected_captured
    return low_defaults * FN_COST + review_count * REVIEW_COST


def _oracle_capture(y: np.ndarray, review_count: int) -> int:
    """Oracle upper-bound default capture at capacity K."""

    return int(min(review_count, int(np.sum(y))))


def _top_k_metrics(
    dataset: str,
    model: str,
    split: str,
    y: np.ndarray,
    score: np.ndarray,
    capacity_pct: float,
    v2_reference_cost: float,
    selection_role: str,
    note: str,
) -> dict[str, Any]:
    """Compute top-K review prioritization metrics."""

    n = len(y)
    review_count = int(np.ceil(n * capacity_pct))
    order = np.argsort(-score, kind="mergesort")
    selected = order[:review_count]
    not_selected = order[review_count:]
    total_defaults = int(np.sum(y))
    defaults_captured = int(np.sum(y[selected]))
    nondefaults_reviewed = int(review_count - defaults_captured)
    low_defaults = int(np.sum(y[not_selected]))
    auto_count = n - review_count
    precision = _safe_divide(defaults_captured, review_count)
    capture = _safe_divide(defaults_captured, total_defaults)
    base_rate = _safe_divide(total_defaults, n)
    lift = _safe_divide(precision, base_rate)
    review_adjusted_cost = low_defaults * FN_COST + review_count * REVIEW_COST
    random_expected_defaults = total_defaults * review_count / n
    random_cost = _random_cost(y, review_count)
    oracle_defaults = _oracle_capture(y, review_count)
    oracle_capture = _safe_divide(oracle_defaults, total_defaults)
    return {
        "dataset": dataset,
        "split": split,
        "model": model,
        "policy_type": "top_k_review",
        "capacity_pct": capacity_pct,
        "review_count": review_count,
        "total_defaults": total_defaults,
        "defaults_captured_in_review": defaults_captured,
        "default_capture_rate": capture,
        "precision_at_k": precision,
        "recall_at_k": capture,
        "lift_at_k": lift,
        "nondefaults_reviewed": nondefaults_reviewed,
        "false_alarm_count": nondefaults_reviewed,
        "manual_review_rate": _safe_divide(review_count, n),
        "auto_decision_rate": _safe_divide(auto_count, n),
        "low_risk_default_count": low_defaults,
        "low_risk_default_rate": _safe_divide(low_defaults, auto_count),
        "high_risk_count": 0,
        "high_risk_precision": np.nan,
        "t_low": np.nan,
        "t_high": np.nan,
        "review_adjusted_cost": review_adjusted_cost,
        "random_expected_defaults_captured": random_expected_defaults,
        "oracle_defaults_captured": oracle_defaults,
        "oracle_capture_rate": oracle_capture,
        "net_value_vs_random": random_cost - review_adjusted_cost,
        "net_value_vs_v2": v2_reference_cost - review_adjusted_cost,
        "selection_role": selection_role,
        "test_set_used_for_selection": False,
        "candidate_note": note,
        "comment": "Top-K review only; no automatic high-risk rejection bucket.",
    }


def _band_counts(y: np.ndarray, score: np.ndarray, t_low: float, t_high: float) -> dict[str, int]:
    """Return low/manual/high decision counts for a score band."""

    low = score < t_low
    high = score >= t_high
    manual = (~low) & (~high)
    default = y == 1
    nondefault = ~default
    return {
        "low_default": int(np.sum(low & default)),
        "low_nondefault": int(np.sum(low & nondefault)),
        "manual_default": int(np.sum(manual & default)),
        "manual_nondefault": int(np.sum(manual & nondefault)),
        "high_default": int(np.sum(high & default)),
        "high_nondefault": int(np.sum(high & nondefault)),
        "manual_count": int(np.sum(manual)),
        "high_count": int(np.sum(high)),
        "low_count": int(np.sum(low)),
    }


def _band_cost_from_counts(counts: dict[str, int]) -> float:
    """Compute review-adjusted cost for a three-bucket band."""

    return counts["high_nondefault"] * FP_COST + counts["low_default"] * FN_COST + counts["manual_count"] * REVIEW_COST


def _select_capacity_band(y: np.ndarray, score: np.ndarray, capacity_pct: float) -> tuple[float, float, dict[str, int]]:
    """Select a validation capacity band with exactly K% manual-review load."""

    n = len(y)
    review_count = int(np.ceil(n * capacity_pct))
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    best: tuple[float, float, dict[str, int], tuple[float, int, int, int]] | None = None
    for start in range(0, n - review_count + 1):
        end = start + review_count
        t_low = 0.0 if start == 0 else float(sorted_score[start])
        t_high = 1.0 + 1e-9 if end >= n else float(sorted_score[end])
        if t_low >= t_high:
            continue
        counts = _band_counts(y, score, t_low, t_high)
        if counts["manual_count"] == 0:
            continue
        sort_key = (
            _band_cost_from_counts(counts),
            counts["low_default"],
            counts["high_nondefault"],
            -counts["manual_default"],
        )
        if best is None or sort_key < best[3]:
            best = (t_low, t_high, counts, sort_key)
    if best is None:
        raise RuntimeError("No capacity band could be selected.")
    return best[0], best[1], best[2]


def _capacity_band_metrics(
    dataset: str,
    model: str,
    split: str,
    y: np.ndarray,
    score: np.ndarray,
    capacity_pct: float,
    t_low: float,
    t_high: float,
    v2_reference_cost: float,
    selection_role: str,
    note: str,
) -> dict[str, Any]:
    """Compute fixed capacity-band metrics."""

    n = len(y)
    total_defaults = int(np.sum(y))
    counts = _band_counts(y, score, t_low, t_high)
    review_count = counts["manual_count"]
    review_defaults = counts["manual_default"]
    review_nondefaults = counts["manual_nondefault"]
    low_count = counts["low_count"]
    high_count = counts["high_count"]
    high_total = counts["high_default"] + counts["high_nondefault"]
    precision = _safe_divide(review_defaults, review_count)
    capture = _safe_divide(review_defaults, total_defaults)
    base_rate = _safe_divide(total_defaults, n)
    lift = _safe_divide(precision, base_rate)
    review_adjusted_cost = _band_cost_from_counts(counts)
    random_cost = _random_cost(y, review_count)
    oracle_defaults = _oracle_capture(y, review_count)
    return {
        "dataset": dataset,
        "split": split,
        "model": model,
        "policy_type": "three_bucket_capacity_band",
        "capacity_pct": capacity_pct,
        "review_count": review_count,
        "total_defaults": total_defaults,
        "defaults_captured_in_review": review_defaults,
        "default_capture_rate": capture,
        "precision_at_k": precision,
        "recall_at_k": capture,
        "lift_at_k": lift,
        "nondefaults_reviewed": review_nondefaults,
        "false_alarm_count": review_nondefaults + counts["high_nondefault"],
        "manual_review_rate": _safe_divide(review_count, n),
        "auto_decision_rate": _safe_divide(n - review_count, n),
        "low_risk_default_count": counts["low_default"],
        "low_risk_default_rate": _safe_divide(counts["low_default"], low_count),
        "high_risk_count": high_count,
        "high_risk_precision": _safe_divide(counts["high_default"], high_total),
        "t_low": t_low,
        "t_high": t_high,
        "review_adjusted_cost": review_adjusted_cost,
        "random_expected_defaults_captured": total_defaults * review_count / n,
        "oracle_defaults_captured": oracle_defaults,
        "oracle_capture_rate": _safe_divide(oracle_defaults, total_defaults),
        "net_value_vs_random": random_cost - review_adjusted_cost,
        "net_value_vs_v2": v2_reference_cost - review_adjusted_cost,
        "selection_role": selection_role,
        "test_set_used_for_selection": False,
        "candidate_note": note,
        "comment": "Three-bucket band; high-risk bucket cost included separately from manual review.",
    }


def _v2_reference_cost(dataset: str, bundles_by_dataset_model: dict[tuple[str, str], ScoreBundle], split: str) -> float:
    """Compute V2 review-adjusted cost on validation or test."""

    spec = V2_BANDS[dataset]
    bundle = bundles_by_dataset_model[(dataset, spec["model"])]
    y = bundle.validation_y if split == "validation" else bundle.test_y
    score = bundle.validation_score if split == "validation" else bundle.test_score
    counts = _band_counts(y, score, spec["t_low"], spec["t_high"])
    return _band_cost_from_counts(counts)


def _build_validation_rows(
    bundles: list[ScoreBundle],
) -> tuple[pd.DataFrame, dict[tuple[str, str, float, str], tuple[float, float]]]:
    """Compute validation rows and store selected band thresholds."""

    by_key = {(b.dataset, b.model): b for b in bundles}
    threshold_lookup: dict[tuple[str, str, float, str], tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        v2_cost = _v2_reference_cost(bundle.dataset, by_key, "validation")
        for capacity in CAPACITY_LEVELS:
            rows.append(
                _top_k_metrics(
                    bundle.dataset,
                    bundle.model,
                    "validation",
                    bundle.validation_y,
                    bundle.validation_score,
                    capacity,
                    v2_cost,
                    "validation_candidate",
                    bundle.note,
                )
            )
            t_low, t_high, _ = _select_capacity_band(bundle.validation_y, bundle.validation_score, capacity)
            threshold_lookup[(bundle.dataset, bundle.model, capacity, "three_bucket_capacity_band")] = (t_low, t_high)
            rows.append(
                _capacity_band_metrics(
                    bundle.dataset,
                    bundle.model,
                    "validation",
                    bundle.validation_y,
                    bundle.validation_score,
                    capacity,
                    t_low,
                    t_high,
                    v2_cost,
                    "validation_candidate",
                    bundle.note,
                )
            )
    return pd.DataFrame(rows), threshold_lookup


def _select_locked(validation_rows: pd.DataFrame) -> pd.DataFrame:
    """Select locked test policies using validation evidence only."""

    locked_rows = []
    for dataset in ["taiwan", "heloc"]:
        subset = validation_rows[
            (validation_rows["dataset"] == dataset)
            & (validation_rows["capacity_pct"].isin(LOCKED_CAPACITIES))
        ].copy()
        for capacity in LOCKED_CAPACITIES:
            cap_subset = subset[subset["capacity_pct"] == capacity].copy()
            for policy_type in ["top_k_review", "three_bucket_capacity_band"]:
                policy_subset = cap_subset[cap_subset["policy_type"] == policy_type].copy()
                if policy_subset.empty:
                    continue
                policy_subset = policy_subset.sort_values(
                    [
                        "net_value_vs_random",
                        "default_capture_rate",
                        "precision_at_k",
                        "review_adjusted_cost",
                    ],
                    ascending=[False, False, False, True],
                )
                locked_rows.append(policy_subset.iloc[0])
    locked = pd.DataFrame(locked_rows).reset_index(drop=True)
    locked["selection_role"] = "locked_by_validation"
    return locked


def _evaluate_locked_test(
    locked_validation: pd.DataFrame,
    bundles: list[ScoreBundle],
    threshold_lookup: dict[tuple[str, str, float, str], tuple[float, float]],
) -> pd.DataFrame:
    """Evaluate validation-locked policies on held-out test."""

    by_key = {(b.dataset, b.model): b for b in bundles}
    rows = []
    for _, locked in locked_validation.iterrows():
        dataset = locked["dataset"]
        model = locked["model"]
        capacity = float(locked["capacity_pct"])
        policy = locked["policy_type"]
        bundle = by_key[(dataset, model)]
        v2_cost = _v2_reference_cost(dataset, by_key, "test")
        if policy == "top_k_review":
            row = _top_k_metrics(
                dataset,
                model,
                "locked_test",
                bundle.test_y,
                bundle.test_score,
                capacity,
                v2_cost,
                "locked_test_evaluation",
                bundle.note,
            )
        else:
            t_low, t_high = threshold_lookup[(dataset, model, capacity, policy)]
            row = _capacity_band_metrics(
                dataset,
                model,
                "locked_test",
                bundle.test_y,
                bundle.test_score,
                capacity,
                t_low,
                t_high,
                v2_cost,
                "locked_test_evaluation",
                bundle.note,
            )
        row["locked_from_validation_model"] = model
        row["locked_from_validation_policy_type"] = policy
        row["locked_from_validation_capacity_pct"] = capacity
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(table: pd.DataFrame, dataset: str, metric: str, output_name: str, ylabel: str) -> None:
    """Plot validation top-K capacity curves."""

    subset = table[
        (table["dataset"] == dataset)
        & (table["split"] == "validation")
        & (table["policy_type"] == "top_k_review")
    ].copy()
    plt.figure(figsize=(10, 6))
    for model, group in subset.groupby("model"):
        group = group.sort_values("capacity_pct")
        plt.plot(group["capacity_pct"] * 100, group[metric], marker="o", linewidth=1.4, label=model)
    plt.xlabel("Review capacity (% of portfolio)")
    plt.ylabel(ylabel)
    plt.title(f"{dataset.upper()} validation {ylabel}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / output_name, dpi=180)
    plt.close()


def _comparison_table(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Create compact model comparison at key capacities."""

    validation = all_rows[(all_rows["split"] == "validation") & (all_rows["policy_type"] == "top_k_review")].copy()
    rows = []
    for (dataset, model), group in validation.groupby(["dataset", "model"]):
        by_cap = group.set_index("capacity_pct")
        def val(cap: float, col: str) -> float:
            return float(by_cap.loc[cap, col]) if cap in by_cap.index else np.nan
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "top_5_capture": val(0.05, "default_capture_rate"),
                "top_10_capture": val(0.10, "default_capture_rate"),
                "top_15_capture": val(0.15, "default_capture_rate"),
                "top_20_capture": val(0.20, "default_capture_rate"),
                "top_25_capture": val(0.25, "default_capture_rate"),
                "top_30_capture": val(0.30, "default_capture_rate"),
                "precision_at_20": val(0.20, "precision_at_k"),
                "lift_at_20": val(0.20, "lift_at_k"),
                "net_value_vs_random_at_20": val(0.20, "net_value_vs_random"),
                "test_set_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "top_20_capture", "lift_at_20"], ascending=[True, False, False])


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render a small dataframe as Markdown without optional dependencies."""

    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *body])


def _write_summary(all_rows: pd.DataFrame, comparison: pd.DataFrame, locked_test: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Write the capacity review summary."""

    lines = [
        "# Capacity-Aware Review Weakness Closing",
        "",
        "## Protocol",
        "- Existing fitted models/probability rankers only.",
        "- All model/capacity candidates are computed on validation.",
        "- Held-out test is evaluated only for policies locked from validation.",
        "- No new model training, feature engineering, threshold learning on test, or resampling.",
        "",
    ]
    for dataset in ["taiwan", "heloc"]:
        val_top20 = comparison[comparison["dataset"] == dataset].sort_values(
            ["top_20_capture", "lift_at_20", "precision_at_20"], ascending=[False, False, False]
        ).head(5)
        lines.extend([f"## {dataset.upper()} validation top-20 ranking", ""])
        lines.append(
            _markdown_table(
                val_top20,
                ["model", "top_10_capture", "top_20_capture", "top_25_capture", "top_30_capture", "precision_at_20", "lift_at_20"],
            )
        )
        lines.append("")

        test_rows = locked_test[locked_test["dataset"] == dataset].sort_values(
            ["capacity_pct", "policy_type", "net_value_vs_random"], ascending=[True, True, False]
        )
        lines.extend([f"## {dataset.upper()} locked-test capacity policies", ""])
        lines.append(
            _markdown_table(
                test_rows,
                [
                    "model",
                    "policy_type",
                    "capacity_pct",
                    "default_capture_rate",
                    "precision_at_k",
                    "lift_at_k",
                    "low_risk_default_rate",
                    "review_adjusted_cost",
                ],
            )
        )
        lines.append("")

    missing = inventory[~inventory["available"]]
    lines.extend(
        [
            "## Answers",
            "",
            "1. Taiwan at 20% capacity: use the validation top-20 table and locked-test rows above; it is a workload-reduction candidate, not an automatic V2 replacement.",
            "2. Taiwan at 25% capacity is closer to the V2 29.5% review load and should be preferred if recall loss at 20% is too high.",
            "3. HELOC at 20% capacity is plausible only if lower review workload is more important than maximum default capture.",
            "4. SCRE ranking is useful only if its validation Lift@K exceeds CatBoost/Scorecard at the same K; check `capacity_model_comparison.csv`.",
            "5. Capacity policy can support or replace the V2 band only after a final audit; for now it is a V3 candidate / appendix evidence.",
            "6. Operational recommendation: discuss 20% and 25% capacity as lower-workload screening alternatives, not automatic rejection policies.",
            "",
            "## Availability notes",
        ]
    )
    if missing.empty:
        lines.append("- All requested non-deep candidate artifacts loaded successfully.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- {row['dataset']} {row['model']}: {row['note']}")
    (OUT_DIR / "capacity_review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the capacity-aware weakness-closing analysis."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles, inventory = _load_scores()
    validation_rows, threshold_lookup = _build_validation_rows(bundles)
    locked_validation = _select_locked(validation_rows)
    locked_test = _evaluate_locked_test(locked_validation, bundles, threshold_lookup)
    all_rows = pd.concat([validation_rows, locked_test], ignore_index=True)
    comparison = _comparison_table(all_rows)

    all_rows[all_rows["dataset"] == "taiwan"].to_csv(OUT_DIR / "capacity_review_taiwan.csv", index=False)
    all_rows[all_rows["dataset"] == "heloc"].to_csv(OUT_DIR / "capacity_review_heloc.csv", index=False)
    comparison.to_csv(OUT_DIR / "capacity_model_comparison.csv", index=False)
    inventory.to_csv(OUT_DIR / "capacity_candidate_inventory.csv", index=False)

    _plot_curves(all_rows, "taiwan", "default_capture_rate", "cumulative_gains_taiwan.png", "Default capture rate")
    _plot_curves(all_rows, "heloc", "default_capture_rate", "cumulative_gains_heloc.png", "Default capture rate")
    _plot_curves(all_rows, "taiwan", "lift_at_k", "lift_curve_taiwan.png", "Lift@K")
    _plot_curves(all_rows, "heloc", "lift_at_k", "lift_curve_heloc.png", "Lift@K")
    _write_summary(all_rows, comparison, locked_test, inventory)


if __name__ == "__main__":
    main()
