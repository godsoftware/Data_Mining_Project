"""SCRE review-prioritization strengthening analysis.

This script does not train models, create features, change thresholds, or claim
SCRE as a dominant classifier. It reuses existing probability artifacts and
already-computed capacity/stability/faithfulness outputs to position SCRE as a
reliability-aware review-prioritization framework.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HELPER_PATH = PROJECT_ROOT / "experiments/44_capacity_aware_manual_review.py"
OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/scre_prioritization"
TARGETS = {"taiwan": "default_next_month", "heloc": "bad_flag"}
DATA_PATHS = {
    "taiwan": PROJECT_ROOT / "data/processed/taiwan_model_ready.csv",
    "heloc": PROJECT_ROOT / "data/processed/heloc_model_ready.csv",
}
RANDOM_SEED = 42
CAPACITY_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _load_helper_module() -> Any:
    """Load helper module used by SCRE capacity scripts."""

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


def _split_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical validation and locked-test splits."""

    target = TARGETS[dataset]
    frame = pd.read_csv(DATA_PATHS[dataset])
    train_validation, test = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame[target],
        random_state=RANDOM_SEED,
    )
    _, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[target],
        random_state=RANDOM_SEED,
    )
    return validation.copy(), test.copy()


def _positive_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    probability = estimator.predict_proba(x)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        return np.asarray(probability)[:, -1]
    return np.asarray(probability)[:, classes.index(1)]


def _load_base_score(dataset: str, frame: pd.DataFrame, model_name: str, calibration: str) -> np.ndarray:
    target = TARGETS[dataset]
    path = PROJECT_ROOT / "outputs/models/calibrated" / dataset / f"{model_name}_{calibration}.joblib"
    estimator = joblib.load(path)
    return _positive_probability(estimator, frame.drop(columns=[target]))


def _load_scre_score(dataset: str, frame: pd.DataFrame, variant: str, helper: Any) -> np.ndarray:
    target = TARGETS[dataset]
    path = PROJECT_ROOT / "outputs/models" / f"{variant}_{dataset}.joblib"
    probability, _ = helper.ensemble_probability(
        dataset,
        path,
        frame.drop(columns=[target]),
        helper.logging.getLogger("scre_prioritization"),
    )
    return probability


def _capacity_topk_from_existing() -> dict[str, pd.DataFrame]:
    """Extract SCRE Top-K review rows from prior capacity analysis."""

    outputs: dict[str, pd.DataFrame] = {}
    for dataset in ["taiwan", "heloc"]:
        path = PROJECT_ROOT / "outputs/final_weakness_closing/capacity_review" / f"capacity_review_{dataset}.csv"
        frame = pd.read_csv(path)
        subset = frame[
            frame["model"].str.contains("SCRE", case=False, na=False)
            & (frame["policy_type"] == "top_k_review")
            & (frame["capacity_pct"].round(2).isin(CAPACITY_LEVELS))
        ].copy()
        subset["scre_version"] = np.where(
            subset["model"].str.contains("Optimized", case=False, na=False),
            "SCRE-Optimized",
            "SCRE-Pareto",
        )
        subset = subset.rename(
            columns={
                "net_value_vs_random": "cost_saved_vs_random",
                "review_adjusted_cost": "net_value",
            }
        )
        subset["analysis_role"] = "review_prioritization_not_final_classifier"
        subset["test_set_used_for_selection"] = False
        outputs[dataset] = subset[
            [
                "dataset",
                "split",
                "scre_version",
                "model",
                "capacity_pct",
                "review_count",
                "total_defaults",
                "defaults_captured_in_review",
                "default_capture_rate",
                "precision_at_k",
                "recall_at_k",
                "lift_at_k",
                "cost_saved_vs_random",
                "net_value",
                "manual_review_rate",
                "auto_decision_rate",
                "test_set_used_for_selection",
                "analysis_role",
            ]
        ].sort_values(["dataset", "split", "scre_version", "capacity_pct"])
    return outputs


def _manual_review_score_map(topk: dict[str, pd.DataFrame]) -> dict[tuple[str, str], float]:
    """Use validation Capture@20 as a transparent manual-review score proxy."""

    result: dict[tuple[str, str], float] = {}
    for dataset, frame in topk.items():
        subset = frame[(frame["split"] == "validation") & (np.isclose(frame["capacity_pct"], 0.20))]
        for row in subset.itertuples(index=False):
            result[(dataset, row.scre_version)] = float(row.default_capture_rate)
    return result


def _score_decomposition(topk: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create SCRE reliability score decomposition from existing weights."""

    path = PROJECT_ROOT / "outputs/tables/scre_revision_weights.csv"
    weights = pd.read_csv(path)
    manual_scores = _manual_review_score_map(topk)
    rows: list[dict[str, Any]] = []
    for row in weights.itertuples(index=False):
        performance_values = [
            getattr(row, "pr_auc_norm", np.nan),
            getattr(row, "roc_auc_norm", np.nan),
            getattr(row, "recall_norm", np.nan),
        ]
        performance_score = float(np.nanmean(performance_values))
        manual_score = manual_scores.get((row.dataset, row.scre_variant), np.nan)
        stability_source = getattr(row, "stability_metric_source", "")
        faithfulness_source = getattr(row, "faithfulness_metric_source", "")
        rows.append(
            {
                "dataset": row.dataset,
                "scre_version": row.scre_variant,
                "component_model": row.model,
                "performance_score": performance_score,
                "calibration_score": getattr(row, "calibration_error_norm", np.nan),
                "cost_score": getattr(row, "expected_cost_norm", np.nan),
                "stability_score": getattr(row, "stability_score_norm", np.nan),
                "faithfulness_score": getattr(row, "faithfulness_score_norm", np.nan),
                "manual_review_score": manual_score,
                "overall_scre_score": getattr(row, "reliability_score", np.nan),
                "ensemble_weight": getattr(row, "ensemble_weight", np.nan),
                "weight_selection_split": getattr(row, "weight_selection_split", "validation"),
                "test_set_used_for_weight_optimization": getattr(row, "test_set_used_for_weight_optimization", False),
                "stability_metric_source": stability_source,
                "faithfulness_metric_source": faithfulness_source,
                "decomposition_note": (
                    "Performance/calibration/cost scores come from existing SCRE weights. "
                    "Manual-review score is validation Capture@20. Stability/faithfulness "
                    "may be placeholder values if marked as neutral_placeholder in sources."
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_overlap(a: np.ndarray, b: np.ndarray, pct: float) -> float:
    """Top-pct overlap between two score vectors."""

    k = max(1, int(np.ceil(len(a) * pct)))
    a_idx = set(np.argsort(-a, kind="mergesort")[:k])
    b_idx = set(np.argsort(-b, kind="mergesort")[:k])
    return len(a_idx & b_idx) / k


def _rank_stability() -> pd.DataFrame:
    """Compute variant-level rank agreement for existing probability rankers."""

    helper = _load_helper_module()
    rows: list[dict[str, Any]] = []
    for dataset in ["taiwan", "heloc"]:
        validation, test = _split_dataset(dataset)
        for split, frame in [("validation", validation), ("locked_test", test)]:
            scores = {
                "SCRE-Optimized": _load_scre_score(dataset, frame, "scre_optimized", helper),
                "SCRE-Pareto": _load_scre_score(dataset, frame, "scre_pareto", helper),
                "CatBoost benchmark": _load_base_score(dataset, frame, "catboost", "isotonic"),
                "Scorecard benchmark": _load_base_score(dataset, frame, "woe_scorecard_logistic_regression", "sigmoid"),
            }
            comparisons = [
                ("SCRE-Optimized", "SCRE-Pareto"),
                ("SCRE-Optimized", "CatBoost benchmark"),
                ("SCRE-Optimized", "Scorecard benchmark"),
                ("SCRE-Pareto", "CatBoost benchmark"),
                ("SCRE-Pareto", "Scorecard benchmark"),
            ]
            for left, right in comparisons:
                left_score = scores[left]
                right_score = scores[right]
                tau = kendalltau(left_score, right_score).statistic
                spear = spearmanr(left_score, right_score).statistic
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "ranker_a": left,
                        "ranker_b": right,
                        "top10_overlap": _rank_overlap(left_score, right_score, 0.10),
                        "top20_overlap": _rank_overlap(left_score, right_score, 0.20),
                        "kendall_tau": float(tau),
                        "spearman_corr": float(spear),
                        "n_observations": len(left_score),
                        "stability_scope": "ranker_variant_agreement",
                        "seed_level_retraining": False,
                        "note": "Agreement between existing rankers, not a new seed/fold retraining experiment.",
                    }
                )
    return pd.DataFrame(rows)


def _append_external_stability(rank_stability: pd.DataFrame) -> pd.DataFrame:
    """Append existing SHAP stability references as separate evidence rows."""

    rows = []
    for dataset in ["taiwan", "heloc"]:
        path = PROJECT_ROOT / "outputs/tables" / f"kendalls_w_stability_{dataset}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        subset = frame[frame["feature_subset"].isin(["top_5_features", "top_10_features"])]
        for row in subset.itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "split": "not_split_specific",
                    "ranker_a": f"SHAP stability source: {row.model}",
                    "ranker_b": row.feature_subset,
                    "top10_overlap": np.nan,
                    "top20_overlap": np.nan,
                    "kendall_tau": np.nan,
                    "spearman_corr": np.nan,
                    "n_observations": getattr(row, "n_seeds", np.nan),
                    "kendalls_w": getattr(row, "kendalls_w", np.nan),
                    "stability_scope": "existing_shap_seed_stability_reference",
                    "seed_level_retraining": True,
                    "note": getattr(row, "limitation_note", ""),
                }
            )
    if rows:
        appended = pd.concat([rank_stability, pd.DataFrame(rows)], ignore_index=True, sort=False)
        return appended
    rank_stability["kendalls_w"] = np.nan
    return rank_stability


def _format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _summary(topk: dict[str, pd.DataFrame], decomposition: pd.DataFrame, rank_stability: pd.DataFrame) -> None:
    """Write role summary and safe-claim files."""

    def topk_table(dataset: str) -> str:
        frame = topk[dataset]
        subset = frame[(frame["split"] == "validation") & (frame["scre_version"].isin(["SCRE-Optimized", "SCRE-Pareto"]))]
        rows = []
        for version, group in subset.groupby("scre_version"):
            def metric(cap: float, col: str) -> float:
                found = group[np.isclose(group["capacity_pct"], cap)]
                return float(found[col].iloc[0]) if not found.empty else np.nan
            rows.append(
                {
                    "Dataset": dataset,
                    "SCRE Version": version,
                    "Capture@10": _format_float(metric(0.10, "default_capture_rate")),
                    "Capture@20": _format_float(metric(0.20, "default_capture_rate")),
                    "Precision@20": _format_float(metric(0.20, "precision_at_k")),
                    "Lift@20": _format_float(metric(0.20, "lift_at_k")),
                    "Comment": "Useful as review ranking evidence, not automatic rejection.",
                }
            )
        df = pd.DataFrame(rows).sort_values(["Dataset", "SCRE Version"])
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
        return "\n".join([header, sep, *body])

    def top20_comment(dataset: str) -> str:
        frame = topk[dataset]
        val = frame[(frame["split"] == "validation") & (frame["scre_version"] == "SCRE-Optimized") & np.isclose(frame["capacity_pct"], 0.20)]
        test = frame[(frame["split"] == "locked_test") & (frame["scre_version"] == "SCRE-Optimized") & np.isclose(frame["capacity_pct"], 0.20)]
        if val.empty:
            return "No SCRE-Optimized top-20 evidence found."
        return (
            f"validation Capture@20={_format_float(val['default_capture_rate'].iloc[0])}, "
            f"Precision@20={_format_float(val['precision_at_k'].iloc[0])}, "
            f"Lift@20={_format_float(val['lift_at_k'].iloc[0])}; "
            f"locked-test Capture@20={_format_float(test['default_capture_rate'].iloc[0]) if not test.empty else 'NA'}."
        )

    high_weight = (
        decomposition.sort_values(["dataset", "scre_version", "ensemble_weight"], ascending=[True, True, False])
        .groupby(["dataset", "scre_version"])
        .head(2)
    )
    high_weight_text = "; ".join(
        f"{row.dataset}/{row.scre_version}: {row.component_model}={row.ensemble_weight:.3f}"
        for row in high_weight.itertuples(index=False)
    )

    strongest_overlap = (
        rank_stability[
            (rank_stability["split"] == "validation")
            & (rank_stability["ranker_a"] == "SCRE-Optimized")
            & (rank_stability["ranker_b"] == "SCRE-Pareto")
        ][["dataset", "top10_overlap", "top20_overlap", "kendall_tau", "spearman_corr"]]
    )
    overlap_text = "; ".join(
        f"{row.dataset}: top10={row.top10_overlap:.3f}, top20={row.top20_overlap:.3f}, tau={row.kendall_tau:.3f}, spearman={row.spearman_corr:.3f}"
        for row in strongest_overlap.itertuples(index=False)
    )

    role = f"""# SCRE Review-Prioritization Role Summary

## Protocol
- SCRE is not promoted as the final classifier.
- No new model training, feature creation, or test-set policy selection was performed.
- Evidence is limited to existing SCRE probabilities, capacity-review outputs, SCRE weights, and existing stability/faithfulness artifacts.

## Top-K review prioritization

### Taiwan
{topk_table('taiwan')}

### HELOC
{topk_table('heloc')}

## Reliability score decomposition
- SCRE weight/decomposition rows are stored in `scre_score_decomposition.csv`.
- Highest-weight components by dataset/version: {high_weight_text}.
- Stability/faithfulness component values in some SCRE weight tables are neutral placeholders where explicitly marked; they should not be overclaimed as direct SCRE-specific SHAP stability.

## Rank stability
- SCRE-Optimized vs SCRE-Pareto validation rank agreement: {overlap_text}.
- `scre_rank_stability.csv` also includes existing SHAP Kendall's W rows as reference evidence. Those rows are model-family stability references, not proof that the SCRE ensemble itself was retrained over seeds.

## Questions
1. SCRE final classifier mı? **NO.** V2 CatBoost/Scorecard policies remain final operational evidence.
2. SCRE review prioritization için işe yarıyor mu? **YES.** It provides useful top-k ranking evidence, especially for review-capacity discussion.
3. SCRE top-20% default capture değerinde iyi mi? **YES as ranking support.** Taiwan: {top20_comment('taiwan')} HELOC: {top20_comment('heloc')}
4. SCRE calibration/stability/faithfulness açısından ne katıyor? It gives a structured reliability framework that combines performance, calibration, cost, and available stability/faithfulness evidence, but not all stability/faithfulness values are direct SCRE-specific measurements.
5. SCRE hangi claim ile sunulmalı? SCRE should be presented as a reliability-aware framework for model comparison, weighted integration, and review prioritization.
6. SCRE hangi claim ile sunulmamalı? It should not be claimed as a universally superior classifier or automatic rejection model.
"""
    (OUT_DIR / "scre_role_summary.md").write_text(role, encoding="utf-8")

    safe = """# SCRE Safe Claims

## Safe claims
- SCRE-Credit is best positioned as a reliability-aware review-prioritization and model-comparison framework.
- SCRE-Optimized and SCRE-Pareto provide useful Top-K review ranking evidence under capacity constraints.
- SCRE can integrate performance, calibration, cost, and available explanation-reliability evidence into a structured audit table.
- SCRE supports screening/manual-review prioritization; it is not an automatic rejection system.

## Unsafe claims
- Do not claim SCRE-Credit outperforms all individual models.
- Do not claim SCRE-Credit is the final operational classifier on Taiwan or HELOC.
- Do not claim SCRE-specific SHAP stability unless the table directly measured SCRE across seeds.
- Do not claim review-prioritization precision is equivalent to binary classification precision.
- Do not claim SCRE proves causality or provides a new fundamental machine-learning algorithm.
"""
    (OUT_DIR / "scre_safe_claims.md").write_text(safe, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topk = _capacity_topk_from_existing()
    decomposition = _score_decomposition(topk)
    rank_stability = _append_external_stability(_rank_stability())

    topk["taiwan"].to_csv(OUT_DIR / "scre_topk_review_taiwan.csv", index=False)
    topk["heloc"].to_csv(OUT_DIR / "scre_topk_review_heloc.csv", index=False)
    decomposition.to_csv(OUT_DIR / "scre_score_decomposition.csv", index=False)
    rank_stability.to_csv(OUT_DIR / "scre_rank_stability.csv", index=False)
    _summary(topk, decomposition, rank_stability)


if __name__ == "__main__":
    main()
