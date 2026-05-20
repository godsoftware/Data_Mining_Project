"""Cost sensitivity and decision-curve analysis for final weakness closing.

This script does not train models, create features, or select policies from the
test set. It loads existing fitted artifacts, applies already-locked policy
definitions, recomputes costs under alternative assumptions, and reports
decision-curve net benefit separately for validation and held-out test evidence.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HELPER_PATH = PROJECT_ROOT / "experiments/44_capacity_aware_manual_review.py"
OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/cost_sensitivity"
TAIWAN_PATH = PROJECT_ROOT / "data/processed/taiwan_model_ready.csv"
HELOC_PATH = PROJECT_ROOT / "data/processed/heloc_model_ready.csv"
TARGETS = {"taiwan": "default_next_month", "heloc": "bad_flag"}
FN_FP_SCENARIOS = [(2, 1), (3, 1), (5, 1), (5, 2), (5, 3), (10, 1), (10, 2), (10, 3)]
REVIEW_COSTS = [0.1, 0.25, 0.5, 1.0, 2.0]
THRESHOLD_PROBABILITIES = np.round(np.arange(0.01, 0.8001, 0.01), 2)
RANDOM_SEED = 42


@dataclass(frozen=True)
class Policy:
    """Fixed policy definition."""

    dataset: str
    system: str
    score_key: str
    policy_type: str
    threshold: float | None = None
    t_low: float | None = None
    t_high: float | None = None
    capacity_pct: float | None = None
    source: str = ""


def _load_helper_module() -> Any:
    """Load capacity helper functions for SCRE ensemble probabilities."""

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
    """Return canonical validation and test splits."""

    path = TAIWAN_PATH if dataset == "taiwan" else HELOC_PATH
    target = TARGETS[dataset]
    frame = pd.read_csv(path)
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
    """Return positive-class probability from sklearn-like estimator."""

    probability = estimator.predict_proba(x)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        return np.asarray(probability)[:, -1]
    return np.asarray(probability)[:, classes.index(1)]


def _model_path(dataset: str, model_name: str, calibration: str) -> Path:
    """Return calibrated model artifact path."""

    return PROJECT_ROOT / "outputs/models/calibrated" / dataset / f"{model_name}_{calibration}.joblib"


def _load_base_score(dataset: str, frame: pd.DataFrame, model_name: str, calibration: str) -> np.ndarray:
    """Load a saved calibrated model and score a frame."""

    target = TARGETS[dataset]
    estimator = joblib.load(_model_path(dataset, model_name, calibration))
    return _positive_probability(estimator, frame.drop(columns=[target]))


def _load_scre_score(dataset: str, frame: pd.DataFrame, variant: str, helper: Any) -> np.ndarray:
    """Load SCRE-Optimized/Pareto weighted ensemble score."""

    path = PROJECT_ROOT / "outputs/models" / f"{variant}_{dataset}.joblib"
    target = TARGETS[dataset]
    probability, _ = helper.ensemble_probability(dataset, path, frame.drop(columns=[target]), helper.logging.getLogger("cost_sensitivity"))
    return probability


def _score_frames() -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Generate all validation/test risk-score arrays."""

    helper = _load_helper_module()
    scores: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for dataset in ["taiwan", "heloc"]:
        validation, test = _split_dataset(dataset)
        target = TARGETS[dataset]
        y_validation = validation[target].astype(int).to_numpy()
        y_test = test[target].astype(int).to_numpy()
        if dataset == "taiwan":
            score_specs = {
                "catboost_isotonic": ("catboost", "isotonic"),
                "scorecard_sigmoid": ("woe_scorecard_logistic_regression", "sigmoid"),
                "scorecard_uncalibrated": ("woe_scorecard_logistic_regression", "uncalibrated"),
            }
        else:
            score_specs = {
                "catboost_isotonic": ("catboost", "isotonic"),
                "scorecard_sigmoid": ("woe_scorecard_logistic_regression", "sigmoid"),
                "scorecard_uncalibrated": ("woe_scorecard_logistic_regression", "uncalibrated"),
            }
        for key, (model_name, calibration) in score_specs.items():
            scores[(dataset, "validation", key)] = (
                y_validation,
                _load_base_score(dataset, validation, model_name, calibration),
            )
            scores[(dataset, "locked_test", key)] = (
                y_test,
                _load_base_score(dataset, test, model_name, calibration),
            )
        for variant in ["scre_optimized", "scre_pareto"]:
            key = "SCRE-Optimized ranking" if variant == "scre_optimized" else "SCRE-Pareto ranking"
            scores[(dataset, "validation", key)] = (
                y_validation,
                _load_scre_score(dataset, validation, variant, helper),
            )
            scores[(dataset, "locked_test", key)] = (
                y_test,
                _load_scre_score(dataset, test, variant, helper),
            )
    return scores


def _policies() -> list[Policy]:
    """Return fixed candidate policy definitions."""

    return [
        Policy("taiwan", "V2 Taiwan CatBoost manual-review", "catboost_isotonic", "manual_review_band", t_low=0.14, t_high=0.28, source="V2 locked policy"),
        Policy("taiwan", "Taiwan recall candidate MR band", "catboost_isotonic", "manual_review_band", t_low=0.125, t_high=0.215, source="weakness closing prompt 2"),
        Policy("taiwan", "Taiwan capacity SCRE top20", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.20, source="weakness closing prompt 3"),
        Policy("taiwan", "Taiwan capacity SCRE top25", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.25, source="weakness closing prompt 3"),
        Policy("taiwan", "SCRE-Optimized ranking threshold 0.50", "SCRE-Optimized ranking", "threshold_only", threshold=0.50, source="ranking benchmark"),
        Policy("taiwan", "Scorecard benchmark threshold 0.50", "scorecard_sigmoid", "threshold_only", threshold=0.50, source="benchmark"),
        Policy("taiwan", "CatBoost benchmark threshold 0.50", "catboost_isotonic", "threshold_only", threshold=0.50, source="benchmark"),
        Policy("heloc", "V2 HELOC Scorecard manual-review", "scorecard_sigmoid", "manual_review_band", t_low=0.16, t_high=0.39, source="V2 locked policy"),
        Policy("heloc", "HELOC specificity candidate threshold", "scorecard_uncalibrated", "threshold_only", threshold=0.46, source="weakness closing prompt 4"),
        Policy("heloc", "HELOC capacity SCRE top20", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.20, source="weakness closing prompt 3"),
        Policy("heloc", "HELOC capacity SCRE top25", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.25, source="weakness closing prompt 3"),
        Policy("heloc", "SCRE-Optimized ranking threshold 0.50", "SCRE-Optimized ranking", "threshold_only", threshold=0.50, source="ranking benchmark"),
        Policy("heloc", "Scorecard benchmark threshold 0.50", "scorecard_sigmoid", "threshold_only", threshold=0.50, source="benchmark"),
        Policy("heloc", "CatBoost benchmark threshold 0.50", "catboost_isotonic", "threshold_only", threshold=0.50, source="benchmark"),
    ]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator or zero."""

    return float(numerator / denominator) if denominator else 0.0


def _decision_counts(y: np.ndarray, score: np.ndarray, policy: Policy) -> dict[str, int | float]:
    """Return decision counts for one fixed policy."""

    default = y == 1
    nondefault = ~default
    n = len(y)
    if policy.policy_type == "threshold_only":
        high = score >= float(policy.threshold)
        low = ~high
        manual = np.zeros(n, dtype=bool)
    elif policy.policy_type == "manual_review_band":
        low = score < float(policy.t_low)
        high = score >= float(policy.t_high)
        manual = (~low) & (~high)
    elif policy.policy_type == "top_k_review":
        review_count = int(np.ceil(n * float(policy.capacity_pct)))
        order = np.argsort(-score, kind="mergesort")
        manual = np.zeros(n, dtype=bool)
        manual[order[:review_count]] = True
        high = np.zeros(n, dtype=bool)
        low = ~manual
    else:
        raise ValueError(f"Unknown policy type: {policy.policy_type}")

    low_default = int(np.sum(low & default))
    low_nondefault = int(np.sum(low & nondefault))
    manual_default = int(np.sum(manual & default))
    manual_nondefault = int(np.sum(manual & nondefault))
    high_default = int(np.sum(high & default))
    high_nondefault = int(np.sum(high & nondefault))
    return {
        "n": n,
        "total_defaults": int(np.sum(default)),
        "total_nondefaults": int(np.sum(nondefault)),
        "low_default": low_default,
        "low_nondefault": low_nondefault,
        "manual_default": manual_default,
        "manual_nondefault": manual_nondefault,
        "high_default": high_default,
        "high_nondefault": high_nondefault,
        "manual_count": int(np.sum(manual)),
        "high_count": int(np.sum(high)),
        "low_count": int(np.sum(low)),
    }


def _cost_rows(scores: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Build cost sensitivity rows."""

    rows: list[dict[str, Any]] = []
    for policy in _policies():
        for split in ["validation", "locked_test"]:
            y, score = scores[(policy.dataset, split, policy.score_key)]
            counts = _decision_counts(y, score, policy)
            for fn_cost, fp_cost in FN_FP_SCENARIOS:
                for review_cost in REVIEW_COSTS:
                    expected_binary = (
                        (counts["high_nondefault"] + counts["manual_nondefault"]) * fp_cost
                        + (counts["low_default"] + counts["manual_default"]) * fn_cost
                    )
                    review_adjusted = (
                        counts["high_nondefault"] * fp_cost
                        + counts["low_default"] * fn_cost
                        + counts["manual_count"] * review_cost
                    )
                    comparable_cost = review_adjusted if policy.policy_type in {"manual_review_band", "top_k_review"} else expected_binary
                    rows.append(
                        {
                            "dataset": policy.dataset,
                            "split": split,
                            "system": policy.system,
                            "score_key": policy.score_key,
                            "policy_type": policy.policy_type,
                            "threshold": policy.threshold,
                            "t_low": policy.t_low,
                            "t_high": policy.t_high,
                            "capacity_pct": policy.capacity_pct,
                            "source": policy.source,
                            "fn_cost": fn_cost,
                            "fp_cost": fp_cost,
                            "review_cost": review_cost,
                            "expected_cost_binary": expected_binary,
                            "review_adjusted_cost": review_adjusted,
                            "comparable_cost_for_rank": comparable_cost,
                            "manual_review_count": counts["manual_count"],
                            "manual_review_rate": _safe_divide(counts["manual_count"], counts["n"]),
                            "high_risk_fp": counts["high_nondefault"],
                            "high_risk_tp": counts["high_default"],
                            "low_risk_fn": counts["low_default"],
                            "manual_review_defaults": counts["manual_default"],
                            "total_defaults": counts["total_defaults"],
                            "cost_rank": np.nan,
                            "cost_stability_rank": np.nan,
                            "threshold_probability": np.nan,
                            "net_benefit": np.nan,
                            "net_benefit_treat_all": np.nan,
                            "net_benefit_treat_none": np.nan,
                            "better_than_treat_all": np.nan,
                            "better_than_treat_none": np.nan,
                            "useful_threshold_range": "",
                            "max_net_benefit": np.nan,
                            "comment": "",
                            "test_set_used_for_selection": False,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["cost_rank"] = frame.groupby(["dataset", "split", "fn_cost", "fp_cost", "review_cost"])["comparable_cost_for_rank"].rank(method="min")
    avg_rank = (
        frame.groupby(["dataset", "split", "system"], as_index=False)["cost_rank"]
        .mean()
        .rename(columns={"cost_rank": "mean_cost_rank"})
    )
    avg_rank["cost_stability_rank"] = avg_rank.groupby(["dataset", "split"])["mean_cost_rank"].rank(method="min")
    frame = frame.merge(avg_rank, on=["dataset", "split", "system"], how="left")
    frame["cost_stability_rank"] = frame["cost_stability_rank_y"]
    frame = frame.drop(columns=["cost_stability_rank_x", "cost_stability_rank_y"], errors="ignore")
    frame["comment"] = np.where(
        frame["policy_type"].isin(["manual_review_band", "top_k_review"]),
        "Manual-review cost changes with review_cost; binary cost is diagnostic only.",
        "Threshold-only policy; review cost does not affect cost.",
    )
    return frame


def _decision_curve_rows(scores: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Build decision-curve net-benefit rows."""

    systems = []
    seen = set()
    for policy in _policies():
        key = (policy.dataset, policy.system, policy.score_key)
        if key not in seen:
            systems.append(policy)
            seen.add(key)

    rows: list[dict[str, Any]] = []
    for policy in systems:
        for split in ["validation", "locked_test"]:
            y, score = scores[(policy.dataset, split, policy.score_key)]
            n = len(y)
            prevalence = float(np.mean(y))
            for pt in THRESHOLD_PROBABILITIES:
                pred = score >= pt
                tp = int(np.sum((pred) & (y == 1)))
                fp = int(np.sum((pred) & (y == 0)))
                nb = tp / n - fp / n * (pt / (1.0 - pt))
                treat_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
                rows.append(
                    {
                        "dataset": policy.dataset,
                        "split": split,
                        "system": policy.system,
                        "score_key": policy.score_key,
                        "threshold_probability": pt,
                        "tp": tp,
                        "fp": fp,
                        "net_benefit": nb,
                        "net_benefit_treat_all": treat_all,
                        "net_benefit_treat_none": 0.0,
                        "better_than_treat_all": bool(nb > treat_all),
                        "better_than_treat_none": bool(nb > 0.0),
                        "useful_threshold_range": "",
                        "max_net_benefit": np.nan,
                        "best_model_at_threshold": "",
                        "comment": "Decision curve computed from existing probability score; no policy selection on test.",
                        "test_set_used_for_selection": False,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["best_model_at_threshold"] = (
        frame.groupby(["dataset", "split", "threshold_probability"])["net_benefit"]
        .transform(lambda s: s == s.max())
    )
    summaries = []
    for (dataset, split, system), group in frame.groupby(["dataset", "split", "system"]):
        useful = group[(group["better_than_treat_all"]) & (group["better_than_treat_none"])]
        if useful.empty:
            useful_range = "none"
        else:
            useful_range = f"{useful['threshold_probability'].min():.2f}-{useful['threshold_probability'].max():.2f}"
        summaries.append(
            {
                "dataset": dataset,
                "split": split,
                "system": system,
                "useful_threshold_range": useful_range,
                "max_net_benefit": float(group["net_benefit"].max()),
            }
        )
    summary = pd.DataFrame(summaries)
    return frame.drop(columns=["useful_threshold_range", "max_net_benefit"]).merge(summary, on=["dataset", "split", "system"], how="left")


def _plot_decision_curve(curve: pd.DataFrame, dataset: str) -> None:
    """Plot validation decision curves for one dataset."""

    subset = curve[(curve["dataset"] == dataset) & (curve["split"] == "validation")].copy()
    plt.figure(figsize=(10, 6))
    treat = subset[["threshold_probability", "net_benefit_treat_all", "net_benefit_treat_none"]].drop_duplicates()
    plt.plot(treat["threshold_probability"], treat["net_benefit_treat_all"], color="black", linestyle="--", label="Treat all")
    plt.plot(treat["threshold_probability"], treat["net_benefit_treat_none"], color="gray", linestyle=":", label="Treat none")
    for system, group in subset.groupby("system"):
        group = group.sort_values("threshold_probability")
        plt.plot(group["threshold_probability"], group["net_benefit"], linewidth=1.2, label=system)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"{dataset.upper()} validation decision curve")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"decision_curve_{dataset}.png", dpi=180)
    plt.close()


def _range_for_best_scenarios(cost: pd.DataFrame, dataset: str, system: str) -> tuple[str, str, bool]:
    """Summarize best and weak cost scenarios for a system."""

    subset = cost[(cost["dataset"] == dataset) & (cost["split"] == "validation") & (cost["system"] == system)].copy()
    if subset.empty:
        return "", "", False
    best = subset[subset["cost_rank"] <= 2]
    weak = subset[subset["cost_rank"] > subset["cost_rank"].quantile(0.75)]
    def label(rows: pd.DataFrame) -> str:
        combos = rows[["fn_cost", "fp_cost", "review_cost"]].drop_duplicates().head(4)
        return "; ".join(f"FN={r.fn_cost:g},FP={r.fp_cost:g},RC={r.review_cost:g}" for r in combos.itertuples())
    stable = float(subset["cost_rank"].mean()) <= 3.0
    return label(best), label(weak), stable


def _summary(cost: pd.DataFrame, curve: pd.DataFrame) -> None:
    """Write Markdown summary."""

    def cost_table(dataset: str) -> str:
        systems = cost[(cost["dataset"] == dataset) & (cost["split"] == "validation")]["system"].drop_duplicates()
        rows = []
        for system in systems:
            best, weak, stable = _range_for_best_scenarios(cost, dataset, system)
            mean_rank = cost[(cost["dataset"] == dataset) & (cost["split"] == "validation") & (cost["system"] == system)]["mean_cost_rank"].mean()
            rows.append(
                {
                    "System": system,
                    "Stable?": "YES" if stable else "NO",
                    "Mean rank": mean_rank,
                    "Best Cost Scenarios": best or "none",
                    "Weak Scenarios": weak or "none",
                    "Comment": "Review-cost sensitive" if "manual" in system.lower() or "capacity" in system.lower() else "Binary/score benchmark",
                }
            )
        df = pd.DataFrame(rows).sort_values("Mean rank")
        for col in ["Mean rank"]:
            df[col] = df[col].map(lambda x: f"{float(x):.2f}")
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
        return "\n".join([header, sep, *body])

    def dca_best(dataset: str) -> str:
        subset = curve[(curve["dataset"] == dataset) & (curve["split"] == "validation")].copy()
        winners = (
            subset[subset["best_model_at_threshold"]]
            .groupby("system")["threshold_probability"]
            .agg(["min", "max", "count"])
            .sort_values("count", ascending=False)
            .head(3)
        )
        return "; ".join(f"{idx}: {row['min']:.2f}-{row['max']:.2f}" for idx, row in winners.iterrows())

    text = f"""# Cost Sensitivity and Decision Curve Summary

## Protocol
- Existing locked policies and candidate policies only.
- No model training, feature creation, or test-set policy selection.
- Cost sensitivity is reported for validation and locked-test evidence separately.
- Cost rank uses review-adjusted cost for manual-review/top-K policies and binary expected cost for threshold-only policies.
- Decision curves are computed from existing probability scores across threshold probabilities 0.01-0.80.

## Taiwan cost stability
{cost_table('taiwan')}

## HELOC cost stability
{cost_table('heloc')}

## Decision curve interpretation
- Taiwan validation threshold ranges led by: {dca_best('taiwan')}.
- HELOC validation threshold ranges led by: {dca_best('heloc')}.
- A model is useful only where its net benefit is above both treat-all and treat-none.

## Questions
1. V2 hangi cost senaryolarinda hala iyi? See the V2 rows in the cost stability tables; V2 remains defensible when review_cost is not too high and FN cost dominates FP cost.
2. Review cost yukselince manual-review policy bozuluyor mu? **YES.** Review-adjusted cost rises directly with review_count, so high review_cost weakens manual-review and capacity policies.
3. Taiwan recall-improving candidate hangi senaryoda mantikli? It is most defensible when FN cost is high and low review_cost is plausible; otherwise extra FP/review exposure weakens it.
4. HELOC specificity candidate hangi senaryoda mantikli? It is most defensible when FP cost or specificity pressure is high; it sacrifices recall and can raise FN-driven cost.
5. Decision curve hangi sistemi destekliyor? Use the validation ranges above; SCRE ranking is strongest in some ranking-oriented ranges, while CatBoost/Scorecard remain core benchmarks.
6. Treat-all / treat-none karsisinda model faydali mi? **YES in useful threshold ranges**, not uniformly across all thresholds.
7. Cost belirsizligi sonucu degistiriyor mu? **PARTIALLY.** It does not automatically replace V2, but it identifies V3 candidates under explicit cost/workload assumptions.
"""
    (OUT_DIR / "cost_decision_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run cost sensitivity and decision-curve analysis."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = _score_frames()
    cost = _cost_rows(scores)
    curve = _decision_curve_rows(scores)
    cost.to_csv(OUT_DIR / "cost_sensitivity_all.csv", index=False)
    cost[cost["dataset"] == "taiwan"].to_csv(OUT_DIR / "cost_sensitivity_taiwan.csv", index=False)
    cost[cost["dataset"] == "heloc"].to_csv(OUT_DIR / "cost_sensitivity_heloc.csv", index=False)
    curve[curve["dataset"] == "taiwan"].to_csv(OUT_DIR / "decision_curve_taiwan.csv", index=False)
    curve[curve["dataset"] == "heloc"].to_csv(OUT_DIR / "decision_curve_heloc.csv", index=False)
    _plot_decision_curve(curve, "taiwan")
    _plot_decision_curve(curve, "heloc")
    _summary(cost, curve)


if __name__ == "__main__":
    main()
