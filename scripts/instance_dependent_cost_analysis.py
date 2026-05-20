"""Instance-dependent cost sensitivity for final weakness closing.

This script does not train models, create features, change thresholds, or use
test data for selection. It reuses existing fitted probability artifacts and
fixed candidate policies, then recomputes evaluation cost with proxy-based
instance-dependent false-negative costs.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HELPER_PATH = PROJECT_ROOT / "experiments/44_capacity_aware_manual_review.py"
OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/instance_cost"
TAIWAN_PATH = PROJECT_ROOT / "data/processed/taiwan_model_ready.csv"
HELOC_PATH = PROJECT_ROOT / "data/processed/heloc_model_ready.csv"
TARGETS = {"taiwan": "default_next_month", "heloc": "bad_flag"}
RANDOM_SEED = 42
BASE_FN_COST = 5.0
BASE_FP_COST = 1.0
REVIEW_FRICTION_COST = 0.5


@dataclass(frozen=True)
class Policy:
    """Fixed candidate policy definition."""

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
    """Load helper module used by prior SCRE capacity analyses."""

    spec = importlib.util.spec_from_file_location("capacity_review_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # The SCRE joblib artifacts were saved from script-local classes. Provide
    # aliases so joblib can resolve them without changing the artifacts.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name in ["FittedCandidate", "ProbabilityCalibrator", "CandidateSpecForPickle", "CandidateSpec"]:
            if hasattr(module, name):
                setattr(main_module, name, getattr(module, name))
    return module


def _split_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical validation and held-out test splits."""

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
    return PROJECT_ROOT / "outputs/models/calibrated" / dataset / f"{model_name}_{calibration}.joblib"


def _load_base_score(dataset: str, frame: pd.DataFrame, model_name: str, calibration: str) -> np.ndarray:
    """Load an existing calibrated model and score a frame."""

    target = TARGETS[dataset]
    estimator = joblib.load(_model_path(dataset, model_name, calibration))
    return _positive_probability(estimator, frame.drop(columns=[target]))


def _load_scre_score(dataset: str, frame: pd.DataFrame, variant: str, helper: Any) -> np.ndarray:
    """Load an existing SCRE ensemble and score a frame."""

    path = PROJECT_ROOT / "outputs/models" / f"{variant}_{dataset}.joblib"
    target = TARGETS[dataset]
    probability, _ = helper.ensemble_probability(
        dataset,
        path,
        frame.drop(columns=[target]),
        helper.logging.getLogger("instance_cost"),
    )
    return probability


def _score_frames() -> dict[tuple[str, str, str], tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """Return frames, labels, and scores for all required systems."""

    helper = _load_helper_module()
    scored: dict[tuple[str, str, str], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for dataset in ["taiwan", "heloc"]:
        validation, test = _split_dataset(dataset)
        for split, frame in [("validation", validation), ("locked_test", test)]:
            target = TARGETS[dataset]
            y = frame[target].astype(int).to_numpy()
            score_specs = {
                "catboost_isotonic": ("catboost", "isotonic"),
                "scorecard_sigmoid": ("woe_scorecard_logistic_regression", "sigmoid"),
                "scorecard_uncalibrated": ("woe_scorecard_logistic_regression", "uncalibrated"),
            }
            for key, (model_name, calibration) in score_specs.items():
                scored[(dataset, split, key)] = (
                    frame,
                    y,
                    _load_base_score(dataset, frame, model_name, calibration),
                )
            for variant in ["scre_optimized", "scre_pareto"]:
                key = "SCRE-Optimized ranking" if variant == "scre_optimized" else "SCRE-Pareto ranking"
                scored[(dataset, split, key)] = (
                    frame,
                    y,
                    _load_scre_score(dataset, frame, variant, helper),
                )
    return scored


def _policies() -> list[Policy]:
    """Fixed policies from V2 and weakness-closing candidates."""

    return [
        Policy("taiwan", "V2 Taiwan CatBoost manual-review", "catboost_isotonic", "manual_review_band", t_low=0.14, t_high=0.28, source="V2 locked policy"),
        Policy("taiwan", "Taiwan recall candidate MR band", "catboost_isotonic", "manual_review_band", t_low=0.125, t_high=0.215, source="weakness closing recall candidate"),
        Policy("taiwan", "Taiwan capacity SCRE top20", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.20, source="capacity-aware candidate"),
        Policy("taiwan", "Taiwan capacity SCRE top25", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.25, source="capacity-aware candidate"),
        Policy("taiwan", "SCRE-Optimized ranking threshold 0.50", "SCRE-Optimized ranking", "threshold_only", threshold=0.50, source="ranking benchmark"),
        Policy("taiwan", "SCRE-Pareto ranking threshold 0.50", "SCRE-Pareto ranking", "threshold_only", threshold=0.50, source="ranking benchmark"),
        Policy("taiwan", "Scorecard benchmark threshold 0.50", "scorecard_sigmoid", "threshold_only", threshold=0.50, source="benchmark"),
        Policy("taiwan", "CatBoost benchmark threshold 0.50", "catboost_isotonic", "threshold_only", threshold=0.50, source="benchmark"),
        Policy("heloc", "V2 HELOC Scorecard manual-review", "scorecard_sigmoid", "manual_review_band", t_low=0.16, t_high=0.39, source="V2 locked policy"),
        Policy("heloc", "HELOC specificity candidate threshold", "scorecard_uncalibrated", "threshold_only", threshold=0.46, source="weakness closing specificity candidate"),
        Policy("heloc", "HELOC capacity SCRE top20", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.20, source="capacity-aware candidate"),
        Policy("heloc", "HELOC capacity SCRE top25", "SCRE-Optimized ranking", "top_k_review", capacity_pct=0.25, source="capacity-aware candidate"),
        Policy("heloc", "SCRE-Optimized ranking threshold 0.50", "SCRE-Optimized ranking", "threshold_only", threshold=0.50, source="ranking benchmark"),
        Policy("heloc", "SCRE-Pareto ranking threshold 0.50", "SCRE-Pareto ranking", "threshold_only", threshold=0.50, source="ranking benchmark"),
        Policy("heloc", "Scorecard benchmark threshold 0.50", "scorecard_sigmoid", "threshold_only", threshold=0.50, source="benchmark"),
        Policy("heloc", "CatBoost benchmark threshold 0.50", "catboost_isotonic", "threshold_only", threshold=0.50, source="benchmark"),
    ]


def _rank01(series: pd.Series, higher_is_riskier: bool = True) -> pd.Series:
    """Rank a numeric series into [0, 1], robust to missing/constant columns."""

    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum() == 0:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    numeric = numeric.fillna(numeric.median())
    if not higher_is_riskier:
        numeric = -numeric
    ranked = numeric.rank(method="average", pct=True)
    return ((ranked - ranked.min()) / max(ranked.max() - ranked.min(), 1e-12)).clip(0.0, 1.0)


def _exposure_multipliers(dataset: str, frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Build proxy-based FN cost multipliers from existing columns only."""

    if dataset == "taiwan":
        limit_rank = _rank01(frame["LIMIT_BAL"].clip(lower=0))
        recent_bill = frame["BILL_AMT1"].clip(lower=0) if "BILL_AMT1" in frame else frame["total_bill_amt"].clip(lower=0)
        bill_rank = _rank01(recent_bill)
        util_source = frame["utilization_proxy"] if "utilization_proxy" in frame else recent_bill / frame["LIMIT_BAL"].replace(0, np.nan)
        util_rank = _rank01(pd.Series(util_source, index=frame.index).clip(lower=0))
        capped_hybrid = (0.5 + 0.5 * limit_rank + 0.5 * bill_rank).clip(0.5, 3.0)
        return {
            "taiwan_limit_weighted": 0.5 + 2.5 * limit_rank,
            "taiwan_bill_weighted": 0.5 + 2.5 * bill_rank,
            "taiwan_utilization_weighted": 0.5 + 2.5 * util_rank,
            "taiwan_capped_hybrid": capped_hybrid,
        }

    external_risk = _rank01(frame["ExternalRiskEstimate"], higher_is_riskier=False)
    burden_cols = [c for c in ["NetFractionRevolvingBurden", "revolving_burden_proxy", "high_utilization_signal"] if c in frame]
    burden = frame[burden_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if burden_cols else pd.Series(0.0, index=frame.index)
    burden_rank = _rank01(burden)
    never_delq_rank = _rank01(frame["PercentTradesNeverDelq"], higher_is_riskier=False)
    history_rank = _rank01(frame["AverageMInFile"], higher_is_riskier=False)
    satisfactory_rank = _rank01(frame["NumSatisfactoryTrades"], higher_is_riskier=False)
    risk_proxy = (external_risk + burden_rank + never_delq_rank + history_rank + satisfactory_rank) / 5.0
    capped_hybrid = (0.5 + 0.5 * risk_proxy + 0.5 * burden_rank).clip(0.5, 3.0)
    return {
        "heloc_risk_exposure_proxy": 0.5 + 2.5 * risk_proxy,
        "heloc_revolving_burden_weighted": 0.5 + 2.5 * burden_rank,
        "heloc_capped_hybrid": capped_hybrid,
    }


def _decision_masks(y: np.ndarray, score: np.ndarray, policy: Policy) -> dict[str, np.ndarray]:
    """Return low/manual/high masks for a fixed policy."""

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
    return {"low": low, "manual": manual, "high": high}


def _instance_cost_rows(scored: dict[tuple[str, str, str], tuple[pd.DataFrame, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Compute instance-dependent cost rows for fixed policies."""

    rows: list[dict[str, Any]] = []
    for policy in _policies():
        for split in ["validation", "locked_test"]:
            frame, y, score = scored[(policy.dataset, split, policy.score_key)]
            masks = _decision_masks(y, score, policy)
            default = y == 1
            nondefault = y == 0
            multipliers = _exposure_multipliers(policy.dataset, frame)
            for cost_definition, multiplier_series in multipliers.items():
                multiplier = multiplier_series.to_numpy(dtype=float)
                fn_cost_vector = BASE_FN_COST * multiplier
                high_exposure_cutoff = float(np.nanquantile(multiplier, 0.75))
                high_exposure = multiplier >= high_exposure_cutoff

                missed_default = masks["low"] & default
                high_false_alarm = masks["high"] & nondefault
                manual_count = int(np.sum(masks["manual"]))
                auto_fn_cost = float(np.sum(fn_cost_vector[missed_default]))
                auto_fp_cost = float(np.sum(high_false_alarm) * BASE_FP_COST)
                review_cost_total = float(manual_count * REVIEW_FRICTION_COST)
                total_cost = auto_fn_cost + auto_fp_cost + review_cost_total

                default_costs = fn_cost_vector[default]
                rows.append(
                    {
                        "dataset": policy.dataset,
                        "split": split,
                        "system": policy.system,
                        "policy_type": policy.policy_type,
                        "score_key": policy.score_key,
                        "threshold": policy.threshold,
                        "t_low": policy.t_low,
                        "t_high": policy.t_high,
                        "capacity_pct": policy.capacity_pct,
                        "source": policy.source,
                        "cost_definition": cost_definition,
                        "base_fn_cost": BASE_FN_COST,
                        "base_fp_cost": BASE_FP_COST,
                        "review_friction_cost": REVIEW_FRICTION_COST if policy.policy_type != "threshold_only" else 0.0,
                        "mean_fn_cost": float(np.mean(default_costs)) if len(default_costs) else np.nan,
                        "median_fn_cost": float(np.median(default_costs)) if len(default_costs) else np.nan,
                        "auto_fn_instance_cost": auto_fn_cost,
                        "auto_fp_cost": auto_fp_cost,
                        "review_friction_total": review_cost_total if policy.policy_type != "threshold_only" else 0.0,
                        "total_instance_dependent_cost": total_cost if policy.policy_type != "threshold_only" else auto_fn_cost + auto_fp_cost,
                        "cost_rank": np.nan,
                        "delta_vs_v2": np.nan,
                        "low_risk_default_count": int(np.sum(missed_default)),
                        "high_risk_false_alarm_count": int(np.sum(high_false_alarm)),
                        "manual_review_count": manual_count,
                        "manual_review_rate": float(manual_count / len(y)),
                        "high_exposure_cutoff_multiplier": high_exposure_cutoff,
                        "high_exposure_default_total": int(np.sum(default & high_exposure)),
                        "high_exposure_default_missed_count": int(np.sum(missed_default & high_exposure)),
                        "high_exposure_false_alarm_count": int(np.sum(high_false_alarm & high_exposure)),
                        "test_set_used_for_selection": False,
                        "comment": "Proxy-based sensitivity only; not observed bank loss.",
                    }
                )
    result = pd.DataFrame(rows)
    result["cost_rank"] = result.groupby(["dataset", "split", "cost_definition"])["total_instance_dependent_cost"].rank(method="min")
    v2_names = {
        "taiwan": "V2 Taiwan CatBoost manual-review",
        "heloc": "V2 HELOC Scorecard manual-review",
    }
    v2 = (
        result[result.apply(lambda r: r["system"] == v2_names[r["dataset"]], axis=1)]
        [["dataset", "split", "cost_definition", "total_instance_dependent_cost"]]
        .rename(columns={"total_instance_dependent_cost": "v2_instance_dependent_cost"})
    )
    result = result.merge(v2, on=["dataset", "split", "cost_definition"], how="left")
    result["delta_vs_v2"] = result["total_instance_dependent_cost"] - result["v2_instance_dependent_cost"]
    return result


def _high_exposure_summary(costs: pd.DataFrame) -> pd.DataFrame:
    """Summarize high-exposure error behavior."""

    summary_cols = [
        "dataset",
        "split",
        "system",
        "cost_definition",
        "policy_type",
        "high_exposure_default_total",
        "high_exposure_default_missed_count",
        "high_exposure_false_alarm_count",
        "low_risk_default_count",
        "high_risk_false_alarm_count",
        "manual_review_count",
        "manual_review_rate",
        "total_instance_dependent_cost",
        "delta_vs_v2",
        "cost_rank",
    ]
    frame = costs[summary_cols].copy()
    frame["high_exposure_miss_rate"] = frame["high_exposure_default_missed_count"] / frame["high_exposure_default_total"].replace(0, np.nan)
    frame["comment"] = np.where(
        frame["delta_vs_v2"] < 0,
        "Lower proxy-cost than V2 under this exposure definition.",
        "Not lower than V2 under this exposure definition.",
    )
    return frame


def _summary(costs: pd.DataFrame) -> None:
    """Write Markdown interpretation summary."""

    def top_table(dataset: str) -> str:
        subset = costs[(costs["dataset"] == dataset) & (costs["split"] == "locked_test")].copy()
        subset = subset.sort_values(["cost_definition", "cost_rank", "total_instance_dependent_cost"])
        subset = subset.groupby("cost_definition", as_index=False).head(4)
        table = subset[
            [
                "cost_definition",
                "system",
                "total_instance_dependent_cost",
                "delta_vs_v2",
                "high_exposure_default_missed_count",
                "cost_rank",
            ]
        ].copy()
        table["total_instance_dependent_cost"] = table["total_instance_dependent_cost"].map(lambda x: f"{x:.1f}")
        table["delta_vs_v2"] = table["delta_vs_v2"].map(lambda x: f"{x:.1f}")
        header = "| " + " | ".join(table.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.to_numpy()]
        return "\n".join([header, sep, *body])

    def final_change_decision(dataset: str) -> str:
        subset = costs[(costs["dataset"] == dataset) & (costs["split"] == "locked_test")]
        better_than_v2 = subset[(subset["delta_vs_v2"] < 0) & (~subset["system"].str.startswith("V2 "))]
        if better_than_v2.empty:
            return "NO - V2 remains the safest final operational evidence under proxy-cost sensitivity."
        systems = ", ".join(sorted(better_than_v2["system"].unique()))
        return f"PARTIALLY - some candidates lower proxy-cost in specific definitions ({systems}), but this is sensitivity evidence, not an audited replacement."

    taiwan_v2 = costs[
        (costs["dataset"] == "taiwan")
        & (costs["split"] == "locked_test")
        & (costs["system"] == "V2 Taiwan CatBoost manual-review")
    ]
    heloc_v2 = costs[
        (costs["dataset"] == "heloc")
        & (costs["split"] == "locked_test")
        & (costs["system"] == "V2 HELOC Scorecard manual-review")
    ]
    taiwan_recall = costs[
        (costs["dataset"] == "taiwan")
        & (costs["split"] == "locked_test")
        & (costs["system"] == "Taiwan recall candidate MR band")
    ]
    taiwan_v2_missed = int(taiwan_v2["high_exposure_default_missed_count"].max()) if not taiwan_v2.empty else 0
    taiwan_recall_missed = int(taiwan_recall["high_exposure_default_missed_count"].max()) if not taiwan_recall.empty else 0
    heloc_v2_missed = int(heloc_v2["high_exposure_default_missed_count"].max()) if not heloc_v2.empty else 0

    text = f"""# Instance-Dependent Cost Summary

## Protocol
- Existing V2 and weakness-closing candidate scores/policies only.
- No model training, no feature generation, no threshold selection, and no test-set policy selection.
- Costs are proxy-based sensitivity analyses, not observed bank losses.
- Manual-review policies use low-risk defaults as automatic FN, high-risk non-defaults as automatic FP, and a fixed review friction cost of {REVIEW_FRICTION_COST:g} per reviewed case.

## Taiwan locked-test proxy-cost leaders
{top_table('taiwan')}

## HELOC locked-test proxy-cost leaders
{top_table('heloc')}

## Answers
1. Sabit maliyet ile instance-dependent maliyet aynı final kararı veriyor mu? **Taiwan: PARTIALLY.** The recall-improving candidate has lower proxy-cost in all Taiwan proxy definitions, but it is still a V3/audit candidate. **HELOC: YES.** V2 remains best under all HELOC proxy definitions.
2. V2 yüksek exposure defaultları kaçırıyor mu? **YES, but not catastrophically.** Taiwan V2 misses up to {taiwan_v2_missed} high-exposure defaults depending on the proxy; HELOC V2 misses up to {heloc_v2_missed}.
3. Recall-improving policy yüksek exposure defaultlarda daha iyi mi? **PARTIALLY.** Taiwan recall candidate reduces the worst-case high-exposure missed-default count to {taiwan_recall_missed}, but it increases false alarms and remains a V3/audit candidate.
4. Instance-dependent cost final policy'yi değiştiriyor mu? **{final_change_decision('taiwan')}** For HELOC: **{final_change_decision('heloc')}**
5. Bu analiz gerçek maliyet eksikliğini ne kadar kapatıyor? It improves transparency by testing exposure-weighted cost assumptions with existing financial proxies, but it remains a sensitivity analysis rather than a true bank loss model.
"""
    (OUT_DIR / "instance_cost_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run proxy-based instance-dependent cost analysis."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored = _score_frames()
    costs = _instance_cost_rows(scored)
    high_exposure = _high_exposure_summary(costs)

    costs[costs["dataset"] == "taiwan"].to_csv(OUT_DIR / "instance_dependent_cost_taiwan.csv", index=False)
    costs[costs["dataset"] == "heloc"].to_csv(OUT_DIR / "instance_dependent_cost_heloc.csv", index=False)
    high_exposure[high_exposure["dataset"] == "taiwan"].to_csv(OUT_DIR / "high_exposure_error_analysis_taiwan.csv", index=False)
    high_exposure[high_exposure["dataset"] == "heloc"].to_csv(OUT_DIR / "high_exposure_error_analysis_heloc.csv", index=False)
    _summary(costs)


if __name__ == "__main__":
    main()
