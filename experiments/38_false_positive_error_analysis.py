"""Analyze false positive profiles for selected frozen decision policies.

No model is trained and no feature is generated. This script reuses existing
probability outputs and selected policy thresholds/bands to compare TP, FP, FN,
and TN feature profiles.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_threshold_tools() -> Any:
    """Load frozen probability helpers from the precision threshold script."""

    script_path = PROJECT_ROOT / "experiments" / "32_precision_constrained_threshold_search.py"
    spec = importlib.util.spec_from_file_location("precision_threshold_tools", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tools = _load_threshold_tools()

DECISION_REVISION_DIR: Path = tools.DECISION_REVISION_DIR
TAIWAN_FEATURES = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "delay_count",
    "severe_delay_count",
    "max_delay",
    "avg_delay",
    "recent_delay",
    "LIMIT_BAL",
    "utilization_proxy",
    "payment_to_bill_ratio",
    "total_bill_amt",
    "total_pay_amt",
    "AGE",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]
HELOC_FEATURES = [
    "ExternalRiskEstimate",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "NetFractionRevolvingBurden",
    "NumBank2NatlTradesWHighUtilization",
    "delinquency_intensity",
    "trade_activity_ratio",
    "recent_inquiry_pressure",
    "revolving_burden_proxy",
    "installment_burden_proxy",
    "credit_history_length_proxy",
    "negative_trade_signal",
    "high_utilization_signal",
]


@dataclass(frozen=True)
class PolicySpec:
    """One frozen decision policy to analyze."""

    system: str
    model: str
    policy_type: str
    threshold: float | None = None
    t_low: float | None = None
    t_high: float | None = None
    source: str = ""


def parse_threshold_or_band(text: str) -> dict[str, float]:
    """Parse threshold or band text from eligibility tables."""

    values: dict[str, float] = {}
    for part in str(text).split(";"):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        values[key.strip()] = float(value.strip())
    return values


def current_catboost_policy(dataset: str) -> PolicySpec:
    """Return old Best CatBoost cost-threshold policy."""

    table = pd.read_csv(DECISION_REVISION_DIR / f"current_threshold_baseline_{dataset}.csv")
    row = table.loc[
        table["model"].eq("Best CatBoost") & table["threshold_type"].eq("cost_optimal_current")
    ].iloc[0]
    return PolicySpec(
        system="Old Best CatBoost cost-threshold",
        model="Best CatBoost",
        policy_type="binary_threshold",
        threshold=float(row["threshold"]),
        source=f"current_threshold_baseline_{dataset}.csv",
    )


def precision_catboost_policy(dataset: str) -> PolicySpec:
    """Return best eligible precision-constrained CatBoost policy."""

    eligible = pd.read_csv(DECISION_REVISION_DIR / f"eligible_model_selection_{dataset}.csv")
    rows = eligible.loc[
        eligible["model"].eq("Best CatBoost")
        & eligible["policy_type"].str.startswith("precision_constraint")
        & eligible["eligible_basic"].astype(bool)
    ].copy()
    if rows.empty:
        table = pd.read_csv(DECISION_REVISION_DIR / f"precision_constrained_test_results_{dataset}.csv")
        rows = table.loc[
            table["model"].eq("Best CatBoost")
            & table["feasible_on_validation"].astype(bool)
            & (table["test_precision"] >= 0.40)
        ].copy()
        rows = rows.sort_values(["validation_expected_cost", "test_precision"], ascending=[True, False])
        threshold = float(rows.iloc[0]["selected_threshold_validation"])
        policy_type = str(rows.iloc[0]["constraint_name"])
    else:
        rows = rows.sort_values(["rank_basic", "expected_cost", "precision"])
        parsed = parse_threshold_or_band(str(rows.iloc[0]["threshold_or_band"]))
        threshold = parsed["threshold"]
        policy_type = str(rows.iloc[0]["policy_type"])
    return PolicySpec(
        system="Best precision-constrained CatBoost",
        model="Best CatBoost",
        policy_type=f"binary_threshold:{policy_type}",
        threshold=threshold,
        source=f"eligible_model_selection_{dataset}.csv",
    )


def eligible_policy(dataset: str, system: str, predicate) -> PolicySpec:
    """Return top eligible policy matching a predicate."""

    eligible = pd.read_csv(DECISION_REVISION_DIR / f"eligible_model_selection_{dataset}.csv")
    rows = eligible.loc[eligible["eligible_basic"].astype(bool)].copy()
    rows = rows.loc[predicate(rows)].copy()
    if rows.empty:
        raise ValueError(f"No eligible policy found for {dataset}/{system}.")
    rows = rows.sort_values(["rank_basic", "expected_cost", "precision"], ascending=[True, True, False])
    row = rows.iloc[0]
    parsed = parse_threshold_or_band(str(row["threshold_or_band"]))
    if "t_high" in parsed:
        return PolicySpec(
            system=system,
            model=str(row["model"]),
            policy_type=str(row["policy_type"]),
            t_low=parsed.get("t_low"),
            t_high=parsed["t_high"],
            source=f"eligible_model_selection_{dataset}.csv",
        )
    return PolicySpec(
        system=system,
        model=str(row["model"]),
        policy_type=str(row["policy_type"]),
        threshold=parsed["threshold"],
        source=f"eligible_model_selection_{dataset}.csv",
    )


def selected_policies(dataset: str) -> list[PolicySpec]:
    """Return selected systems requested by the prompt."""

    policies = [
        current_catboost_policy(dataset),
        precision_catboost_policy(dataset),
        eligible_policy(
            dataset,
            "Best manual-review high-risk bucket",
            lambda df: df["policy_type"].str.startswith("manual_review_band"),
        ),
        eligible_policy(
            dataset,
            "Best SCRE policy",
            lambda df: df["model"].isin(["SCRE-Optimized", "SCRE-Pareto", "Old SCRE-Credit"]),
        ),
        eligible_policy(
            dataset,
            "Best Scorecard policy",
            lambda df: df["model"].eq("Best Scorecard"),
        ),
    ]
    return policies


def policy_probability(provider: Any, dataset: str, policy: PolicySpec) -> np.ndarray:
    """Return test probability for one selected display model."""

    plan = tools.model_plan(dataset)
    if policy.model not in plan:
        raise KeyError(f"Policy model {policy.model!r} is not available in frozen provider plan.")
    source_model, _ = plan[policy.model]
    probability = tools.probability_for_display(provider, dataset, policy.model, source_model, "test")
    return np.asarray(probability, dtype=float)


def group_labels(y_true: pd.Series, probability: np.ndarray, policy: PolicySpec) -> pd.Series:
    """Assign TP/FP/FN/TN labels for one policy."""

    y = np.asarray(y_true, dtype=int)
    if policy.t_high is not None:
        predicted_positive = probability >= float(policy.t_high)
    elif policy.threshold is not None:
        predicted_positive = probability >= float(policy.threshold)
    else:
        raise ValueError(f"Policy has no threshold: {policy}")

    labels = np.empty(len(y), dtype=object)
    labels[predicted_positive & (y == 1)] = "TP"
    labels[predicted_positive & (y == 0)] = "FP"
    labels[(~predicted_positive) & (y == 1)] = "FN"
    labels[(~predicted_positive) & (y == 0)] = "TN"
    return pd.Series(labels)


def feature_summary(system: str, features: pd.DataFrame, labels: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Summarize requested feature distributions by error group."""

    rows: list[dict[str, Any]] = []
    available = [column for column in columns if column in features.columns]
    for group in ["TP", "FP", "FN", "TN"]:
        group_features = features.loc[labels.eq(group), available]
        for feature in available:
            series = pd.to_numeric(group_features[feature], errors="coerce")
            rows.append(
                {
                    "system": system,
                    "error_group": group,
                    "feature": feature,
                    "mean": float(series.mean()) if len(series) else np.nan,
                    "median": float(series.median()) if len(series) else np.nan,
                    "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else np.nan,
                    "min": float(series.min()) if len(series) else np.nan,
                    "max": float(series.max()) if len(series) else np.nan,
                    "q25": float(series.quantile(0.25)) if len(series) else np.nan,
                    "q75": float(series.quantile(0.75)) if len(series) else np.nan,
                    "missing_count": int(series.isna().sum()),
                    "group_size": int(len(series)),
                }
            )
    return pd.DataFrame(rows)


def pooled_smd(a: pd.Series, b: pd.Series) -> float:
    """Return absolute standardized mean difference between two numeric series."""

    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    pooled = np.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if np.isclose(float(a.mean()), float(b.mean())) else np.nan
    return abs(float(a.mean()) - float(b.mean())) / pooled


def similarity_table(system: str, features: pd.DataFrame, labels: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Compare FP group similarity to TP and TN groups."""

    rows: list[dict[str, Any]] = []
    available = [column for column in columns if column in features.columns]
    fp = features.loc[labels.eq("FP")]
    tp = features.loc[labels.eq("TP")]
    tn = features.loc[labels.eq("TN")]
    for feature in available:
        fp_tp = pooled_smd(fp[feature], tp[feature])
        fp_tn = pooled_smd(fp[feature], tn[feature])
        score = fp_tn - fp_tp if pd.notna(fp_tp) and pd.notna(fp_tn) else np.nan
        rows.append(
            {
                "system": system,
                "feature": feature,
                "fp_vs_tp_smd": fp_tp,
                "fp_vs_tn_smd": fp_tn,
                "similarity_score": score,
                "closer_to": "TP" if pd.notna(score) and score > 0 else "TN" if pd.notna(score) and score < 0 else "tie",
                "interpretation": "positive score means FP is closer to TP than TN",
            }
        )
    table = pd.DataFrame(rows)
    overall = {
        "system": system,
        "feature": "__overall__",
        "fp_vs_tp_smd": float(table["fp_vs_tp_smd"].mean()),
        "fp_vs_tn_smd": float(table["fp_vs_tn_smd"].mean()),
    }
    overall["similarity_score"] = overall["fp_vs_tn_smd"] - overall["fp_vs_tp_smd"]
    overall["closer_to"] = "TP" if overall["similarity_score"] > 0 else "TN" if overall["similarity_score"] < 0 else "tie"
    overall["interpretation"] = "mean across analyzed features"
    return pd.concat([table, pd.DataFrame([overall])], ignore_index=True)


def fp_profile(system: str, features: pd.DataFrame, labels: pd.Series, dataset: str) -> dict[str, Any]:
    """Build compact FP profile indicators."""

    fp = features.loc[labels.eq("FP")].copy()
    output: dict[str, Any] = {
        "dataset": dataset,
        "system": system,
        "fp_count": int(len(fp)),
    }
    if dataset == "taiwan":
        output.update(
            {
                "fp_pay0_delayed_rate": float((fp["PAY_0"] > 0).mean()) if len(fp) else np.nan,
                "fp_delay_count_ge_2_rate": float((fp["delay_count"] >= 2).mean()) if len(fp) else np.nan,
                "fp_high_utilization_rate": float((fp["utilization_proxy"] >= 0.80).mean()) if len(fp) else np.nan,
                "fp_low_payment_to_bill_rate": float((fp["payment_to_bill_ratio"] <= 0.10).mean()) if len(fp) else np.nan,
                "fp_mean_PAY_0": float(fp["PAY_0"].mean()) if len(fp) else np.nan,
                "fp_mean_delay_count": float(fp["delay_count"].mean()) if len(fp) else np.nan,
                "fp_mean_utilization_proxy": float(fp["utilization_proxy"].mean()) if len(fp) else np.nan,
                "fp_median_payment_to_bill_ratio": float(fp["payment_to_bill_ratio"].median()) if len(fp) else np.nan,
            }
        )
    else:
        output.update(
            {
                "fp_high_utilization_signal_rate": float((fp["high_utilization_signal"] >= 1).mean()) if len(fp) else np.nan,
                "fp_negative_trade_signal_rate": float((fp["negative_trade_signal"] >= 1).mean()) if len(fp) else np.nan,
                "fp_mean_external_risk_estimate": float(fp["ExternalRiskEstimate"].mean()) if len(fp) else np.nan,
                "fp_mean_revolving_burden_proxy": float(fp["revolving_burden_proxy"].mean()) if len(fp) else np.nan,
            }
        )
    return output


def build_dataset_analysis(dataset: str, provider: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build FP profile, feature summary, similarity, and system comparison tables."""

    split = provider.splits[dataset]
    features = split.X_test.reset_index(drop=True).copy()
    y_test = split.y_test.reset_index(drop=True).astype(int)
    feature_columns = TAIWAN_FEATURES if dataset == "taiwan" else HELOC_FEATURES
    profile_rows: list[dict[str, Any]] = []
    summary_tables: list[pd.DataFrame] = []
    similarity_tables: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

    for policy in selected_policies(dataset):
        probability = policy_probability(provider, dataset, policy)
        labels = group_labels(y_test, probability, policy)
        profile_rows.append(fp_profile(policy.system, features, labels, dataset))
        summary_tables.append(feature_summary(policy.system, features, labels, feature_columns))
        similarity = similarity_table(policy.system, features, labels, feature_columns)
        similarity_tables.append(similarity)
        counts = labels.value_counts()
        overall = similarity.loc[similarity["feature"].eq("__overall__")].iloc[0]
        fp_count = int(counts.get("FP", 0))
        comparison_rows.append(
            {
                "dataset": dataset,
                "system": policy.system,
                "model": policy.model,
                "policy_type": policy.policy_type,
                "threshold": policy.threshold,
                "t_low": policy.t_low,
                "t_high": policy.t_high,
                "tp": int(counts.get("TP", 0)),
                "fp": fp_count,
                "fn": int(counts.get("FN", 0)),
                "tn": int(counts.get("TN", 0)),
                "overall_fp_vs_tp_smd": float(overall["fp_vs_tp_smd"]),
                "overall_fp_vs_tn_smd": float(overall["fp_vs_tn_smd"]),
                "overall_similarity_score": float(overall["similarity_score"]),
                "fp_closer_to": str(overall["closer_to"]),
                "fp_looks_risky": bool(overall["closer_to"] == "TP"),
                "source": policy.source,
            }
        )

    return (
        pd.DataFrame(profile_rows),
        pd.concat(summary_tables, ignore_index=True),
        pd.concat(similarity_tables, ignore_index=True),
        pd.DataFrame(comparison_rows),
    )


def plot_taiwan_distributions(summary: pd.DataFrame) -> Path:
    """Plot selected Taiwan error-group mean distributions by system."""

    path = DECISION_REVISION_DIR / "error_group_distributions_taiwan.png"
    features = ["PAY_0", "delay_count", "utilization_proxy", "payment_to_bill_ratio"]
    systems = summary["system"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, feature in zip(axes.ravel(), features):
        subset = summary.loc[summary["feature"].eq(feature)].copy()
        for group in ["TP", "FP", "FN", "TN"]:
            values = [
                subset.loc[subset["system"].eq(system) & subset["error_group"].eq(group), "mean"].iloc[0]
                for system in systems
            ]
            offsets = {"TP": -0.27, "FP": -0.09, "FN": 0.09, "TN": 0.27}
            ax.bar(np.arange(len(systems)) + offsets[group], values, width=0.18, label=group)
        ax.set_title(feature)
        ax.set_xticks(np.arange(len(systems)))
        ax.set_xticklabels([short_system_name(system) for system in systems], rotation=25, ha="right")
        ax.set_ylabel("Group mean")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Taiwan Error Group Feature Means", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def short_system_name(system: str) -> str:
    """Shorten system labels for plots."""

    return (
        system.replace("Old Best CatBoost cost-threshold", "Old Cat")
        .replace("Best precision-constrained CatBoost", "Prec Cat")
        .replace("Best manual-review high-risk bucket", "Manual")
        .replace("Best SCRE policy", "SCRE")
        .replace("Best Scorecard policy", "Scorecard")
    )


def write_interpretation(
    dataset: str,
    profile: pd.DataFrame,
    similarity: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Path:
    """Write markdown interpretation for one dataset."""

    suffix = "" if dataset == "taiwan" else "_heloc"
    path = DECISION_REVISION_DIR / f"false_positive_interpretation{suffix}.md"
    lines = [
        f"# False Positive Error Analysis: {dataset.upper()}",
        "",
        "No model was trained. Existing selected probability policies were evaluated on the held-out test split.",
        "",
        "## System Comparison",
        "",
        "| System | FP Count | FP Closer To | FP Looks Risky? | Overall FP-vs-TP SMD | Overall FP-vs-TN SMD | Comment |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in comparison.sort_values("fp").itertuples(index=False):
        comment = (
            "FP profile is closer to true defaults than true non-defaults."
            if row.fp_closer_to == "TP"
            else "FP profile is closer to true non-defaults; alarms may be less meaningful."
        )
        lines.append(
            f"| {row.system} | {row.fp} | {row.fp_closer_to} | {'YES' if row.fp_looks_risky else 'NO'} | "
            f"{row.overall_fp_vs_tp_smd:.4f} | {row.overall_fp_vs_tn_smd:.4f} | {comment} |"
        )
    lines.extend(["", "## FP Profile Indicators", ""])
    if dataset == "taiwan":
        lines.extend(
            [
                "| System | PAY_0 delayed | delay_count>=2 | high utilization | low payment/bill | Mean delay_count |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in profile.itertuples(index=False):
            lines.append(
                f"| {row.system} | {row.fp_pay0_delayed_rate:.4f} | {row.fp_delay_count_ge_2_rate:.4f} | "
                f"{row.fp_high_utilization_rate:.4f} | {row.fp_low_payment_to_bill_rate:.4f} | "
                f"{row.fp_mean_delay_count:.4f} |"
            )
    else:
        lines.extend(
            [
                "| System | High utilization signal | Negative trade signal | Mean ExternalRiskEstimate | Mean revolving burden |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in profile.itertuples(index=False):
            lines.append(
                f"| {row.system} | {row.fp_high_utilization_signal_rate:.4f} | "
                f"{row.fp_negative_trade_signal_rate:.4f} | {row.fp_mean_external_risk_estimate:.4f} | "
                f"{row.fp_mean_revolving_burden_proxy:.4f} |"
            )

    closer_tp_count = int(comparison["fp_closer_to"].eq("TP").sum())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Systems where FP is closer to TP than TN: {closer_tp_count}/{len(comparison)}.",
            "- Low precision should not automatically be read as meaningless alarms; FP borrowers can still exhibit high-risk characteristics.",
            "- Automatic rejection is not appropriate from FP profile alone. Manual review is the safer operational framing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_outputs(
    taiwan_profile: pd.DataFrame,
    taiwan_summary: pd.DataFrame,
    taiwan_similarity: pd.DataFrame,
    taiwan_comparison: pd.DataFrame,
    heloc_profile: pd.DataFrame,
    heloc_summary: pd.DataFrame,
    heloc_similarity: pd.DataFrame,
    heloc_comparison: pd.DataFrame,
) -> list[Path]:
    """Write all requested false-positive artifacts."""

    paths = [
        DECISION_REVISION_DIR / "false_positive_profile_taiwan.csv",
        DECISION_REVISION_DIR / "error_group_feature_summary_taiwan.csv",
        DECISION_REVISION_DIR / "fp_tp_tn_similarity_taiwan.csv",
        DECISION_REVISION_DIR / "false_positive_profile_heloc.csv",
        DECISION_REVISION_DIR / "error_group_feature_summary_heloc.csv",
        DECISION_REVISION_DIR / "fp_tp_tn_similarity_heloc.csv",
        DECISION_REVISION_DIR / "false_positive_system_comparison_taiwan.csv",
        DECISION_REVISION_DIR / "false_positive_system_comparison_heloc.csv",
    ]
    taiwan_profile.to_csv(paths[0], index=False)
    taiwan_summary.to_csv(paths[1], index=False)
    taiwan_similarity.to_csv(paths[2], index=False)
    heloc_profile.to_csv(paths[3], index=False)
    heloc_summary.to_csv(paths[4], index=False)
    heloc_similarity.to_csv(paths[5], index=False)
    taiwan_comparison.to_csv(paths[6], index=False)
    heloc_comparison.to_csv(paths[7], index=False)
    paths.extend(
        [
            write_interpretation("taiwan", taiwan_profile, taiwan_similarity, taiwan_comparison),
            write_interpretation("heloc", heloc_profile, heloc_similarity, heloc_comparison),
            plot_taiwan_distributions(taiwan_summary),
        ]
    )
    return paths


def main() -> None:
    """Run false-positive error analysis."""

    splits = {
        "taiwan": tools.load_split("taiwan"),
        "heloc": tools.load_split("heloc"),
    }
    provider = tools.ProbabilityProvider(splits)
    taiwan_profile, taiwan_summary, taiwan_similarity, taiwan_comparison = build_dataset_analysis("taiwan", provider)
    heloc_profile, heloc_summary, heloc_similarity, heloc_comparison = build_dataset_analysis("heloc", provider)
    paths = write_outputs(
        taiwan_profile,
        taiwan_summary,
        taiwan_similarity,
        taiwan_comparison,
        heloc_profile,
        heloc_summary,
        heloc_similarity,
        heloc_comparison,
    )
    for path in paths:
        print(path)

    print("\nTaiwan FP comparison:")
    print(
        taiwan_comparison[
            ["system", "fp", "fp_closer_to", "overall_fp_vs_tp_smd", "overall_fp_vs_tn_smd"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
