"""Run decision curve / net benefit analysis on frozen model probabilities.

No model is trained here. The analysis evaluates existing calibrated model
probabilities on the held-out test split across clinically/operationally
interpretable threshold probabilities.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_threshold_tools() -> Any:
    """Load shared frozen-probability helpers from the threshold experiment."""

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
MODEL_CANDIDATES = [
    "Best CatBoost",
    "Best XGBoost",
    "Best Scorecard",
    "SCRE-Optimized",
    "SCRE-Pareto",
]
THRESHOLD_PROBABILITIES = np.round(np.arange(0.01, 0.8001, 0.01), 2)


def confusion_counts(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, int]:
    """Return TP/FP/TN/FN counts for one threshold probability."""

    prediction = probability >= threshold
    positive = y_true == 1
    negative = ~positive
    return {
        "tp": int(np.sum(prediction & positive)),
        "fp": int(np.sum(prediction & negative)),
        "tn": int(np.sum((~prediction) & negative)),
        "fn": int(np.sum((~prediction) & positive)),
    }


def net_benefit_values(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold_probability: float,
) -> dict[str, float | int | bool]:
    """Compute model, treat-all, and treat-none net benefit."""

    counts = confusion_counts(y_true, probability, threshold_probability)
    n = float(len(y_true))
    prevalence = float(np.mean(y_true))
    threshold_odds = threshold_probability / (1.0 - threshold_probability)
    net_benefit = counts["tp"] / n - counts["fp"] / n * threshold_odds
    treat_all = prevalence - (1.0 - prevalence) * threshold_odds
    treat_none = 0.0
    standardized = net_benefit / prevalence if prevalence > 0 else np.nan
    return {
        **counts,
        "net_benefit": float(net_benefit),
        "net_benefit_treat_all": float(treat_all),
        "net_benefit_treat_none": float(treat_none),
        "standardized_net_benefit": float(standardized),
        "is_better_than_treat_all": bool(net_benefit > treat_all),
        "is_better_than_treat_none": bool(net_benefit > treat_none),
    }


def build_dataset_results(dataset: str, provider: Any) -> pd.DataFrame:
    """Build decision-curve rows for one dataset on the test split."""

    plan = tools.model_plan(dataset)
    split = provider.splits[dataset]
    y_test = np.asarray(split.y_test, dtype=int)
    rows: list[dict[str, Any]] = []

    for model in MODEL_CANDIDATES:
        source_model, calibration_type = plan[model]
        probability = tools.probability_for_display(provider, dataset, model, source_model, "test")
        probability = np.asarray(probability, dtype=float)
        for threshold_probability in THRESHOLD_PROBABILITIES:
            values = net_benefit_values(y_test, probability, float(threshold_probability))
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "evaluation_split": "test",
                    "calibration_type": calibration_type,
                    "threshold_probability": float(threshold_probability),
                    **values,
                }
            )

    table = pd.DataFrame(rows)
    table["rank_at_threshold"] = (
        table.groupby(["dataset", "threshold_probability"])["net_benefit"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return table


def contiguous_ranges(
    values: pd.DataFrame,
    active_column: str,
    range_type: str,
) -> list[dict[str, Any]]:
    """Return contiguous true ranges for one model and condition."""

    active = values.loc[values[active_column].astype(bool)].copy()
    if active.empty:
        return []

    active = active.sort_values("threshold_probability")
    ranges: list[dict[str, Any]] = []
    current: list[pd.Series] = []
    previous_threshold: float | None = None
    for row in active.itertuples(index=False):
        threshold = float(row.threshold_probability)
        if previous_threshold is None or abs(threshold - previous_threshold - 0.01) <= 1e-9:
            current.append(pd.Series(row._asdict()))
        else:
            ranges.append(range_summary(current, range_type))
            current = [pd.Series(row._asdict())]
        previous_threshold = threshold
    if current:
        ranges.append(range_summary(current, range_type))
    return ranges


def range_summary(rows: list[pd.Series], range_type: str) -> dict[str, Any]:
    """Summarize one contiguous threshold range."""

    block = pd.DataFrame(rows)
    best = block.sort_values(["net_benefit", "threshold_probability"], ascending=[False, True]).iloc[0]
    return {
        "dataset": str(best["dataset"]),
        "model": str(best["model"]),
        "range_type": range_type,
        "threshold_start": float(block["threshold_probability"].min()),
        "threshold_end": float(block["threshold_probability"].max()),
        "n_thresholds": int(len(block)),
        "best_threshold_probability": float(best["threshold_probability"]),
        "max_net_benefit": float(best["net_benefit"]),
        "mean_net_benefit": float(block["net_benefit"].mean()),
        "best_rank_in_range": int(block["rank_at_threshold"].min()),
    }


def model_ranges(table: pd.DataFrame) -> pd.DataFrame:
    """Build useful net-benefit ranges for every model."""

    rows: list[dict[str, Any]] = []
    enriched = table.copy()
    enriched["better_than_both"] = (
        enriched["is_better_than_treat_all"].astype(bool)
        & enriched["is_better_than_treat_none"].astype(bool)
    )
    for (_, model), group in enriched.groupby(["dataset", "model"]):
        rows.extend(contiguous_ranges(group, "is_better_than_treat_all", "better_than_treat_all"))
        rows.extend(contiguous_ranges(group, "is_better_than_treat_none", "better_than_treat_none"))
        rows.extend(contiguous_ranges(group, "better_than_both", "better_than_both"))
    return pd.DataFrame(rows)


def winner_ranges(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize contiguous threshold ranges where one model has the best NB."""

    winners = (
        table.loc[table["rank_at_threshold"].eq(1)]
        .sort_values(["dataset", "threshold_probability", "model"])
        .groupby(["dataset", "threshold_probability"], as_index=False)
        .agg(
            best_model=("model", lambda values: "Tie: " + "; ".join(values) if len(values) > 1 else str(values.iloc[0])),
            better_than_treat_all=("is_better_than_treat_all", "all"),
            better_than_treat_none=("is_better_than_treat_none", "all"),
            max_net_benefit=("net_benefit", "max"),
        )
    )
    rows: list[dict[str, Any]] = []
    for dataset, group in winners.groupby("dataset"):
        current: list[pd.Series] = []
        previous_key: tuple[str, bool, bool] | None = None
        previous_threshold: float | None = None
        for row in group.itertuples(index=False):
            key = (
                str(row.best_model),
                bool(row.better_than_treat_all),
                bool(row.better_than_treat_none),
            )
            threshold = float(row.threshold_probability)
            contiguous = previous_threshold is None or abs(threshold - previous_threshold - 0.01) <= 1e-9
            if previous_key is None or (key == previous_key and contiguous):
                current.append(pd.Series(row._asdict()))
            else:
                rows.append(winner_range_summary(dataset, current))
                current = [pd.Series(row._asdict())]
            previous_key = key
            previous_threshold = threshold
        if current:
            rows.append(winner_range_summary(dataset, current))
    return pd.DataFrame(rows)


def winner_range_summary(dataset: str, rows: list[pd.Series]) -> dict[str, Any]:
    """Summarize one best-model threshold range."""

    block = pd.DataFrame(rows)
    best = block.sort_values(["max_net_benefit", "threshold_probability"], ascending=[False, True]).iloc[0]
    return {
        "dataset": dataset,
        "threshold_start": float(block["threshold_probability"].min()),
        "threshold_end": float(block["threshold_probability"].max()),
        "best_model": str(best["best_model"]),
        "better_than_treat_all": bool(best["better_than_treat_all"]),
        "better_than_treat_none": bool(best["better_than_treat_none"]),
        "max_net_benefit": float(best["max_net_benefit"]),
        "best_threshold_probability": float(best["threshold_probability"]),
        "n_thresholds": int(len(block)),
    }


def fmt(value: float) -> str:
    """Format a numeric value for markdown."""

    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def interval_text(start: float, end: float) -> str:
    """Format threshold interval text."""

    if abs(start - end) <= 1e-9:
        return f"{start:.2f}"
    return f"{start:.2f}-{end:.2f}"


def plot_decision_curve(dataset: str, table: pd.DataFrame) -> Path:
    """Save one decision curve figure."""

    path = DECISION_REVISION_DIR / f"decision_curve_{dataset}.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODEL_CANDIDATES:
        rows = table.loc[table["model"].eq(model)].sort_values("threshold_probability")
        ax.plot(rows["threshold_probability"], rows["net_benefit"], label=model, linewidth=2)

    reference = table.loc[table["model"].eq(MODEL_CANDIDATES[0])].sort_values("threshold_probability")
    ax.plot(
        reference["threshold_probability"],
        reference["net_benefit_treat_all"],
        color="black",
        linestyle="--",
        label="Treat all",
    )
    ax.axhline(0.0, color="gray", linestyle=":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision Curve Analysis: {dataset.upper()}")
    ax.set_xlim(0.01, 0.80)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def peak_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return the peak net benefit row for each dataset/model."""

    return (
        table.sort_values(["dataset", "model", "net_benefit", "threshold_probability"], ascending=[True, True, False, True])
        .groupby(["dataset", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def bool_text(value: bool) -> str:
    """Human-readable boolean."""

    return "YES" if bool(value) else "NO"


def write_summary(dataset: str, table: pd.DataFrame, winners: pd.DataFrame, ranges: pd.DataFrame) -> Path:
    """Write a dataset-specific markdown summary."""

    dataset_winners = winners.loc[winners["dataset"].eq(dataset)].copy()
    dataset_ranges = ranges.loc[ranges["dataset"].eq(dataset)].copy()
    peaks = peak_table(table.loc[table["dataset"].eq(dataset)]).copy()

    lines = [
        f"# Decision Curve Summary: {dataset.upper()}",
        "",
        "Decision curves are evaluated on the held-out test split. No model training or threshold optimization is performed here.",
        "",
        "## Net Benefit Winner Ranges",
        "",
        "| Threshold Range | Best Model | Better Than Treat-All? | Better Than Treat-None? | Max Net Benefit | Comment |",
        "|---|---|---|---|---:|---|",
    ]
    for row in dataset_winners.itertuples(index=False):
        both = bool(row.better_than_treat_all) and bool(row.better_than_treat_none)
        comment = "Useful decision-support range" if both else "Model is top-ranked but not superior to both defaults"
        lines.append(
            f"| {interval_text(row.threshold_start, row.threshold_end)} | {row.best_model} | "
            f"{bool_text(row.better_than_treat_all)} | {bool_text(row.better_than_treat_none)} | "
            f"{fmt(row.max_net_benefit)} | {comment} |"
        )

    lines.extend(
        [
            "",
            "## Peak Net Benefit By Model",
            "",
            "| Model | Best Threshold Probability | Max Net Benefit | Better Than Treat-All? | Better Than Treat-None? |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in peaks.itertuples(index=False):
        lines.append(
            f"| {row.model} | {fmt(row.threshold_probability)} | {fmt(row.net_benefit)} | "
            f"{bool_text(row.is_better_than_treat_all)} | {bool_text(row.is_better_than_treat_none)} |"
        )

    lines.extend(["", "## Useful Ranges By Model", ""])
    for model in MODEL_CANDIDATES:
        both = dataset_ranges.loc[
            dataset_ranges["model"].eq(model) & dataset_ranges["range_type"].eq("better_than_both")
        ]
        if both.empty:
            lines.append(f"- {model}: no threshold interval beats both treat-all and treat-none.")
        else:
            intervals = ", ".join(
                interval_text(row.threshold_start, row.threshold_end)
                for row in both.itertuples(index=False)
            )
            lines.append(f"- {model}: beats both defaults over {intervals}.")

    if dataset == "taiwan":
        catboost_015 = table.loc[
            table["dataset"].eq("taiwan")
            & table["model"].eq("Best CatBoost")
            & table["threshold_probability"].eq(0.15)
        ].iloc[0]
        scre_vs_cat = scre_better_than_catboost_ranges(table, "taiwan")
        lines.extend(
            [
                "",
                "## Taiwan CatBoost 0.15 Check",
                "",
                f"- Net benefit at pt=0.15: {fmt(catboost_015['net_benefit'])}.",
                f"- Better than treat-all: {bool_text(catboost_015['is_better_than_treat_all'])}.",
                f"- Better than treat-none: {bool_text(catboost_015['is_better_than_treat_none'])}.",
                f"- Rank at pt=0.15: {int(catboost_015['rank_at_threshold'])}.",
                "",
                "## SCRE-Optimized vs CatBoost",
                "",
                f"- SCRE-Optimized has higher net benefit than Best CatBoost over: {scre_vs_cat}.",
            ]
        )
    else:
        scorecard_ranges = scorecard_meaningful_ranges(table, "heloc")
        lines.extend(
            [
                "",
                "## HELOC Scorecard Check",
                "",
                f"- Scorecard beats both default strategies over: {scorecard_ranges}.",
            ]
        )

    path = DECISION_REVISION_DIR / f"decision_curve_summary_{dataset}.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def scre_better_than_catboost_ranges(table: pd.DataFrame, dataset: str) -> str:
    """Return intervals where SCRE-Optimized beats Best CatBoost."""

    pivot = table.loc[
        table["dataset"].eq(dataset) & table["model"].isin(["Best CatBoost", "SCRE-Optimized"])
    ].pivot(index="threshold_probability", columns="model", values="net_benefit")
    active = pivot.loc[pivot["SCRE-Optimized"] > pivot["Best CatBoost"]].reset_index()
    return ranges_from_thresholds(active["threshold_probability"].tolist())


def scorecard_meaningful_ranges(table: pd.DataFrame, dataset: str) -> str:
    """Return intervals where Scorecard beats both defaults."""

    rows = table.loc[
        table["dataset"].eq(dataset)
        & table["model"].eq("Best Scorecard")
        & table["is_better_than_treat_all"].astype(bool)
        & table["is_better_than_treat_none"].astype(bool)
    ].copy()
    return ranges_from_thresholds(rows["threshold_probability"].tolist())


def ranges_from_thresholds(thresholds: list[float]) -> str:
    """Format contiguous ranges from threshold values."""

    if not thresholds:
        return "none"
    values = sorted(float(value) for value in thresholds)
    ranges: list[tuple[float, float]] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if abs(value - previous - 0.01) <= 1e-9:
            previous = value
            continue
        ranges.append((start, previous))
        start = value
        previous = value
    ranges.append((start, previous))
    return ", ".join(interval_text(start, end) for start, end in ranges)


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> list[Path]:
    """Write all requested decision-curve artifacts."""

    DECISION_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([taiwan, heloc], ignore_index=True)
    ranges = model_ranges(combined)
    winners = winner_ranges(combined)

    paths = [
        DECISION_REVISION_DIR / "decision_curve_results_taiwan.csv",
        DECISION_REVISION_DIR / "decision_curve_results_heloc.csv",
        DECISION_REVISION_DIR / "net_benefit_model_ranges.csv",
        DECISION_REVISION_DIR / "decision_curve_winner_ranges.csv",
    ]
    taiwan.to_csv(paths[0], index=False)
    heloc.to_csv(paths[1], index=False)
    ranges.to_csv(paths[2], index=False)
    winners.to_csv(paths[3], index=False)
    paths.extend(
        [
            write_summary("taiwan", combined, winners, ranges),
            write_summary("heloc", combined, winners, ranges),
            plot_decision_curve("taiwan", taiwan),
            plot_decision_curve("heloc", heloc),
        ]
    )
    return paths


def main() -> None:
    """Run decision curve analysis."""

    splits = {
        "taiwan": tools.load_split("taiwan"),
        "heloc": tools.load_split("heloc"),
    }
    provider = tools.ProbabilityProvider(splits)
    taiwan = build_dataset_results("taiwan", provider)
    heloc = build_dataset_results("heloc", provider)
    paths = write_outputs(taiwan, heloc)
    for path in paths:
        print(path)

    winners = winner_ranges(pd.concat([taiwan, heloc], ignore_index=True))
    for dataset in ["taiwan", "heloc"]:
        print(f"\n{dataset.upper()} net benefit winner ranges:")
        print(
            winners.loc[winners["dataset"].eq(dataset)][
                [
                    "threshold_start",
                    "threshold_end",
                    "best_model",
                    "better_than_treat_all",
                    "better_than_treat_none",
                    "max_net_benefit",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
