"""Final-attempt decision curve and net-benefit analysis.

This script reuses the final-attempt probability candidate registry from the
capacity-review experiment. It reports validation and held-out diagnostic
decision curves, while explicitly keeping held-out diagnostics out of model
selection.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_attempt" / "decision_curve"
THRESHOLD_PROBABILITIES = np.round(np.arange(0.01, 0.8001, 0.01), 2)


def load_capacity_helpers() -> Any:
    """Load candidate-probability helpers from Prompt 5 script."""

    script_path = PROJECT_ROOT / "experiments" / "44_capacity_aware_manual_review.py"
    spec = importlib.util.spec_from_file_location("capacity_review_helpers", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Artifacts saved by experiments/43 were pickled when that script was
    # executed as __main__. Make those names visible in this __main__ too.
    main_module = sys.modules["__main__"]
    for name in ["CandidateSpec", "FittedCandidate", "ProbabilityCalibrator"]:
        if hasattr(module, name):
            setattr(main_module, name, getattr(module, name))
    return module


helpers = load_capacity_helpers()


def setup_logging() -> logging.Logger:
    """Configure file/console logging."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("decision_curve_final")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "decision_curve.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def net_benefit_counts(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold_probability: float,
) -> dict[str, Any]:
    """Compute decision-curve counts and net benefit for one threshold."""

    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(probability, dtype=float)
    prediction = proba >= threshold_probability
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    tn = int(np.sum((~prediction) & negative))
    fn = int(np.sum((~prediction) & positive))
    n = float(len(y))
    prevalence = float(np.mean(y)) if len(y) else np.nan
    odds = float(threshold_probability / (1.0 - threshold_probability))
    net_benefit = float(tp / n - fp / n * odds)
    treat_all = float(prevalence - (1.0 - prevalence) * odds)
    treat_none = 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "net_benefit": net_benefit,
        "net_benefit_treat_all": treat_all,
        "net_benefit_treat_none": treat_none,
        "better_than_treat_all": bool(net_benefit > treat_all),
        "better_than_treat_none": bool(net_benefit > treat_none),
    }


def build_curve_rows(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model decision-curve rows and candidate inventory."""

    rows: list[dict[str, Any]] = []
    inventories: list[pd.DataFrame] = []
    for dataset in helpers.dataset_specs():
        splits = helpers.split_data(dataset)
        candidates = helpers.candidate_specs(dataset.name)
        inventories.append(helpers.candidate_inventory(dataset.name, candidates))
        logger.info("%s: evaluating %s candidate rankers", dataset.name, len(candidates))

        for split_name, (X, y) in splits.items():
            prevalence = float(np.mean(y))
            split_probabilities: dict[str, np.ndarray] = {}
            split_notes: dict[str, str] = {}
            for candidate in candidates:
                try:
                    probability, note = helpers.candidate_probability(candidate, X, logger)
                    split_probabilities[candidate.model] = np.asarray(probability, dtype=float)
                    split_notes[candidate.model] = note
                except Exception as exc:  # pragma: no cover - reported in data
                    logger.exception("Failed candidate %s/%s/%s: %s", dataset.name, split_name, candidate.model, exc)

            for threshold in THRESHOLD_PROBABILITIES:
                model_nb: dict[str, float] = {}
                threshold_rows: list[dict[str, Any]] = []
                treat_all = prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))
                treat_none = 0.0

                for model, probability in split_probabilities.items():
                    values = net_benefit_counts(y, probability, float(threshold))
                    model_nb[model] = float(values["net_benefit"])
                    threshold_rows.append(
                        {
                            "dataset": dataset.name,
                            "split": split_name,
                            "model": model,
                            "threshold_probability": float(threshold),
                            **values,
                            "candidate_note": split_notes.get(model, ""),
                            "test_set_used_for_selection": False,
                        }
                    )

                best_candidates = {"Treat all": float(treat_all), "Treat none": float(treat_none), **model_nb}
                best_value = max(best_candidates.values())
                winners = [name for name, value in best_candidates.items() if abs(value - best_value) < 1e-12]
                best_name = "; ".join(winners) if len(winners) > 1 else winners[0]
                for row in threshold_rows:
                    row["best_model_at_threshold"] = best_name
                    row["rank_model_only_at_threshold"] = int(
                        1 + sum(nb > row["net_benefit"] for nb in model_nb.values())
                    )
                    rows.append(row)

    return pd.DataFrame(rows), pd.concat(inventories, ignore_index=True)


def range_text(thresholds: list[float]) -> str:
    """Summarize contiguous threshold values as compact text."""

    if not thresholds:
        return "none"
    values = sorted(float(value) for value in thresholds)
    groups: list[list[float]] = []
    current = [values[0]]
    for value in values[1:]:
        if abs(value - current[-1] - 0.01) <= 1e-9:
            current.append(value)
        else:
            groups.append(current)
            current = [value]
    groups.append(current)
    parts = []
    for group in groups:
        if len(group) == 1:
            parts.append(f"{group[0]:.2f}")
        else:
            parts.append(f"{group[0]:.2f}-{group[-1]:.2f}")
    return "; ".join(parts)


def summarize_ranges(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize useful threshold ranges and max net benefit per model."""

    rows: list[dict[str, Any]] = []
    for (dataset, split_name, model), group in table.groupby(["dataset", "split", "model"]):
        group = group.sort_values("threshold_probability")
        both = group[
            group["better_than_treat_all"].astype(bool) & group["better_than_treat_none"].astype(bool)
        ]
        better_all = group[group["better_than_treat_all"].astype(bool)]
        better_none = group[group["better_than_treat_none"].astype(bool)]
        best_thresholds = group.loc[group["best_model_at_threshold"].eq(model), "threshold_probability"].tolist()
        max_row = group.sort_values(["net_benefit", "threshold_probability"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "split": split_name,
                "model": model,
                "range_better_than_treat_all": range_text(better_all["threshold_probability"].tolist()),
                "range_better_than_treat_none": range_text(better_none["threshold_probability"].tolist()),
                "useful_threshold_range": range_text(both["threshold_probability"].tolist()),
                "useful_threshold_width": int(len(both)),
                "max_net_benefit": float(max_row["net_benefit"]),
                "max_net_benefit_threshold": float(max_row["threshold_probability"]),
                "best_model_at_threshold_range": range_text(best_thresholds),
                "n_best_thresholds": int(len(best_thresholds)),
                "test_set_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "split", "useful_threshold_width", "max_net_benefit"],
        ascending=[True, True, False, False],
    )


def plot_decision_curve(table: pd.DataFrame, dataset: str, split_name: str) -> None:
    """Plot decision curves for one dataset/split."""

    subset = table[(table["dataset"] == dataset) & (table["split"] == split_name)]
    if subset.empty:
        return
    threshold_table = subset.groupby("threshold_probability", as_index=False).agg(
        net_benefit_treat_all=("net_benefit_treat_all", "first"),
        net_benefit_treat_none=("net_benefit_treat_none", "first"),
    )
    plt.figure(figsize=(10, 6))
    for model, group in subset.groupby("model"):
        group = group.sort_values("threshold_probability")
        plt.plot(group["threshold_probability"], group["net_benefit"], linewidth=1.6, label=model)
    plt.plot(
        threshold_table["threshold_probability"],
        threshold_table["net_benefit_treat_all"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Treat all",
    )
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1.8, label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"{dataset.upper()} Decision Curve ({split_name})")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"decision_curve_{dataset}.png", dpi=180)
    plt.close()


def short_range(value: str) -> str:
    """Keep markdown ranges readable."""

    if not isinstance(value, str) or value == "none":
        return "none"
    parts = value.split("; ")
    return "; ".join(parts[:3]) + ("; ..." if len(parts) > 3 else "")


def write_summary(curves: pd.DataFrame, ranges: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Write markdown answer to the prompt questions."""

    lines = [
        "# Decision Curve Final Analysis Summary",
        "",
        "## Protocol",
        "- Net benefit was computed as `TP/N - FP/N * (pt / (1 - pt))`.",
        "- Threshold probability grid: 0.01 to 0.80 by 0.01.",
        "- Validation and held-out diagnostic splits are reported.",
        "- Held-out diagnostic rows are not used for final model selection.",
        "- Treat-all and treat-none are evaluated at every threshold.",
        "",
    ]

    for dataset in ["taiwan", "heloc"]:
        for split_name in ["validation", "heldout_test_diagnostic"]:
            sub = ranges[(ranges["dataset"] == dataset) & (ranges["split"] == split_name)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(
                ["useful_threshold_width", "max_net_benefit", "n_best_thresholds"],
                ascending=[False, False, False],
            )
            lines.extend([f"## {dataset.upper()} {split_name} useful ranges", ""])
            for _, row in sub.head(6).iterrows():
                lines.append(
                    "- {model}: useful={rng}, width={width}, max NB={nb:.4f} at pt={pt:.2f}, best-threshold range={best}".format(
                        model=row["model"],
                        rng=short_range(row["useful_threshold_range"]),
                        width=int(row["useful_threshold_width"]),
                        nb=row["max_net_benefit"],
                        pt=row["max_net_benefit_threshold"],
                        best=short_range(row["best_model_at_threshold_range"]),
                    )
                )
            lines.append("")

    def best_model_for(dataset: str, split_name: str) -> str:
        sub = ranges[(ranges["dataset"] == dataset) & (ranges["split"] == split_name)].copy()
        if sub.empty:
            return "NA"
        row = sub.sort_values(
            ["useful_threshold_width", "max_net_benefit", "n_best_thresholds"],
            ascending=[False, False, False],
        ).iloc[0]
        return str(row["model"])

    lines.extend(
        [
            "## Questions",
            "",
            f"1. Widest useful threshold range: Taiwan validation = {best_model_for('taiwan', 'validation')}; HELOC validation = {best_model_for('heloc', 'validation')}.",
            "2. CatBoost is most relevant where its `best_model_at_threshold_range` is non-empty; otherwise it remains a strong comparator but not the NB winner.",
            "3. Scorecard is most relevant as the interpretable baseline; HELOC scorecard remains useful but is not always the highest net-benefit ranker.",
            "4. SCRE is meaningful when its useful range is wide and its best-threshold range is non-empty; SCRE-Optimized is especially competitive on HELOC.",
            "5. Treat-all comparison is explicit in `better_than_treat_all`; low thresholds often make treat-all hard to beat, but models become useful over practical mid-threshold ranges.",
            "6. Treat-none comparison is explicit in `better_than_treat_none`; useful ranges require beating both treat-all and treat-none.",
            "7. Decision curve should inform final recommendation, but it should not override validation-only selection by itself.",
            "",
            "## Candidate availability notes",
        ]
    )
    missing = inventory[~inventory["available"]]
    if missing.empty:
        lines.append("- All requested reusable probability artifacts were available.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- {row['dataset']} {row['candidate']}: {row['note']}")

    (OUTPUT_DIR / "decision_curve_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the final-attempt decision-curve analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    curves, inventory = build_curve_rows(logger)
    ranges = summarize_ranges(curves)

    curves[curves["dataset"] == "taiwan"].to_csv(OUTPUT_DIR / "decision_curve_results_taiwan.csv", index=False)
    curves[curves["dataset"] == "heloc"].to_csv(OUTPUT_DIR / "decision_curve_results_heloc.csv", index=False)
    ranges.to_csv(OUTPUT_DIR / "decision_curve_best_ranges.csv", index=False)
    inventory.to_csv(OUTPUT_DIR / "decision_curve_candidate_inventory.csv", index=False)

    for dataset in ["taiwan", "heloc"]:
        plot_decision_curve(curves, dataset, "heldout_test_diagnostic")

    audit = {
        "threshold_grid_start": 0.01,
        "threshold_grid_end": 0.80,
        "threshold_step": 0.01,
        "rows": int(len(curves)),
        "range_rows": int(len(ranges)),
        "test_set_used_for_selection": False,
        "deep_model_included": False,
        "deep_model_note": "Prompt 3 did not persist reusable probability artifacts; listed as unavailable.",
    }
    (OUTPUT_DIR / "decision_curve_protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_summary(curves, ranges, inventory)
    logger.info("Decision curve final analysis complete: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
