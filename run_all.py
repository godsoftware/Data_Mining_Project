"""Run the complete reproducible SCRE-Credit experiment workflow."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_registry import finish_run, start_run
from src.config.paths import EXPERIMENT_LOGS_DIR, MODELS_DIR, OUTPUTS_DIR, TABLES_DIR
from src.utils.logging_utils import create_experiment_logger


PYTHON = sys.executable


@dataclass(frozen=True)
class WorkflowStep:
    """One experiment script invocation."""

    script: str
    args: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Return a compact label for logging and inventory metadata."""

        return Path(self.script).stem

    def command(self) -> list[str]:
        """Return the executable command list."""

        return [PYTHON, str(PROJECT_ROOT / self.script), *self.args]


WORKFLOW_STEPS: list[WorkflowStep] = [
    WorkflowStep("experiments/00_environment_check.py"),
    WorkflowStep("experiments/01_prepare_taiwan.py"),
    WorkflowStep("experiments/02_prepare_heloc.py"),
    WorkflowStep("experiments/03_feature_engineering.py"),
    WorkflowStep("experiments/04_create_splits.py"),
    WorkflowStep("experiments/05_train_baselines.py"),
    WorkflowStep("experiments/06_train_scorecard.py"),
    WorkflowStep("experiments/07_train_boosting_models.py"),
    WorkflowStep("experiments/08_optuna_tuning.py"),
    WorkflowStep("experiments/09_calibration.py"),
    WorkflowStep("experiments/10_threshold_cost_analysis.py"),
    WorkflowStep("experiments/11_train_scre_credit.py"),
    WorkflowStep("experiments/12_statistical_tests.py"),
    WorkflowStep("experiments/13_explainability.py"),
    WorkflowStep("experiments/14_shap_stability.py"),
    WorkflowStep("experiments/15_background_sensitivity.py"),
    WorkflowStep("experiments/16_faithfulness.py"),
    WorkflowStep("experiments/17_subgroup_reliability.py"),
    WorkflowStep("experiments/18_external_validation_heloc.py"),
    WorkflowStep("experiments/19_ablation_study.py"),
]


def parse_args() -> argparse.Namespace:
    """Parse run-all options."""

    parser = argparse.ArgumentParser(description="Run the complete SCRE-Credit pipeline.")
    parser.add_argument(
        "--start-at",
        default=None,
        choices=[step.label for step in WORKFLOW_STEPS],
        help="Resume from the named step label, for example 08_optuna_tuning.",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        choices=[step.label for step in WORKFLOW_STEPS],
        help="Stop after the named step label.",
    )
    return parser.parse_args()


def selected_steps(start_at: str | None, stop_after: str | None) -> list[WorkflowStep]:
    """Select a contiguous subset of workflow steps."""

    labels = [step.label for step in WORKFLOW_STEPS]
    start = labels.index(start_at) if start_at else 0
    stop = labels.index(stop_after) + 1 if stop_after else len(WORKFLOW_STEPS)
    if start >= stop:
        raise ValueError("--start-at must come before or equal --stop-after.")
    return WORKFLOW_STEPS[start:stop]


def utc_timestamp() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_step(step: WorkflowStep, step_index: int, total_steps: int) -> dict[str, object]:
    """Run one workflow step, write stdout/stderr log, and return execution metadata."""

    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENT_LOGS_DIR / f"run_all_{step_index:02d}_{step.label}.log"
    command = step.command()
    start = time.perf_counter()
    start_utc = utc_timestamp()

    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"# Step {step_index}/{total_steps}: {step.script}\n")
        log_handle.write(f"# Started UTC: {start_utc}\n")
        log_handle.write(f"# Command: {' '.join(command)}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    end = time.perf_counter()
    duration = end - start
    status = "completed" if completed.returncode == 0 else "failed"
    with log_path.open("a", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"\n# Finished UTC: {utc_timestamp()}\n")
        log_handle.write(f"# Return code: {completed.returncode}\n")
        log_handle.write(f"# Duration seconds: {duration:.2f}\n")

    return {
        "step_index": step_index,
        "step_total": total_steps,
        "script": step.script,
        "step_label": step.label,
        "command": " ".join(command),
        "status": status,
        "return_code": completed.returncode,
        "duration_seconds": round(duration, 3),
        "log_path": str(log_path),
        "started_utc": start_utc,
        "finished_utc": utc_timestamp(),
    }


def output_category(path: Path) -> str:
    """Classify an output artifact."""

    if path.suffix.lower() == ".csv":
        return "table"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure"
    if path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        return "model"
    return "other"


def inventory_roots() -> list[Path]:
    """Return directories scanned by the final output inventory."""

    return [TABLES_DIR, OUTPUTS_DIR / "figures", MODELS_DIR, OUTPUTS_DIR / "shap", OUTPUTS_DIR / "lime"]


def create_final_output_inventory(path: Path = TABLES_DIR / "final_output_inventory.csv") -> Path:
    """Write an inventory of all table, figure, and model artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in inventory_roots():
        if not root.exists():
            continue
        for artifact in sorted(root.rglob("*")):
            if not artifact.is_file() or artifact in seen:
                continue
            category = output_category(artifact)
            if category not in {"table", "figure", "model"}:
                continue
            stat = artifact.stat()
            rows.append(
                {
                    "artifact_type": category,
                    "relative_path": artifact.relative_to(PROJECT_ROOT).as_posix(),
                    "absolute_path": str(artifact),
                    "file_name": artifact.name,
                    "extension": artifact.suffix.lower(),
                    "size_bytes": int(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                }
            )
            seen.add(artifact)

    rows.sort(key=lambda row: (str(row["artifact_type"]), str(row["relative_path"])))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "artifact_type",
                "relative_path",
                "absolute_path",
                "file_name",
                "extension",
                "size_bytes",
                "modified_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_step_summary(rows: list[dict[str, object]]) -> Path:
    """Save run_all step-level execution summary."""

    path = TABLES_DIR / "run_all_step_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step_index",
                "step_total",
                "script",
                "step_label",
                "command",
                "status",
                "return_code",
                "duration_seconds",
                "log_path",
                "started_utc",
                "finished_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    """Run all configured experiment scripts in order."""

    args = parse_args()
    logger = create_experiment_logger("run_all", EXPERIMENT_LOGS_DIR / "run_all.log")
    steps = selected_steps(args.start_at, args.stop_after)
    run_id = start_run(
        "run_all",
        dataset="both",
        params={
            "start_at": args.start_at,
            "stop_after": args.stop_after,
            "n_steps": len(steps),
            "workflow": [step.script for step in steps],
        },
        tags=["complete-workflow", "prompt-22"],
    )

    step_rows: list[dict[str, object]] = []
    try:
        for index, step in enumerate(steps, start=1):
            logger.info("Starting step %s/%s: %s", index, len(steps), step.script)
            print(f"\n=== Step {index}/{len(steps)}: {step.script} ===")
            row = run_step(step, index, len(steps))
            step_rows.append(row)
            write_step_summary(step_rows)
            if row["return_code"] != 0:
                message = (
                    f"Workflow failed at {step.script} "
                    f"(return_code={row['return_code']}). See log: {row['log_path']}"
                )
                logger.error(message)
                finish_run(
                    run_id,
                    status="failed",
                    metrics={"failed_step": step.script, "return_code": row["return_code"]},
                    artifacts={"step_summary": TABLES_DIR / "run_all_step_summary.csv"},
                    notes=message,
                )
                raise RuntimeError(message)
            logger.info("Completed step %s/%s: %s", index, len(steps), step.script)

        inventory_path = create_final_output_inventory()
        summary_path = write_step_summary(step_rows)
        logger.info("Workflow completed. Final inventory: %s", inventory_path)
        finish_run(
            run_id,
            metrics={
                "completed_steps": len(step_rows),
                "failed_steps": 0,
            },
            artifacts={
                "final_output_inventory": inventory_path,
                "step_summary": summary_path,
                "run_all_log": EXPERIMENT_LOGS_DIR / "run_all.log",
            },
        )
        print(f"\nCompleted {len(step_rows)} steps.")
        print(f"Final output inventory: {inventory_path}")
    except Exception as exc:
        if step_rows:
            write_step_summary(step_rows)
        if not (TABLES_DIR / "final_output_inventory.csv").exists():
            create_final_output_inventory()
        logger.exception("run_all failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
