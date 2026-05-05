"""Create a homework/report freeze package without running experiments."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import OUTPUTS_DIR, TABLES_DIR


DELIVERY_DIR = OUTPUTS_DIR / "homework_delivery_pack"
PYTHON_EXE = Path(r"C:\Users\AKTS\anaconda3\envs\tf210win\python.exe")


@dataclass(frozen=True)
class CopyResult:
    """Result of copying one delivery-pack artifact."""

    label: str
    source: Path
    destination: Path | None
    status: str


def run_command(command: list[str], cwd: Path = PROJECT_ROOT) -> tuple[int, str]:
    """Run a read-only precheck command and return exit code plus combined output."""

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except FileNotFoundError as exc:
        return 127, str(exc)


def write_text(path: Path, content: str) -> Path:
    """Write a UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def copy_file(label: str, source: Path, destination: Path) -> CopyResult:
    """Copy one file if it exists and report the result."""

    if not source.exists():
        return CopyResult(label, source, None, "MISSING")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return CopyResult(label, source, destination, "COPIED")


def relative_to_project(path: Path | None) -> str:
    """Return a project-relative path string where possible."""

    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def copy_delivery_files() -> list[CopyResult]:
    """Copy required tables, documents, and selected figures to the delivery pack."""

    results: list[CopyResult] = []

    table_files = [
        "final_decision_matrix.csv",
        "scre_revision_global_comparison.csv",
        "final_winners_taiwan.csv",
        "final_winners_heloc.csv",
        "statistical_tests_cleaned.csv",
        "kendalls_w_stability_taiwan.csv",
        "kendalls_w_stability_heloc.csv",
        "claim_control_matrix.csv",
    ]
    for name in table_files:
        results.append(copy_file(name, TABLES_DIR / name, DELIVERY_DIR / "tables" / name))

    doc_files = [
        OUTPUTS_DIR / "final_positioning_statement.md",
        OUTPUTS_DIR / "final_revision_audit.md",
        PROJECT_ROOT / "PROJECT_SCOPE.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "environment.yml",
        PROJECT_ROOT / "run_all.py",
    ]
    for source in doc_files:
        results.append(copy_file(source.name, source, DELIVERY_DIR / "project_metadata" / source.name))

    figure_sources = [
        OUTPUTS_DIR / "figures" / "roc_curves.png",
        OUTPUTS_DIR / "figures" / "precision_recall_curves.png",
        OUTPUTS_DIR / "figures" / "calibration_curve.png",
        OUTPUTS_DIR / "figures" / "calibration_curve_taiwan.png",
        OUTPUTS_DIR / "figures" / "calibration_curve_heloc.png",
        OUTPUTS_DIR / "figures" / "cost_curve.png",
        OUTPUTS_DIR / "figures" / "cost_curve_taiwan.png",
        OUTPUTS_DIR / "figures" / "cost_curve_heloc.png",
        OUTPUTS_DIR / "figures" / "kendalls_w_comparison.png",
        OUTPUTS_DIR / "figures" / "shap_rank_stability_heatmap.png",
        OUTPUTS_DIR / "figures" / "manual_review_band_distribution_taiwan.png",
        OUTPUTS_DIR / "figures" / "manual_review_band_distribution_heloc.png",
        OUTPUTS_DIR / "figures" / "taiwan_vs_heloc_metrics.png",
        OUTPUTS_DIR / "figures" / "taiwan_vs_heloc_calibration.png",
        OUTPUTS_DIR / "figures" / "ablation_results_barplot.png",
        OUTPUTS_DIR / "figures" / "ablation_cost_vs_auc.png",
        OUTPUTS_DIR / "shap" / "global_bar.png",
        OUTPUTS_DIR / "shap" / "global_beeswarm.png",
        OUTPUTS_DIR / "shap" / "external_validation_heloc" / "global_bar.png",
        OUTPUTS_DIR / "shap" / "external_validation_heloc" / "global_beeswarm.png",
    ]
    for source in figure_sources:
        if source.is_relative_to(OUTPUTS_DIR / "shap"):
            relative = source.relative_to(OUTPUTS_DIR / "shap")
            destination = DELIVERY_DIR / "figures" / "shap" / relative
        else:
            destination = DELIVERY_DIR / "figures" / source.name
        results.append(copy_file(source.name, source, destination))

    return results


def write_precheck_files() -> None:
    """Write environment and tree precheck artifacts into the delivery pack."""

    command_log: list[str] = []

    commands = {
        "pwd": ["powershell", "-NoProfile", "-Command", "pwd"],
        "tree_L3_requested": ["tree", "-L", "3"],
        "tree_windows_fallback": ["tree", "/A", "/F"],
        "python_version": [str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable), "--version"],
        "pip_freeze": [str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable), "-m", "pip", "freeze"],
    }
    for name, command in commands.items():
        code, output = run_command(command)
        command_log.append(f"$ {' '.join(command)}\nexit_code={code}\n{output}\n")
        write_text(DELIVERY_DIR / "environment" / f"{name}.txt", output)

    write_text(DELIVERY_DIR / "environment" / "precheck_command_log.txt", "\n".join(command_log))


def write_delivery_inventory(copy_results: list[CopyResult]) -> Path:
    """Write a CSV inventory of files included in the delivery pack."""

    inventory_path = DELIVERY_DIR / "delivery_pack_file_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "source", "destination", "status", "bytes"],
        )
        writer.writeheader()
        for result in copy_results:
            bytes_count = result.destination.stat().st_size if result.destination and result.destination.exists() else 0
            writer.writerow(
                {
                    "label": result.label,
                    "source": relative_to_project(result.source),
                    "destination": relative_to_project(result.destination),
                    "status": result.status,
                    "bytes": bytes_count,
                }
            )
    return inventory_path


def result_summary(dataset: str) -> pd.DataFrame:
    """Read final comparison results for one dataset."""

    return pd.read_csv(TABLES_DIR / f"scre_revision_comparison_{dataset}.csv")


def row_for_model(table: pd.DataFrame, model: str) -> pd.Series:
    """Return one model row from a final comparison table."""

    rows = table.loc[table["model"].eq(model)]
    if rows.empty:
        raise ValueError(f"Missing model row: {model}")
    return rows.iloc[0]


def fmt(value: object) -> str:
    """Format numeric values for audit text."""

    number = float(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    return f"{number:.4f}"


def create_zip(date_stamp: str) -> Path:
    """Create a zip archive for the delivery pack."""

    zip_base = OUTPUTS_DIR / f"SCRE_Credit_homework_delivery_pack_{date_stamp}"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", DELIVERY_DIR))
    return zip_path


def should_ignore_backup(directory: str, names: list[str]) -> set[str]:
    """Ignore temporary/cache directories when copying the freeze backup."""

    ignored_names = {
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
    }
    return {name for name in names if name in ignored_names or name.endswith(".tmp")}


def create_backup(timestamp: str) -> Path:
    """Create the timestamped project backup outside the project directory."""

    backup_root = PROJECT_ROOT.parent / "project_backups"
    backup_path = backup_root / f"SCRE_Credit_homework_freeze_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        raise FileExistsError(f"Backup path already exists: {backup_path}")

    backup_path.mkdir(parents=True)
    folder_names = ["src", "experiments", "data", "outputs", "notebooks", "tests", "archive"]
    file_names = ["requirements.txt", "environment.yml", "README.md", "PROJECT_SCOPE.md", "run_all.py", ".gitignore", ".gitattributes"]

    for folder_name in folder_names:
        source = PROJECT_ROOT / folder_name
        if source.exists():
            shutil.copytree(source, backup_path / folder_name, ignore=should_ignore_backup)

    for file_name in file_names:
        source = PROJECT_ROOT / file_name
        if source.exists():
            shutil.copy2(source, backup_path / file_name)

    return backup_path


def build_audit(
    freeze_dt: datetime,
    project_version: str,
    backup_path: Path,
    zip_path: Path,
    copy_results: list[CopyResult],
) -> str:
    """Build the homework freeze audit markdown."""

    taiwan = result_summary("taiwan")
    heloc = result_summary("heloc")
    taiwan_cat = row_for_model(taiwan, "Best CatBoost")
    taiwan_scre_opt = row_for_model(taiwan, "SCRE-Optimized")
    heloc_scorecard = row_for_model(heloc, "Best Scorecard")
    heloc_scre_opt = row_for_model(heloc, "SCRE-Optimized")
    missing = [result for result in copy_results if result.status != "COPIED"]
    can_use = "YES" if not missing else "NO"

    included_lines = [
        f"- `{relative_to_project(result.destination)}` from `{relative_to_project(result.source)}`"
        for result in copy_results
        if result.status == "COPIED"
    ]
    missing_lines = [
        f"- `{relative_to_project(result.source)}` ({result.label})"
        for result in missing
    ] or ["- None"]

    return f"""# Homework Freeze Audit

## Freeze Metadata

- Freeze date: {freeze_dt.strftime('%Y-%m-%d %H:%M')}
- Project version: {project_version}
- Git commit hash: not available; Git command was not available in PATH.
- Backup path: `{backup_path}`
- Delivery zip: `{zip_path}`

## Main Datasets

- Primary dataset: UCI Default of Credit Card Clients / Taiwan.
- External validation dataset: FICO HELOC.

## Final Taiwan Result Summary

- Taiwan operational winner: CatBoost.
- CatBoost ROC-AUC: {fmt(taiwan_cat['roc_auc'])}
- CatBoost PR-AUC: {fmt(taiwan_cat['pr_auc'])}
- CatBoost recall: {fmt(taiwan_cat['recall'])}
- CatBoost expected cost FN=5, FP=1: {fmt(taiwan_cat['expected_cost'])}
- Taiwan PR-AUC winner: SCRE-Optimized, PR-AUC {fmt(taiwan_scre_opt['pr_auc'])}, expected cost {fmt(taiwan_scre_opt['expected_cost'])}.

## Final HELOC Result Summary

- HELOC cost/interpretable winner: Scorecard.
- Scorecard ROC-AUC: {fmt(heloc_scorecard['roc_auc'])}
- Scorecard PR-AUC: {fmt(heloc_scorecard['pr_auc'])}
- Scorecard recall: {fmt(heloc_scorecard['recall'])}
- Scorecard expected cost FN=5, FP=1: {fmt(heloc_scorecard['expected_cost'])}
- HELOC SCRE-Optimized PR-AUC: {fmt(heloc_scre_opt['pr_auc'])}, expected cost {fmt(heloc_scre_opt['expected_cost'])}.

## Final Model Decision

- Taiwan operational winner: CatBoost.
- Taiwan PR-AUC winner: SCRE-Optimized.
- HELOC cost/interpretable winner: Scorecard.
- Proposed framework: SCRE-Optimized.

## Safe Claim

Best single calibrated CatBoost provides the strongest Taiwan operational result under the explicit FN=5, FP=1 cost scenario, while SCRE-Credit provides a structured reliability-aware framework for comparing and integrating performance, calibration, cost, stability, and faithfulness evidence.

## Unsafe Claims

- Do not claim SCRE-Credit outperforms all individual models.
- Do not claim SCRE-Credit is the best model on Taiwan.
- Do not claim SCRE-Credit is the best model on HELOC.
- Do not claim SHAP/LIME evidence is causal.
- Do not claim HELOC proves universal external robustness.
- Do not claim this is a new fundamental machine-learning algorithm.

## Files Included In Delivery Pack

{chr(10).join(included_lines)}

## Missing Files

{chr(10).join(missing_lines)}

## Can This Version Be Used For Homework Report?

{can_use}
"""


def main() -> None:
    """Create the homework freeze delivery pack, zip archive, and backup copy."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=None, help="Optional timestamp override in YYYYMMDD_HHMM format.")
    args = parser.parse_args()

    if args.timestamp:
        freeze_dt = datetime.strptime(args.timestamp, "%Y%m%d_%H%M")
        timestamp = args.timestamp
    else:
        freeze_dt = datetime.now()
        timestamp = freeze_dt.strftime("%Y%m%d_%H%M")
    date_stamp = timestamp.split("_")[0]
    project_version = f"homework_v1_scre_credit_{date_stamp}"

    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    write_precheck_files()
    copy_results = copy_delivery_files()

    planned_backup_path = PROJECT_ROOT.parent / "project_backups" / f"SCRE_Credit_homework_freeze_{timestamp}"
    planned_zip_path = OUTPUTS_DIR / f"SCRE_Credit_homework_delivery_pack_{date_stamp}.zip"

    audit_path = DELIVERY_DIR / "HOMEWORK_FREEZE_AUDIT.md"
    summary_path = DELIVERY_DIR / "FREEZE_SUMMARY.txt"
    inventory_path = DELIVERY_DIR / "delivery_pack_file_inventory.csv"
    final_copy_results = [
        *copy_results,
        CopyResult("delivery_pack_file_inventory.csv", inventory_path, inventory_path, "COPIED"),
        CopyResult("HOMEWORK_FREEZE_AUDIT.md", audit_path, audit_path, "COPIED"),
        CopyResult("FREEZE_SUMMARY.txt", summary_path, summary_path, "COPIED"),
    ]

    audit_text = build_audit(
        freeze_dt=freeze_dt,
        project_version=project_version,
        backup_path=planned_backup_path,
        zip_path=planned_zip_path,
        copy_results=final_copy_results,
    )
    write_text(audit_path, audit_text)
    write_text(
        summary_path,
        "\n".join(
            [
                f"project_version={project_version}",
                "git_commit=not_created_git_unavailable",
                "git_tag=not_created_git_unavailable",
                f"backup_path={planned_backup_path}",
                f"zip_path={planned_zip_path}",
                f"audit_path={audit_path}",
            ]
        ),
    )
    write_delivery_inventory(final_copy_results)
    write_delivery_inventory(final_copy_results)

    zip_path = create_zip(date_stamp)
    backup_path = create_backup(timestamp)

    print(f"project_version={project_version}")
    print("git_commit=not_created_git_unavailable")
    print("git_tag=not_created_git_unavailable")
    print(f"backup_path={backup_path}")
    print(f"zip_path={zip_path}")
    print(f"audit_path={audit_path}")
    print(f"inventory_path={inventory_path}")


if __name__ == "__main__":
    main()
