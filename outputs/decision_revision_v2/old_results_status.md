# Old Results Status

Generated: 2026-05-05 11:10:36
Archived prior decision revision outputs: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project\outputs\decision_revision_archive_20260505_111036`

## Status

The previous decision-revision outputs are retained and backed up, but they are now classified as **diagnostic retrospective evidence only**.

## Why

The old CatBoost manual-review result looks operationally strong, especially because it reduces false positives and keeps manual-review workload near the intended operational band. However, the old final recommendation and final operational ranks were derived from tables that include held-out test metrics. That makes the result useful for auditing and diagnosis, but not valid as final model-selection evidence under the stricter validation-only protocol.

## How The Old Results May Be Used

- They may be used to understand failure modes of the old cost-threshold model.
- They may be used to motivate the need for precision constraints and manual review.
- They may be used as retrospective diagnostic evidence.
- They may not be used as the final selection proof.

## What Must Happen Next

The CatBoost manual-review policy can still be used if it is selected again under Selection Protocol V2. That means the policy must be chosen from validation-only evidence, locked in a new selection log, and only then evaluated on the test set.

## Required Label For Existing Rankings

All old final ranking outputs should be labeled:

**Diagnostic retrospective evidence. Not valid as validation-only final model-selection evidence.**

## Current Interpretation

- Old CatBoost manual-review result: promising, but not final selection evidence.
- Old SCRE/Scorecard manual-review results: useful diagnostics and framework comparisons, but not final selection evidence.
- Old final recommendation: superseded by Selection Protocol V2 until a validation-only candidate registry and locked final policy are produced.
