"""Utilities for comparing SHAP and LIME local explanations."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


RULE_SPLIT_PATTERN = re.compile(r"\s*(<=|>=|<|>|=)\s*")


def normalize_lime_rule_feature(rule: str, known_features: list[str]) -> str:
    """Extract a raw feature name from a LIME rule string."""

    cleaned = RULE_SPLIT_PATTERN.split(str(rule), maxsplit=1)[0].strip()
    if cleaned in known_features:
        return cleaned

    for feature in sorted(known_features, key=len, reverse=True):
        if str(rule).strip().startswith(feature):
            return feature
    return cleaned


def shap_lime_agreement(
    shap_contributions: pd.DataFrame,
    lime_contributions: pd.DataFrame,
    known_features: list[str],
    top_k: int = 10,
) -> dict[str, float | int]:
    """Compare local SHAP and LIME explanations with top-k overlap."""

    shap_top = (
        shap_contributions.assign(abs_value=lambda frame: frame["shap_value"].abs())
        .sort_values("abs_value", ascending=False)
        .head(top_k)
    )
    lime = lime_contributions.copy()
    lime["feature"] = lime["feature_rule"].map(
        lambda rule: normalize_lime_rule_feature(rule, known_features)
    )
    lime_top = lime.assign(abs_value=lambda frame: frame["lime_weight"].abs()).sort_values(
        "abs_value", ascending=False
    ).head(top_k)

    shap_set = set(shap_top["feature"])
    lime_set = set(lime_top["feature"])
    overlap = shap_set & lime_set
    union = shap_set | lime_set

    sign_agreements = []
    shap_sign = shap_top.set_index("feature")["shap_value"]
    lime_sign = lime_top.set_index("feature")["lime_weight"]
    for feature in overlap:
        sign_agreements.append(float(np.sign(shap_sign.loc[feature]) == np.sign(lime_sign.loc[feature])))

    return {
        "top_k": int(top_k),
        "shap_top_k": int(len(shap_set)),
        "lime_top_k": int(len(lime_set)),
        "overlap_count": int(len(overlap)),
        "jaccard": float(len(overlap) / len(union)) if union else np.nan,
        "sign_agreement": float(np.mean(sign_agreements)) if sign_agreements else np.nan,
    }

