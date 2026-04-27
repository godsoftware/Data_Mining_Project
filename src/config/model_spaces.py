"""Model pools and search-space metadata for SCRE-Credit."""

TAIWAN_MODEL_POOL = [
    "M1_logistic_regression",
    "M2_woe_scorecard_logistic",
    "M3_random_forest",
    "M4_xgboost",
    "M5_lightgbm",
    "M6_catboost",
    "M7_monotonic_xgboost",
    "M8_monotonic_lightgbm",
    "M9_smotenc_xgboost",
    "M10_class_weighted_lightgbm",
]

HELOC_MODEL_POOL = [
    "M1_logistic_regression",
    "M2_woe_scorecard_logistic",
    "M3_xgboost",
    "M4_lightgbm",
    "M5_monotonic_xgboost",
    "M5_monotonic_lightgbm",
]

RELIABILITY_WEIGHTS = {
    "pr_auc_norm": 0.20,
    "roc_auc_norm": 0.15,
    "recall_norm": 0.15,
    "calibration_norm": 0.15,
    "cost_norm": 0.15,
    "stability_norm": 0.10,
    "faithfulness_norm": 0.10,
}

XGBOOST_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.04, 0.06, 0.08],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}

LIGHTGBM_SEARCH_SPACE = {
    "n_estimators": [300, 500, 700, 900],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.02, 0.04, 0.06],
    "min_child_samples": [20, 40, 80, 120],
}

CATBOOST_SEARCH_SPACE = {
    "iterations": [300, 500, 700],
    "depth": [4, 5, 6],
    "learning_rate": [0.02, 0.04, 0.06],
    "l2_leaf_reg": [1, 3, 5, 7],
}

