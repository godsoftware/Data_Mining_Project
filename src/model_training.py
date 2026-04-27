"""Model pipelines, resampling strategies, and hyperparameter search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config.settings import RANDOM_SEED
from data_preprocessing import PROJECT_ROOT, TARGET_COLUMN

try:  # Optional but recommended for this project.
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
except ImportError:  # pragma: no cover
    SMOTE = None
    SMOTEENN = None
    ImbalancedPipeline = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


@dataclass
class TrainTestData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> TrainTestData:
    """Create a stratified train/test split before fitting any transforms."""

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return TrainTestData(X_train, X_test, y_train, y_test)


def _one_hot_encoder() -> OneHotEncoder:
    """Create an encoder compatible with older and newer sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical variables."""

    categorical = [column for column in CATEGORICAL_COLUMNS if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", _one_hot_encoder(), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, X: pd.DataFrame, sampler=None):
    """Build a sklearn/imblearn pipeline with preprocessing inside CV folds."""

    steps = [("preprocessor", build_preprocessor(X))]
    if sampler is not None:
        if ImbalancedPipeline is None:
            raise ImportError("Install imbalanced-learn to use SMOTE/SMOTE-ENN.")
        steps.append(("sampler", sampler))
        steps.append(("model", model))
        return ImbalancedPipeline(steps)

    steps.append(("model", model))
    return Pipeline(steps)


def candidate_models(random_state: int = RANDOM_SEED) -> dict[str, object]:
    """Return baseline and tree-based candidate models."""

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=random_state,
        ),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            random_state=random_state,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    return models


def sampler_options(random_state: int = RANDOM_SEED) -> dict[str, object | None]:
    """Return imbalance-handling options. SMOTE options require imbalanced-learn."""

    samplers: dict[str, object | None] = {"no_resampling": None}
    if SMOTE is not None and SMOTEENN is not None:
        samplers["smote"] = SMOTE(random_state=random_state)
        samplers["smote_enn"] = SMOTEENN(random_state=random_state)
    return samplers


def hyperparameter_spaces() -> dict[str, dict[str, list]]:
    """Search spaces for RandomizedSearchCV."""

    spaces = {
        "logistic_regression": {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["l1", "l2"],
        },
        "logistic_regression_balanced": {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["l1", "l2"],
        },
        "decision_tree": {
            "model__max_depth": [3, 4, 5, 6, 8, 10, None],
            "model__min_samples_leaf": [20, 50, 100, 200],
        },
        "random_forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [4, 6, 8, 10, None],
            "model__min_samples_leaf": [10, 20, 50, 100],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.02, 0.04, 0.06, 0.08],
            "model__max_iter": [150, 250, 400],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    }

    if XGBClassifier is not None:
        spaces["xgboost"] = {
            "model__n_estimators": [250, 400, 600],
            "model__learning_rate": [0.02, 0.04, 0.06, 0.08],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.75, 0.85, 1.0],
            "model__colsample_bytree": [0.75, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5],
        }

    if LGBMClassifier is not None:
        spaces["lightgbm"] = {
            "model__n_estimators": [300, 500, 700],
            "model__learning_rate": [0.02, 0.04, 0.06],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [20, 50, 100],
            "model__subsample": [0.8, 0.9, 1.0],
        }

    return spaces


def tune_model(
    model_name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    random_state: int = RANDOM_SEED,
    scoring: str = "roc_auc",
) -> RandomizedSearchCV:
    """Tune a pipeline with stratified cross-validation."""

    spaces = hyperparameter_spaces()
    if model_name not in spaces:
        raise KeyError(f"No hyperparameter space defined for {model_name}.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spaces[model_name],
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True,
    )
    return search.fit(X_train, y_train)


def save_model(estimator, model_name: str) -> Path:
    """Persist a fitted model pipeline."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(estimator, path)
    return path
