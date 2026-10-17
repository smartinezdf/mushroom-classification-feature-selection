"""
Mushroom Classification — Feature Selection and Explainable ML.

Run:
    python mushroom_classification.py

Optional:
    python mushroom_classification.py path/to/mushrooms.csv

This script uses five-fold cross validation only. It reports:
1. Accuracy using all features.
2. Mutual information ranking.
3. SHAP feature ranking.
4. Accuracy using five MI-selected features and five SHAP-selected features.

Feature selection is repeated using only training data in each fold.
Full-data rankings are also printed for interpretation.
Dependencies: numpy, pandas, scikit-learn, shap (see requirements.txt).
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COLUMN = "class"
UCI_MUSHROOM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mushroom/agaricus-lepiota.data"
)
MUSHROOM_COLUMNS = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]


def find_data_file() -> Path | None:
    """Find mushrooms.csv in common locations."""
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find file from argument: {candidate}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "mushrooms.csv",
        script_dir / "mushroom.csv",
        Path.cwd() / "mushrooms.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_mushroom_data() -> tuple[pd.DataFrame, str]:
    data_path = find_data_file()
    if data_path is not None:
        return pd.read_csv(data_path), str(data_path)

    # This allows running only the .py file if internet access is
    # available. If not, the script still explains how to run it locally.
    try:
        with urllib.request.urlopen(UCI_MUSHROOM_URL, timeout=20) as response:
            df = pd.read_csv(response, header=None, names=MUSHROOM_COLUMNS)
        return df, UCI_MUSHROOM_URL
    except urllib.error.URLError as exc:
        raise FileNotFoundError(
            "Could not find mushrooms.csv locally and could not download the "
            "UCI Mushroom data. Put mushrooms.csv in the same folder as "
            "mushroom_classification.py or run: python mushroom_classification.py path/to/mushrooms.csv"
        ) from exc


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model(feature_names: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), feature_names),
        ],
        remainder="drop",
    )

    classifier = ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("one_hot", preprocessor),
            ("classifier", classifier),
        ]
    )


def five_fold_accuracy(X: pd.DataFrame, y: np.ndarray, feature_names: list[str]) -> np.ndarray:
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    model = make_model(feature_names)
    return cross_val_score(
        model,
        X[feature_names],
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )


def mutual_information_scores(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    # Mutual information needs numeric inputs. Each categorical feature is
    # encoded as integer categories and marked as discrete.
    X_encoded = X.apply(lambda col: pd.factorize(col)[0])
    scores = mutual_info_classif(
        X_encoded,
        y,
        discrete_features=True,
        random_state=RANDOM_STATE,
    )
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def get_positive_class_shap_values(shap_values) -> np.ndarray:
    """Return SHAP values for the positive/poisonous class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 2:
        return shap_array
    if shap_array.ndim == 3:
        if shap_array.shape[-1] == 2:
            return shap_array[:, :, 1]
        if shap_array.shape[0] == 2:
            return shap_array[1, :, :]

    raise ValueError(f"Unexpected SHAP value shape: {shap_array.shape}")


def shap_scores_by_original_feature(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The required package 'shap' is not installed. Install it with: "
            "python -m pip install shap"
        ) from exc

    X_one_hot = pd.get_dummies(X, prefix_sep="=", dtype=float)
    model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_one_hot, y)

    explainer = shap.TreeExplainer(model)
    shap_values = get_positive_class_shap_values(explainer.shap_values(X_one_hot))
    abs_shap = np.abs(shap_values)

    grouped_scores: dict[str, float] = {}
    encoded_columns = list(X_one_hot.columns)
    for original_feature in X.columns:
        matching_indices = [
            idx
            for idx, encoded_name in enumerate(encoded_columns)
            if encoded_name.startswith(f"{original_feature}=")
        ]
        grouped_scores[original_feature] = abs_shap[:, matching_indices].sum(axis=1).mean()

    return pd.Series(grouped_scores).sort_values(ascending=False)


def print_scores(title: str, scores: np.ndarray) -> None:
    print(title)
    print("-" * len(title))
    for fold_number, score in enumerate(scores, start=1):
        print(f"Fold {fold_number}: {score:.6f}")
    print(f"Mean accuracy: {scores.mean():.6f}")
    print(f"Std accuracy:  {scores.std():.6f}")
    print()


def five_fold_selected_accuracy(X, y, method):
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = []
    predictions = np.empty_like(y)

    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # Evaluation rows and labels never enter feature selection.
        if method == "mi":
            ranking = mutual_information_scores(X_train, y_train)
        elif method == "shap":
            ranking = shap_scores_by_original_feature(X_train, y_train)
        else:
            raise ValueError("method must be 'mi' or 'shap'")

        selected = list(ranking.head(5).index)
        model = make_model(selected)
        model.fit(X_train[selected], y_train)
        predicted = model.predict(X_test[selected])
        predictions[test_index] = predicted
        scores.append(accuracy_score(y_test, predicted))
        print(f"{method.upper()} fold {fold} selected: {selected}", flush=True)

    return np.asarray(scores), predictions


def main() -> None:
    df, data_source = load_mushroom_data()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column named '{TARGET_COLUMN}'.")

    # The UCI Mushroom data uses '?' for some unknown stalk-root values.
    # Treat it as its own category instead of deleting rows.
    df = df.astype(str).replace("?", "missing")

    X = df.drop(columns=[TARGET_COLUMN])
    y = LabelEncoder().fit_transform(df[TARGET_COLUMN])
    all_features = list(X.columns)

    print(f"Data source: {data_source}")
    print(f"Rows: {len(df)}")
    print(f"Total original features: {len(all_features)}")
    print()

    full_feature_scores = five_fold_accuracy(X, y, all_features)
    print_scores("Five-fold CV accuracy using ALL features", full_feature_scores)

    mi_scores = mutual_information_scores(X, y)
    print("Top 5 features by mutual information")
    print("------------------------------------")
    for rank, (feature, score) in enumerate(mi_scores.head(5).items(), start=1):
        print(f"{rank}. {feature}: {score:.6f}")
    print()

    shap_scores = shap_scores_by_original_feature(X, y)
    top_5_shap_features = list(shap_scores.head(5).index)

    print("Top 5 features by SHAP mean absolute value")
    print("------------------------------------------")
    for rank, (feature, score) in enumerate(shap_scores.head(5).items(), start=1):
        print(f"{rank}. {feature}: {score:.6f}")
    print()

    print("Selecting features within each training fold...")
    mi_cv_scores, _ = five_fold_selected_accuracy(X, y, "mi")
    print_scores(
        "Five-fold CV accuracy using 5 MI features selected within each fold",
        mi_cv_scores,
    )
    top_5_scores, fold_predictions = five_fold_selected_accuracy(X, y, "shap")
    print_scores(
        "Five-fold CV accuracy using 5 SHAP features selected within each fold",
        top_5_scores,
    )

    print("Final summary")
    print("-------------")
    print(f"Full-feature mean CV accuracy: {full_feature_scores.mean():.6f}")
    print(f"Top-5 MI mean CV accuracy: {mi_cv_scores.mean():.6f}")
    print(f"Top-5 SHAP mean CV accuracy: {top_5_scores.mean():.6f}")
    print(f"SHAP out-of-fold accuracy: {accuracy_score(y, fold_predictions):.6f}")
    print(f"Full-data top 5 SHAP features: {top_5_shap_features}")


if __name__ == "__main__":
    main()
