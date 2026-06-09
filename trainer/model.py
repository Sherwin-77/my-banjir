import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from trainer.config import (
    OUT_MODEL_PATH,
    RIVER_DATA_PATH,
    SELECTED_FEATURES,
    TARGET_FEATURE,
)
from trainer.datafill.river import parse_river

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)

def print_top_coefficients(optimized_model, top_n=12):
    poly = optimized_model.named_steps["poly"]
    clf = optimized_model.named_steps["clf"]

    # transformed_features = preprocess.get_feature_names_out()
    model_features = poly.get_feature_names_out(SELECTED_FEATURES)
    coefficients = clf.coef_.ravel()

    coef_df = pd.DataFrame(
        {
            "feature": model_features,
            "coef": coefficients,
            "abs_coef": np.abs(coefficients),
        }
    ).sort_values("abs_coef", ascending=False)

    print("\nTop coefficients:")
    print(coef_df[["feature", "coef"]].head(top_n).to_string(index=False))

def train():
    if not os.path.exists(RIVER_DATA_PATH):
        print("Data with river distances not found. Running river parsing...")
        parse_river()

    df = load_data(RIVER_DATA_PATH)
    X = df[SELECTED_FEATURES]
    y = df[TARGET_FEATURE]

    # Print target balance
    print("Target distribution (counts):")
    print(y.value_counts(dropna=False))

    print("\nTarget distribution (percentages):")
    print(
        (y.value_counts(normalize=True, dropna=False) * 100).round(2).astype(str) + "%"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()), 
            (
                "clf",
                LogisticRegression(
                    max_iter=10000,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "poly__degree": [1, 2],
        "clf__penalty": ["l1", "l2", None],
        "clf__C": [0.01, 0.1, 1, 10, 100],
    }

    inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        return_train_score=True,
        n_jobs=-1,
        verbose=2,
    )
    print("Starting GridSearchCV on training set...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\nGridSearchCV complete.")
    print("Best params:", grid.best_params_)
    print("Best CV (validation) ROC AUC:", grid.best_score_)

    results = pd.DataFrame(grid.cv_results_)
    best_idx = grid.best_index_
    mean_train = results.loc[best_idx, "mean_train_score"]
    mean_val = results.loc[best_idx, "mean_test_score"]
    std_train = results.loc[best_idx, "std_train_score"]
    std_val = results.loc[best_idx, "std_test_score"]
    print(
        f"\nFor best params -> mean_train_roc_auc = {mean_train:.4f} (±{std_train:.4f}), "
        f"mean_val_roc_auc = {mean_val:.4f} (±{std_val:.4f})"
    )

    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)
    test_roc = roc_auc_score(y_test, y_test_proba)
    print("\nFinal test set ROC AUC:", test_roc)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix (test set):")
    print(cm)

    joblib.dump(best_model, OUT_MODEL_PATH)
    print(f"\nSaved best estimator to: {OUT_MODEL_PATH}")

    print_top_coefficients(best_model, top_n=20)


if __name__ == "__main__":
    train()
