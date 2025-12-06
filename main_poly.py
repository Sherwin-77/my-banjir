"""
main_poly_grid.py

- Train / validation / test split (80/20).
- Pipeline: PolynomialFeatures -> StandardScaler -> LogisticRegression (saga).
- GridSearchCV over penalty (l1, l2), C, and polynomial degree.
- Returns train vs validation scores (from CV), prints best params and final test evaluation.
- Exports the trained best estimator to /mnt/data/logreg_poly_best.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from datafill.config import OUT_MODEL_PATH, RIVER_DATA_PATH
from datafill.river import parse_river

SELECTED_FEATURES = [
    "avg_rainfall",
    "max_rainfall",
    "elevation",
    "slope",
    "distance_to_river",
]
TARGET_FEATURE = "banjir"

RANDOM_STATE = 42
TEST_SIZE = 0.4
CV_FOLDS = 5


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)


def main():
    if not os.path.exists(RIVER_DATA_PATH):
        print("Data with river distances not found. Running river parsing...")
        parse_river()

    df = load_data(RIVER_DATA_PATH)
    X = df[SELECTED_FEATURES]
    y = df[TARGET_FEATURE]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", max_iter=20000, n_jobs=-1, random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "poly__degree": [1, 2], 
        "clf__penalty": ["l1", "l2"], 
        "clf__C": [0.01, 0.1, 1, 10]
    }

    # Use StratifiedKFold for inner CV
    inner_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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

    best_model = grid.best_estimator_
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

    best_params = grid.best_params_
    if best_params.get("clf__penalty") == "l1":
        poly_step = best_model.named_steps["poly"]
        clf_step = best_model.named_steps["clf"]

        feat_names = poly_step.get_feature_names_out(SELECTED_FEATURES)

        coefs = clf_step.coef_.ravel()
        # Note: coefficients correspond to scaled features, so to interpret sizes you may rescale back if needed
        nonzero_idx = np.where(np.abs(coefs) > 1e-8)[0]
        kept_feats = [(feat_names[i], coefs[i]) for i in nonzero_idx]
        kept_df = pd.DataFrame(kept_feats, columns=["feature", "coef"]).sort_values(
            by="coef", key=lambda s: np.abs(s), ascending=False
        )
        print("\nNon-zero coefficients (L1 selected features):")
        print(kept_df.to_string(index=False))


if __name__ == "__main__":
    main()
