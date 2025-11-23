import os

import pandas as pd

from datafill.config import RIVER_DATA_PATH
from datafill.river import parse_river
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel


SELECTED_FEATURES = [
    "avg_rainfall",
    "max_rainfall",
    "elevation",
    "slope",
    "distance_to_river",
]
TARGET_FEATURE = "banjir"

def train():
    if not os.path.exists(RIVER_DATA_PATH):
        print("Data with river distances not found. Running river parsing...")
        parse_river()

    print("Loading data...")
    df = pd.read_csv(RIVER_DATA_PATH)
    X = df[SELECTED_FEATURES]
    y = df[TARGET_FEATURE]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )
    # create polynomial interactions (degree 2) but be careful with dimensionality
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # pipeline with logistic regression (we'll grid search penalty and C)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=5000))
    ])

    param_grid = {
        "clf__penalty": ["l2","l1"],
        "clf__C": [0.01, 0.1, 1, 10, 100]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train_poly, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV ROC AUC:", grid.best_score_)

    # pick best estimator, evaluate on test
    best = grid.best_estimator_
    y_pred = best.predict(X_test_poly)
    print(classification_report(y_test, y_pred))

    return pipeline
    

if __name__ == "__main__":
    model = train()