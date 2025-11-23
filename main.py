import os

import pandas as pd

from datafill.config import RIVER_DATA_PATH
from datafill.river import parse_river
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("classifier", LogisticRegression(solver="liblinear", max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    return pipeline
    

if __name__ == "__main__":
    model = train()