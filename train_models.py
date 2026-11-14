"""
Train multiclass and binary IMDb rating classifiers.
Run from the repository root: python src/train_models.py
Place the Kaggle CSV at data/movies.csv.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "movies.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPECTED_COLUMNS = [
    "Title", "Rating", "Year", "Month", "Certificate", "Runtime",
    "Directors", "Stars", "Genre", "Filming_location", "Budget",
    "Income", "Country_of_origin"
]

def numeric_from_text(value):
    """Extract a numeric value from strings such as '$1,200,000' or '142 min'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan

def load_and_prepare(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload the Kaggle CSV and save it as data/movies.csv."
        )
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df.drop_duplicates().copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df[df["Rating"].between(0, 10)].copy()

    for col in ["Year", "Month", "Runtime", "Budget", "Income"]:
        df[col] = df[col].apply(numeric_from_text)

    # Keep only the predictors used in the project; exclude target leakage and
    # high-cardinality identifier/free-text columns.
    feature_cols = [
        "Year", "Month", "Certificate", "Runtime", "Genre",
        "Budget", "Income", "Country_of_origin"
    ]
    X = df[feature_cols].copy()
    return df, X

def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

def evaluate_task(df, X, task_name, y, models):
    # Remove rows with missing targets and align features.
    valid = y.notna()
    X_task = X.loc[valid]
    y_task = y.loc[valid].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_task, y_task, test_size=0.2, random_state=42, stratify=y_task
    )
    rows = []
    for model_name, estimator in models:
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", estimator)
        ])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        scores = {
            "Model": model_name,
            "Classification Type": task_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        rows.append(scores)
        print(f"\n{task_name} — {model_name}")
        print(classification_report(y_test, predictions, zero_division=0))
    return rows

def main():
    df, X = load_and_prepare()
    rating = df["Rating"]
    multiclass_y = pd.cut(
        rating, bins=[-np.inf, 5, 7, 8.5, np.inf],
        labels=["Poor", "Average", "Good", "Excellent"],
        right=False
    )
    # Match boundary definitions: <5, [5,7), [7,8.5), [8.5,10].
    binary_y = pd.Series(
        np.where(rating < 7, "Low-rated", "High-rated"), index=df.index
    )

    multiclass_models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
    ]
    binary_models = [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVM", SVC(kernel="rbf", class_weight="balanced")),
    ]

    results = []
    results += evaluate_task(df, X, "Multiclass", multiclass_y, multiclass_models)
    results += evaluate_task(df, X, "Binary", binary_y, binary_models)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_comparison_recomputed.csv", index=False)

    ax = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
        kind="bar", figsize=(11, 6), ylim=(0, 1), title="Recomputed Model Performance"
    )
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_recomputed.png", dpi=180)
    print("\nSaved results/model_comparison_recomputed.csv and .png")

if __name__ == "__main__":
    main()
