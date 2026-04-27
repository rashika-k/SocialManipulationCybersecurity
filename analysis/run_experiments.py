import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_STATE = 42
TEST_SIZES = [0.10, 0.20, 0.30, 0.40]
DATA_PATH = Path("data/social_cyber_engineering_dataset.csv")
OUT_DIR = Path("reports")


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 1),
                    min_df=3,
                    max_features=2500,
                ),
                "messages",
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_models():
    return {
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=25, min_samples_leaf=5
        ),
        "NaiveBayes": MultinomialNB(alpha=0.5),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=120,
            n_jobs=1,
            max_depth=None,
            min_samples_leaf=2,
        ),
        "LinearSVM": LinearSVC(
            random_state=RANDOM_STATE,
            max_iter=5000,
            class_weight="balanced"
        ),
    }


def evaluate_splits(df, feature_cols, target_col, numeric_cols, categorical_cols):
    split_rows = []
    X = df[feature_cols]
    y = df[target_col]

    for test_size in TEST_SIZES:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        for model_name, model in build_models().items():
            clf = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                    ("model", model),
                ]
            )
            start_train = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time_s = time.perf_counter() - start_train

            start_pred = time.perf_counter()
            y_pred = clf.predict(X_test)
            pred_time_s = time.perf_counter() - start_pred

            split_rows.append(
                {
                    "model": model_name,
                    "train_size": round(1.0 - test_size, 2),
                    "test_size": round(test_size, 2),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_weighted": precision_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "train_time_s": train_time_s,
                    "predict_time_s": pred_time_s,
                }
            )
    return pd.DataFrame(split_rows)


def evaluate_cv(df, feature_cols, target_col, numeric_cols, categorical_cols):
    cv_rows = []
    X = df[feature_cols]
    y = df[target_col]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }

    for model_name, model in build_models().items():
        clf = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )
        scores = cross_validate(
            clf,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        cv_rows.append(
            {
                "model": model_name,
                "cv_folds": 10,
                "accuracy_mean": np.mean(scores["test_accuracy"]),
                "accuracy_std": np.std(scores["test_accuracy"]),
                "precision_weighted_mean": np.mean(scores["test_precision_weighted"]),
                "recall_weighted_mean": np.mean(scores["test_recall_weighted"]),
                "f1_weighted_mean": np.mean(scores["test_f1_weighted"]),
                "fit_time_mean_s": np.mean(scores["fit_time"]),
                "score_time_mean_s": np.mean(scores["score_time"]),
            }
        )
    return pd.DataFrame(cv_rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    target_col = "manipulation_type"
    dropped_cols = ["conversation_id", "original_conversation_id", "source_dataset"]

    feature_cols = [c for c in df.columns if c not in dropped_cols + [target_col]]

    numeric_cols = [
        "conversation_length",
        "attack_intensity",
        "victim_confusion_score",
        "pressure_score",
        "urgency_score",
        "threat_level",
        "self_doubt_score",
        "employee_extraversion",
        "employee_emotional_resilience",
        "employee_assertiveness",
        "avg_response_delay_seconds",
        "word_count_total",
        "question_count",
        "denial_count",
    ]

    categorical_cols = [
        "is_attack",
        "communication_channel",
        "relationship_context",
        "organizational_pretext",
        "employee_attachment_style",
        "escalation_pattern",
    ]

    split_df = evaluate_splits(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    cv_df = evaluate_cv(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    split_df.to_csv(OUT_DIR / "model_results_splits.csv", index=False)
    cv_df.to_csv(OUT_DIR / "model_results_cv10.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "dataset_path": str(DATA_PATH),
        "target": target_col,
        "dropped_columns": dropped_cols,
        "test_sizes": TEST_SIZES,
        "notes": "Missing values handled in-pipeline via imputers; raw dataset unchanged.",
    }
    (OUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Saved:", OUT_DIR / "model_results_splits.csv")
    print("Saved:", OUT_DIR / "model_results_cv10.csv")
    print("Saved:", OUT_DIR / "experiment_metadata.json")


if __name__ == "__main__":
    main()
