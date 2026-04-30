from pathlib import Path
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


Ran_Seed = 42
Data_Set = Path(__file__).resolve().parent / "Data" / "social_cyber_engineering_dataset.csv"

T_Split = [0.30, 0.20, 0.10]
CrossV_Folds = 10

Target_Col = "is_attack"

Risk_Col = [
    "attack_intensity",
    "victim_confusion_score",
    "pressure_score",
    "urgency_score",
    "threat_level",
    "self_doubt_score"
]


def clean_mes_txt(value):

    if pd.isna(value):
        return ""

    text = str(value)

    for item in ["[", "]", "{", "}", "'", '"', "speaker", "text", "sentiment_score", ":", ","]:
        text = text.replace(item, " ")

    text = re.sub(r"\b-?\d+(\.\d+)?\b", " ", text)

    return " ".join(text.split())


def load_dataset():

    df = pd.read_csv(Data_Set)

    required_col = ["messages", "is_attack", "manipulation_type"]
    missing_col = [col for col in required_col if col not in df.columns]

    if missing_col:
        raise ValueError(f"Missing required columns: {missing_col}")

    df = df.dropna(subset=["messages", "is_attack"]).copy()
    df["is_attack"] = df["is_attack"].astype(int)
    df["clean_text"] = df["messages"].apply(clean_mes_txt)

    before_count = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    after_count = len(df)

    print("==============================")
    print("Dataset loaded successfully.")
    print("Dataset:", Data_Set)
    print("Rows after cleaning:", len(df))
    print("Duplicate messages removed:", before_count - after_count)
    print("Total columns:", len(df.columns))
    print("\nAttack count:")
    print(df["is_attack"].value_counts())
    print("\nManipulation type counts:")
    print(df["manipulation_type"].value_counts())
    print("==============================")

    return df


def add_data_risk_scr(df):

    avail_risk_columns = [col for col in Risk_Col if col in df.columns]

    df["dataset_risk_score"] = (
        df[avail_risk_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .mean(axis=1)
    )

    return df


def build_pipeline(model):
    """Wrap any classifier in a TF-IDF -> model pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("model", model)
    ])


def print_results(model_name, y_test, predictions):

    print(f"{model_name} Results\n")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("==============================")
    print("Classification Report:")
    print("==============================")
    print(classification_report(y_test, predictions, zero_division=0))


def run_all_splits(df):

    X = df["clean_text"]
    y = df[Target_Col]

    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=Ran_Seed, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=Ran_Seed, max_depth=8)
    }

    summary_rows = []

    for test_size in T_Split:
        train_pct = int((1 - test_size) * 100)
        test_pct = int(test_size * 100)

        print("==============================")
        print(f"Train/Test Split: {train_pct}% train / {test_pct}% test")
        print("Features: TF-IDF on clean_text (max 5,000 features), all models")
        print("==============================")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=Ran_Seed,
            stratify=y
        )

        row = {"Split": f"{train_pct}/{test_pct}"}

        for model_name, model in models.items():
            clf = build_pipeline(model)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            row[model_name] = round(acc, 4)
            print_results(model_name, y_test, predictions)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("==============================")
    print("Accuracy Summary")
    print("==============================")
    print(summary_df.to_string(index=False))


def print_cv_results(model_name, scores):

    print(f"{model_name} CV Results\n")
    print("Accuracy: ", round(scores["test_accuracy"].mean(), 4))
    print("Precision:", round(scores["test_precision"].mean(), 4))
    print("Recall:   ", round(scores["test_recall"].mean(), 4))
    print("F1 Score: ", round(scores["test_f1"].mean(), 4))
    print("==============================")


def run_cross_validation(df):

    print("==============================")
    print("10-Fold Cross-Validation")
    print("Features: TF-IDF on clean_text (max 5,000 features), all models")
    print("==============================")

    X = df["clean_text"]
    y = df[Target_Col]

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    cv = StratifiedKFold(n_splits=CrossV_Folds, shuffle=True, random_state=Ran_Seed)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=Ran_Seed, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=Ran_Seed, max_depth=8)
    }

    for model_name, model in models.items():
        clf = build_pipeline(model)
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        print_cv_results(model_name, scores)


def train_txt_risk_mdl(df):
    """Train all three regressors to predict risk score from text only."""

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import Ridge

    X = df["clean_text"]
    y = df["dataset_risk_score"]

    vect = TfidfVectorizer(stop_words="english", max_features=6000)
    X_vec = vect.fit_transform(X)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=Ran_Seed,
            max_depth=8
        ),
        "Decision Tree": DecisionTreeRegressor(
            random_state=Ran_Seed,
            max_depth=8
        ),
        "Ridge (text baseline)": Ridge(alpha=1.0)
    }

    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_vec, y)
        trained[name] = mdl

    return trained, vect


def calculate_risk_score(row):

    avail_risk_columns = [col for col in Risk_Col if col in row.index]

    if not avail_risk_columns:
        return 0.0

    values = pd.to_numeric(row[avail_risk_columns], errors="coerce").fillna(0)
    return values.mean()


def get_risk_lvl(score):

    if score >= 0.65:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"


def show_sample_alert(df):

    sample = df.sample(1, random_state=Ran_Seed).iloc[0]

    risk_score = calculate_risk_score(sample)
    risk_level = get_risk_lvl(risk_score)

    print("==============================")
    print("Sample Alert")
    print("==============================")
    print("Conversation ID:", sample.get("conversation_id", "N/A"))
    print("Actual Label:   ", "Attack" if sample["is_attack"] == 1 else "Not Attack")
    print("Manipulation Type:", sample["manipulation_type"])
    print("Risk Score:     ", round(risk_score, 3))
    print("Risk Level:     ", risk_level)
    print("\nMessage Text:")
    print(sample["clean_text"])


def analyze_cus_msg_risk(risk_models, risk_vec):

    print("==============================")
    print("Custom Risk Detection")
    print("==============================")

    choose = input("Would you like to enter a custom message to determine risk? (Y/N): ")

    if choose.lower() != "y":
        print("Skipping custom message check.")
        return

    user_msg = input("Enter the message: ")

    clean_msg = clean_mes_txt(user_msg)
    msg_vec = risk_vec.transform([clean_msg])

    print("==============================")
    scores = []
    for model_name, mdl in risk_models.items():
        risk_scr = float(mdl.predict(msg_vec)[0])
        risk_scr = max(0.0, min(1.0, risk_scr))
        risk_lvl = get_risk_lvl(risk_scr)
        scores.append(risk_scr)
        print(f"{model_name}")
        print(f"  Risk Score: {round(risk_scr, 4)}")
        print(f"  Risk Level: {risk_lvl}")
        print("------------------------------")

    avg_score = max(0.0, min(1.0, sum(scores) / len(scores)))
    print(f"Ensemble Average")
    print(f"  Risk Score: {round(avg_score, 4)}")
    print(f"  Risk Level: {get_risk_lvl(avg_score)}")
    print("==============================")


def main():

    df = load_dataset()
    df = add_data_risk_scr(df)

    run_all_splits(df)
    run_cross_validation(df)

    show_sample_alert(df)

    risk_mdl, risk_vect = train_txt_risk_mdl(df)
    analyze_cus_msg_risk(risk_mdl, risk_vect)


if __name__ == "__main__":
    main()