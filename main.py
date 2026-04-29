from pathlib import Path
import re
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


Ran_Seed = 42
Data_Set = Path(__file__).resolve().parent / "Data" / "social_cyber_engineering_dataset.csv"

T_Split = [0.30, 0.20, 0.10]
CrossV_Folds = 10

Target_Col = "is_attack"


Num_Feat = [

    "conversation_length",
    "word_count_total",
    "question_count",
    "denial_count"
]


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
    print("Rows after cleaning performed:", len(df))
    print("Duplicate Messages Removed From Data:", before_count - after_count)
    print("Total Columns:", len(df.columns))

    print("Attack count:")
    print(df["is_attack"].value_counts())

    print("Manipulation type counts:\n")
    print(df["manipulation_type"].value_counts())
    print("==============================")
    return df


def get_numeric_feat(df):
  
    available_features = [col for col in Num_Feat if col in df.columns]

    if not available_features:
        raise ValueError("No numeric features were found.")

    X = df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[Target_Col]

    return X, y, available_features


def train_naive_bayes(df, test_size):
  
    X = df["clean_text"]
    y = df[Target_Col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state= Ran_Seed,
        stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    X_train_text = vectorizer.fit_transform(X_train)
    X_test_text = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_text, y_train)

    predictions = model.predict(X_test_text)
    accuracy = accuracy_score(y_test, predictions)

    print("Naive Bayes Results\n")
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\n")
    print("==============================")
    print("Classification Report:\n")
    print("==============================")
    print(classification_report(y_test, predictions, zero_division=0))

    return accuracy


def train_decision_tree(df, test_size):
  
    X, y, features_used = get_numeric_feat(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=Ran_Seed,
        stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=Ran_Seed,
        max_depth=6
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Decision Tree Results\n")
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("==============================")
    print("Classification Report:")
    print("==============================")
    print(classification_report(y_test, predictions, zero_division=0))

    return accuracy


def train_random_forest(df, test_size):

    X, y, features_used = get_numeric_feat(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=Ran_Seed,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=Ran_Seed,
        max_depth=8
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Random Forest Results\n")
    print("Features used:", features_used)
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("==============================")
    print("Classification Report:")
    print("==============================")

    print(classification_report(y_test, predictions, zero_division=0))

    return accuracy


def run_all_splits(df):

    results = []

    for test_size in T_Split:
        train_percent = int((1 - test_size) * 100)
        test_percent = int(test_size * 100)

        print("==============================")
        print(f"Train/Test Split: {train_percent}% train / {test_percent}% test")
        print("==============================")

        nb_accuracy = train_naive_bayes(df, test_size)
        dt_accuracy = train_decision_tree(df, test_size)
        rf_accuracy = train_random_forest(df, test_size)

        results.append({
            "Split": f"{train_percent}/{test_percent}",
            "Naive Bayes": round(nb_accuracy, 4),
            "Decision Tree": round(dt_accuracy, 4),
            "Random Forest": round(rf_accuracy, 4)
        })

    results_df = pd.DataFrame(results)

    print("==============================")
    print("Accuracy Summary")
    print("==============================")
    print(results_df.to_string(index=False))


def run_cross_validation(df):

    print("==============================")
    print("10-Fold Cross-Validation")
    print("==============================")

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    cv = StratifiedKFold(
        n_splits=CrossV_Folds,
        shuffle=True,
        random_state=Ran_Seed
    )

    y = df[Target_Col]


    nb_pipe = [
        ("Naive Bayes", df["clean_text"], MultinomialNB(), "text")
    ]

    for model_name, X_data, model, feat_type in nb_pipe:
        if feat_type == "text":
            from sklearn.pipeline import Pipeline

            pipel = Pipeline([
                ("TF-IDF", TfidfVectorizer(stop_words="english", max_features=5000)),
                ("model", model)
            ])

            scores = cross_validate(
                pipel,
                X_data,
                y,
                cv=cv,
                scoring=scoring,
            )

            print_cv_results(model_name, scores)


    X_numeric, y_numeric, features_used = get_numeric_feat(df)

    numeric_models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=Ran_Seed,
            max_depth=6
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=Ran_Seed,
            max_depth=8
        )
    }

    print("Numeric features used for Decision Tree and Random Forest:")
    print(features_used)

    for model_name, model in numeric_models.items():
        scores = cross_validate(
            model,
            X_numeric,
            y_numeric,
            cv=cv,
            scoring=scoring,
        )

        print_cv_results(model_name, scores)

def add_data_risk_scr(df):

    avail_risk_columns = [col for col in Risk_Col if col in df.columns]

    
    df["dataset_risk_score"] = (
        df[avail_risk_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .mean(axis=1)
    )

    return df

def train_txt_risk_mdl(df):

    x = df["clean_text"]
    y = df["dataset_risk_score"]

    vect = TfidfVectorizer(stop_words="english", max_features=6000)

    x_txt = vect.fit_transform(x)

    mdl = RandomForestRegressor(
        n_estimators=100,
        random_state=Ran_Seed,
        max_depth=8
    )

    mdl.fit(x_txt, y)

    return mdl, vect


def print_cv_results(model_name, scores):

    print(f"{model_name} CV Results\n")
    print("Accuracy:", round(scores["test_accuracy"].mean(), 4))
    print("Precision:", round(scores["test_precision"].mean(), 4))
    print("Recall:", round(scores["test_recall"].mean(), 4))
    print("F1 Score:", round(scores["test_f1"].mean(), 4))


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
   
    sample = df.sample(1).iloc[0]

    risk_score = calculate_risk_score(sample)
    risk_level = get_risk_lvl(risk_score)
    

    print("==============================")
    print("Sample Alert")
    print("==============================")
    print("Conversation ID:", sample.get("conversation_id", "N/A"))
    print("Actual Label:", "Attack" if sample["is_attack"] == 1 else "Not Attack")
    print("Manipulation Type:", sample["manipulation_type"])
    print("Risk Score:", round(risk_score, 3))
    print("Risk Level:", risk_level)
    print("Sample Message Being Determined:")
    print(sample["clean_text"])

def analyze_cus_msg_risk(risk_model, risk_vec):
    
    print("==============================")
    print("Custom Risk Detection Message")
    print("==============================")

    choose = input("Would you like to enter a custom message to determine risk? Enter Y for Yes or N for No: ")

    if choose.lower() != "y":
        print("Skipping Custom Message...")
        return
    user_msg = input("Please Enter the custom message for risk detection: ")

    clean_msg = clean_mes_txt(user_msg)
    msg_vec = risk_vec.transform([clean_msg])

    risk_scr = risk_model.predict(msg_vec)[0]

    if risk_scr < 0:
        risk_scr = 0
    
    if risk_scr > 1:
        risk_scr = 1

    risk_lvl = get_risk_lvl(risk_scr)

    print("==============================")
    print("Detected Risk Score: ", round(risk_scr, 4))
    print("Detected Risk Level: ", risk_lvl)
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
