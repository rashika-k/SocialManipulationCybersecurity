import ast
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RANDOM_SEED = 42
DATASET_FILE = "social_cyber_engineering_dataset.csv"


def load_dataset(file_path):
    """Load the CSV dataset."""
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())

    return df


def extract_message_text(message_value):
    """
    Converts the messages column into plain text.

    The messages column contains list-like text such as:
    [{'speaker': 'A', 'text': 'message here'}, ...]
    """
    try:
        messages = ast.literal_eval(message_value)

        if isinstance(messages, list):
            text_parts = []

            for msg in messages:
                if isinstance(msg, dict) and "text" in msg:
                    text_parts.append(str(msg["text"]))

            return " ".join(text_parts)

    except Exception:
        return str(message_value)

    return str(message_value)


def prepare_text_data(df):
    """Prepare message text and labels for Naive Bayes."""
    df["clean_text"] = df["messages"].apply(extract_message_text)

    X_text = df["clean_text"]
    y = df["is_attack"]

    return X_text, y


def prepare_numeric_data(df):
    """Prepare numeric risk and behavior features for Decision Tree and Random Forest."""
    numeric_features = [
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
        "denial_count"
    ]

    X_numeric = df[numeric_features].copy()
    X_numeric = X_numeric.fillna(0)

    y = df["is_attack"]

    return X_numeric, y, numeric_features


def train_naive_bayes(X_text, y):
    """Train and evaluate a Naive Bayes model using message text."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    print("\n==============================")
    print("Naive Bayes Results")
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model, vectorizer


def train_decision_tree(X_numeric, y):
    """Train and evaluate a Decision Tree model using numeric features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=RANDOM_SEED,
        max_depth=6
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n==============================")
    print("Decision Tree Results")
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


def train_random_forest(X_numeric, y):
    """Train and evaluate a Random Forest model using numeric features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        max_depth=8
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n==============================")
    print("Random Forest Results")
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


def calculate_risk_score(row):
    """Calculate a simple average risk score from risk-related columns."""
    risk_columns = [
        "attack_intensity",
        "victim_confusion_score",
        "pressure_score",
        "urgency_score",
        "threat_level",
        "self_doubt_score"
    ]

    values = []

    for column in risk_columns:
        value = row.get(column, 0)

        if pd.isna(value):
            value = 0

        values.append(float(value))

    return sum(values) / len(values)


def get_risk_level(score):
    """Convert a numeric risk score into Low, Medium, or High."""
    if score >= 0.65:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"


def get_training_recommendation(manipulation_type):
    """Give a training recommendation based on the manipulation type."""
    recommendations = {
        "phishing": "Phishing awareness and suspicious link training",
        "credential_harvesting": "Password safety and credential protection training",
        "impersonation": "Sender verification and identity confirmation training",
        "pretexting": "Pretexting and social engineering awareness training",
        "baiting": "Suspicious offer and attachment safety training",
        "invoice_fraud": "Invoice verification and financial fraud training",
        "guilt_tripping": "Emotional manipulation awareness training",
        "gaslighting": "Psychological pressure recognition training",
        "love_bombing": "Manipulative trust-building awareness training",
        "charm_flattery": "Flattery-based manipulation awareness training",
        "direct_coercion": "Threat and coercion response training",
        "passive_aggressive": "Indirect pressure recognition training",
        "neutral": "General security awareness training",
        "benign_business": "No additional training required"
    }

    return recommendations.get(
        manipulation_type,
        "General social engineering awareness training"
    )


def show_sample_alert(df):
    """Show one sample alert from the dataset."""
    sample = df.sample(1, random_state=RANDOM_SEED).iloc[0]

    risk_score = calculate_risk_score(sample)
    risk_level = get_risk_level(risk_score)
    recommendation = get_training_recommendation(sample["manipulation_type"])

    print("\n==============================")
    print("Sample Alert Output")
    print("==============================")
    print("Conversation ID:", sample["conversation_id"])
    print("Source Dataset:", sample["source_dataset"])
    print("Channel:", sample["communication_channel"])
    print("Actual Attack Label:", "Attack" if sample["is_attack"] == 1 else "Not Attack")
    print("Manipulation Type:", sample["manipulation_type"])
    print("Risk Score:", round(risk_score, 3))
    print("Risk Level:", risk_level)
    print("Training Recommendation:", recommendation)

    print("\nMessage Text:")
    print(extract_message_text(sample["messages"]))


def main():
    df = load_dataset(DATASET_FILE)

    print("\nAttack label counts:")
    print(df["is_attack"].value_counts())

    print("\nManipulation type counts:")
    print(df["manipulation_type"].value_counts())

    print("\nSource dataset counts:")
    print(df["source_dataset"].value_counts())

    X_text, y_text = prepare_text_data(df)
    X_numeric, y_numeric, numeric_features = prepare_numeric_data(df)

    print("\nNumeric features used:")
    print(numeric_features)

    train_naive_bayes(X_text, y_text)
    train_decision_tree(X_numeric, y_numeric)
    train_random_forest(X_numeric, y_numeric)

    show_sample_alert(df)


if __name__ == "__main__":
    main()
