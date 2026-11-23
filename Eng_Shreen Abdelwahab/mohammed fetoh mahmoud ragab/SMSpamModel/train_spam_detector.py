"""
SMS Spam Detection Model
Dataset: SMSSpamCollection
Algorithm: TF-IDF Vectorizer + Multinomial Naive Bayes Pipeline
"""

import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def load_dataset(filepath="SMSSpamCollection"):
    """Load the SMS dataset handling tab delimiter and quotes."""
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(
        filepath,
        sep="\t",
        names=["label", "message"],
        quoting=3,  # QUOTE_NONE: avoids issue with unescaped quotes in messages
        encoding="utf-8"
    )
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    return df


def build_pipeline():
    """Create an NLP classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.1))
    ])


def train_and_evaluate(df, model_save_path="spam_detector_model.joblib"):
    """Split data, train pipeline, evaluate performance, and save the model."""
    X = df["message"]
    y = df["label"]

    # Stratified split to preserve ham/spam ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    model = build_pipeline()
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(f"              Predicted Ham    Predicted Spam")
    print(f"Actual Ham         {cm[0][0]:<16} {cm[0][1]}")
    print(f"Actual Spam        {cm[1][0]:<16} {cm[1][1]}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully to '{model_save_path}'.\n")

    return model


def test_samples(model):
    """Run live inference on sample messages."""
    sample_messages = [
        "Hey, are we still meeting for lunch today at 1pm?",
        "URGENT! You have won a 1-week FREE holiday to Hawaii. Call 09061701461 to claim now!",
        "Can you send me the report when you get a chance?",
        "WINNER!! As a valued customer, you get £900 prize reward! Claim code: KL341. Valid 12 hours.",
        "Don't forget Mom's birthday tomorrow!"
    ]

    print("=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    for msg in sample_messages:
        prediction = model.predict([msg])[0]
        # predict_proba returns [prob_ham, prob_spam] or ordered by model.classes_
        classes = list(model.classes_)
        probs = model.predict_proba([msg])[0]
        prob_dict = dict(zip(classes, probs))
        confidence = prob_dict[prediction] * 100

        print(f"Message: \"{msg}\"")
        print(f"-> Prediction: [{prediction.upper()}] (Confidence: {confidence:.2f}%)\n")


if __name__ == "__main__":
    df = load_dataset()
    model = train_and_evaluate(df)
    test_samples(model)
