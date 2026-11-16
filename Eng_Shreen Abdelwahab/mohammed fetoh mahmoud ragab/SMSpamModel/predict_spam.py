
import sys
import joblib


def load_model(model_path="spam_detector_model.joblib"):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run `python train_spam_detector.py` first to train and save the model.")
        sys.exit(1)


def predict_message(model, text: str):
    pred = model.predict([text])[0]
    classes = list(model.classes_)
    probs = model.predict_proba([text])[0]
    prob_dict = dict(zip(classes, probs))
    return pred, prob_dict


def interactive_mode(model):
    print("=" * 60)
    print(" SMS Spam Detector - Interactive Mode")
    print(" Enter a message to check if it's HAM or SPAM.")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter message: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            label, probs = predict_message(model, user_input)
            confidence = probs[label] * 100
            print(f"Result: [{label.upper()}] with {confidence:.2f}% confidence")
            print(f"Details: Ham: {probs.get('ham', 0)*100:.2f}%, Spam: {probs.get('spam', 0)*100:.2f}%")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    model = load_model()

    if len(sys.argv) > 1:
        # Message provided as command line arguments
        message = " ".join(sys.argv[1:])
        label, probs = predict_message(model, message)
        confidence = probs[label] * 100
        print(f"\nMessage: \"{message}\"")
        print(f"Result : [{label.upper()}] ({confidence:.2f}% confidence)")
        print(f"Details: Ham: {probs.get('ham', 0)*100:.2f}%, Spam: {probs.get('spam', 0)*100:.2f}%\n")
    else:
        interactive_mode(model)
