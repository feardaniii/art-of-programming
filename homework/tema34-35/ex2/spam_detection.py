from __future__ import annotations

import argparse
import math
import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DETECTION_MODES = {
    "balanced": {"proba_threshold": 0.50, "margin_threshold": 0.00},
    "strict": {"proba_threshold": 0.40, "margin_threshold": -0.50},
}


# Step 1: Text normalization used by all vectorizers/models.
def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s$#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Step 2: Find a usable dataset automatically if user does not pass --data.
def find_dataset_path(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "art-of-programming/homework/tema34-35/ex2/spam.csv",
        repo_root / "art-of-programming/34-35_NLP_and_nltk/tasks/spam.csv",
        repo_root / "art-of-programming/Students Tasks/GigiC/tema_nlp/spam.csv",
        repo_root / "spam.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# Step 3: Load and standardize dataset to columns: label (0/1), text.
def load_dataset(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if df is None:
        raise ValueError(f"Could not decode dataset with supported encodings: {csv_path}") from last_error

    # Common format in SMS Spam Collection.
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "text"]
    # Alternative common naming.
    elif {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]].copy()
    else:
        # Fallback: first two columns are expected to be label/text.
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least two columns for label and text.")
        df = df.iloc[:, :2].copy()
        df.columns = ["label", "text"]

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df[df["text"].str.len() > 0]

    if df.empty:
        raise ValueError("Dataset became empty after cleaning.")
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both ham and spam classes.")

    return df


def augment_with_phishing_examples(df: pd.DataFrame) -> pd.DataFrame:
    spam_like = [
        "Immediate action required your account will be suspended unless you verify now",
        "Security alert suspicious login attempt verify your account immediately",
        "Invoice #78122 payment failed update billing details to avoid service interruption",
        "Your netflix account will be suspended in 24 hours verify your information now",
        "Geek Squad renewal charge detected call now if this was not you",
        "Your package is on hold confirm address and pay small redelivery fee",
        "Unusual sign in detected from new device confirm identity now",
        "Final warning your mailbox storage is full click to upgrade immediately",
        "Bank verification needed login now to prevent account lock",
        "You won a reward card claim now by confirming your card details",
        "Payment declined update card now to keep subscription active",
        "Tax refund available submit details to receive payment today",
        "We detected unauthorized activity confirm your password immediately",
        "Urgent notice account will be disabled unless details are confirmed",
        "Crypto investment opportunity guaranteed high return act now",
        "Invoice immediate attention required your $499.99 charge for geek squad is processing",
        "Invoice #78391 immediate attention required your $499.99 charge for geek squad is processing",
    ]

    ham_like = [
        "Hi team attached is the monthly invoice for your review",
        "Reminder we have a project meeting tomorrow at 10 am",
        "Please review the document and share feedback by Friday",
        "Can we reschedule our call to next week",
        "Your order has shipped and will arrive tomorrow",
        "I will be late to the office due to traffic",
        "Thanks for your message I will reply in the afternoon",
        "Dinner at 8 works for me see you then",
        "Please send me the updated presentation file",
        "The report looks good I only have a few minor comments",
        "Lets catch up next month when you are available",
        "Your appointment is confirmed for Monday morning",
        "I received your transfer thank you",
        "Can you share the meeting notes when ready",
        "Please call me when you are free",
    ]

    extra = pd.DataFrame(
        {
            "label": [1] * len(spam_like) + [0] * len(ham_like),
            "text": [normalize_text(t) for t in (spam_like + ham_like)],
        }
    )
    combined = pd.concat([df, extra], ignore_index=True)
    return combined


# Step 4: Train requested model variations and compare metrics.
def evaluate_models(df: pd.DataFrame) -> tuple[list[dict], dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    experiments = [
        (
            "TF-IDF + LogisticRegression",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        (
            "CountVectorizer + LogisticRegression",
            CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        (
            "CountVectorizer + LinearSVC",
            CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95),
            LinearSVC(random_state=42),
        ),
    ]

    results: list[dict] = []
    best_entry: dict | None = None

    for name, vectorizer, model in experiments:
        pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])
        pipeline.fit(x_train, y_train)
        y_pred = predict_labels(pipeline, x_test.tolist())

        entry = {
            "name": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_spam": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall_spam": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1_spam": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "pipeline": pipeline,
        }
        results.append(entry)

        if best_entry is None:
            best_entry = entry
        else:
            if (entry["recall_spam"], entry["f1_spam"], entry["accuracy"]) > (
                best_entry["recall_spam"],
                best_entry["f1_spam"],
                best_entry["accuracy"],
            ):
                best_entry = entry

    if best_entry is None:
        raise RuntimeError("No model was trained.")

    return results, best_entry


# Step 5: Save the best model for fast reuse.
def save_model(best_entry: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "spam_best_model.joblib"
    payload = {
        "model_name": best_entry["name"],
        "pipeline": best_entry["pipeline"],
    }
    joblib.dump(payload, model_path)
    return model_path


# Step 6: Print a compact, readable comparison table.
def print_results_table(results: list[dict]) -> None:
    print("\nModel comparison (spam-focused metrics):")
    print("-" * 94)
    print(f"{'Model':40s} {'Accuracy':>10s} {'Precision':>12s} {'Recall':>12s} {'F1':>10s}")
    print("-" * 94)
    for row in results:
        print(
            f"{row['name']:40s} "
            f"{row['accuracy']:10.4f} "
            f"{row['precision_spam']:12.4f} "
            f"{row['recall_spam']:12.4f} "
            f"{row['f1_spam']:10.4f}"
        )


def _estimate_confidence(pipeline: Pipeline, message: str) -> float | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = pipeline.predict_proba([message])[0][1]
        return float(proba)
    if hasattr(model, "decision_function"):
        margin = float(pipeline.decision_function([message])[0])
        return 1.0 / (1.0 + math.exp(-abs(margin)))
    return None


def _predict_binary_with_threshold(pipeline: Pipeline, cleaned_message: str, mode: str = "balanced") -> int:
    cfg = DETECTION_MODES.get(mode, DETECTION_MODES["balanced"])
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = float(pipeline.predict_proba([cleaned_message])[0][1])
        return int(proba >= cfg["proba_threshold"])
    if hasattr(model, "decision_function"):
        margin = float(pipeline.decision_function([cleaned_message])[0])
        return int(margin >= cfg["margin_threshold"])
    return int(pipeline.predict([cleaned_message])[0])


def predict_labels(pipeline: Pipeline, messages: list[str], mode: str = "balanced") -> list[int]:
    return [_predict_binary_with_threshold(pipeline, normalize_text(msg), mode=mode) for msg in messages]


def _predict_label_and_confidence(
    pipeline: Pipeline, raw_message: str, mode: str = "balanced"
) -> tuple[str, float | None]:
    clean = normalize_text(raw_message)
    pred = _predict_binary_with_threshold(pipeline, clean, mode=mode)
    label = "spam" if pred == 1 else "ham"
    confidence = _estimate_confidence(pipeline, clean)
    return label, confidence


def _print_prediction_result(index: int | None, label: str, confidence: float | None, message: str) -> None:
    idx = f"{index}. " if index is not None else ""
    if confidence is None:
        print(f"{idx}{label} | {message}")
    else:
        print(f"{idx}{label} ({confidence:.2%}) | {message}")


# Step 7A: Single-message interactive mode.
def interactive_single_predict(pipeline: Pipeline, mode: str) -> None:
    print("\nSingle message mode")
    print("Type one message per line. Type 'exit' to return to menu.")
    print(f"Detection mode: {mode}")

    while True:
        raw = input("\nMessage> ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("Returning to menu.")
            break
        if not raw:
            print("Please type a non-empty message.")
            continue

        label, confidence = _predict_label_and_confidence(pipeline, raw, mode=mode)
        _print_prediction_result(index=None, label=label, confidence=confidence, message=raw)


def _extract_batch_messages_from_csv(path: Path) -> list[str]:
    raw_df = pd.read_csv(path)
    lowered = {col.lower(): col for col in raw_df.columns}

    if "text" in lowered:
        series = raw_df[lowered["text"]]
    elif "v2" in lowered:
        series = raw_df[lowered["v2"]]
    elif raw_df.shape[1] >= 1:
        series = raw_df.iloc[:, -1]
    else:
        return []

    return [str(v).strip() for v in series.dropna().tolist() if str(v).strip()]


def _read_batch_messages_from_prompt() -> list[str]:
    print("\nPaste messages (one per line). Submit an empty line to finish.")
    messages: list[str] = []
    while True:
        line = input()
        if not line.strip():
            break
        messages.append(line.strip())
    return messages


# Step 7B: Batch mode for multiple messages at once.
def interactive_batch_predict(pipeline: Pipeline, mode: str) -> None:
    print("\nBatch mode")
    print(f"Detection mode: {mode}")
    print("Choose input source:")
    print("1. CSV path")
    print("2. Paste lines")

    source = input("Source> ").strip()
    messages: list[str] = []

    if source == "1":
        csv_input = input("CSV path> ").strip().strip('"')
        csv_path = Path(csv_input)
        if not csv_path.exists():
            print("CSV file not found.")
            return
        try:
            messages = _extract_batch_messages_from_csv(csv_path)
        except Exception as exc:
            print(f"Could not read CSV: {exc}")
            return
    elif source == "2":
        messages = _read_batch_messages_from_prompt()
    else:
        print("Invalid option.")
        return

    if not messages:
        print("No messages found.")
        return

    print("\nBatch predictions:")
    for i, msg in enumerate(messages, start=1):
        label, confidence = _predict_label_and_confidence(pipeline, msg, mode=mode)
        _print_prediction_result(index=i, label=label, confidence=confidence, message=msg)


def choose_detection_mode() -> str:
    while True:
        print("\nChoose detection mode:")
        print("1. Balanced (fewer false positives)")
        print("2. Strict anti-phishing (higher spam sensitivity)")
        mode_option = input("Mode> ").strip()
        if mode_option == "1":
            return "balanced"
        if mode_option == "2":
            return "strict"
        print("Invalid option. Please choose 1 or 2.")


# Step 7C: UX menu that lets the user choose one-by-one or dataset mode.
def interactive_predict_menu(pipeline: Pipeline) -> None:
    mode = choose_detection_mode()

    while True:
        print("\nWhat would you like to use?")
        print("1. Single line analyzer (spam/ham)")
        print("2. Dataset / batch analyzer")
        print("3. Change detection mode")
        print("4. Exit")
        print(f"Current mode: {mode}")
        option = input("Choice> ").strip()

        if option == "1":
            interactive_single_predict(pipeline, mode=mode)
        elif option == "2":
            interactive_batch_predict(pipeline, mode=mode)
        elif option == "3":
            mode = choose_detection_mode()
        elif option == "4":
            print("Exiting interactive mode.")
            return
        else:
            print("Invalid option. Please choose 1, 2, 3 or 4.")


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "art-of-programming").exists():
            return parent
    return start


# Main flow: load data -> compare models -> pick best -> save -> interactive menu.
def main() -> None:
    parser = argparse.ArgumentParser(description="Spam/Ham text classifier.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset with label/text columns.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive prediction mode.")
    parser.add_argument("--no-augment", action="store_true", help="Disable built-in phishing augmentation examples.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(script_dir)

    dataset_path = Path(args.data) if args.data else find_dataset_path(repo_root)
    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(
            "No dataset found. Provide --data path/to/spam.csv.\n"
            "Expected columns: v1,v2 or label,text."
        )

    print(f"Using dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    if not args.no_augment:
        df = augment_with_phishing_examples(df)
        print("Applied phishing-oriented augmentation examples.")
    print(f"Loaded {len(df)} rows. Spam ratio: {df['label'].mean():.2%}")

    results, best_entry = evaluate_models(df)
    print_results_table(results)

    print(
        f"\nBest model: {best_entry['name']} "
        f"(Recall spam: {best_entry['recall_spam']:.4f}, F1 spam: {best_entry['f1_spam']:.4f})"
    )
    model_path = save_model(best_entry, script_dir / "artifacts")
    print(f"Saved best model to: {model_path}")

    if not args.no_interactive:
        interactive_predict_menu(best_entry["pipeline"])


if __name__ == "__main__":
    main()
