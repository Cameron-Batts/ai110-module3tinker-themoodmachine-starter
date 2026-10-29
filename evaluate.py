"""Reproduce known-sample rules and leakage-controlled two-fold ML evaluation."""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer

LABELS = ["negative", "neutral", "mixed", "positive"]


def make_model():
    return make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000, random_state=42))


def evaluate():
    rules = MoodAnalyzer()
    rule_predictions = [rules.predict_label(text) for text in SAMPLE_POSTS]
    predictions = [None] * len(SAMPLE_POSTS)
    folds = []
    split = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (train, test) in enumerate(split.split(SAMPLE_POSTS, TRUE_LABELS), 1):
        model = make_model()
        model.fit([SAMPLE_POSTS[i] for i in train], [TRUE_LABELS[i] for i in train])
        labels = model.predict([SAMPLE_POSTS[i] for i in test])
        for index, prediction in zip(test, labels):
            predictions[int(index)] = str(prediction)
        folds.append({"fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
                      "vocabulary_size": len(model.named_steps["countvectorizer"].vocabulary_)})
    assert all(prediction is not None for prediction in predictions)
    return {
        "samples": len(SAMPLE_POSTS), "labels": LABELS,
        "rule_known_samples": {"correct": sum(a == b for a, b in zip(rule_predictions, TRUE_LABELS)),
            "accuracy": accuracy_score(TRUE_LABELS, rule_predictions), "predictions": rule_predictions},
        "ml_out_of_fold": {"correct": sum(a == b for a, b in zip(predictions, TRUE_LABELS)),
            "accuracy": accuracy_score(TRUE_LABELS, predictions),
            "macro_f1": f1_score(TRUE_LABELS, predictions, labels=LABELS, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(TRUE_LABELS, predictions, labels=LABELS).tolist(),
            "predictions": predictions, "folds": folds},
        "cases": [{"text": text, "label": label, "rule_prediction": rule, "oof_prediction": ml}
            for text, label, rule, ml in zip(SAMPLE_POSTS, TRUE_LABELS, rule_predictions, predictions)],
        "limits": ["Fourteen authored examples are too few for broad accuracy conclusions.",
                   "Rules and examples were developed together; the rule score is not held-out accuracy.",
                   "ML vectorization and fitting occur inside each training fold, but the dataset was already used during project development.",
                   "Emoji labels are context-dependent; sarcasm remains a known rule failure."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="output")
    args = parser.parse_args()
    record = evaluate()
    print(f"Rule known-sample result: {record['rule_known_samples']['correct']}/{record['samples']}")
    print(f"ML out-of-fold result: {record['ml_out_of_fold']['correct']}/{record['samples']}")
    print(f"ML macro F1: {record['ml_out_of_fold']['macro_f1']:.3f}")
    print("Two stratified folds; vectorizer fitted only on each fold's training data.")
    print("Small development dataset; these are not population accuracy estimates.")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
