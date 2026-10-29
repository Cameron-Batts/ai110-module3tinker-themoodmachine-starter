# Mood Machine

Cameron Batts

An explainable text-classification experiment comparing word/emoji rules with a small bag-of-words logistic-regression model. The labels describe authored examples, not a person's actual emotional state.

## Current evaluation

```bash
python3 -m pip install -r requirements.txt
python3 evaluate.py --json validation/evaluation.json
python3 -m pytest -q
```

Verified results:

```text
Rule known-sample result: 13/14
ML out-of-fold result: 7/14
ML macro F1: 0.467
7 passed
```

The rule score is measured on examples used during development. The ML result uses two stratified folds: vocabulary building and model fitting occur inside each training fold, and each example receives one prediction while excluded from that fold's training data. The 14-example dataset was already used during project development, so this is not an independent external validation set. The two metrics have different evaluation conditions and should not be used to declare one approach universally better.

The historical 100% ML figure was training accuracy. It remains a demonstration of fitting, not evidence of generalization.

## Rule behavior

The classifier recognizes positive and negative words, selected emoji signals, immediate negation and mixed positive/negative signals. Emoji interpretation depends on context; the authored laughing-emoji example is treated as positive amusement. Sarcasm remains a known failure: the traffic example is still read literally.

## Interactive exploration

```bash
python3 main.py
python3 ml_experiments.py
```

These commands provide interactive rule and ML demonstrations. Use evaluate.py for the out-of-fold evaluation rather than relying on a training score.

## Files

- mood_analyzer.py: inspectable rules.
- dataset.py: original 14 labeled examples and vocabularies.
- evaluate.py: fold-isolated ML evaluation and parseable results.
- validation/evaluation.json: predictions, confusion matrix, fold indices and limitations.
- tests/test_evaluation.py: rule and evaluation-isolation regressions.
- model_card.md: data, bias and interpretation limits.

The original starter's provenance remains visible through this repository's GitHub fork lineage.
