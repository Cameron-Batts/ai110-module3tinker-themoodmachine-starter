# Mood Machine Model Card

Cameron Batts

## Intended use

A small text-labeling experiment for exploring explainable rules, failure cases and evaluation. Outputs are positive, negative, neutral or mixed labels for text. They do not establish a person's actual mood, mental health or suitability for a consequential decision.

## Data

The original 14 authored examples and labels are retained: six positive, four negative, two neutral and two mixed. They include slang, negation, an emoji, sarcasm and mixed sentiment. Labels reflect interpretation; the laughing emoji was labeled as positive amusement but can mean something else in context. This dataset is not representative of languages, dialects or real-world usage.

## Rule design

Positive and negative vocabulary contributes unit signals. An immediate not, never or no reverses the following sentiment signal. Positive and negative signals together produce mixed. Selected emoji signals are now preserved, while custom empty vocabularies remain empty.

The known laughing-emoji failure is fixed under its authored label. The traffic sarcasm example remains incorrectly positive. The rules do not understand intent, longer negation scope or conversational context.

## ML design and fold isolation

A CountVectorizer and LogisticRegression pipeline is fitted inside each of two shuffled stratified folds with seed 42. Every example is evaluated once while excluded from that fold's fitting data. Vocabulary is never fitted on the held-out fold. Training/test indices and predictions are saved in validation/evaluation.json.

There are only one or a few training examples per class in each fold. Although preprocessing is fold-isolated, the same dataset was already available during project development. This is a development-data estimate, not external validation.

## Current results

| Evaluation | Result | Interpretation |
|---|---:|---|
| Rule classifier, 14 known development examples | 13/14 | Checks familiar examples used to develop the rules |
| ML, out-of-fold predictions across 14 examples | 7/14 | Two-fold estimate on the same tiny development dataset |
| ML macro F1 | 0.467 | Includes all four labels with zero-division handling |
| Regression suite | 7 passed | Checks rules and fold isolation, not population accuracy |

The earlier rule score was 12/14 before emoji handling. The earlier ML score of 14/14 was training accuracy. These figures have different evaluation conditions and cannot support a general claim that one model is better.

## Limitations and responsible use

Sarcasm, ambiguous emoji, unfamiliar wording and dialect variation remain difficult. Labels can misrepresent a person's message. Neither system should diagnose emotions or support consequential decisions about people. Real feedback analysis would require consent, privacy protection, a larger separately labeled evaluation set and review of ambiguous cases.

## Next steps

Collect a separate evaluation set before further model tuning; use multiple labelers where interpretation is ambiguous; compare error categories and class-level recall rather than emphasizing an attractive training score.
