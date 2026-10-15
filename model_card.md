# Model Card: Mood Machine

## 1. Model Overview

I compared a rule-based classifier with a machine learning classifier.

Both classify short messages as positive, negative, neutral, or mixed. This is a learning project for exploring how data and modeling choices affect predictions.

The rule-based model uses word lists and simple negation. The ML model learns patterns from labeled examples using word counts and logistic regression.

## 2. Data

The dataset contains 14 posts: the original 6 plus 8 added examples.

It includes 6 positive, 4 negative, 2 neutral, and 2 mixed labels. The added examples include slang, an emoji, sarcasm, and mixed feelings.

I used AI-suggested examples and labels. These labels reflect an interpretation of each message, not an objective measurement of someone's mood. For example, I treated 😂 as positive amusement, but its meaning could change with context.

The dataset is small, favors positive examples, and does not represent many languages, dialects, or communication styles.

## 3. Rule-Based Model

The model lowercases text and extracts words without punctuation. Each positive word adds 1 and each negative word subtracts 1.

If a mood word immediately follows "not," "never," or "no," its score is reversed. If both positive and negative signals appear, the label is mixed. Otherwise, the score determines positive, negative, or neutral.

This approach is easy to inspect and explain. It correctly handled "I am not happy about this" and "Feeling tired but kind of hopeful."

It still reads sarcasm literally, ignores emojis, and misses unfamiliar words. Its negation rule only checks the immediately preceding word.

## 4. ML Model

The ML model uses CountVectorizer to turn text into word-count features and LogisticRegression to learn from SAMPLE_POSTS and TRUE_LABELS.

It was trained and evaluated on the same 14 posts. I did not measure ML performance before expanding the dataset, so I cannot claim how much adding examples changed its accuracy.

It can learn patterns without a manually written rule for each phrase. However, it ignores word order and can fit a tiny dataset without learning patterns that work reliably on new messages.

The default vectorizer does not represent the laughing emoji. A correct label for that example does not prove the model understood the emoji.

## 5. Evaluation

The rule-based model correctly classified 12 of 14 posts, approximately 86%.

The ML model correctly classified all 14 training posts, giving 100% training accuracy. This was not a held-out test, and the rules were also developed using the sample data. These results do not establish accuracy on new messages.

Correct examples included:

- "I am not happy about this" was negative in both models.
- "Feeling tired but kind of hopeful" was mixed in both models.
- "The meeting starts at noon" was neutral in both models.

Rule-based failures included:

- "I absolutely love getting stuck in traffic" was positive instead of the intended sarcastic negative.
- "😂" was neutral instead of the chosen positive label.

In informal checks, the ML model labeled "I am happy" positive, "I am not happy" negative, and "I am not sad" positive.

It labeled "Great, another delayed train" positive, missing the intended sarcasm. Getting the training sarcasm example right did not mean it could recognize a new sarcastic message.

## 6. Limitations

The dataset is too small to support broad conclusions. Neither model reliably understands context, sarcasm, or a person's actual emotional state.

The rule-based model depends on its vocabulary and simple rules. The ML model may rely on word associations that do not hold outside its training examples.

## 7. Ethical Considerations

Incorrect labels could misrepresent someone's message, especially when slang, dialect, sarcasm, or distress is involved.

Personal messages should be handled with consent and privacy protections. This classroom model should not be used to diagnose mental health or make consequential decisions about people.

## 8. Ideas for Improvement

- Add more varied examples and more neutral and mixed posts.
- Use a separate test set that is not used to develop rules or train the ML model.
- Review ambiguous labels with multiple people.
- Preserve emojis and improve negation handling.
- Try phrase features that retain some word-order information.
- Compare errors across different writing styles.