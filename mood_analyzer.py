"""Rule-based mood classification with simple negation."""

import re

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

POSITIVE_EMOJI = {"😂", "😊", "🙂", "😀", "😁", "😄", "😍", "❤"}
NEGATIVE_EMOJI = {"😞", "😢", "😭", "😠", "😡", "😔", "💔"}


class MoodAnalyzer:
    def __init__(self, positive_words=None, negative_words=None):
        if positive_words is None:
            positive_words = [*POSITIVE_WORDS, *POSITIVE_EMOJI]
        if negative_words is None:
            negative_words = [*NEGATIVE_WORDS, *NEGATIVE_EMOJI]

        self.positive_words = {word.lower() for word in positive_words}
        self.negative_words = {word.lower() for word in negative_words}

    def preprocess(self, text):
        return [token.rstrip("\ufe0f") for token in re.findall(
            r"[a-z]+(?:'[a-z]+)?|[😂😊🙂😀😁😄😍😞😢😭😠😡😔💔❤]\ufe0f?", text.lower())]

    def _word_scores(self, text):
        tokens = self.preprocess(text)
        results = []

        for index, token in enumerate(tokens):
            value = 0

            if token in self.positive_words:
                value += 1
            if token in self.negative_words:
                value -= 1

            negated = (
                index > 0
                and tokens[index - 1] in {"not", "never", "no"}
            )

            if negated:
                value = -value

            if value != 0:
                description = token
                if negated:
                    description = f"{tokens[index - 1]} {token}"
                results.append((description, value))

        return results

    def score_text(self, text):
        return sum(value for _, value in self._word_scores(text))

    def predict_label(self, text):
        results = self._word_scores(text)
        has_positive = any(value > 0 for _, value in results)
        has_negative = any(value < 0 for _, value in results)

        if has_positive and has_negative:
            return "mixed"

        score = self.score_text(text)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def explain(self, text):
        results = self._word_scores(text)
        positive = [word for word, value in results if value > 0]
        negative = [word for word, value in results if value < 0]

        return (
            f"Score = {self.score_text(text)} "
            f"(positive: {positive}, negative: {negative})"
        )
