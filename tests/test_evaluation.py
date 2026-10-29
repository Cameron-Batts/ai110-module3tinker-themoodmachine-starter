"""Small, explicit checks for rule behavior and fold isolation."""

from mood_analyzer import MoodAnalyzer
from evaluate import evaluate, make_model
from dataset import SAMPLE_POSTS


def test_emoji_signal_is_preserved():
    model = MoodAnalyzer()
    assert model.predict_label("😂") == "positive"
    assert model.predict_label("😢") == "negative"


def test_mixed_emoji_and_word_signals():
    assert MoodAnalyzer().predict_label("happy 😢") == "mixed"


def test_custom_empty_vocabularies_stay_empty():
    assert MoodAnalyzer([], []).predict_label("happy 😂") == "neutral"


def test_negation_still_works():
    model = MoodAnalyzer()
    assert model.predict_label("not happy") == "negative"
    assert model.predict_label("not sad") == "positive"


def test_rule_sarcasm_limitation_stays_visible():
    assert MoodAnalyzer().predict_label("I absolutely love getting stuck in traffic") == "positive"


def test_test_vocabulary_is_not_fitted():
    model = make_model()
    model.fit(["pleasant good", "awful bad", "neutral ordinary", "mixed feelings"],
              ["positive", "negative", "neutral", "mixed"])
    model.predict(["sentinelunseenword"])
    assert "sentinelunseenword" not in model.named_steps["countvectorizer"].vocabulary_


def test_every_sample_is_evaluated_once_without_train_overlap():
    record = evaluate()
    test_indices = []
    for fold in record["ml_out_of_fold"]["folds"]:
        assert not set(fold["train_indices"]) & set(fold["test_indices"])
        test_indices.extend(fold["test_indices"])
    assert sorted(test_indices) == list(range(len(SAMPLE_POSTS)))
    assert len(test_indices) == len(set(test_indices))
    assert len(record["ml_out_of_fold"]["predictions"]) == len(SAMPLE_POSTS)
