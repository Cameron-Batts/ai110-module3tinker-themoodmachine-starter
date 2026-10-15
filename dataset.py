"""Word lists and labeled examples for the Mood Machine lab."""

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
    "grateful",
    "lit",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "frustrating",
    "disappointed",
    "miserable",
    "annoying",
]

SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "Lowkey proud of myself today",
    "This party is lit",
    "That was frustrating and annoying",
    "Feeling grateful but tired",
    "I absolutely love getting stuck in traffic",
    "😂",
    "The meeting starts at noon",
    "I am not sad anymore",
]

TRUE_LABELS = [
    "positive",  # I love this class so much
    "negative",  # Today was a terrible day
    "mixed",     # Feeling tired but kind of hopeful
    "neutral",   # This is fine
    "positive",  # So excited for the weekend
    "negative",  # I am not happy about this
    "positive",  # Lowkey proud of myself today
    "positive",  # This party is lit
    "negative",  # That was frustrating and annoying
    "mixed",     # Feeling grateful but tired
    "negative",  # Sarcasm about getting stuck in traffic
    "positive",  # Laughing emoji, interpreted as amusement
    "neutral",   # The meeting starts at noon
    "positive",  # I am not sad anymore
]

assert len(SAMPLE_POSTS) == len(TRUE_LABELS)