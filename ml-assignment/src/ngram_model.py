import random
import re
from collections import defaultdict, Counter


class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Mapping: (word1, word2) -> Counter(next_word)
        self.trigram_counts = defaultdict(Counter)
        self.tokens = []  # store tokens for fallback when text is too short

    def _clean_and_tokenize(self, text):
        """
        Cleans and tokenizes the text into words.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
        tokens = text.split()
        return tokens

    def fit(self, text):
        """
        Trains the trigram model on the given text.
        """
        # Reset model
        self.trigram_counts = defaultdict(Counter)
        self.tokens = []

        if not text:
            return

        tokens = self._clean_and_tokenize(text)
        self.tokens = tokens

        # If not enough tokens for a trigram, no training needed
        if len(tokens) < 3:
            return

        # Count trigrams
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            self.trigram_counts[(w1, w2)][w3] += 1

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.
        """
        # If no tokens learned (empty text)
        if not self.tokens:
            return ""

        # If text too short to form trigrams → return cleaned text
        if len(self.trigram_counts) == 0:
            return " ".join(self.tokens[:max_length])

        # Randomly choose a starting context (bigram)
        context = random.choice(list(self.trigram_counts.keys()))
        w1, w2 = context
        generated = [w1, w2]

        # Generate next tokens
        for _ in range(max_length - 2):
            next_word_counts = self.trigram_counts.get((w1, w2), None)
            if not next_word_counts:
                break

            # Probabilistic sampling
            words, counts = zip(*next_word_counts.items())
            total = sum(counts)
            probabilities = [c / total for c in counts]
            next_word = random.choices(words, probabilities)[0]

            generated.append(next_word)
            w1, w2 = w2, next_word

        return " ".join(generated)
