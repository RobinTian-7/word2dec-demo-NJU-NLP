from collections import Counter
from pathlib import Path
import pickle

class Vocabulary:
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self._word_freq = Counter()

    def build(self, corpus):
        self._word_freq = Counter()

        for sentence in corpus:
            self._word_freq.update(sentence)

        filtered_words = [
            word for word, freq in self._word_freq.items() if freq >= self.min_count
        ]
        filtered_words.sort(key=lambda word: (-self._word_freq[word], word))

        self.word2idx = {
            word: idx for idx, word in enumerate(filtered_words)
        }
        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }
        return self

    def word_freq(self):
        return {
            word: self._word_freq[word]
            for word in self.word2idx
        }

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def save_cache(self, cache_path):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "min_count": self.min_count,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "word_freq": dict(self._word_freq),
        }

        with cache_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_cache(cls, cache_path):
        cache_path = Path(cache_path)
        with cache_path.open("rb") as f:
            payload = pickle.load(f)

        vocab = cls(min_count=payload["min_count"])
        vocab.word2idx = payload["word2idx"]
        vocab.idx2word = {
            int(idx): word for idx, word in payload["idx2word"].items()
        }
        vocab._word_freq = Counter(payload["word_freq"])
        return vocab

    
