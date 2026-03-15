import random
from collections import Counter

import numpy as np


def compute_discard_probabilities(word_freq, t=1e-5):
    total_tokens = sum(word_freq.values())
    if total_tokens <= 0:
        return {}

    discard_probs = {}
    for word, count in word_freq.items():
        freq = count / total_tokens
        if freq <= 0:
            discard_probs[word] = 0.0
            continue

        discard = 1.0 - np.sqrt(t / freq)
        discard_probs[word] = float(np.clip(discard, 0.0, 1.0))
    return discard_probs


def generate_training_pairs(corpus, word2idx, window_size=5, discard_probs=None, rng=None):
    pairs = []
    rng = rng or random

    for sentence in corpus:
        filtered_sentence = []
        for token in sentence:
            if token not in word2idx:
                continue

            if discard_probs is not None and rng.random() < discard_probs.get(token, 0.0):
                continue

            filtered_sentence.append(token)

        for i, center in enumerate(filtered_sentence):
            dynamic_window = rng.randint(1, window_size)
            start = max(0, i - dynamic_window)
            end = min(len(filtered_sentence), i + dynamic_window + 1)

            for j in range(start, end):
                if j == i:
                    continue
                pairs.append((word2idx[center], word2idx[filtered_sentence[j]]))
    return pairs


class SkipGram:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        limit = 0.5 / max(1, embedding_dim)
        self.W_center = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        self.W_context = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))

    def softmax(self, x):
        x = np.asarray(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, center_idx):
        center_embedding = self.W_center[center_idx]
        scores = self.W_context @ center_embedding
        return self.softmax(scores)

    def train(self, pairs, epochs=100, lr=0.05):
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for center_idx, context_idx in pairs:
                probs = self.forward(center_idx)
                total_loss += -np.log(probs[context_idx] + 1e-12)

                d_scores = probs.copy()
                d_scores[context_idx] -= 1

                center_embedding = self.W_center[center_idx].copy()
                grad_center = d_scores @ self.W_context
                grad_context = np.outer(d_scores, center_embedding)

                self.W_center[center_idx] -= lr * grad_center
                self.W_context -= lr * grad_context

            avg_loss = total_loss / max(1, len(pairs))
            loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        return loss_history


class SkipGramNegativeSampling:
    def __init__(self, vocab_size, embedding_dim, negative_samples=10):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        limit = 0.5 / max(1, embedding_dim)
        self.W_center = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        self.W_context = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
        self.neg_table = None

    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def build_neg_sampling_table(self, word_freq: Counter, table_size=int(1e6), power=0.75):
        if not word_freq:
            raise ValueError("word_freq is empty; build the vocabulary before training")

        freq = np.fromiter(word_freq.values(), dtype=np.float64)
        powered_freq = freq ** power
        probs = powered_freq / np.sum(powered_freq)
        cumulative = np.cumsum(probs)

        table = np.zeros(table_size, dtype=np.int32)
        j = 0
        for i in range(table_size):
            x = i / table_size
            while j < len(cumulative) - 1 and x > cumulative[j]:
                j += 1
            table[i] = j
        self.neg_table = table

    def sample_negative(self, center_idx, context_idx, k):
        if self.neg_table is None:
            raise ValueError("Negative sampling table has not been built")

        neg_samples = []
        while len(neg_samples) < k:
            neg_idx = int(self.neg_table[np.random.randint(len(self.neg_table))])
            if neg_idx != center_idx and neg_idx != context_idx:
                neg_samples.append(neg_idx)
        return neg_samples

    def forward(self, center_idx, context_idx, neg_idxs):
        center_embedding = self.W_center[center_idx]
        context_embedding = self.W_context[context_idx]
        neg_embeddings = self.W_context[neg_idxs]

        pos_score = self.sigmoid(np.dot(center_embedding, context_embedding))
        neg_score = self.sigmoid(-neg_embeddings @ center_embedding)

        loss = -np.log(pos_score + 1e-12)
        loss -= np.sum(np.log(neg_score + 1e-12))
        return loss, pos_score, neg_score

    def train(
        self,
        pairs,
        word_freq,
        epochs=5,
        initial_lr=0.025,
        min_lr_ratio=1e-4,
        report_every=100000,
        table_size=int(1e6),
    ):
        self.build_neg_sampling_table(word_freq, table_size=table_size)

        total_steps = max(1, epochs * len(pairs))
        current_step = 0
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0
            for center_idx, context_idx in pairs:
                current_lr = max(
                    initial_lr * (1 - current_step / total_steps),
                    initial_lr * min_lr_ratio,
                )

                neg_idxs = self.sample_negative(center_idx, context_idx, self.negative_samples)
                loss, pos_score, neg_score = self.forward(center_idx, context_idx, neg_idxs)
                total_loss += float(loss)

                center_embedding = self.W_center[center_idx].copy()
                context_embedding = self.W_context[context_idx].copy()
                neg_embeddings = self.W_context[neg_idxs].copy()

                pos_grad = pos_score - 1.0
                neg_grad = 1.0 - neg_score

                grad_center = pos_grad * context_embedding
                grad_center += np.sum(neg_grad[:, np.newaxis] * neg_embeddings, axis=0)

                self.W_center[center_idx] -= current_lr * grad_center
                self.W_context[context_idx] -= current_lr * pos_grad * center_embedding
                self.W_context[neg_idxs] -= current_lr * (
                    neg_grad[:, np.newaxis] * center_embedding[np.newaxis, :]
                )

                current_step += 1
                if report_every and current_step % report_every == 0:
                    avg_loss = total_loss / current_step
                    print(
                        f"step={current_step}/{total_steps} "
                        f"lr={current_lr:.6f} avg_loss={avg_loss:.6f}"
                    )

            avg_epoch_loss = total_loss / max(1, len(pairs))
            loss_history.append(avg_epoch_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {avg_epoch_loss:.6f}, "
                f"Last LR: {current_lr:.6f}"
            )
        return loss_history

    def get_embeddings(self):
        return (self.W_center + self.W_context) / 2.0
