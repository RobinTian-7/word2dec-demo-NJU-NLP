import random
import numpy as np
from collections import Counter

def generate_training_pairs(corpus, word2idx, window_size=5):
    pairs = []
    for sentence in corpus:
        for i, center in enumerate(sentence):
            dynamic_window = random.randint(1, window_size)
            for j in range(max(0, i - dynamic_window), min(len(sentence), i + dynamic_window + 1)):
                if j != i and sentence[j] in word2idx and center in word2idx:
                    pairs.append((word2idx[center], word2idx[sentence[j]]))
    return pairs

class SkipGram:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W_center = np.random.rand(vocab_size, embedding_dim)
        self.W_context = np.random.rand(vocab_size, embedding_dim)

    def softmax(self, x):
        x = np.array(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, center_idx):
        center_embedding = self.W_center[center_idx]
        scores = self.W_context @ center_embedding
        probs = self.softmax(scores)
        return probs

    def train(self, pairs, epochs=100,lr=0.05):
        for epoch in range(epochs):

            total_loss = 0
            #forward
            for center_idx, context_idx in pairs:
                prob = self.forward(center_idx)
                loss = -np.log(prob[context_idx])
                total_loss += loss
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs)}')
            #backward
            for center_idx, context_idx in pairs:
                probs = self.forward(center_idx)
                probs[context_idx] -= 1
                center_embedding = self.W_center[center_idx]
                self.W_center[center_idx] -= lr * (probs @ self.W_context)
                self.W_context -= lr * np.outer(probs, center_embedding)

class SkipGramNegativeSampling():
    def __init__(self, vocab_size, embedding_dim, negative_samples=10):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.W_center = np.random.rand(vocab_size, embedding_dim)
        self.W_context = np.random.rand(vocab_size, embedding_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def build_neg_sampling_table(self, word_freq: Counter, table_size=int(1e8), power=0.75):

        vocab = list(word_freq.keys())
        freq = np.array([word_freq[word] for word in vocab], dtype=np.float32)

        powered_freq = freq ** power
        probs = powered_freq / np.sum(powered_freq)

        cumulative = np.cumsum(probs)
        table = np.zeros(table_size, dtype=np.int32)
        j = 0
        for i in range(table_size):
            x = i+1 / table_size
            while x > cumulative[j]:
                j += 1
            table[i] = j
        self.neg_table = table

    
    def sample_nagative(self, center_idx, context_idx, k):
        neg_samples = []
        while len(neg_samples) < k:
          neg_idx = self.neg_table[np.random.randint(len(self.neg_table))]
          if neg_idx != center_idx and neg_idx != context_idx:
                neg_samples.append(neg_idx)
        return neg_samples
    
    def forward(self, center_idx, context_idx, neg_idxs):
        center_embedding = self.W_center[center_idx]
        context_embedding = self.W_context[context_idx]
        neg_embeddings = self.W_context[neg_idxs]

        pos_score = self.sigmoid(np.dot(center_embedding, context_embedding))
        neg_score = self.sigmoid(-np.dot(center_embedding, neg_embeddings.T))

        loss = - (np.log(pos_score) + np.log(neg_score))
        
        return loss, pos_score, neg_score
    
    def train(self, pairs, word_freq, epochs=100, lr=0.05):
        self.build_neg_sampling_table(word_freq)
        for epoch in range(epochs):
            total_loss = 0
            for center_idx, context_idx in pairs:
                neg_idxs = self.sample_nagative(center_idx, context_idx, self.negative_samples)
                loss, pos_score, neg_score = self.forward(center_idx, context_idx, neg_idxs)
                total_loss += loss

                #backward
                center_embedding = self.W_center[center_idx]
                context_embedding = self.W_context[context_idx]
                neg_embeddings = self.W_context[neg_idxs]

                pos_grad = pos_score - 1
                neg_grad = neg_score


                self.W_center[center_idx] -= lr * (pos_grad * context_embedding + np.sum(neg_grad[:, np.newaxis] * neg_embeddings, axis=0))
                self.W_context[context_idx] -= lr * pos_grad * center_embedding
                for i, neg_idx in enumerate(neg_idxs):
                    self.W_context[neg_idx] -= lr * neg_grad[i] * center_embedding
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs)}')