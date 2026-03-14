import random
import numpy as np

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
