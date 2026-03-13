import random

def generate_training_pairs(corpus, word2idx, window_size=5):
    pairs = []
    for sentence in corpus:
        for i, center in enumerate(sentence):
            dynamic_window = random.randint(1, window_size)
            for j in range(max(0, i - dynamic_window), min(len(sentence), i + dynamic_window + 1)):
                if j != i and sentence[j] in word2idx and center in word2idx:
                    pairs.append((word2idx[center], word2idx[sentence[j]]))
    return pairs
