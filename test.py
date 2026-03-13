import model.skipgram as skipgram
import model.vocabulary as vocab
import data.preprocess as preprocess

if __name__ == "__main__":
    preprogressor = preprocess.Preprocessor(min_token_length=2)
    corpus = preprogressor.build_corpus_with_cache(
        "database/cnews.train.txt",
        "cache/cnews_train_corpuspkl",
    )
    vocabs = vocab.Vocabulary(min_count = 5).build(corpus)
    training_pairs = skipgram.generate_training_pairs(corpus, vocabs.word2idx, window_size=5)

    for i in range(100):
        center_idx, context_idx = training_pairs[i]
        center_word = vocabs.idx2word[center_idx]
        context_word = vocabs.idx2word[context_idx]
        print(f"样本 {i+1}: (中心词: '{center_word}', 上下文词: '{context_word}')")