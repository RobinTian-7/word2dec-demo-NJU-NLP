from pathlib import Path
import json
import pickle
import re
import jieba


class Preprocessor:
    
    def __init__(self, keep_numbers=False, min_token_length=1):
        self.keep_numbers = keep_numbers
        self.min_token_length = min_token_length

    def load_raw_texts(self, path):
        texts = []
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue

                _, text = parts
                texts.append(text)

        return texts

    def clean_text(self, text):
        if not text:
            return ""

        text = text.strip()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\s+", " ", text)

        if self.keep_numbers:
            text = re.sub(
                r"[^0-9A-Za-z\u4e00-\u9fff，。！？；：“”‘’（）《》、,.!?;:()\- ]",
                "",
                text,
            )
        else:
            text = re.sub(r"\d+", "", text)
            text = re.sub(
                r"[^A-Za-z\u4e00-\u9fff，。！？；：“”‘’（）《》、,.!?;:()\- ]",
                "",
                text,
            )

        return re.sub(r"\s+", " ", text).strip()

    def split_sentences(self, text):
        if not text:
            return []

        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\.{3,}", "。", text)
        text = re.sub(r"…+", "。", text)
        text = re.sub(r"[。！？?!；;]+", "。", text)

        return [sentence.strip() for sentence in text.split("。") if sentence.strip()]

    def tokenize(self, sentence):
        if not sentence:
            return []

        tokens = jieba.lcut(sentence)
        clean_tokens = []

        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if len(token) < self.min_token_length:
                continue
            if re.fullmatch(r"[，、。！？；：“”‘’（）《》,.!?;:()\-]+", token):
                continue
            clean_tokens.append(token)

        return clean_tokens

    def build_corpus(self, path):
        corpus = []

        for text in self.load_raw_texts(path):
            clean = self.clean_text(text)
            if not clean:
                continue

            for sentence in self.split_sentences(clean):
                tokens = self.tokenize(sentence)
                if tokens:
                    corpus.append(tokens)

        return corpus

    def cache_config(self):
        return {
            "keep_numbers": self.keep_numbers,
            "min_token_length": self.min_token_length,
        }

    def save_corpus_cache(self, corpus, cache_path, source_path):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "source_path": str(Path(source_path).resolve()),
            "config": self.cache_config(),
            "corpus": corpus,
        }

        with cache_path.open("wb") as f:
            pickle.dump(payload, f)

    def load_corpus_cache(self, cache_path):
        cache_path = Path(cache_path)
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        return payload["corpus"]

    def build_corpus_with_cache(self, path, cache_path):
        cache_path = Path(cache_path)
        source_path = Path(path)
        meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

        config = self.cache_config()
        source_stat = source_path.stat()
        source_meta = {
            "source_path": str(source_path.resolve()),
            "source_size": source_stat.st_size,
            "source_mtime": source_stat.st_mtime,
            "config": config,
        }

        if cache_path.exists() and meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta == source_meta:
                return self.load_corpus_cache(cache_path)

        corpus = self.build_corpus(source_path)
        self.save_corpus_cache(corpus, cache_path, source_path)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(source_meta, f, ensure_ascii=False, indent=2)

        return corpus



