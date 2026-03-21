import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

import data.preprocess as preprocess
import model.skipgram as skipgram
import model.vocabulary as vocab
from torch.utils.data import DataLoader
import torch
from dataset import NegativeSamplingCollator, PairsDataset, build_noise_table
from skipgram_torch import SkipGramNS
import model.skipgram as skipgram_utils


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Skip-Gram with negative sampling.")
    parser.add_argument("--data-path", default="database/cnews.train.txt")
    parser.add_argument("--cache-path", default="cache/cnews_train_corpuspkl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--min-token-length", type=int, default=1)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--neg-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subsample-threshold", type=float, default=1e-5)
    parser.add_argument("--table-size", type=int, default=int(1e6))
    parser.add_argument("--report-every", type=int, default=100000)
    parser.add_argument("--max-sentences", type=int, default=10000000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parttern", default="pytorch")
    return parser.parse_args()


def save_training_outputs(output_dir, model, vocabs, loss_history, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", model.get_embeddings())
    np.save(output_dir / "center_embeddings.npy", model.W_center)
    np.save(output_dir / "context_embeddings.npy", model.W_context)
    np.save(output_dir / "loss_history.npy", np.asarray(loss_history, dtype=np.float64))

    with (output_dir / "idx2word.json").open("w", encoding="utf-8") as f:
        json.dump(vocabs.idx2word, f, ensure_ascii=False, indent=2)

    with (output_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    if args.parttern == "handmade":
        random.seed(args.seed)
        np.random.seed(args.seed)

        preprocessor = preprocess.Preprocessor(min_token_length=args.min_token_length)
        corpus = preprocessor.build_corpus_with_cache(args.data_path, args.cache_path)
        if args.max_sentences is not None:
            corpus = corpus[: args.max_sentences]
        
        vocabs = vocab.Vocabulary(min_count=args.min_count).build(corpus)
        word_freq = vocabs.word_freq()
        discard_probs = skipgram.compute_discard_probabilities(
            word_freq,
            t=args.subsample_threshold,
        )

        sampling_rng = random.Random(args.seed)
        training_pairs = skipgram.generate_training_pairs(
            corpus,
            vocabs.word2idx,
            window_size=args.window_size,
            discard_probs=discard_probs,
            rng=sampling_rng,
        )

        if not training_pairs:
            raise ValueError("No training pairs generated; reduce subsampling or min_count.")

        sorted_discard = sorted(discard_probs.items(), key=lambda item: item[1], reverse=True)
        high_examples = sorted_discard[:10]
        low_examples = sorted(discard_probs.items(), key=lambda item: item[1])[:10]

        print(f"Sentences used: {len(corpus)}")
        print(f"Vocabulary size: {vocabs.vocab_size}")
        print(f"Training pairs after subsampling: {len(training_pairs)}")
        print(f"Top discard probabilities: {high_examples}")
        print(f"Low discard probabilities: {low_examples}")
        if "的" in discard_probs:
            print(f"Discard probability for '的': {discard_probs['的']:.6f}")

        model = skipgram.SkipGramNegativeSampling(
            vocab_size=vocabs.vocab_size,
            embedding_dim=args.embedding_dim,
            negative_samples=args.neg_samples,
        )
        loss_history = model.train(
            training_pairs,
            word_freq,
            epochs=args.epochs,
            initial_lr=args.lr,
            report_every=args.report_every,
            table_size=args.table_size,
        )

        output_dir = Path(args.output_dir)
        metadata = {
            "data_path": str(Path(args.data_path).resolve()),
            "cache_path": str(Path(args.cache_path).resolve()),
            "sentences_used": len(corpus),
            "vocab_size": vocabs.vocab_size,
            "training_pairs": len(training_pairs),
            "embedding_dim": args.embedding_dim,
            "window_size": args.window_size,
            "neg_samples": args.neg_samples,
            "epochs": args.epochs,
            "initial_lr": args.lr,
            "subsample_threshold": args.subsample_threshold,
            "table_size": args.table_size,
            "seed": args.seed,
            "loss_history": [float(loss) for loss in loss_history],
            "top_discard_examples": high_examples,
            "low_discard_examples": low_examples,
            "discard_probability_of_的": float(discard_probs["的"]) if "的" in discard_probs else None,
        }
        save_training_outputs(output_dir, model, vocabs, loss_history, metadata)


    if args.parttern == "pytorch":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = resolve_device()
        print(f"Device: {device}")

        preprocessor = preprocess.Preprocessor(min_token_length=args.min_token_length)
        corpus = preprocessor.build_corpus_with_cache(args.data_path, args.cache_path)
        if args.max_sentences:
            corpus = corpus[: args.max_sentences]

        vocabs = vocab.Vocabulary(min_count=args.min_count).build(corpus)
        word_freq = vocabs.word_freq()
        discard_probs = skipgram_utils.compute_discard_probabilities(word_freq, t=args.subsample_threshold)

        pairs = skipgram_utils.generate_training_pairs(
            corpus,
            vocabs.word2idx,
            window_size=args.window_size,
            discard_probs=discard_probs,
            rng=random.Random(args.seed),
        )

        print(f"Vocab size: {vocabs.vocab_size}")
        print(f"Training pairs: {len(pairs)}")

        noise_table = build_noise_table(
            {idx: word_freq[vocabs.idx2word[idx]] for idx in range(vocabs.vocab_size)},
            table_size=args.table_size,
        )

        dataset = PairsDataset(pairs, noise_table, neg_samples=args.neg_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

        model = SkipGramNS(vocabs.vocab_size, args.embedding_dim).to(device)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)

        total_steps = args.epochs * len(loader)
        step = 0
        loss_history = []

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0

            for center, pos, neg in loader:
                progress = step / total_steps
                lr = max(args.lr * (1 - progress), args.lr * 1e-2)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                center = center.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                loss = model(center, pos, neg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step += 1

            avg = epoch_loss / len(loader)
            loss_history.append(avg)
            print(f"Epoch {epoch+1}/{args.epochs}  loss={avg:.4f}  lr={lr:.6f}")

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        embeddings = model.get_embeddings().detach().cpu().numpy()
        np.save(out / "embeddings.npy", embeddings)
        np.save(out / "center_embeddings.npy", model.center.weight.detach().cpu().numpy())
        np.save(out / "context_embeddings.npy", model.context.weight.detach().cpu().numpy())
        np.save(out / "loss_history.npy", np.array(loss_history))

        with (out / "idx2word.json").open("w", encoding="utf-8") as f:
            json.dump(vocabs.idx2word, f, ensure_ascii=False, indent=2)

        metadata = {
            "vocab_size": vocabs.vocab_size,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "neg_samples": args.neg_samples,
            "loss_history": loss_history,
            "device": str(device),
        }
        with (out / "train_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Saved to {out.resolve()}")

if __name__ == "__main__":
    main()
