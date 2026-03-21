import numpy as np
import torch
from torch.utils.data import Dataset


class PairsDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        noise_table: np.ndarray,
        neg_samples: int = 5,
    ):
        arr = np.asarray(pairs, dtype=np.int32)
        self.centers = arr[:, 0]
        self.contexts = arr[:, 1]
        self.noise_table = np.asarray(noise_table, dtype=np.int32)
        self.neg_samples = neg_samples

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        return int(self.centers[idx]), int(self.contexts[idx])


class NegativeSamplingCollator:
    def __init__(self, noise_table: np.ndarray, neg_samples: int = 5):
        self.noise_table = np.asarray(noise_table, dtype=np.int64)
        self.neg_samples = neg_samples

    def __call__(self, batch: list[tuple[int, int]]):
        centers = np.fromiter((center for center, _ in batch), dtype=np.int64, count=len(batch))
        contexts = np.fromiter((context for _, context in batch), dtype=np.int64, count=len(batch))
        negatives = self._sample_negatives(centers, contexts)

        return (
            torch.from_numpy(centers),
            torch.from_numpy(contexts),
            torch.from_numpy(negatives),
        )

    def _sample_negatives(self, centers: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        batch_size = len(centers)
        negatives = np.empty((batch_size, self.neg_samples), dtype=np.int64)
        filled = np.zeros(batch_size, dtype=np.int32)

        while True:
            pending = np.flatnonzero(filled < self.neg_samples)
            if pending.size == 0:
                break

            needed = self.neg_samples - filled[pending]
            draw_width = max(self.neg_samples * 2, int(needed.max()) * 2)
            sampled = self.noise_table[
                np.random.randint(0, len(self.noise_table), size=(pending.size, draw_width))
            ]
            valid = (sampled != centers[pending, None]) & (sampled != contexts[pending, None])

            for row_offset, row_idx in enumerate(pending):
                row_values = sampled[row_offset][valid[row_offset]]
                take = min(row_values.size, self.neg_samples - filled[row_idx])
                if take == 0:
                    continue

                start = filled[row_idx]
                negatives[row_idx, start : start + take] = row_values[:take]
                filled[row_idx] += take

        return negatives


def build_noise_table(word_freq: dict, table_size: int = 1_000_000, power: float = 0.75) -> np.ndarray:
    words = list(word_freq.keys())
    counts = np.array([word_freq[w] for w in words], dtype=np.float64)
    counts = counts ** power
    counts /= counts.sum()
    indices = np.arange(len(words), dtype=np.int32)
    table = np.random.choice(indices, size=table_size, p=counts).astype(np.int32)
    return table
