import bisect
import warnings
from typing import (
    Iterable,
    List,
    TypeVar,
)
from torch.utils.data import Dataset, IterableDataset, Sampler
import random
import math
import numpy as np

T_co = TypeVar('T_co', covariant=True)


class CustomConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(CustomConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        assert hasattr(datasets[0], "batch_size")
        self.batch_size = datasets[0].batch_size

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def reset_dataset(self, shuffled_idx):
        # dtu

        for di in range(len(self.datasets)):
            self.datasets[di].idx_map = {}

        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, sid)
            # 找到真实idx: sample_idx
            if dataset_idx == 0:
                sample_idx = sid
            else:
                sample_idx = sid - self.cumulative_sizes[dataset_idx - 1]
            self.datasets[dataset_idx].idx_map[sample_idx] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1


class BalancedRandomSampler(Sampler):
    def __init__(self, concat_datasets, rank=0, num_replicas=1, shuffle=True):
        # 默认使用concat_datasets中最少的数据作为每个dataset的在每个epoch中的样本数
        self.cumulative_sizes = concat_datasets.cumulative_sizes
        self.n_sample_every_scene = [len(d) for d in concat_datasets.datasets]
        self.n_sample_per_scene = min(self.n_sample_every_scene)
        self.rank = rank
        self.epoch = 0
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"
                             " [0, {}]".format(rank, num_replicas - 1))

        self.n_scene = len(concat_datasets.datasets)
        total_size = self.n_scene * self.n_sample_per_scene
        if total_size % self.num_replicas != 0:
            self.num_samples = math.ceil((total_size - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(total_size / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        print(f'scene num: {len(concat_datasets.datasets)}, original sample nums: {self.n_sample_every_scene}, '
              f'balanced sample nums: {self.n_sample_per_scene}*{len(concat_datasets.datasets)}')
        print(f'Training dataset rank{rank + 1}/{self.num_replicas}, '
              f'samples for each epoch:{self.num_samples}/{self.total_size}')

    def __iter__(self):
        new_list = []
        # deterministically shuffle based on epoch and seed
        random.seed(self.epoch)
        for i, n in enumerate(self.n_sample_every_scene):
            new_idxs = np.arange(n)
            if i > 0:
                new_idxs += self.cumulative_sizes[i - 1]
            new_idxs = list(new_idxs)
            if self.shuffle:
                random.shuffle(new_idxs)
            new_list.extend(new_idxs[:self.n_sample_per_scene])

        # global shuffle
        if self.shuffle:
            random.shuffle(new_list)

        # split for each process
        indices = new_list[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
