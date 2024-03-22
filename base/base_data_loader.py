from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate, val_dataset=None):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        # self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(dataset=dataset, **self.init_kwargs)
        self.val_dataset = val_dataset

    def split_validation(self):
        if self.val_dataset is None:
            return None
        else:
            return DataLoader(dataset=self.val_dataset, **self.init_kwargs)
