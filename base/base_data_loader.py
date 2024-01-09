import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SequentialSampler, RandomSampler, SubsetRandomSampler, WeightedRandomSampler, BatchSampler

class BaseDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size, shuffle, validation_split,
        num_workers, collate_fn=default_collate
        ):
        
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        
        # 생성자에서 데이터 로더 생성하기 위한 파라미터 선언
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        """ 
        like
        mnist_train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
            )
        """
        super().__init__(sampler=self.sampler, **self.init_kwargs)
    
    # 샘플러 구현
    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples) # 데이터 셋 크기 만큼 np.arange 0 ~ 데이터셋 크기 - 1

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            # 0보단 크고 데이터 셋 길이보다는 작아야 한다.
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split) # self.n_samples = len(dataset)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler        
    
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        