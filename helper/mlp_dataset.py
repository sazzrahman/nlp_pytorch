import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from custom_vectorizer import SurnameVectorizer



class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer
        self.train_df = surname_df[surname_df.split == 'train']
        self.train_size = self.train_df.shape[0]
        self.val_df = surname_df[surname_df.split == 'val']
        self.validation_size = self.val_df.shape[0]
        self.test_df = surname_df[surname_df.split == 'test']
        self.test_size = self.test_df.shape[0]
        self._target_size = None
        self._target_df = None
        self._target_split = None
        self._lookup_dict = dict(train=(self.train_df, self.train_size),
                                 validation=(self.val_df, self.validation_size),
                                 test=(self.test_df, self.test_size))
        self.set_split(split='train')
        class_counts = surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        vectorizer = SurnameVectorizer.from_dataframe(surname_df)
        return cls(surname_df, vectorizer)

    def save_vectorizer(self, path):
        # implement to_serializable method
        with open(path, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def load_vectorizer(self):
        # implement from_serializable method
        pass

    def get_vectorizer(self):
        # calls the protected attribute vectorizer
        return self._vectorizer

    def set_split(self, split='train'):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        # Essential for DataLoader object
        return self._target_size

    def __getitem__(self, index):
        # Essential for DataLoader object
        """
        Priamry entry point for PyTorch DataSets
        :return:
        """
        row = self._target_df.iloc[index]
        # Input Vectors
        surname_vector = self._vectorizer.vectorize(row.surname)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
        return {'x_surname': surname_vector, 'y_nationality': nationality_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size