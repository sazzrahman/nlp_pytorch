import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Vocabulary(object):
    def __init__(self, token_to_index=None, add_unk=True, unk_token="<UNK>"):
        self.unk_index = None
        if token_to_index is None:
            token_to_index = {}
        self._token_to_index = token_to_index
        self._idx_to_token = {v: k for k, v in self._token_to_index.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.add_token(unk_token)
            self.unk_index = self.lookup_token(unk_token)

    def to_serializable(self):
        serializable = dict(
            token_to_idx=self._token_to_index,
            add_unk=self._add_unk,
            unk_token=self._unk_token
        )
        return serializable

    @classmethod
    def from_serializable(cls, content):
        return cls(**content)

    def add_token(self, token):
        try:
            index = self._token_to_index[token]
        except KeyError:
            index = len(self._token_to_index)
            self._token_to_index[token] = index
            self._idx_to_token[index] = token

    def add_many(self):
        pass

    def lookup_token(self, token):
        """
        access function private attribute lookup token
        :param token:
        :return: the index of a given token from the vocab
        """
        if self.unk_index >= 0:
            # if an unk index is available return the default value
            # the get method falls to zero if index is not found
            return self._token_to_index.get(token, self.unk_index)
        else:
            return self._token_to_index[token]

    def lookup_index(self, idx):
        return self._idx_to_token[idx]

    def __str__(self):
        pass

    def __len__(self):
        return len(self._token_to_index)


class SurnameVectorizer(object):
    def __init__(self, surname_vocab, nationality_vocab):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname):
        # only surname needs to be vectorized
        vocab = self.surname_vocab
        one_hot = np.zeros(len(vocab))
        onehot_index = [vocab.lookup_token(letter) for letter in surname]
        one_hot[onehot_index] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, surname_df):
        surname_vocab = Vocabulary(unk_token="@")
        # the prediction variables
        nationality_vocab = Vocabulary(add_unk=False)
        for i, row in surname_df.iterrows():
            for letter in row["surname"]:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row["nationality"])
        return cls(surname_vocab, nationality_vocab)

    @classmethod
    def from_serializable(cls):
        pass

    def to_serializable(self):
        serializable = dict(
            surname_vocab=self.surname_vocab.to_serializable(),
            nationality_vocab=self.nationality_vocab.to_serializable()
        )
        return serializable


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


class SurnameClassifier(nn.Module):
    # Was not tested independently
    # the call method will take care of the forward unit
    def __init__(self, input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)
        # super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        x_in = x_in.float() # necessary for input processing
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector


class ModelUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def make_train_state(args):
        state_dict = dict(
            stop_early=False,
            early_stopping_step=0,
            early_stopping_best_val=1e8,
            learing_rate=args.learning_rate,
            epoch_index=0,
            train_loss=[],
            train_acc=[],
            val_loss=[],
            val_acc=[],
            test_loss=-1,
            test_acc=-1,
            model_filename=args.model_state_file
        )
        return state_dict

    @staticmethod
    def update_train_state(args,model,train_state):

        # Save model for the very first time
        if train_state['epoch_index']==0:
            torch.save(model.state_dict(),train_state['model_filename'])
            train_state['stop_early'] = False

        elif train_state["epoch_index"]>=1:
            loss_tm1, loss_t = train_state['val_loss'][-2:]
            # if loss is worse
            if loss_t >= train_state['early_stopping_best_val']:
                train_state['early_stopping_step'] +=1
            # if loss is better
            else:
                if loss_t < train_state['early_stopping_best_val']:
                    torch.save(model.state_dict(), train_state['model_filename'])
                    # reset early stopping step
                    train_state['early_stopping_step'] = 0
            train_state['stop_early'] = train_state['early_stopping_step'] >= args.early_stopping_criteria
        return train_state



    @staticmethod
    def compute_accuracy(y_pred,y_target):
        # It is a vector operation, not a single operation
        _, y_pred_indices = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct/ len(y_pred_indices) * 100


    @staticmethod
    def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
        # Implement torch Dataloader function for this
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

    @staticmethod
    def set_seed_everywhere(seed, cuda):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def handle_dirs(save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
