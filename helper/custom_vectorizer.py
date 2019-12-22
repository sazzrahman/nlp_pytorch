import numpy as np
from custom_vocab import Vocabulary


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