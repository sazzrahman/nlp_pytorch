import numpy as np
from cnn_vocab import Vocabulary


class SurnameVectorizer(object):
    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    def vectorize(self, surname):
        # only surname needs to be vectorized
        one_hot_matrix_size = (len(self.surname_vocab), self._max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

        # traverse the matrix
        for position_index, character in enumerate(surname):
            character_index = self.surname_vocab.lookup_token(character)
            one_hot_matrix[character_index, position_index] = 1

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df):
        """
        Manually adds token for each row of a dataframe
        retrun : Instantiated class
        """
        # initiate surname vocab with unk token
        surname_vocab = Vocabulary(unk_token="@")
        # initiate nationality vocab without unk token
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0
        for i, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row["surname"]))
            for letter in row["surname"]:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row["nationality"])
        return cls(surname_vocab, nationality_vocab, max_surname_length)

    @classmethod
    def from_serializable(cls):
        pass

    def to_serializable(self):
        serializable = dict(
            surname_vocab=self.surname_vocab.to_serializable(),
            nationality_vocab=self.nationality_vocab.to_serializable()
        )
        return serializable
