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
        """
        input_dict must contain keys:
         token_to_index,
         add_unk,
         unk_token
        """
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
        """
        this will call the __len__ method
        """
        return f"<Vocabulary>(size={len(self)})"

    def __len__(self):
        return len(self._token_to_index)