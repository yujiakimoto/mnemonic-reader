#!/usr/bin/env python
"""Data loading helpers."""

import unicodedata

# ------------------------------------------------------------------------------
# Dictionary class for tokens.
# ------------------------------------------------------------------------------


class Dictionary(object):
    """
    Dictionary class for tokens; stores mapping from word -> integer ID and vice versa.
    """
    EOF = '<EOF>'   # End of file
    UNK = '<UNK>'   # Unknown token

    def __init__(self):
        self.tok2idx = {self.EOF: 0, self.UNK: 1}
        self.idx2tok = {0: self.EOF, 1: self.UNK}

    def __len__(self):
        return len(self.tok2idx)

    def __iter__(self):
        return iter(self.tok2idx)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.idx2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2idx

    def __getitem__(self, key):
        if type(key) == int:
            return self.idx2tok.get(key, self.UNK)
        elif type(key) == str:
            return self.tok2idx.get(self.normalize(key), self.tok2idx.get(self.UNK))

    def __setitem__(self, key, value):
        if type(key) == int and type(value) == str:
            self.idx2tok[key] = value
        elif type(key) == str and type(value) == int:
            self.tok2idx[key] = value
        else:
            raise ValueError('Invalid (key, value) types.')

    def tokens(self):
        """Return all words in dictionary (except special tokens)."""
        tokens = [k for k in self.tok2idx if k not in {self.EOF, self.UNK}]
        return tokens

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)
