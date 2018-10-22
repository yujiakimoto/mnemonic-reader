#!/usr/bin/env python
"""Implementation of the Mnemonic Reader"""

import torch
import torch.nn as nn
import layers


class MnemonicReader(nn.Module):
    """
    TODO: Documentation
    """

    def __init__(self, args):
        super(MnemonicReader, self).__init__()

        # Model configuration
        self.args = args

        # Word/Character embeddings
        self.w_embedding = nn.Embedding(args.w_vocab_size, args.w_embedding_dim, padding_idx=0)
        self.c_embedding = nn.Embedding(args.c_vocab_size, args.c_embedding_dim, padding_idx=0)

        # Char-RNN
        self.char_rnn = None

        # Encoder
        self.encoder = None

        # Interaction

        # Answer pointer
        self.ptr = None

    def forward(self,
                p_word, p_char, p_feat, p_mask,
                q_word, q_char, q_feat, q_mask):
        """

        :param p_word:
        :param p_char:
        :param p_feat:
        :param p_mask:
        :param q_word:
        :param q_char:
        :param q_feat:
        :param q_mask:
        :return:
        """

        # Embed both paragraph and question
        p_word_emb = self.w_embedding(p_word)
        p_char_emb = self.c_embedding(p_char)
        q_word_emb = self.w_embedding(q_word)
        q_char_emb = self.c_embedding(q_char)

        # Dropout on embeddings
        # TODO: is this used?

        # Generate char-RNN features
        p_char_enc = self.char_rnn(p_char_emb, p_mask)
        q_char_enc = self.char_rnn(q_char_emb, q_mask)

        # Combine inputs
        p_input = [p_word_emb, p_char_enc]
        q_input = [q_word_emb, q_char_enc]
        # TODO: are additional features used? (e.g. Pos)

        # Encode paragraph, question
        p_encoded = self.encoder(torch.cat(p_input, 2), p_mask)
        q_encoded = self.encoder(torch.cat(q_input, 2), q_mask)

        # Align and aggregate

        # Predict
