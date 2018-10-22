#!/usr/bin/env python
"""Implementations of attention mechanisms"""

import torch.nn as nn


class SeqAttnMatch(nn.Module):
    """
    TODO: Document
    """
    def __init__(self):
        super(SeqAttnMatch, self).__init__()
        pass

    def forward(self, x, y, y_mask):
        pass


class SelfAttnMatch(nn.Module):
    """
    TODO: Document
    """

    def __init__(self):
        super(SelfAttnMatch, self).__init__()
        pass

    def forward(self, x, x_mask):
        pass


class BiLinearSeqAttn(nn.Module):
    """
    TODO: Document
    """

    def __init__(self):
        super(BiLinearSeqAttn, self).__init__()
        pass

    def forward(self, x, y, x_mask):
        pass


class LinearSeqAttn(nn.Module):
    """
    TODO: Document
    """

    def __init__(self):
        super(LinearSeqAttn, self).__init__()
        pass

    def forward(self, x, x_mask):
        pass


class NonLinearSeqAttn(nn.Module):
    """
    TODO: Document
    """

    def __init__(self):
        super(NonLinearSeqAttn, self).__init__()
        pass

    def forward(self, x, x_mask):
        pass