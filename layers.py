#!/usr/bin/env python
"""Implementations of abstracted layers/modules"""

import torch
import torch.nn as nn


class StackedBiRNN(nn.Module):
    """
    TODO: Documentation
    """
    def __init__(self):
        super(StackedBiRNN, self).__init__()
        pass

    def forward(self, x, x_mask):
        pass


class FullyConnected(nn.Module):
    """
    TODO: Documentation
    """
    def __init__(self):
        super(FullyConnected, self).__init__()
        pass

    def forward(self, x):
        pass


class PointerNetwork(nn.Module):
    """
    TODO: Documentation
    """
    def __init__(self):
        super(PointerNetwork, self).__init__()
        pass

    def forward(self, x, y, x_mask, y_mask):
        pass


class MemoryAnswerPointer(nn.Module):
    """
    TODO: Documentation
    """
    def __init__(self):
        super(MemoryAnswerPointer, self).__init__()
        pass

    def forward(self, x, y, x_mask, y_mask):
        pass

