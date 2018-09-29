#!/usr/bin/env python
"""Base class for NLP models"""

import copy
import torch


class NLPModel(object):
    """
    TODO: Documentation
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self):
        pass

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Saving & Loading
    # --------------------------------------------------------------------------

    def save(self, filename: str):
        pass

    def checkpoint(self, filename: str, epoch: int):
        pass

    @classmethod
    def load(cls, filename: str):
        model = cls()
        return model

    @classmethod
    def load_checkpoint(cls, filename: str):
        model = cls()
        epoch = 0
        return model, epoch
