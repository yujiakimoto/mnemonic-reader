#!/usr/bin/env python
"""Implementations of abstracted layers/modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StackedBiRNN(nn.Module):
    """
    TODO: Documentation
    """
    def __init__(self, n_input, n_hidden, n_layers, rnn_cell=nn.LSTM,
                 dropout=0, dropout_output=False, concat_layers=False, padding=False):
        super(StackedBiRNN, self).__init__()
        self.padding = padding
        self.dropout = dropout
        self.dropout_output = dropout_output
        self.n_layers = n_layers
        self.concat_layers = concat_layers
        self.RNNs = nn.ModuleList()
        for i in range(n_layers):
            n_input = n_input if i == 0 else 2 * n_hidden
            self.RNNs.append(rnn_cell(n_input, n_hidden, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask):
        # Pad if specified, or if during evaluation
        if self.padding or not self.training:
            output = self._forward_padded(x, x_mask)
        # No padding necessary
        else:
            output = self._forward_unpadded(x)
        return output.contiguous()

    def _forward_unpadded(self, x):
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.n_layers):
            rnn_input = outputs[-1]
            # Apply dropout to hidden input
            if self.dropout > 0:
                rnn_input = F.dropout(rnn_input, p=self.dropout, training=self.training)
            # Forward
            rnn_output = self.RNNs[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return output

    def _forward_padded(self, x, x_mask):
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.n_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout > 0:
                dropout_input = F.dropout(rnn_input.data, p=self.dropout, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.RNNs[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout > 0:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return output


class FullyConnected(nn.Module):
    """
    TODO: Documentation
    """
    def __init__(self, n_input, n_hidden, n_output, dropout=0):
        super(FullyConnected, self).__init__()
        self.dropout = dropout
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        return x


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

