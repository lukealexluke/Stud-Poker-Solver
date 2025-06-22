# Adapted from Sandholm Lab's "ESCHER.py"
# Written for Python 3 and newer
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReservoirBuffer:

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        # Called when an element is potentially added to the buffer
        # Args:
        #   element (object): data to be added to the buffer (potentially)

        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
            self._add_calls += 1

    def sample(self, num_samples):
        # Returns 'num_samples' uniformly sampled from the buffer
        # Args:
        #   num_samples (int): how many samples to draw
        # Returns:
        #   An iterable over 'num_samples' random elements of the buffer

        if len(self._data) < num_samples:
            raise ValueError(f'{num_samples} could not be sampled from size {len(self._data)}')
        return random.sample(self._data, num_samples)
    
    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    @property
    def data(self):
        return self._data

    def get_data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)

    def get_num_calls(self):
        return self._add_calls
    

class SkipDense(nn.Module):
    """Dense Layer with skip connection."""

    def __init__(self, input_dim, units):
        super().__init__()
        self.hidden = nn.Linear(input_dim, units)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity='relu')

    def forward(self, x):
        return self.hidden(x) + x
    

class PolicyNetwork(nn.Module):

    def __init__(self, input_size, policy_network_layers, num_actions, activation='leakyrelu'):
        super().__init__()

        self._input_size = input_size
        self._num_actions = num_actions

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        
        self.hidden = nn.ModuleList()
        prev_units = input_size

        for units in policy_network_layers[:-1]:
            if prev_units == units:
                self.hidden.append(SkipDense(units, units))
            else:
                self.hidden.append(nn.Linear(prev_units, units))
            prev_units = units

        self.lastlayer = nn.Linear(prev_units, policy_network_layers[-1])
        nn.init.kaiming_normal_(self.lastlayer.weight, nonlinearity='relu')
        self.normalization = nn.LayerNorm(policy_network_layers[-1])
        self.out_layer = nn.Linear(policy_network_layers[-1], num_actions)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):
        # Applies Policy Network
        # Args:
        #   inputs: tuple representing (infostate, legal_action_mask)
        # Returns:
        #   Action probabilities
        x, mask = inputs
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.lastlayer(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.out_layer(x)
        # Mask illegal actions
        x = torch.where(mask == 1, x, torch.full_like(x, -1e20))
        return self.softmax(x)
    

class RegretNetwork(nn.Module):

    def __init__(self, input_size, regret_network_layers, num_actions, activation='leakyrelu'):
        super().__init__()

        self._input_size = input_size
        self._num_actions = num_actions

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = activation
        
        self.hidden = nn.ModuleList()
        prev_units = input_size
        for units in regret_network_layers[:-1]:
            if prev_units == units:
                self.hidden.append(SkipDense(units, units))
            else:
                self.hidden.append(nn.Linear(prev_units, units))
            prev_units = units

        self.lastlayer = nn.Linear(prev_units, regret_network_layers[-1])
        nn.init.kaiming_normal_(self.lastlayer.weight, nonlinearity='relu')
        self.normalization = nn.LayerNorm(regret_network_layers[-1])
        self.out_layer = nn.Linear(regret_network_layers[-1], num_actions)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):
        # Applies Regret Network
        # Args:
        #   inputs: tuple representing (infostate, legal_action_mask)
        # Returns:
        #   Cumulative regret for each info_state action
        x, mask = inputs
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.lastlayer(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.out_layer(x)
        # Mask illegal actions
        x = mask * x
        return x


class ValueNetwork(nn.Module):

    def __init__(self, input_size, val_network_layers, activation='leakyrelu'):
        super().__init__()
        self._input_size = input_size

        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = activation

        self.hidden = nn.ModuleList()
        prev_units = input_size
        for units in val_network_layers[:-1]:
            if prev_units == units:
                self.hidden.append(SkipDense(units, units))
            else:
                self.hidden.append(nn.Linear(prev_units, units))
            prev_units = units

        self.lastlayer = nn.Linear(prev_units, val_network_layers[-1])
        nn.init.kaiming_normal_(self.lastlayer.weight, nonlinearity='relu')
        self.normalization = nn.LayerNorm(val_network_layers[-1])
        self.out_layer = nn.Linear(val_network_layers[-1], 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Applies Value Network
        # Args:
        #   inputs: tuple representing (infostate, legal_action_mask)
        # Returns:
        #   expected utility at specified history with current policy
        x, mask = inputs
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.lastlayer(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.out_layer(x)
        return x
