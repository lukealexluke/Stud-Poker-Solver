"""
Initialize history value function q
2 Initialize policy πi for both players
3 for t = 1, ..., T do
4   Retrain history value function q on data from π
5   Reinitialize regret networks R0, R1
6   for update player i ∈ {0, 1} do
7       for P trajectories do
8           Get trajectory τ using sampling distribution (Equation 4)
9           for each state s ∈ τ do
10              for each action a do
11                  Estimate immediate cf-regret
                    ˆr(π, s, a|z) = qi(π, z[s], a|θ) − ∑a πi(s, a)qi(π, z[s], a|θ)
12              Add (s, ˆr(π, s)) to cumulative regret buffer
13              Add (s, a′) to average policy buffer where a′ is action taken at state s in trajectory τ
14      Train regret network Ri on cumulative regret buffer
15 Train average policy network ¯πφ on average policy buffer
16 return average policy network ¯πφ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SkipLayer(nn.Module):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.layer = nn.Linear(units, units)
        self.hidden = nn.Linear(units, units)
    
    def forward(self, x):
        return self.hidden(x) + x

class HistoryValueNetwork(nn.Module):

    def __init__(self, input_size, nodes_network_layers, activation='leaky_relu', **kwargs):

        super().__init__()

        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else: self.activation = activation

        # Used to be a normal list, is now a ModuleList to handle append better   -Trevor
        self.hidden_layers = nn.ModuleList()

        # The loop tried to define each layer based on the current and next node in nodes_network_layers, 
        # but the first layer has no explicit input size. Leading to size mis-match   -Trevor
        prev_layer = input_size
        for units in nodes_network_layers:
            if prev_layer == units:
                self.hidden_layers.append(SkipLayer(units))
            else:
                self.hidden_layers.append(nn.Linear(prev_layer, units))
            prev_layer = units

        self.normalization = nn.LayerNorm(nodes_network_layers[-1])
        self.last_layer = nn.Linear(nodes_network_layers[-1], nodes_network_layers[-1])
        self.out_layer = nn.Linear(nodes_network_layers[-1], 1)

    def forward(self, inputs: tuple):

        x, mask = inputs
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.normalization(x)
        x = self.last_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)

        return x


class PolicyNetwork(nn.Module):

    pass

