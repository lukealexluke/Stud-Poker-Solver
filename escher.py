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

    def __init__(self, input_size, nodes_network_layers, activation, **kwargs):

        super().__init__()
        self.activation = activation
        self.hidden_layers = []

        prev_layer = 0
        for i in range(len(nodes_network_layers) - 1):
            if prev_layer == nodes_network_layers[i]:
                self.hidden_layers.append(SkipLayer(prev_layer))
            else:
                self.hidden_layers.append(nn.Linear(nodes_network_layers[i], nodes_network_layers[i+1]))
            prev_layer = nodes_network_layers[i]

        self.normalization = nn.LayerNorm(nodes_network_layers[-1])
        self.last_layer = nn.Linear(nodes_network_layers[-1], nodes_network_layers[-1])
        self.out_layer = nn.Linear(nodes_network_layers[-1], 1)

    def forward(self, inputs: tuple):
        # !! input should consist of information state, and all legal actions (masked)

        x, mask = inputs
        for layer in self.hidden_layers:
            # !! change later so activation can be any method (relu, leaky, etc.)
            x = F.relu(layer(x))

        x = self.normalization(x)
        x = self.last_layer(x)
        x = F.relu(x) # !! change activation here too
        x = self.out_layer(x)

        return x
