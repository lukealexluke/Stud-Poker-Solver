import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import pyspiel

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

model = PolicyNetwork(11, (256, 128), 2)
model.load_state_dict(torch.load('tmp/results/2025_07_19_18_34_13/policy_nodes_340465', weights_only=True)) 


game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()
state.apply_action(2)    # deal player 0 a King
state.apply_action(1)    # deal player 1 a Queen

print("Current player: ", state.current_player())
info_state = torch.tensor(state.information_state_tensor())
mask = torch.tensor(state.legal_actions_mask())

model.eval()
with torch.no_grad():
    print("Policy: ", model((info_state, mask)))

arr = np.load('tmp/results/_convs.npy')
print("Convergence: ", arr)