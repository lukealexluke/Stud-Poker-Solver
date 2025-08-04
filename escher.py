# Original code from OpenSpiel's DeepCFR implementation
# ESCHER logic Adapted from Sandholm Lab's "ESCHER.py", and OpenSpiel
# Written for Python 3 and newer with PyTorch
import random
import pickle
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import collections
from datetime import datetime
import time
from itertools import islice, product

import policy
import exploitability

import pyspiel
import time

import seven_card_stud

class InterDataset(Dataset):
    
    def __init__(self, buffer_data, transform=None):
        self.buffer_data = buffer_data
        self.transform = transform

    def __len__(self):
        return len(self.buffer_data)
    
    def __getitem__(self, idx):
        raw = self.buffer_data[idx]
        return self.transform(raw) if self.transform else raw
    
def infinite_dataloader(dataloader):
    # Pytorch doesn't have an equivalent to tf.data.repeat(count=None), so this is the alternative
    while True:
        for batch in dataloader:
            yield batch
    

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


class MemoryDataset(Dataset):
    def __init__(self, buffer_data, transform=None):
        self.data = buffer_data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            return self.transform(item)
        return item
    
def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
    

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


class ESCHER(policy.Policy):

    def __init__(self,
                 game,
                 policy_network_layers = (256, 128),
                 regret_network_layers = (256, 128),
                 value_network_layers = (256, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 130000,
                 num_val_fn_traversals: int = 100,
                 learning_rate: float = 1e-3,
                 batch_size_regret: int = 10000,
                 batch_size_value: int = 2024,
                 batch_size_average_policy: int = 10000,
                 memory_capacity: int = 100_000,
                 policy_network_train_steps: int = 15000,
                 regret_network_train_steps: int = 5000,
                 value_network_train_steps: int = 4048,
                 check_exploitability_every: int = 20,
                 reinitialize_regret_networks: bool = True,
                 reinitialize_value_network: bool = True,
                 save_regret_networks: str = None,
                 append_legal_actions_mask: bool = False,
                 save_average_policy_memories: str = None,
                 save_policy_weights: bool = True,
                 expl: float = 1.0,
                 val_expl: float = 0.01,
                 importance_sampling_threshold: float = 100.0,
                 importance_sampling: bool = True,
                 clear_value_buffer: bool = True,
                 val_bootstrap: bool = False,
                 use_balanced_probs: bool = False,
                 val_op_prob = 0.,
                 infer_device: str = 'cpu',
                 debug_val: bool = False,
                 play_against_random: bool = False,
                 train_device: str = 'cpu',
                 experiment_string: str = None,
                 all_actions: bool = True,
                 random_policy_path = None,
                 *args, **kwargs):
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        self._game = game
        self._save_policy_weights = save_policy_weights
        self._compute_exploitability = False # set to false since stud is too complex to compute nash
        self._play_against_random = play_against_random
        self._append_legal_actions_mask = append_legal_actions_mask
        self._num_random_games = 2000
        self._batch_size_regret = batch_size_regret
        self._batch_size_value = batch_size_value
        self._batch_size_average_policy = batch_size_average_policy
        self._policy_network_train_steps = policy_network_train_steps
        self._regret_network_train_steps = regret_network_train_steps
        self._value_network_train_steps = value_network_train_steps
        self._policy_network_layers = policy_network_layers
        self._regret_network_layers = regret_network_layers
        self._value_network_layers = value_network_layers
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        hist_state = np.append(self._root_node.information_state_tensor(0),
                               self._root_node.information_state_tensor(1))
        

        self._value_embedding_size = len(hist_state)
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._num_val_fn_traversals = num_val_fn_traversals
        self._reinitialize_regret_networks = reinitialize_regret_networks
        self._reinit_value_network = reinitialize_value_network
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._learning_rate = learning_rate
        self._save_regret_networks = save_regret_networks
        self._save_average_policy_memories = save_average_policy_memories
        self._infer_device = infer_device
        self._train_device = train_device
        self._memories_save_path = None
        self._memories_savefile = None
        self._check_exploitability_every = check_exploitability_every
        self._expl = expl
        self._val_expl = val_expl
        self._importance_sampling = importance_sampling
        self._importance_sampling_threshold = importance_sampling_threshold
        self._clear_value_buffer = clear_value_buffer
        self._nodes_visited = 0
        self._example_info_state = [None, None]
        self._example_hist_state = None
        self._example_legal_actions_mask = [None, None]
        self._squared_errors = []
        self._squared_errors_child = []
        self._balanced_probs = {}
        self._use_balanced_probs = use_balanced_probs
        self._val_op_prob = val_op_prob
        self._val_bootstrap = val_bootstrap
        self._debug_val = debug_val
        self._experiment_string = experiment_string
        self._all_actions = all_actions
        self._random_policy_path = random_policy_path

        if self._save_regret_networks:
            os.makedirs(self._save_regret_networks, exist_ok=True)
        
        if self._save_average_policy_memories:
            path = self._save_average_policy_memories

            if os.path.isdir(path):
                self._memories_save_path = os.path.join(path, 'average_policy_memories.pt')
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._memories_save_path = path
        self._reinitialize_policy_network()

        self._regret_networks = []
        self._regret_networks_train = []
        self._loss_regrets = []
        self._optimizer_regrets = []
        self._regret_train_step = []

        self._infer_device = torch.device(self._infer_device)
        self._train_device = torch.device(self._train_device)

        for player in range(self._num_players):

            net = RegretNetwork(
                self._embedding_size,
                self._regret_network_layers,
                self._num_actions
            ).to(self._infer_device)
            self._regret_networks.append(net)

            net2 = RegretNetwork(
                self._embedding_size,
                self._regret_network_layers,
                self._num_actions
            ).to(self._train_device)
            self._regret_networks_train.append(net2)

            self._loss_regrets.append(torch.nn.MSELoss(reduction='none'))
            self._optimizer_regrets.append(torch.optim.AdamW(net2.parameters(), lr=self._learning_rate, weight_decay=0.0001)) # !! if i reinitialize the net i need to also reinitialize this as well, move in to reinit logic
            self._regret_train_step.append(self._regret_train(player)) # !! this needs to be replaced with something else

        self._create_memories(memory_capacity)

        self._val_network = ValueNetwork(self._value_embedding_size, self._value_network_layers)
        self._val_network_train = ValueNetwork(self._value_embedding_size, self._value_network_layers)
        self._loss_value = torch.nn.MSELoss()
        self._optimizer_value = torch.optim.AdamW(self._val_network_train.parameters(), lr=self._learning_rate, weight_decay=0.0001)
        self._value_train_step = self._value_train()
        self._value_test_step = self._value_test()


    def _reinitialize_policy_network(self):
        self._policy_network = PolicyNetwork(self._embedding_size,
                                             self._policy_network_layers,
                                             self._num_actions)
        self._optimizer_policy = torch.optim.AdamW(self._policy_network.parameters(), lr=self._learning_rate, weight_decay=0.0001)
        self._loss_policy = torch.nn.MSELoss()

    def _reinitialize_regret_network(self, player):
        self._regret_networks_train[player] = RegretNetwork(
            self._embedding_size, self._regret_network_layers,
            self._num_actions)
        self._optimizer_regrets[player] = torch.optim.AdamW(self._regret_networks_train[player].parameters(), lr=self._learning_rate, weight_decay=0.0001)
        self._regret_train_step[player] = self._regret_train(player)

    def get_example_info_state(self, player):
        return self._example_info_state[player]
    
    def get_example_hist_state(self):
        return self._example_hist_state
    
    def get_example_legal_actions_mask(self, player):
        return self._example_legal_actions_mask[player]
    
    def _reinitialize_value_network(self):
        self._val_network_train = ValueNetwork(
            self._value_embedding_size, self._value_network_layers)
        self._optimizer_value = torch.optim.AdamW(self._val_network_train.parameters(), lr=self._learning_rate, weight_decay=0.0001)
        self._value_train_step = self._value_train()

    @property
    def regret_buffers(self):
        return self._regret_memories
    
    @property
    def average_policy_buffer(self):
        return self._average_policy_memories
    
    def clear_regret_buffers(self):
        for p in range(self._num_players):
            self._regret_memories[p].clear()

    def _create_memories(self, memory_capacity):
        self._average_policy_memories = ReservoirBuffer(memory_capacity)
        self._regret_memories = [
            ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._value_memory = ReservoirBuffer(memory_capacity)
        self._value_memory_test = ReservoirBuffer(memory_capacity)

    def get_val_weights(self):
        return self._val_network.state_dict()
    
    def set_val_weights(self, weights):
        self._val_network.load_state_dict(weights)

    def get_num_calls(self):
        num_calls = 0
        for p in range(self._num_players):
            num_calls += self._regret_memories[p].get_num_calls()
        return num_calls
    
    def set_iteration(self, iteration):
        self._iteration = iteration

    def get_weights(self): # !! rename this function to get_regret_weights?
        regret_weights = [self._regret_networks[player].state_dict() for player in range(self._num_players)]
        return regret_weights
    
    def get_policy_weights(self):
        policy_weights = self._policy_network.state_dict()
        return policy_weights
    
    def set_policy_weights(self, policy_weights):
        self._reinitialize_policy_network()
        self._policy_network.load_state_dict(policy_weights)

    def get_regret_memories(self, player):
        return self._regret_memories[player].get_data()
    
    def get_value_memory(self):
        return self._value_memory.get_data()
    
    def clear_value_memory(self):
        self._value_memory.clear()

    def get_value_memory_test(self):
        return self._value_memory_test.get_data()
    
    def get_average_policy_memories(self):
        return self._average_policy_memories.get_data()
    
    def get_num_nodes(self):
        return self._nodes_visited
    
    def get_squared_errors(self):
        return self._squared_errors
    
    def reset_squared_errors(self):
        self._squared_errors = []

    def get_squared_errors_child(self):
        return self._squared_errors_child
    
    def reset_squared_errors_child(self):
        self._squared_errors_child = []

    def clear_val_memories_test(self):
        self._value_memory_test.clear()

    def clear_val_memories(self):
        self._value_memory.clear()

    def traverse_game_tree_n_times(self, n, p, train_regret = False, train_value = False,
                                   track_mean_squares = True, on_policy_prob = 0, expl = 0.6, val_test = False):
        for i in range(n):
            if i > 0:
                track_mean_squares = False
            self._traverse_game_tree(self._root_node, p, my_reach=1.0, opp_reach=1.0, sample_reach=1.0,
                                     my_sample_reach=1.0, train_regret=train_regret, train_value=train_value,
                                     track_mean_squares=track_mean_squares, on_policy_prob=on_policy_prob,
                                     expl=expl, val_test=val_test)
    
    def init_regret_net(self):
        # initialize regret net for each player
        for p in range(self._num_players):
            example_info_state = self.get_example_info_state(p)
            example_legal_actions_mask = self.get_example_legal_actions_mask(p)
            self.traverse_game_tree_n_times(1, p, track_mean_squares=False)
            self._init_main_regret_network(example_info_state, example_legal_actions_mask, p)
        

    def init_val_net(self):
        example_hist_state = self.get_example_hist_state()
        example_legal_actions_mask = self.get_example_legal_actions_mask(0)
        self._init_main_val_network(example_hist_state, example_legal_actions_mask) # !! fixed, was originally regret_net on accident

    def play_game_against_random(self):
        # play one game per player
        reward = 0
        for player in [0, 1]:
            state = self._game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    aidx = np.random.choice(range(len(outcomes)), p=probs)
                    action = outcomes[aidx]
                else:
                    cur_player = state.current_player()
                    legal_actions = state.legal_actions(cur_player)
                    legal_actions_mask = torch.tensor(state.legal_actions_mask(cur_player), dtype=torch.float32)
                    obs = torch.tensor(state.observation_tensor(), dtype=torch.float32)
                    if len(obs.ndim) == 1:
                        obs = obs.unsqueeze(0)
                    if cur_player == player:
                        with torch.no_grad():
                            probs = self._policy_network((obs, legal_actions_mask))
                            probs = probs.cpu().numpy()[0]
                            probs /= probs.sum()
                            action = np.random.choice(range(state.num_distinct_actions()), p=probs)
                    elif cur_player == 1 - player:
                        action = random.choice(state.legal_actions())
                    else:
                        print("Got player", str(cur_player))
                        break
                state.apply_action(action)
            reward += state.returns()[player]
        return reward
    
    def play_n_games_against_random(self, n):
        total_reward = 0
        for i in range(n):
            reward = self.play_game_against_random()
            total_reward += reward
        return total_reward / (2 * n)
    
    def print_mse(self):
        # for tracking MSE
        squared_errors = self.get_squared_errors()
        self.reset_squared_errors()
        squared_errors_child = self.get_squared_errors_child()
        self.reset_squared_errors_child()
        print(sum(squared_errors) / len(squared_errors), "Mean Squared Errors")
        print(sum(squared_errors_child) / len(squared_errors_child), "Mean Squared Errors Child")

    def solve(self, save_path_convs=None):
        """Main solution logic"""
        regret_losses = collections.defaultdict(list)
        value_losses = []
        str(datetime.now())
        timestr = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        if self._use_balanced_probs:
            self._get_balanced_probs(self._root_node)

        # !! may need to assign torch calls later
        if self._save_average_policy_memories:
            self._memories_tfrecordfile = open(self._memories_filepath, "wb") # !! this needs to be manually closed later with self._memories_file.close()
        convs = []
        nodes = []
        self.traverse_game_tree_n_times(1, 0, track_mean_squares = False)

        for i in range(self._num_iterations + 1): # (Line 3)
            print(f"Iteration #{i}")

            start = time.time()
            if self._experiment_string is not None:
                print(self._experiment_string)

            # initialize weights
            self.init_regret_net()
            self.init_val_net()
            # train value function (Line 4)
            self.traverse_game_tree_n_times(self._num_val_fn_traversals, 0,
                                            train_value=True,
                                            track_mean_squares=False,
                                            on_policy_prob=self._val_op_prob,
                                            expl=self._val_expl)
            self.traverse_game_tree_n_times(20, 0, train_value=True, track_mean_squares=False,
                                            on_policy_prob=self._val_op_prob,
                                            expl=self._val_expl,
                                            val_test = True)
            val_traj_time = time.time()
            print(val_traj_time - start, 'value trajectory time')
            if self._reinit_value_network:
                self._reinitialize_value_network()
            value_losses.append(self._learn_value_network())
            print(value_losses[-1], 'value loss')
            test_loss = self._get_value_test_loss()
            print(test_loss, 'test loss')
            if self._clear_value_buffer:
                self.clear_val_memories_test()
                self.clear_val_memories()
            val_train_time = time.time()
            print(val_train_time - val_traj_time, 'value train time')

            # track_mse = False # !! im not sure the point of this
            # train regret network
            for p in range(self._num_players): # (Line 6)
                regret_start_time = time.time()
                results = []
                self.traverse_game_tree_n_times(self._num_traversals, p,
                                                train_regret = True,
                                                track_mean_squares = False,
                                                expl = self._expl) # (Lines 7-13)
                num_nodes = self.get_num_nodes()
                regret_traj_time = time.time()
                print(regret_traj_time - regret_start_time, 'regret trajectory time')
                if self._reinitialize_regret_networks:
                    self._reinitialize_regret_network(p) # (Line 5)\
                regret_losses[p].append(self._learn_regret_network(p)) # (Line 14)
                if self._save_regret_networks:
                    os.makedirs(self._save_regret_networks, exist_ok = True)
                    self._regret_networks[p].save(
                        os.path.join(self._save_regret_networks, f'regretnet_p{p}_it{self._iteration:04}'))
                print(time.time() - regret_traj_time, 'regret train time')
            # self.print_mse()
            total_regret_time = time.time()
            print(total_regret_time - val_train_time, 'total regret time')

            # check exploitability
            self._iteration += 1
            if i % self._check_exploitability_every == 0:
                exp_start_time = time.time()
                self._reinitialize_policy_network()
                policy_loss = self._learn_average_policy_network()
                if self._save_policy_weights:
                    save_path_model = save_path_convs + "/" + timestr
                    os.makedirs(save_path_model, exist_ok = True)
                    model_path = save_path_model + "/policy_nodes_" + str(num_nodes)
                    torch.save(self._policy_network.state_dict(), model_path)
                    torch.save(self._val_network.state_dict(), save_path_model + f'/value_net_{self._iteration}')
                    torch.save(self._regret_networks[0], save_path_model + f'/regret_net_p0_{self._iteration}')
                    torch.save(self._regret_networks[1], save_path_model + f'/regret_net_p1_{self._iteration}')
                    print("saved policy to ", model_path)
                    print("BUFFER LENGTHS:", len(self._average_policy_memories), len(self._regret_memories[0]), len(self._value_memory))
                    # self.save_policy_network(model_path + "full_model") # !! redundant, remove save_policy_network() function, replace with .save(model, 'model.pth')
                    # print("saved policy to ", model_path + "full_model") # !! redundant call
                if self._play_against_random:
                    start_time = time.time()
                    avg_reward = self.play_n_games_against_random(self._num_random_games)
                    print(avg_reward, "Average reward against random\n")
                if self._compute_exploitability:
                    conv = exploitability.nash_conv(self._game, policy.tabular_policy_from_callable(self._game, self.action_probabilities))
                    convs.append(conv)
                    nodes.append(num_nodes)
                    if save_path_convs:
                        convs_path = save_path_convs + "_convs.npy"
                        nodes_path = save_path_convs + "_nodes.npy"
                        np.save(convs_path, np.array(convs))
                        np.save(nodes_path, np.array(nodes))
                    print("iteration, nodes, nash_conv: ", self._iteration, num_nodes, conv)
                    print(time.time() - exp_start_time, 'exploitability time')
            print("end iter")

        # Train policy network
        policy_loss = self._learn_average_policy_network() # (Line 15)
        return regret_losses, policy_loss, convs, nodes, value_losses # (Line 16)
    
    def save_policy_network(self, outputfolder):
        os.makedirs(outputfolder, exist_ok=True)
        torch.save(self._policy_network.state_dict(), outputfolder)

    def train_policy_network_from_file(self,
                                       save_path,
                                       iteration = None,
                                       batch_size_average_policy = None,
                                       policy_network_train_steps = None,
                                       reinitialize_policy_network = True):
        self._memories_save_path = save_path
        if iteration:
            self._iteration = iteration
        if batch_size_average_policy:
            self._batch_size_average_policy = batch_size_average_policy
        if policy_network_train_steps:
            self._policy_network_train_steps = policy_network_train_steps
        if reinitialize_policy_network:
            self._reinitialize_policy_network()
        policy_loss = self._learn_average_policy_network()
        return policy_loss
    
    def _add_to_average_policy_memory(self, info_state, iteration,
                                      average_policy_action_probs, legal_actions_mask):
        serialized_example = self._serialize_average_policy_memory(
            info_state, iteration, average_policy_action_probs, legal_actions_mask
        )
        if self._save_average_policy_memories:
            with open(self._memories_savefile, "ab") as f:
                f.write(serialized_example + b"<END>")
        else:
            self._average_policy_memories.add(serialized_example)



    def _serialize_average_policy_memory(self, info_state, iteration, action_probs, legal_actions_mask):
        record = {
            'info_state': info_state, # removed tolist()
            'action_probs': action_probs.tolist(),
            'iteration': iteration,
            'legal_actions': legal_actions_mask # removed tolist()
        }
        return pickle.dumps(record)
    
    def _deserialize_average_policy_memory(self, serialized_batch):
        records = pickle.loads(serialized_batch)
        info_states = torch.tensor(records['info_state'], dtype=torch.float32) # removed list comprehension
        action_probs = torch.tensor(records['action_probs'], dtype=torch.float32) # removed list comprehension
        iterations = torch.tensor(records['iteration'], dtype=torch.float32) # removed list comprehension
        legal_actions = torch.tensor(records['legal_actions'], dtype=torch.float32) # removed list comprehension
        return (info_states, action_probs, iterations, legal_actions)

    def _serialize_regret_memory(self, info_state, iteration, samp_regret, legal_actions_mask):
        record = {
            'info_state': info_state, # removed tolist()
            'iteration': iteration,
            'samp_regret': samp_regret.tolist(),
            'legal_actions': legal_actions_mask # removed tolist()
        }
        return pickle.dumps(record)
    
    def _deserialize_regret_memory(self, serialized_batch):
        records = pickle.loads(serialized_batch)
        info_states = torch.tensor(records['info_state'], dtype=torch.float32) # removed list comprehension
        samp_regret = torch.tensor(records['samp_regret'], dtype=torch.float32) # removed list comprehension
        iterations = torch.tensor(records['iteration'], dtype=torch.float32) # removed list comprehension
        legal_actions = torch.tensor(records['legal_actions'], dtype=torch.float32) # removed list comprehension
        return (info_states, samp_regret, iterations, legal_actions)
    
    def _serialize_value_memory(self, hist_state, iteration, samp_value, legal_actions_mask):
        record = {
            'hist_state': hist_state.tolist(),
            'iteration': iteration,
            'samp_value': samp_value, # removed tolist()
            'legal_actions': legal_actions_mask # removed tolist()
        }
        return pickle.dumps(record)

    def _deserialize_value_memory(self, serialized_batch):
        # !! temporarily treating this as a single example
        records = pickle.loads(serialized_batch) # !! adjustment made here
        hist_states = torch.tensor(records['hist_state'], dtype=torch.float32) # removed list comprehension
        samp_values = torch.tensor(records['samp_value'], dtype=torch.float32) # removed list comprehension
        iterations = torch.tensor(records['iteration'], dtype=torch.float32) # removed list comprehension
        legal_actions = torch.tensor(records['legal_actions'], dtype=torch.float32) # removed list comprehension
        return (hist_states, samp_values, iterations, legal_actions)



    def _baseline(self, state, aidx): # !! useless function? check later
        return 0
    
    def _baseline_corrected_child_value(self, state, sampled_aidx,
                                        aidx, child_value, sample_prob):
        # From Eq. 9 of Scmid et al. 2019 (VR-MCCFR @ AAAI)
        baseline = self._baseline(state, aidx)
        if aidx == sampled_aidx:
            return baseline + (child_value - baseline) / sample_prob
        else:
            return baseline
        
    def _exact_value(self, state, update_player):
        state = state.clone()
        if state.is_terminal():
            return state.player_return(update_player)
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            val = 0
            for aidx in range(len(outcomes)):
                new_state = state.child(outcomes[aidx])
                val += probs[aidx] * self._exact_value(new_state, update_player)
            return val
        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        _, policy = self._sample_action_from_regret(state, cur_player)
        val = 0
        for aidx in range(num_legal_actions):
            new_state = state.child(legal_actions[aidx])
            val += policy[aidx] * self._exact_value(new_state, update_player)
        return val
    
    def _get_balanced_probs(self, state):
        if state.is_terminal():
            return 1
        elif state.is_chance_node():
            legal_actions = state.legal_actions()
            num_nodes = 0
            for action in legal_actions:
                num_nodes += self._get_balanced_probs(state.child(action))
            return num_nodes
        else:
            legal_actions = state.legal_actions()
            num_nodes = 0
            balanced_probs = np.zeros((state.num_distinct_actions()))
            for action in legal_actions:
                nodes = self._get_balanced_probs(state.child(action))
                balanced_probs[action] = nodes
                num_nodes += nodes
            self._balanced_probs[state.information_state_string()] = balanced_probs / balanced_probs.sum()
            return num_nodes
        
    def _traverse_game_tree(self, state, player, my_reach, opp_reach, sample_reach,
                            my_sample_reach, train_regret, train_value,
                            on_policy_prob=0, track_mean_squares=True, expl=1.0, val_test=False, last_action=0):
        """Performs a traversal of the game tree using external sampling
        
        Over a traversal, the regret and average_policy memories are populated with
        computed regret values and matched regrets respectively if train_regret = True
        If train_value = True, then we use traversals to train the history value function.

        Args:
           state: current OpenSpiel game state
           player: (int) Player index for this traversal

        Returns:
           Recursively returns expected payoffs for each action
        """
        self._nodes_visited += 1
        if state.is_terminal():
            # Terminal state get returns
            return state.returns()[player], state.returns()[player]
        elif state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = np.random.choice(range(len(outcomes)), p=probs)
            action = outcomes[aidx]
            new_state = state.child(action)
            return self._traverse_game_tree(new_state, player, my_reach,
                                            probs[aidx] * opp_reach, probs[aidx] * sample_reach, my_sample_reach,
                                            train_regret, train_value, expl=expl,
                                            track_mean_squares=track_mean_squares, val_test=val_test,
                                            last_action=action)
        # With probability equal to op_prob, we switch over to on-policy rollout for remainder of a trajectory
        # used for value estimation to get coverage but not needing importance sampling
        if expl != 0.0:
            if np.random.rand() < on_policy_prob:
                expl = 0.0
        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        num_actions = state.num_distinct_actions()
        _, policy = self._sample_action_from_regret(state, state.current_player())

        if cur_player == player or train_value:
            uniform_policy = (np.array(state.legal_actions_mask()) / num_legal_actions)
            if self._use_balanced_probs:
                uniform_policy = self._balanced_probs[state.information_state_string()]
            sample_policy = expl * uniform_policy + (1.0 - expl) * policy
        else:
            sample_policy = policy

        sample_policy /= sample_policy.sum()
        sampled_action = np.random.choice(range(state.num_distinct_actions()), p=sample_policy)
        orig_state = state.clone()
        new_state = state.child(sampled_action)

        child_value = self._estimate_value_from_hist(new_state.clone(), player ,last_action=sampled_action)
        value_estimate = self._estimate_value_from_hist(state.clone(), player, last_action=last_action)
        if track_mean_squares:
            oracle_child_value = self._exact_value(new_state.clone(), player)
            oracle_value_estimate = self._exact_value(state.clone(), player)
            squared_error = (oracle_value_estimate - value_estimate) ** 2
            self._squared_errors.append(squared_error)
            squared_child_error = (oracle_child_value - child_value) ** 2
            self._squared_errors_child.append(squared_child_error)
        if cur_player == player:
            new_my_reach = my_reach * policy[sampled_action]
            new_opp_reach = opp_reach
            new_my_sample_reach = my_sample_reach * sample_policy[sampled_action]
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_action]
            new_my_sample_reach = my_sample_reach
        new_sample_reach = sample_reach * sample_policy[sampled_action]
        iw_sampled_value, sampled_value = self._traverse_game_tree(new_state, player, new_my_reach,
                                                                   new_opp_reach, new_sample_reach, new_my_sample_reach,
                                                                   train_regret, train_value, expl=expl,
                                                                   track_mean_squares=track_mean_squares,
                                                                   val_test=val_test, last_action=sampled_action)
        importance_weighted_sampled_value = iw_sampled_value * policy[sampled_action] / sample_policy[sampled_action]

        # Compute each of the child estimated values
        child_values = np.zeros(num_actions, dtype=np.float64)
        if self._all_actions:
            for aidx in range(num_legal_actions):
                cloned_state = orig_state.clone()
                action = legal_actions[aidx]
                new_cloned_state = cloned_state.child(action)
                child_values[action] = self._estimate_value_from_hist(new_cloned_state.clone(), player, last_action=action) # this may need to be updated (DepreciationWarning)
        else:
            child_values[sampled_action] = child_value / sample_policy[sampled_action]
        if train_regret:
            if cur_player == player:
                cf_action_values = 0 * policy
                for action in range(num_actions):
                    if self._importance_sampling:
                        action_sample_reach = my_sample_reach * sample_policy[sampled_action]
                        cf_value = value_estimate * min(1 / my_sample_reach, self._importance_sampling_threshold)
                        cf_action_value = child_values[action] * min(1/ action_sample_reach, self._importance_sampling_threshold)

                    else:
                        cf_action_value = child_values[action]
                        cf_value = value_estimate
                    cf_action_values[action] = cf_action_value
                
                samp_regret = (cf_action_values - cf_value) * state.legal_actions_mask(player)
                network_input = state.information_state_tensor()
                self._regret_memories[player].add(self._serialize_regret_memory(network_input,
                                                                                self._iteration,
                                                                                samp_regret,
                                                                                state.legal_actions_mask(player)))
            else:
                obs_input = state.information_state_tensor(cur_player)
                self._add_to_average_policy_memory(obs_input, self._iteration,
                                                   policy, state.legal_actions_mask(cur_player))
        # value function predicts value for player 0
        if train_value:
            # if op_prob = 0, then we have importance weighted sampling
            # if op_prob > 0, then we need to wait until expl = 0 to get pure on-policy rollouts
            if on_policy_prob == 0 or expl == 0:
                hist_state = np.append(state.information_state_tensor(0), state.information_state_tensor(1))
                assert player == 0
                if self._val_bootstrap:
                    if self._all_actions:
                        target = policy @ child_values
                    else:
                        target = child_value * policy[sampled_action] / sample_policy[sampled_action]
                elif self._debug_val:
                    target = child_value * policy[sampled_action] / sample_policy[sampled_action]
                    print(target, 'value target')
                else:
                    target = iw_sampled_value
                if val_test:
                    self._value_memory_test.add(
                        self._serialize_value_memory(hist_state, self._iteration, target,
                                                     state.legal_actions_mask(cur_player)))
                else:
                    self._value_memory.add(
                        self._serialize_value_memory(hist_state, self._iteration, target,
                                                     state.legal_actions_mask(cur_player)))
        return importance_weighted_sampled_value, sampled_value

    def _init_main_regret_network(self, info_state, legal_actions_mask, player): # !! is this function in use? Check to make sure
        unsqueezed_info_state = info_state.unsqueeze(0)
        self._regret_networks[player].eval() # set to eval mode
        with torch.no_grad():
            regrets = self._regret_networks[player]((unsqueezed_info_state, legal_actions_mask))[0] #!! removed training=False

    def _init_main_val_network(self, hist_state, legal_actions_mask):  
        unsqueezed_hist_state = torch.from_numpy(hist_state).float().unsqueeze(0) # !! converted to torch tensor
        self._val_network.eval() # !! set to eval mode
        with torch.no_grad():
            self._val_network((unsqueezed_hist_state, legal_actions_mask))[0] # !! removed training=False

    def _get_matched_regrets(self, info_state, legal_actions_mask, player):
        unsqueezed_info_state = info_state.unsqueeze(0)
        self._regret_networks[player].eval() # set to eval mode
        with torch.no_grad():
            regrets = self._regret_networks[player]((unsqueezed_info_state, legal_actions_mask))[0] # !! removed training=False
            regrets = torch.clamp(regrets, min=0)
            summed_regret = torch.sum(regrets)
            if summed_regret > 0:
                matched_regrets = regrets / summed_regret
            else:
                masked_regrets = torch.where(legal_actions_mask == 1, regrets, torch.tensor(-1e20, device=regrets.device))
                max_index = torch.argmax(masked_regrets)
                matched_regrets = F.one_hot(max_index, num_classes=self._num_actions).float()
        return regrets, matched_regrets

    def _get_estimated_value(self, hist_state, legal_actions_mask):
        unsqueezed_hist_state = hist_state.unsqueeze(0)
        self._val_network.eval() # !! set to eval mode
        with torch.no_grad():
            estimated_val = self._val_network((unsqueezed_hist_state, legal_actions_mask))[0] # removed training=False
        return estimated_val

    def _sample_action_from_regret(self, state, player):
        info_state = torch.tensor(state.information_state_tensor(player), dtype=torch.float32)
        legal_actions_mask = torch.tensor(state.legal_actions_mask(player), dtype=torch.float32)

        self._example_info_state[player] = info_state
        self._example_legal_actions_mask[player] = legal_actions_mask

        regrets, matched_regrets = self._get_matched_regrets(
            info_state, legal_actions_mask, player
        )
        return regrets.cpu().numpy(), matched_regrets.cpu().numpy()

    def _estimate_value_from_hist(self, state, player, last_action=0):
        """returns an info state policy by applying regret matching
        
        Args:
           state: current OpenSpiel game state
           player: (int) Player index over which to compute regrets
           
        Returns:
           1. (np-array) regret values for info state actions indexed by action
           2. (np-array) Matched regrets, prob for actions indexed by action
        """
        state = state.clone()
        if state.is_terminal():
            return state.player_return(player)

        hist_state = np.append(state.information_state_tensor(0), state.information_state_tensor(1))
        self._example_hist_state = hist_state
        hist_state = torch.as_tensor(hist_state, dtype=torch.float32)
        if state.current_player() == pyspiel.PlayerId.CHANCE: # pyspiel has a bug when trying to call legal_actions_mask with a player argument during a chance_node
            mask = [0] * game.num_distinct_actions()
            legal_actions_mask = torch.as_tensor(mask, dtype=torch.float32)
        else:
            legal_actions_mask = torch.as_tensor(state.legal_actions_mask(player), dtype=torch.float32)
        estimated_value = self._get_estimated_value(hist_state, legal_actions_mask)
        if player == 1:
            estimated_value = -estimated_value
        return estimated_value.cpu().numpy()
    
    def action_probabilities(self, state):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        legal_actions_mask = torch.as_tensor(
            state.legal_actions_mask(cur_player), dtype=torch.float32
        )
        info_state_vector = torch.as_tensor(
            state.information_state_tensor(), dtype=torch.float32
        )
        if len(info_state_vector.shape) == 1:
            info_state_vector = info_state_vector.unsqueeze(0)
        self._policy_network.eval() # !! set policy_net to evaluation mode
        with torch.no_grad(): # !! added torch_nograd()
            probs = self._policy_network((info_state_vector, legal_actions_mask)) # !! removed training=False
        probs = probs.detach().cpu().numpy() # !! added detach, maybe not the right call though?
        return {action: probs[0][action] for action in legal_actions}
    
    def _get_regret_dataset(self, player):
        data = self.get_regret_memories(player)
        dataset = InterDataset(
            buffer_data=data,
            transform=self._deserialize_regret_memory
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size_regret,
            shuffle=True,
            drop_last=False
        )
        return infinite_dataloader(dataloader)

    def _get_value_dataset(self):
        data = self.get_value_memory()
        dataset = InterDataset(
            buffer_data=data,
            transform = self._deserialize_value_memory
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size_value,
            shuffle=True,
            drop_last=False
        )
        return infinite_dataloader(dataloader)

    def _get_value_dataset_test(self):
        data = self.get_value_memory_test()
        dataset = InterDataset(
            buffer_data=data,
            transform = self._deserialize_value_memory
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size_value,
            shuffle=True,
            drop_last=False
        )
        return infinite_dataloader(dataloader)

    def _get_value_test_loss(self):
        iteration = torch.tensor(self._iteration, dtype=torch.float32)
        data = self._get_value_dataset_test()
        for batch in islice(data, 1):
            main_loss = self._value_test_step(*batch, iteration)
            if self._debug_val:
                print(main_loss, "test loss")

        return main_loss

    def _regret_train(self, player):

        def train_step(info_states, regrets, iterations, masks, iter):
            model = self._regret_networks_train[player]
            model.train()

            self._optimizer_regrets[player].zero_grad()

            preds = model((info_states, masks))
            sample_weights = (iterations * 2.0) / iter
            loss_fn = self._loss_regrets[player]

            raw_losses = loss_fn(preds, regrets)
            weighted_losses = raw_losses * sample_weights.unsqueeze(1) #!! unsqueeze might be unneccesary here, not sure
            main_loss = weighted_losses.mean()

            main_loss.backward()
            self._optimizer_regrets[player].step()

            return main_loss.item()
        
        return train_step
    
    def _value_train(self):

        def train_step(full_hist_states, values, iterations, masks, iter):
            model = self._val_network_train # !! possible mix up, make sure that right model is being selected (see line 1217 as well)
            model.train()

            self._optimizer_value.zero_grad()

            preds = model((full_hist_states, masks))
            preds = preds.squeeze(-1) # !! possible solution
            main_loss = self._loss_value(preds, values) # !! this is the warning line
            main_loss.backward()
            self._optimizer_value.step()

            return main_loss.item()
        
        return train_step
    
    def _value_test(self):

        def train_step(full_hist_states, values, iterations, masks, iter):
            model = self._val_network
            model.train() # was model.eval()

            with torch.no_grad():
                preds = model((full_hist_states, masks))
                preds = preds.squeeze(-1) # !! possible solution?
                main_loss = self._loss_value(preds, values)

            return main_loss.item()
        
        return train_step
    
    def _learn_value_network(self):
        """Perform a Q-network update""" # !! consider edge case when buffer doesnt have enough elements
        iteration = torch.tensor(self._iteration, dtype=torch.float32)
        data = self._get_value_dataset()
        for batch in islice(data, self._value_network_train_steps):
            main_loss = self._value_train_step(*batch, iteration)
            if self._debug_val:
                print(main_loss, "main val loss")

        self._val_network.load_state_dict(
            self._val_network_train.state_dict()
        )
        return main_loss
    
    def _learn_regret_network(self, player):
        """Perform a Q-network update"""
        iteration = torch.tensor(self._iteration, dtype=torch.float32)
        data = self._get_regret_dataset(player)
        for batch in islice(data, self._regret_network_train_steps):
            main_loss = self._regret_train_step[player](*batch, iteration)

        self._regret_networks[player].load_state_dict(
            self._regret_networks_train[player].state_dict()
        )
        return main_loss

    def _get_average_policy_dataset(self):
        if self._memories_save_path:
            raise NotImplemented("Not there yet")
        else:
            data = self.get_average_policy_memories()
        dataset = InterDataset(
            buffer_data=data,
            transform = self._deserialize_average_policy_memory
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self._batch_size_average_policy,
            shuffle=True,
            drop_last=False
        )
        return infinite_dataloader(dataloader)

    def _learn_average_policy_network(self):
        
        def train_step(info_states, action_probs, iterations, masks):
            model = self._policy_network
            model.train()

            self._optimizer_policy.zero_grad()

            preds = model((info_states, masks))
            sample_weights = (iterations * 2.0) / self._iteration
            loss_fn = self._loss_policy

            raw_losses = loss_fn(preds, action_probs)
            weighted_losses = raw_losses * sample_weights.unsqueeze(1) #!! unsqueeze might be unneccesary here, not sure
            main_loss = weighted_losses.mean()

            main_loss.backward()
            self._optimizer_policy.step()

            return main_loss.item()
        
        data = self._get_average_policy_dataset()
        for batch in islice(data, self._policy_network_train_steps):
            main_loss = train_step(*batch)
        return main_loss
        

if __name__ == "__main__":
    # Quick example how to run on Kuhn
    # Hyperparameters not tuned
    print("STARTED:")
    train_device = 'cpu'
    save_path = "./tmp/results/"
    os.makedirs(save_path, exist_ok=True)
    seven_card_stud.register()

    game = pyspiel.load_game("python_scs_poker")

    iters = 30
    num_traversals = 100
    num_val_fn_traversals = 100
    regret_train_steps = 100
    val_train_steps = 100
    policy_net_train_steps = 100
    batch_size_regret = 256
    batch_size_val = 256
    batch_size_pol = 256
    print("escher initializing")
    deep_cfr_solver = ESCHER(
        game,
        num_traversals=int(num_traversals),
        num_iterations=iters,
        check_exploitability_every=1, # was 10
        compute_exploitability=False,
        regret_network_train_steps=regret_train_steps,
        policy_network_train_steps=policy_net_train_steps,
        batch_size_regret=batch_size_regret,
        batch_size_average_policy=batch_size_pol,
        value_network_train_steps=val_train_steps,
        batch_size_value=batch_size_val,
        train_device=train_device,
        learning_rate= 0.001, # smaller learning rate

        val_expl = 0.01, # apparently this parameter does a lot

        policy_network_layers=(64,64,64),
        regret_network_layers=(64,64,64),
        value_network_layers=(64,64,64),
        memory_capacity=100000, # smaller buffer capacity?
    )
    regret, pol_loss, convs, nodes, value_loss = deep_cfr_solver.solve(save_path_convs=save_path)

    print("COMPLETE!!!")
    print("regret losses over iterations:", regret)
    print("policy loss", pol_loss)