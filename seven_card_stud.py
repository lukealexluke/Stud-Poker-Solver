# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum
import eval7 as ev
from itertools import permutations, chain
import numpy as np
import pyspiel
import random

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suits = ['c', 'd', 'h', 's']

deck_dict = {}
index = 0
for suit in suits:
    for rank in ranks:
        deck_dict[index] = f"{rank}{suit}"
        index += 1

def best_hand(hands):
  """Returns index of best hand"""
  tf_cards_0 = [ev.Card(deck_dict[s]) for s in hands[0]]
  tf_cards_1 = [ev.Card(deck_dict[s]) for s in hands[1]]
  if ev.evaluate(tf_cards_0) > ev.evaluate(tf_cards_1):
    return 0
  elif ev.evaluate(tf_cards_0) < ev.evaluate(tf_cards_1):
    return 1
  else: # same rank, go to suits for tiebreaker (three cards only)
    if max(hands[0]) > max(hands[1]):
      return 0
    else:
      return 1

class Action(enum.IntEnum):
  BRING_IN = 0
  COMPLETE = 1
  CALL = 2
  BET = 3
  FOLD = 4
  CHECK = 5


_NUM_PLAYERS = 2
_DECK = frozenset([i for i in range(0,52)])
_GAME_TYPE = pyspiel.GameType(
    short_name="python_scs_poker",
    long_name="Python Seven-Card-Stud Poker",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-320,
    max_utility=320,
    utility_sum=0.0,
    max_game_length=31)  # complete doesnt count as a raise, but there are a maximum of four raises allowed per round of betting


class StudPokerGame(pyspiel.Game):
  """A Python version of Seven-Card-Stud Poker."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return StudPokerState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return StudPokerObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class StudPokerState(pyspiel.State):
  """A python version of the Kuhn poker state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.pot = [1.0, 1.0]
    self._game_over = False
    self._next_player = 0

    self._num_checks = 0 # (number of checks in a given round, ends round when equal to 2)
    self._num_raises = 0 # (number of raises in a given round, ends round when equal to 4)
    self._stakes = [1,2,8,16] # (Ante, Bring-In, Complete/Small-Bet, Big-Bet)
    self.public_cards = [[],[]] # up cards for both players
    self.private_cards = [[],[]] # down cards for both players

    self._third_street_sequence = [] # action sequence for third street
    self._fourth_street_sequence = [] # and so on...
    self._fifth_street_sequence = []
    self._sixth_street_sequence = []
    self._seventh_street_sequence = []

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif len(self.private_cards[1]) < 2 or len(self.public_cards[1]) == 0:
      return pyspiel.PlayerId.CHANCE
    elif self.pick_array() and self.pick_array()[-1] == Action.CALL:
      if self._seventh_street_sequence:
        return pyspiel.PlayerId.TERMINAL
      return pyspiel.PlayerId.CHANCE 
    elif self._num_checks == 2: # !! the issue lies here! _num_raises / _num_checks need to be reset AFTER cards are dealt, doing so in here makes it such that it resets while calling chance nodes, it only fails for later streets since third street has its own logic
      if self._seventh_street_sequence:
        return pyspiel.PlayerId.TERMINAL
      return pyspiel.PlayerId.CHANCE
    # accounted for on apply_action()
    else:
      return self._next_player

  def _legal_actions(self, player): # !! when action goes check check on seventh assertion error comes up
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    # specific state cases
    if not self._third_street_sequence:
      return [Action.BRING_IN, Action.COMPLETE]
    elif self._num_raises == 4:
      return [Action.CALL, Action.FOLD]
    elif self._third_street_sequence[-1] == Action.BRING_IN:
      return [Action.COMPLETE, Action.CALL, Action.FOLD]
    
    movelist = []

    if self.pick_array() and self.pick_array()[-1] == Action.COMPLETE:
      return [Action.CALL, Action.BET, Action.FOLD]
    
    if self._num_raises == 0:
      movelist.append(Action.CHECK)
    else:
      movelist.append(Action.CALL)
      movelist.append(Action.FOLD)
    movelist.append(Action.BET)
    return sorted(movelist)

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    taken_cards = self.public_cards + self.private_cards
    taken_cards = [item for sub_list in taken_cards for item in sub_list]
    outcomes = sorted(_DECK - set(taken_cards))
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      if len(self.private_cards[1]) < 2:
        if len(self.private_cards[0]) > len(self.private_cards[1]):
          self.private_cards[1].append(action)
        else:
          self.private_cards[0].append(action)
      elif len(self.public_cards[0]) == 0:
        self.public_cards[0].append(action)
      elif len(self.public_cards[1]) == 0:
        self.public_cards[1].append(action)
        self._next_player = 1 - best_hand(self.public_cards)

      elif len(self.public_cards[1]) < 4:
        if len(self.public_cards[0]) > len(self.public_cards[1]):
          self.public_cards[1].append(action)
          self._next_player = best_hand(self.public_cards)
          self._num_raises = 0
          self._num_checks = 0
        else:
          self.public_cards[0].append(action)
      else:
        if len(self.private_cards[0]) > len(self.private_cards[1]):
          self.private_cards[1].append(action)
          self._next_player = best_hand(self.public_cards)
          self._num_raises = 0
          self._num_checks = 0
        else:
          self.private_cards[0].append(action)
    else:
      self.pick_array().append(action)
      if action == Action.FOLD or (action == Action.CALL and self._seventh_street_sequence):
        self._game_over = True
      if action == Action.BET:
        if len(self.public_cards[0]) <= 2:
          self.pot[self._next_player] = max(self.pot) + self._stakes[2]
        else:
          self.pot[self._next_player] = max(self.pot) + self._stakes[3]
        self._num_raises += 1
      elif action == Action.BRING_IN:
        self.pot[self._next_player] = self._stakes[1]
      elif action == Action.COMPLETE:
        self.pot[self._next_player] = self._stakes[2]
      elif action == Action.CALL:
        self.pot[self._next_player] = max(self.pot)
        self._num_raises = 0
        self._num_checks = 0
      elif action == Action.CHECK:
        self._num_checks += 1
        if self._num_checks == 2 and self._seventh_street_sequence:
          self._game_over = True

      self._next_player = 1 - self._next_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.BRING_IN:
      return "Bring In"
    elif action == Action.COMPLETE:
      return "Complete"
    elif action == Action.CALL:
      return "Call"
    elif action == Action.CHECK:
      return "Check"
    elif action == Action.BET:
      return "Bet"
    else:
      return "Fold"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self): #!! might not be factoring in ante into the returns correctly
    """Total reward for each player over the course of the game so far."""
    pot = self.pot
    winnings = float(min(pot))
    if not self._game_over:
      return [0., 0.]
    elif pot[0] > pot[1]:
      return [winnings, -winnings]
    elif pot[0] < pot[1]:
      return [-winnings, winnings]
    elif best_hand(self.cards) == 0:
      return [winnings, -winnings]
    else:
      return [-winnings, winnings]
    
  def pick_array(self):
    if len(self.private_cards[1] + self.public_cards[1]) == 3:
      return self._third_street_sequence
    elif len(self.private_cards[1] + self.public_cards[1]) == 4:
      return self._fourth_street_sequence
    elif len(self.private_cards[1] + self.public_cards[1]) == 5:
      return self._fifth_street_sequence
    elif len(self.private_cards[1] + self.public_cards[1]) == 6:
      return self._sixth_street_sequence
    elif len(self.private_cards[1] + self.public_cards[1]) == 7:
      return self._seventh_street_sequence
    else:
      raise ValueError(f"Unexpected number of cards: {len(self.private_cards[1] + self.public_cards[1])}")

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    pub = ' '.join([str(c) for c in self.public_cards[0]]) + ' | ' + ' '.join([str(c) for c in self.public_cards[1]])
    spacer = pub.find('|')
    priv = ' '.join([str(c) for c in self.private_cards[0]]) + ' ' * spacer + '|' + ' ' * spacer + ' '.join([str(c) for c in self.private_cards[1]])
    pub = ' ' * (len(' '.join([str(c) for c in self.private_cards[0]]))) + pub
    bets = chain(self._third_street_sequence, self._fourth_street_sequence, self._fifth_street_sequence, self._sixth_street_sequence, self._seventh_street_sequence)
    bets = ' '.join(map(str, bets))
    return f"Cards:\n PUBLIC CARDS:  {pub}\nPRIVATE CARDS:  {priv}\nBets:\n{bets}\nPot\n{self.pot}"
  
  @property
  def cards(self):
    return [self.private_cards[0] + self.public_cards[0], self.private_cards[1] + self.public_cards[1]]
  
  @property
  def bets(self):
    return self._third_street_sequence + self._fourth_street_sequence + self._fifth_street_sequence + self._sixth_street_sequence + self._seventh_street_sequence


class StudPokerObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", 2, (2,))]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_card_1", 52, (52,)))
      pieces.append(("private_card_2", 52, (52,)))
      pieces.append(("private_card_3", 52, (52,)))
      for i in range(1, 5):
        pieces.append((f"player_upcard_{i}", 52, (52,)))
        pieces.append((f"opp_upcard_{i}", 52, (52,)))
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:
        pieces.append(("third_street", 42, (7, 6)))
        pieces.append(("fourth_street", 36, (6, 6)))
        pieces.append(("fifth_street", 36, (6, 6)))
        pieces.append(("sixth_street", 36, (6, 6)))
        pieces.append(("seventh_street", 36, (6, 6)))
      else:
        pieces.append(("pot_contribution", 2, (2,)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][player] = 1

    if "private_card_1" in self.dict and len(state.private_cards[player]) > 0:
      self.dict["private_card_1"][state.private_cards[player][0]] = 1
    if "private_card_2" in self.dict and len(state.private_cards[player]) > 1:
      self.dict["private_card_2"][state.private_cards[player][1]] = 1
    if "private_card_3" in self.dict and len(state.private_cards[player]) > 2:
      self.dict["private_card_3"][state.private_cards[player][2]] = 1

    if "player_upcard_1" in self.dict and len(state.public_cards[player]) > 0:
      self.dict["player_upcard_1"][state.public_cards[player][0]] = 1
    if "opp_upcard_1" in self.dict and len(state.public_cards[(player + 1) % 2]) > 0:
      self.dict["opp_card_1"][state.public_cards[(player + 1) % 2][0]] = 1
    if "player_upcard_2" in self.dict and len(state.public_cards[player]) > 1:
      self.dict["player_upcard_2"][state.public_cards[player][1]] = 1
    if "opp_upcard_2" in self.dict and len(state.public_cards[(player + 1) % 2]) > 1:
      self.dict["opp_card_2"][state.public_cards[(player + 1) % 2][1]] = 1
    if "player_upcard_3" in self.dict and len(state.public_cards[player]) > 2:
      self.dict["player_upcard_3"][state.public_cards[player][2]] = 1
    if "opp_upcard_3" in self.dict and len(state.public_cards[(player + 1) % 2]) > 2:
      self.dict["opp_card_3"][state.public_cards[(player + 1) % 2][2]] = 1
    if "player_upcard_4" in self.dict and len(state.public_cards[player]) > 3:
      self.dict["player_upcard_4"][state.public_cards[player][3]] = 1
    if "opp_upcard_4" in self.dict and len(state.public_cards[(player + 1) % 2]) > 3:
      self.dict["opp_card_4"][state.public_cards[(player + 1) % 2][3]] = 1

    if "pot_contribution" in self.dict:
      self.dict["pot_contribution"][:] = state.pot

    if "third_street" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["third_street"][turn, action] = 1
    if "fourth_street" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["fourth_street"][turn, action] = 1
    if "fifth_street" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["fifth_street"][turn, action] = 1
    if "sixth_street" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["sixth_street"][turn, action] = 1
    if "seventh_street" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["seventh_street"][turn, action] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")

    if "private_card_1" in self.dict and len(state.private_cards[player]) > 0:
      pieces.append(f"Private Cards:{state.cards[player]}")
    if "private_card_2" in self.dict and len(state.private_cards[player]) > 1:
      pieces.append(f"{state.cards[player]}")
    if "private_card_3" in self.dict and len(state.private_cards[player]) > 2:
      pieces.append(f"{state.cards[player]}")

    if "player_upcard_1" in self.dict and len(state.public_cards[player]) > 0:
      pieces.append(f"Player Upcards:{state.cards[player]}")
    if "player_upcard_2" in self.dict and len(state.public_cards[player]) > 1:
      pieces.append(f"{state.cards[player]}")
    if "player_upcard_3" in self.dict and len(state.public_cards[player]) > 2:
      pieces.append(f"{state.cards[player]}")
    if "player_upcard_4" in self.dict and len(state.public_cards[player]) > 3:
      pieces.append(f"{state.cards[player]}")

    if "opp_upcard_1" in self.dict and len(state.public_cards[(player + 1) % 2]) > 0:
      pieces.append(f"Opponent Upcards:{state.cards[player]}")
    if "opp_upcard_2" in self.dict and len(state.public_cards[(player + 1) % 2]) > 1:
      pieces.append(f"{state.cards[player]}")
    if "opp_upcard_3" in self.dict and len(state.public_cards[(player + 1) % 2]) > 2:
      pieces.append(f"{state.cards[player]}")
    if "opp_upcard_4" in self.dict and len(state.public_cards[(player + 1) % 2]) > 3:
      pieces.append(f"{state.cards[player]}")
    
    if "pot_contribution" in self.dict:
      pieces.append(f"pot[{int(state.pot[0])} {int(state.pot[1])}]")

    if "third_street" in self.dict and state.third_street:
      pieces.append("".join("pb"[b] for b in state.third_street))
    if "fourth_street" in self.dict and state.fourth_street:
      pieces.append("".join("pb"[b] for b in state.fourth_street))
    if "fifth_street" in self.dict and state.fifth_street:
      pieces.append("".join("pb"[b] for b in state.fifth_street))
    if "sixth_street" in self.dict and state.sixth_street:
      pieces.append("".join("pb"[b] for b in state.sixth_street))
    if "seventh_street" in self.dict and state.seventh_street:
      pieces.append("".join("pb"[b] for b in state.seventh_street))

    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, StudPokerGame)
# !! somewhere there is an assertion error with legal actions but im not sure where, its random
game = pyspiel.load_game("python_scs_poker")
print("INITIALIZING GAME")
state = game.new_initial_state()
print("GAME START")
while not state.is_terminal():
  # The state can be three different types: chance node,
  # simultaneous node, or decision node
  if state.is_chance_node():
    # Chance node: sample an outcome
    outcomes = state.chance_outcomes()
    num_actions = len(outcomes)
    print("Chance node, got " + str(num_actions) + " outcomes")
    action_list, prob_list = zip(*outcomes)
    action = np.random.choice(action_list, p=prob_list)
    print("Sampled outcome: ",
          state.action_to_string(state.current_player(), action))
    state.apply_action(action)
  else:
    # Decision node: sample action for the single current player
    action = random.choice(state.legal_actions(state.current_player()))
    action_string = state.action_to_string(state.current_player(), action)
    print("Player ", state.current_player(), ", randomly sampled action: ",
          action_string)
    state.apply_action(action)
  print(str(state))

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))