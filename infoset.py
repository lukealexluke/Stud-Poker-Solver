from stud import Game, SevenCardStud
import numpy as np


# Object which holds the Strategies of each player in a game
class StrategyProfile:

    def __init__(self, game):
        if not isinstance(game, Game):
            raise TypeError("Must call a Stud subclass when creating an instance")

        self.player_1 = InformationTree(game)
        self.player_2 = InformationTree(game)


# Tree which holds the information sets of a player
class InformationTree:

    def __init__(self, game):
        self.game = game
        self.root = InformationSet(game, [])
        
    def search_or_insert(self, infoset, history):
        current_node = self.root

        for element in infoset:
            if element not in current_node.children:
                # Create a new InformationSet if it doesn't already exist
                current_node.children[element] = InformationSet(history)
                # !! update instance to reflect information set not pure history
            current_node = current_node.children[element]


# A node on the information set tree, holds a mixed strategy and a table of regrets
class InformationSet:

    def __init__(self, game: Game, history):

        self.legal_actions = np.array(game.get_legal_actions(history))
        self.policy = np.divide(self.legal_actions, np.sum(self.legal_actions)) # !! modify this later so we can pick a different starting policy
        self.children = {}
        # !! update to reflect proper probabilities