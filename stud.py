# Parent Class that defines basic logic of a Stud Poker game
class StudGame():

    # [Bring-In, Complete, Call, Bet/Raise, Fold, Check, All-In]

    def __init__(self):
        pass

    # Returns True/False for whether or not the hand has ended
    def is_terminal(self, history):
        if not history:
            return False
        elif history[-1] == 4:
            return True
        
        round_count = 0
        length = len(history)
        left = 0

        while left < length:
            if history[left] == 2:
                round_count += 1
            elif history[left:left+2] == [5, 5]:
                round_count += 1
                left += 1
            left += 1

        if round_count == 5:
            return True
        else:
            return False
    
    def get_legal_actions(self, history):
        action_map = {
            0:[0, 1, 1, 0, 1, 0, 0], # Bring-In
            1:[0, 0, 1, 1, 1, 0, 0], # Complete
            2:[0, 0, 0, 1, 0, 1, 0], # Call
            3:[0, 0, 1, 1, 1, 0, 0], # Bet
            5:[0, 0, 0, 1, 0, 1, 0], # Check
            6:[0, 0, 1, 0, 1, 0, 0], # All-In
        }

        if not history: # Start of Hand
            return [1, 1, 0, 0, 0, 0, 0]
        elif self.is_terminal(history): # Hand is Over
            return [0, 0, 0, 0, 0, 0, 0]
        elif self.turn_to_act(history) == 'C': # Chance Node
            return [0, 0, 0, 1, 0, 1, 0]

        last_action = history[-1]
        return action_map.get(last_action, [0, 0, 0, 0, 0, 0, 0])
    
    # Determines who is first to act given a history and set of cards (Note: Player 1 is always assumed to be to the left of the dealer, and will act first if hands are tied)
    def turn_to_act(self, history: list, visible_cards: list):
        pass
        if history[-1] == 2:
            return 'C'
        elif history[-2:] == [5, 5] and history[-3:] != [5, 5, 5]:
            return 'C'
        
        swaps = 0
        first = self.hand_evaluation(visible_cards)
        for i in range(len(visible_cards[1])):
            curr = self.hand_evaluation([card[0:i+1] for card in visible_cards])
            if first != curr:
                swaps += 1
            
        return ((len(history) - swaps) % 2)
    
    # Determines the payoff at a terminal node
    def utility_function(self, history):
        raise NotImplementedError("Method is only defined for subclasses")
    
    # Determines the winning hand from an array of poker hands
    def hand_evaluation(self, player_hands):
        raise NotImplementedError("Method is only defined for subclasses")


# Classic Stud High Game
class SevenCardStud(StudGame):

    def __init__(self):
        super().__init__()
        self.stack = [1000, 1000]
        self.betting_structure = [1,2,4,8] # Ante - BI - (CO/SB) - BB
    
    def utility_function(self, history: list, player_hands: list):
        if not self.is_terminal(self, history):
            raise ValueError(f"hand history was not terminal: {history}")
        
        contribution = [self.betting_structure[0]] * 2

        if history[-1] == 4:
            for action in history:
                pass
            winner = self.turn_to_act(history, [hand[2:] for hand in player_hands])
            pot = sum(contribution)
            contribution = [0, 0]
            contribution[winner] = pot
            return contribution
        else:
            for action in history:
                pass
            pot = sum(contribution)
            contribution = [0, 0]
            contribution[self.hand_evaluation(player_hands)] = pot
            return contribution
        
    
    def hand_evaluation(self, player_hands: list):
        pass
        # given an array of hands (array of arrays) return the index of the best hand (either 0 or 1)


# Lowball variant of Stud
class Razz(StudGame):

    def __init__(self):
        super().__init__()

    
# Stud 8-or-Better (Split-Game)
class StudHiLo(StudGame):

    def __init__(self):
        super().__init__()

    # Stud8 has more complex logic with how it determines who wins what percentage of the pop (e.g. scoops, splits, chops on the low/high)
    def utility_function(self, history: list, player_hands: list):
        pass
