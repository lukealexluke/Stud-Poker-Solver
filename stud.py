import eval7 as ev
import copy as cop
from collections import Counter


# Parent Class that defines basic logic of a Stud Poker game
class Game():

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
            3:[0, 0, 1, 1, 1, 0, 0], # Bet/Raise
            5:[0, 0, 0, 1, 0, 1, 0], # Check
            6:[0, 0, 1, 0, 1, 0, 0], # All-In
        }
        if not history: # Start of Hand
            return [1, 1, 0, 0, 0, 0, 0]
        elif self.is_terminal(history): # Hand is Over
            return [0, 0, 0, 0, 0, 0, 0]
        elif history[-1] == 'C': # Chance Node
            return [0, 0, 0, 1, 0, 1, 0]
        elif history[-4:] == [3, 3, 3, 3]:
            return [0, 0, 1, 0, 1, 0, 0]
        elif history[-4:] == [0, 1, 3, 3]: #!! update limit to one bet and four raises, instead of one bet and three raises
            return [0, 0, 1, 0, 1, 0, 0]

        last_action = history[-1]
        return action_map.get(last_action, [0, 0, 0, 0, 0, 0, 0])
    
    # Determines who is first to act given a history and set of cards (Note: Player 1 is always assumed to be to the left of the dealer, and will act first if hands are tied)
    def turn_to_act(self, history: list, visible_cards: list):

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
            
        return ((len(history) - swaps - history.count('C')) % 2)
    
    # Determines the payoff at a terminal node
    def utility_function(self, history):
        raise NotImplementedError("Method is only defined for subclasses")
    
    # Determines the winning hand from an array of poker hands
    def hand_evaluation(self, player_hands):
        raise NotImplementedError("Method is only defined for subclasses")


# Classic Stud High Game
class SevenCardStud(Game):

    def __init__(self):
        super().__init__()
        self.stack = [1000, 1000]
        self.betting_structure = [1,2,4,8] # Ante - BI - (CO/SB) - BB
        self.card_dict = self.initialize_card_dict()

    def initialize_card_dict(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['c', 'd', 'h', 's']
        card_dict = {}
        index = 1
        for suit in suits:
            for rank in ranks:
                card_dict[index] = f"{rank}{suit}"
                index += 1
        return card_dict

    def number_to_string(self, player_hands: list):
        ret_hands = cop.deepcopy(player_hands)
        for i, hand in enumerate(ret_hands):
            for j, card in enumerate(hand):
                card = self.card_dict[card]
                ret_hands[i][j] = card
        return ret_hands

    def utility_function(self, history: list, player_hands: list):
        if not self.is_terminal(history):
            raise ValueError(f"hand history was not terminal: {history}")

        contribution = [0, 0]
        pot = (2 * self.betting_structure[0])
        round_num = 3
        betsize = 2
        bet_map = {
            0:self.betting_structure[1], # Bring-In
            1:self.betting_structure[2], # Complete
            2:lambda: max(contribution), # Call
            3:lambda: max(contribution) + self.betting_structure[betsize], # Bet
            4:lambda: min(contribution), # Fold
            5:contribution[0], # Check
        }

        for action in history:
            if action == 'C':
                pot += sum(contribution)
                contribution = [0, 0]
                round_num += 1
                if round_num == 5:
                    betsize += 1
            else:
                contribution[contribution.index(min(contribution))] = bet_map[action]() if callable(bet_map[action]) else bet_map[action]

        if history[-1] == 4:
            winner = self.turn_to_act(history, [hand[2:] for hand in player_hands])
            pot += sum(contribution)
            contribution = [0, 0]
            contribution[winner] = pot
            return contribution
        
        else:
            pot += sum(contribution)
            if self.hand_evaluation(player_hands) == 2:
                contribution = [pot / 2] * 2
                return contribution
            contribution = [0, 0]
            contribution[self.hand_evaluation(player_hands)] = pot
            return contribution
        
    
    def hand_evaluation(self, player_hands: list):
        player_hands = self.number_to_string(player_hands)
        player_hands = [[ev.Card(card) for card in entry] for entry in player_hands]
        player_evals = [ev.evaluate(hand) for hand in player_hands]

        if len(player_hands[0]) == 1:
            if player_evals[0] == player_evals[1]:
                return (player_hands.index(max(player_hands)))
            else:
                return player_evals.index(max(player_evals))
        if player_evals[0] == player_evals[1]:
            return 2
        return player_evals.index(max(player_evals))


# Lowball variant of Stud
class Razz(Game):

    def __init__(self):
        super().__init__()
        self.stack = [1000, 1000]
        self.betting_structure = [1,2,4,8] # Ante - BI - (CO/SB) - BB
        self.player_hands = [[],[]]
    
    def set_player_hands(self, hand1: list, hand2: list):
        self.player_hands[0] = hand1
        self.player_hands[1] = hand2 

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
            # !! introduce logic for chopped pots (players have same hand)
            contribution[self.hand_evaluation(player_hands)] = pot
            return contribution
    
    def one_card_winner(self):
        if len(self.player_hands[0]) != 1 or len(self.player_hands[1]) != 1:
            return -1
        
    # Finds best hand from 7 cards
    def best_hand(self, player_num: int, card_count: int):
        sorted_hand = sorted([x % 13 for x in self.player_hands[player_num]])
        final_hand = []
        used_indices = []
        # Put lowest available cards in hand without dupes
        for i in range(len(sorted_hand)):
            if len(final_hand) >= card_count:
                break
            elif sorted_hand[i] not in final_hand:
                final_hand.append(sorted_hand[i])
                used_indices.append(i)
        # If we were unable to complete the hand, make lowest pair/three-of-a-kind
        i = 2
        while i <= 3:
            for j in range(len(sorted_hand)):
                if len(final_hand) >= card_count:
                    break
                if j in used_indices:
                    continue
                if final_hand.count(sorted_hand[j]) < i:
                     final_hand.append(sorted_hand[j])
            i += 1
        return final_hand
    # Counts number of modulo steps to reach answer
    def count_mod_steps(x, n):
        count = 0
        while x >= n:
            x -= n
            count += 1
            if count > 100:
                return -1
        return count
    # Returns 0 if P1 wins, 1 if P2 wins
    def winner(self, card_count):
        # Check with gene to see if this logic is good
        if card_count == 1:
            if self.player_hands[0][0] % 13 > self.player_hands[1][0] % 13:
                return 0
            elif self.player_hands[0][0] % 13 < self.player_hands[1][0] % 13:
                return 1
            else:
                if self.player_hands[0][0] > self.player_hands[1][0]:
                    return 0
                else: return 1
        hand1 = Counter(self.best_hand(0, card_count))
        hand2 = Counter(self.best_hand(1, card_count))
        hand1 = dict(sorted(hand1.items(), key=lambda x: x[1], reverse=True))
        hand2 = dict(sorted(hand2.items(), key=lambda x: x[1], reverse=True))
        hand1_rank = self.find_hand_rank(hand1)
        hand2_rank = self.find_hand_rank(hand2)
        # Compare hand ranks
        if hand1_rank < hand2_rank:
            return 0
        elif hand1_rank > hand2_rank:
            return 1
        rank = hand1_rank
        # Full House tiebreaker
        if rank == 4:
            if list(hand1.keys())[0] < list(hand2.keys())[0]:
                return 0
            return 1
        # ToaK tiebreaker
        elif rank == 3:
            if list(hand1.keys())[0] < list(hand2.keys())[0]:
                return 0
            return 1
        # Two pair tiebreaker
        elif rank == 2:
            if min(list(hand1.keys())[0], list(hand1.keys())[1]) < min(list(hand2.keys())[0], list(hand2.keys())[1]):
                return 0
            elif min(list(hand1.keys())[0], list(hand1.keys())[1]) > min(list(hand2.keys())[0], list(hand2.keys())[1]):
                return 1
            elif max(list(hand1.keys())[0], list(hand1.keys())[1]) < max(list(hand2.keys())[0], list(hand2.keys())[1]):
                return 0
            return 1
        # Pair tiebreaker
        elif rank == 2:
            if list(hand1.keys())[0] < list(hand2.keys())[0]:
                return 0
            return 1
        # Clean tiebreaker
        else:
            if min(hand1) < min(hand2):
                return 0
            return 1
        # Fail-safe return
        return -1
        
    def find_hand_rank(self, hand: dict):
        # ranks -- Full House = 4  ToaK = 3  Two Pair = 2  Pair = 1  Clean = 0
        if list(hand.values()).count(2) == 0 and list(hand.values()).count(3) == 0: return 0
        elif list(hand.values()).count(2) == 1 and list(hand.values()).count(3) == 0: return 1
        elif list(hand.values()).count(2) == 2 and list(hand.values()).count(3) == 0: return 2
        elif list(hand.values()).count(2) == 0 and list(hand.values()).count(3) == 1: return 3
        else: return 4

    
# Stud 8-or-Better (Split-Game)
class StudHiLo(Game):

    def __init__(self):
        super().__init__()

    # Stud8 has more complex logic with how it determines who wins what percentage of the pop (e.g. scoops, splits, chops on the low/high)
    def utility_function(self, history: list, player_hands: list):
        pass

    def hand_evaluation(self, player_hands: list):
        pass