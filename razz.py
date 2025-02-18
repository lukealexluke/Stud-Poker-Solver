from collections import Counter
# Lowball variant of Stud
#         <>   removed "StudGame" to test, "StudGame" should be below the <> in the parentheses

class Razz():

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
        
    
    def one_card_winner():
        pass
    
    
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
                


    
    