# Lowball variant of Stud
#         <>   removed "StudGame" to test, "StudGame" should be below the <> in the parentheses
class Razz():

    def __init__(self):
        super().__init__()
    
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
            # !! introduce logic for chopped pots (players have same hand)
            contribution[self.hand_evaluation(player_hands)] = pot
            return contribution
    
    def hand_evaluation(self, player_hands: list):
        if len(player_hands[0]) == 1:
            self.one_card_evaluation(player_hands)

            # !! update to check for suit
        if player_hands[0] == player_hands[1]:
            # !!TIE HAND, check for suit
            return 2
        elif player_hands[0] > player_hands[1]:
            return 0
        else:
            return 1
    
    def best_hand(self, player_hands: list, player, play_count):
        sorted_hand = sorted([x % 13 for x in player_hands[player]])
        final_hand = []
        used_indices = []
        # Put lowest available cards in hand without dupes
        for i in range(len(sorted_hand)):
            if len(final_hand) >= play_count:
                break
            elif sorted_hand[i] not in final_hand:
                final_hand.append(sorted_hand[i])
                used_indices.append(i)
        # If we were unable to complete the hand, make lowest pair/three-of-a-kind
        i = 2
        while i <= 3:
            for j in range(len(sorted_hand)):
                if len(final_hand) >= play_count:
                    break
                if j in used_indices:
                    continue
                if final_hand.count(sorted_hand[j]) < i:
                     final_hand.append(sorted_hand[j])
            i += 1
        return final_hand

    def count_mod_steps(x, n):
        count = 0
        while x >= n:
            x -= n
            count += 1
            if count > 100:
                return -1
        return count
    
    # CREATE METHOD TO COMPARE TWO HANDS TO FIND WINNER
    # CREATE METHOD TO COMPARE SINGLE CARDS (SUIT MATTERS)
                


    
    