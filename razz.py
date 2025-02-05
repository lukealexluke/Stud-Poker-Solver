# Lowball variant of Stud
class Razz(StudGame):

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
    
    # May become obsolete once best_hand is implemented
    def one_card_evaluation(self, player_hands: list):
        card_1 = player_hands[0][0] % 13
        card_2 = player_hands[1][0] % 13
        if card_1 > card_2: 
            return 0
        elif card_1 == card_2:
            if player_hands[0][0] > player_hands[1][0]: 
                return 0
            else: return 1
        else: return 1
    
    def best_hand(self, player_hands: list, player):
        sorted_hand = sorted([x if x <= 13 else x % 13 for x in player_hands[player]])
        final_hand = []
        for card in sorted_hand:
            if card not in final_hand and len(final_hand) < 5:
                final_hand.append(card)
                


    
    