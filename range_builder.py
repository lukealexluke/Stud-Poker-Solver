import random
import pyspiel
import numpy as np
import pandas as pd
import seven_card_stud
import escher
import torch
from scipy import stats
from itertools import combinations, permutations



ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suits = ['c', 'd', 'h', 's']
deck_dict = {}
index = 0
for suit in suits:
    for rank in ranks:
        deck_dict[index] = f"{rank}{suit}"
        index += 1

seven_card_stud.register()
game = pyspiel.load_game("python_scs_poker")
embedding_size = len(game.new_initial_state().information_state_tensor(0))

num_actions = game.num_distinct_actions()
bot_1 = escher.PolicyNetwork(
    input_size=embedding_size,
    policy_network_layers=(512,512,512),
    num_actions=num_actions,
)
bot_1.load_state_dict(torch.load(f'final_results/policy_iter9_10', weights_only=True)) #10
torch.set_printoptions(precision=4, sci_mode=False)

def cumulative_vector(vec: torch.tensor):
    cumul_vec = vec.clone()
    for i in range(len(cumul_vec)):
        cumul_vec[i] = sum(vec[0:i+1])
    return cumul_vec

# generate combinations of cards
def card_combos(player_upcards, opp_upcards):

    print(player_upcards, opp_upcards)

    deck = [i for i in range(52)]
    for card in player_upcards + opp_upcards:
        deck.remove(card)

    down_combos = combinations(deck, 2)
    return down_combos

# create base_tensor
def base_tensor(player_upcards, opp_upcards, action_sequence):
    # write the history of the hand up to current state below
    state = game.new_initial_state()
    for a in action_sequence:
        state.apply_action(a)

    if state.is_terminal() == True:
        raise ValueError("cannot build a range for a terminal game state")

    base_tensor = state.information_state_tensor()
    # erase hole cards
    base_tensor[2] = 0.0
    base_tensor[15] = 0.0
    base_tensor[19] = 0.0
    base_tensor[32] = 0.0

    return base_tensor, state.legal_actions_mask(), state.current_player()

# create an info_tensor with given information
def info_tensor_construction(player_upcards, opp_upcards, combo, base):

    combo_tensor = base.copy()
    c = sorted(combo)
    combo_tensor[2 + (c[0] % 13)] = 1
    combo_tensor[2 + 13 + (c[0] // 13)] = 1
    combo_tensor[2 + 17 + (c[1] % 13)] = 1
    combo_tensor[2 + 17 + 13 + (c[1] // 13)] = 1

    return combo_tensor

# construct dataframes for range chart
def frame_construction(player_upcards, opp_upcards, actions=[]):
    bot_1.eval()
    with torch.no_grad():
        header_dict = {
            0: "bring-in",
            1: "complete",
            2: "call",
            3: "bet",
            4: "fold",
            5: "check"
        }

        base, mask, cur_player = base_tensor(player_upcards, opp_upcards, action_sequence=actions)
        card_combinations = card_combos(player_upcards, opp_upcards)

        frame_list = []
        for i in range(13):
            frame_list.append(
            pd.DataFrame({
                "dynamic_card": ["A","K","Q","J","T", "9", "8", "7", "6", "5", "4", "3", "2"],
                "bring-in": [0]*13,
                "complete": [0]*13,
                "call": [0]*13,
                "bet": [0]*13,
                "fold": [0]*13,
                "check": [0]*13
            })
            )

        for combo in card_combinations:
            info_tensor = info_tensor_construction(player_upcards, opp_upcards, combo, base)
            probs = bot_1((torch.tensor(info_tensor, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)))
            """
            if combo[0] % 13 == 10 and combo[1] % 13 == 11:
                print(combo)
                print(probs)
            """
            probs = cumulative_vector(probs)

            if combo[0] // 13 == combo[1] // 13: # same suit
                i, j = (12 - (max(combo) % 13)), (12 - (min(combo) % 13))
            else:
                i, j = (12 - (min(combo) % 13)), (12 - (max(combo) % 13))
            for k in range(6):
                frame_list[j][header_dict[k]][i] += float(probs[k])
        
        for frame in frame_list:
            scale = frame['check']
            num_cols = frame.select_dtypes(include='number').columns
            mask = scale.ne(0)
            frame.loc[mask, num_cols] = frame.loc[mask, num_cols].div(scale[mask], axis=0)
            frame.loc[mask, 'check'] = 1.0

    return frame_list, cur_player
                