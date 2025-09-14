import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pyspiel
import torch
import range_builder
import warnings
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seven_card_stud
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
warnings.filterwarnings('ignore')

# This script generates a range chart based on the policy and hand history written in range_builder.py
# The top right triangle contains suited hole cards, and the bottom left triangle is unsuited

# To evaluate a given state, make sure to select a policy to load, and rewrite the base_tensor() method
# in range_builder.py to match the hand you want to evaluate, then put the upcards in the player_cards
# and opponent_cards variables below.

if __name__ == "__main__":
    seven_card_stud.register()
    game = pyspiel.load_game("python_scs_poker")
    state2 = game.new_initial_state()
    state2.apply_action(0)
    state2.apply_action(0)
    state2.apply_action(0)
    state2.apply_action(0)

    action_sequence = [0, 0, 0, 0]
    num_cards = 0
    up_cards = []
    while True:
        try:
            valid = state2.legal_actions()
            if num_cards == 0:
                valid.append(0)
            elif num_cards == 0.5 and action_sequence[-1] != 0:
                valid.append(0)
            if len(valid) > 6:
                x = int(input("Input a card to be dealt: "))
                up_cards.append(x)
                if x in valid:
                    num_cards += 0.5
                    action_sequence.append(x)
                    state2.apply_action(x)
                else:
                    raise ValueError

            elif len(valid) == 0:
                break
            else:
                x = input("Select a legal action: ")
                if x == "X":
                    break
                else:
                    x = int(x)
                if x in valid:
                    action_sequence.append(x)
                    state2.apply_action(x)
                else:
                    raise ValueError
        except:
            print("Invalid input, try again")
            pass

    print(action_sequence)

    player_cards = up_cards[1::2]
    opponent_cards = up_cards[0::2]
    print(player_cards, opponent_cards)

    player_imgs = [plt.imread(f"PNG/{range_builder.deck_dict[i]}.png") for i in player_cards]
    oppponent_imgs = [plt.imread(f"PNG/{range_builder.deck_dict[i]}.png") for i in opponent_cards]
    my_list, cur_player = range_builder.frame_construction(player_cards, opponent_cards, actions=action_sequence)

    card_labels = ["A","K","Q","J","T", "9", "8", "7", "6", "5", "4", "3", "2"]

    sns.set_theme(style="whitegrid")
    c_bringin  = sns.color_palette("pastel")[0]
    c_complete = sns.color_palette("pastel")[1]
    c_call = sns.color_palette("pastel")[2]
    c_bet = sns.color_palette("pastel")[4]
    c_fold = sns.color_palette("pastel")[3]
    c_check = sns.color_palette("pastel")[5]

    fig, axes = plt.subplots(
        nrows=1, ncols=13, figsize=(39, 10), sharex=True, sharey=True, constrained_layout=True
    )

    for i, ax in enumerate(axes):
        sns.barplot(x="check", y="dynamic_card", data=my_list[i],
                    color=c_check, ax=ax,
                    label=("CHECK" if i == 0 else "_nolegend_"))
        sns.barplot(x="fold", y="dynamic_card", data=my_list[i],
                    color=c_fold, ax=ax,
                    label=("FOLD" if i == 0 else "_nolegend_"))
        sns.barplot(x="bet", y="dynamic_card", data=my_list[i],
                    color=c_bet, ax=ax,
                    label=("BET/RAISE" if i == 0 else "_nolegend_"))
        sns.barplot(x="call", y="dynamic_card", data=my_list[i],
                    color=c_call, ax=ax,
                    label=("CALL" if i == 0 else "_nolegend_"))
        sns.barplot(x="complete", y="dynamic_card", data=my_list[i],
                    color=c_complete, ax=ax,
                    label=("COMPLETE" if i == 0 else "_nolegend_"))
        sns.barplot(x="bring-in", y="dynamic_card", data=my_list[i],
                    color=c_bringin, ax=ax,
                    label=("BRING-IN" if i == 0 else "_nolegend_"))
        ax.set(xlim=(0, 1), ylabel="")
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        if i != 0:
            ax.set_yticklabels([])
        sns.despine(ax=ax, left=(i != 0), bottom=True)
        ax.set_title(f"{card_labels[i]}", fontsize=9)

        for j in range(13):
            ax.text(0.5, 0.96 - (0.077 * j), f"{card_labels[j]}{card_labels[i]}{'s' if i > j else ('o' if i < j else '')}", transform=ax.transAxes, ha="center", va="center")

        if i == 0:
            ax.set_ylabel("      |      ".join(map(str, reversed(card_labels))), fontsize=9.68)
            if cur_player == 0:
                ab = AnnotationBbox(
                    OffsetImage(plt.imread("PNG/UP.png"), zoom=0.32),
                    (-1.25, 0.2),
                    xycoords=ax.transAxes,
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                )
            elif cur_player == 1:
                ab = AnnotationBbox(
                    OffsetImage(plt.imread("PNG/DOWN.png"), zoom=0.32),
                    (-1.25, 0.8),
                    xycoords=ax.transAxes,
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                )
            ax.add_artist(ab)

            for j in range(int(num_cards)):
                ab = AnnotationBbox(
                    OffsetImage(player_imgs[j], zoom=0.10),
                    (-1.25, 0.1 + (0.05 * j)),
                    xycoords=ax.transAxes,
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                )
                ax.add_artist(ab)
                ab = AnnotationBbox(
                    OffsetImage(oppponent_imgs[j], zoom=0.10),
                    (-1.25, 0.9 - (0.05 * j)),
                    xycoords=ax.transAxes,
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                )
                ax.add_artist(ab)

    axes[0].legend(ncol=1, loc="center left", frameon=True, borderaxespad=0.0, bbox_to_anchor=(-2, 0.5))

    plt.show()