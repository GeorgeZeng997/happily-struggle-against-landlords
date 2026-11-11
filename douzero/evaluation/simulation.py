from douzero.env.game import GameEnv, EnvCard2RealCard, RealCard2EnvCard, AllEnvCard
from .deep_agent import DeepAgent


def evaluate(landlord, landlord_up, landlord_down):
    user_hand_cards_real = input("Enter your hand cards (e.g. 333456789TJQKA2XD): ").strip().upper()
    user_hand_cards_env = [RealCard2EnvCard[c] for c in user_hand_cards_real if c in RealCard2EnvCard]

    user_position_code = int(
        input("Select your seat [0: landlord_up, 1: landlord, 2: landlord_down]: ").strip()
    )
    user_position = ['landlord_up', 'landlord', 'landlord_down'][user_position_code]

    three_landlord_cards_real = input("Enter the 3 bottom landlord cards (e.g. 2XD): ").strip().upper()
    three_landlord_cards_env = [
        RealCard2EnvCard[c] for c in three_landlord_cards_real if c in RealCard2EnvCard
    ]

    other_hand_cards = []
    for card_val in sorted(set(AllEnvCard)):
        other_hand_cards.extend(
            [card_val] * max(0, AllEnvCard.count(card_val) - user_hand_cards_env.count(card_val))
        )

    card_play_data = {
        'three_landlord_cards': three_landlord_cards_env,
        ['landlord_up', 'landlord', 'landlord_down'][(user_position_code + 0) % 3]: user_hand_cards_env,
        ['landlord_up', 'landlord', 'landlord_down'][(user_position_code + 1) % 3]: (
            other_hand_cards[0:17] if (user_position_code + 1) % 3 != 1 else other_hand_cards[17:]
        ),
        ['landlord_up', 'landlord', 'landlord_down'][(user_position_code + 2) % 3]: (
            other_hand_cards[0:17] if (user_position_code + 1) % 3 == 1 else other_hand_cards[17:]
        ),
    }

    if len(card_play_data["three_landlord_cards"]) != 3:
        print("Error: Bottom landlord cards must contain exactly 3 cards.\n")
        return
    if (
        len(card_play_data["landlord_up"]) != 17
        or len(card_play_data["landlord_down"]) != 17
        or len(card_play_data["landlord"]) != 20
    ):
        print("Error: Invalid initial hand counts (expected 20 for landlord, 17 for farmers).\n")
        return

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down,
    }

    print(f"Initializing DouZero agent for {user_position}...")
    players = [user_position, DeepAgent(user_position, card_play_model_path_dict[user_position])]

    env = GameEnv(players)
    env.card_play_init(card_play_data)
    print("Round started.\n")
    while not env.game_over:
        acting = env.acting_player_position
        if acting == user_position:
            msg = env.step(user_position)
            print(f"[AI {user_position}] Play: {msg['action'] or 'PASS'} | Win Rate: {msg['win_rate']}")
        else:
            raw = input(f"Enter cards played by {acting} (empty for PASS): ").strip().upper()
            opp_action = [RealCard2EnvCard[c] for c in raw if c in RealCard2EnvCard]
            env.step(user_position, opp_action)
    winner_label = "Farmers" if env.winner == "farmer" else "Landlord"
    print(f"Round finished: {winner_label} won!\n")
    env.reset()
