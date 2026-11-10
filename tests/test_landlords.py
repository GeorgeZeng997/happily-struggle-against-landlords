import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import cv2
import numpy as np

import card_tracker
import pos_debug
from douzero.env.env import get_obs
from douzero.env.game import AllEnvCard, GameEnv


class DummyAgent:
    """Deterministic stub agent choosing the first legal non-pass action."""

    def act(self, infoset):
        for move in infoset.legal_actions:
            if move:
                return move, 0.35
        return infoset.legal_actions[0], -0.2


class TestCardTrackerAndCalibration(unittest.TestCase):
    def test_yolo_class_mapping_and_remaining_deck(self) -> None:
        self.assertIsNone(card_tracker.map_yolo_class_name("-1"))
        self.assertEqual(card_tracker.map_yolo_class_name("1"), "A")
        self.assertEqual(card_tracker.map_yolo_class_name("10"), "T")
        self.assertEqual(card_tracker.map_yolo_class_name("13"), "K")
        self.assertEqual(card_tracker.map_yolo_class_name("14"), "X")
        self.assertEqual(card_tracker.map_yolo_class_name("15"), "D")
        self.assertEqual(card_tracker.map_yolo_class_name("7"), "7")

        my_hand = card_tracker.real_cards_to_env("3344556677889922XD")
        played = {"landlord": card_tracker.real_cards_to_env("22AA")}
        rem = card_tracker.compute_remaining_cards(my_hand, played)
        self.assertEqual(rem["D"], 0)
        self.assertEqual(rem["X"], 0)
        self.assertEqual(rem["2"], 0)
        self.assertEqual(rem["A"], 2)
        self.assertEqual(rem["K"], 4)
        self.assertIn("A:2", card_tracker.format_remaining_summary(rem, high_only=True))

    def test_region_csv_validation_and_pos_debug_cli(self) -> None:
        self.assertEqual(card_tracker.parse_region_csv("100, 200, 300, 80"), (100, 200, 300, 80))
        with self.assertRaises(ValueError):
            card_tracker.parse_region_csv("100,200,0,80")

        valid_report = card_tracker.validate_regions_dict(
            card_tracker.DEFAULT_REGIONS_1080P, screen_size=(1920, 1080)
        )
        self.assertTrue(valid_report["valid"])
        self.assertEqual(valid_report["region_count"], 7)

        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "screen.png")
            out_path = os.path.join(tmp, "annotated.png")
            cfg_path = os.path.join(tmp, "regions.json")
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(img_path, canvas)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(card_tracker.DEFAULT_REGIONS_1080P, f)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = pos_debug.main(["--config", cfg_path, "--validate"])
            self.assertEqual(rc, 0)
            parsed = json.loads(buf.getvalue())
            self.assertTrue(parsed["valid"])

            rc2 = pos_debug.main(["--image", img_path, "--config", cfg_path, "--output", out_path])
            self.assertEqual(rc2, 0)
            self.assertTrue(os.path.isfile(out_path))


class TestDouZeroGameEnvEdgeCases(unittest.TestCase):
    def test_game_env_single_pass_history_and_full_obs(self) -> None:
        deck = sorted(AllEnvCard)
        card_play_data = {
            "landlord": deck[:20],
            "landlord_down": deck[20:37],
            "landlord_up": deck[37:54],
            "three_landlord_cards": deck[:3],
        }
        env = GameEnv(["landlord", DummyAgent()])
        env.card_play_init(card_play_data)

        # Verify observation tensor generation runs cleanly without torch
        obs = get_obs(env.game_infoset)
        self.assertIn("x_batch", obs)
        self.assertIn("z_batch", obs)
        self.assertGreater(obs["x_batch"].shape[0], 0)

        # Regression test: single empty pass in action_sequence must not raise IndexError
        env.card_play_action_seq = [[]]
        self.assertEqual(env.get_last_move(), [])
        legal_after_single_pass = env.get_legal_card_play_actions()
        self.assertGreater(len(legal_after_single_pass), 0)

        # Reset and step through landlord -> landlord_down -> landlord_up
        env.reset()
        env.card_play_init(card_play_data)
        res = env.step("landlord")
        self.assertTrue(res["action"])
        self.assertTrue(res["win_rate"].endswith("%"))

        # Opponent passes
        env.step("landlord", [])
        env.step("landlord", [])
        self.assertEqual(env.acting_player_position, "landlord")


if __name__ == "__main__":
    unittest.main()
