# -*- coding: utf-8 -*-

import ctypes
import json
import os
import platform
import sys
import threading
import time

import cv2
import numpy as np
import pyscreeze
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEventLoop, QTime
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from ultralytics import YOLO

from card_tracker import (
    AllEnvCard,
    DEFAULT_REGIONS_1080P,
    EnvCard2RealCard,
    RealCard2EnvCard,
    compute_remaining_cards,
    format_remaining_summary,
    map_yolo_class_name,
    parse_region_csv,
)
from douzero.env.game import GameEnv
from douzero.evaluation.deep_agent import DeepAgent
from MainWindowUI import Ui_Form

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _asset_path(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def _enable_windows_dpi_awareness() -> None:
    if platform.system() != "Windows":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class MyPyQT_Form(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MyPyQT_Form, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(
            QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setFixedSize(self.width(), self.height())
        icon_path = _asset_path("pics", "favicon.ico")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        bg_path = _asset_path("pics", "bg.png")
        if os.path.isfile(bg_path):
            window_pale = QtGui.QPalette()
            window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap(bg_path)))
            self.setPalette(window_pale)

        self.Players = [self.RPlayer, self.Player, self.LPlayer]
        self.counter = QTime()

        # Detection & timing parameters
        self.MyConfidence = 0.95
        self.OtherConfidence = 0.9
        self.WhiteConfidence = 0.9
        self.LandlordFlagConfidence = 0.9
        self.ThreeLandlordCardsConfidence = 0.9
        self.WaitTime = 1
        self.MyFilter = 40
        self.OtherFilter = 25
        self.SleepTime = 0.1
        self.in_game = False
        self._stop_requested = False

        # Screen capture coordinates (x, y, width, height) for 1920x1080
        self.MyHandCardsPos = tuple(DEFAULT_REGIONS_1080P["MyHandCardsPos"])
        self.LPlayedCardsPos = tuple(DEFAULT_REGIONS_1080P["LPlayedCardsPos"])
        self.RPlayedCardsPos = tuple(DEFAULT_REGIONS_1080P["RPlayedCardsPos"])
        self.LandlordFlagPos = [tuple(r) for r in DEFAULT_REGIONS_1080P["LandlordFlagPos"]]
        self.ThreeLandlordCardsPos = tuple(DEFAULT_REGIONS_1080P["ThreeLandlordCardsPos"])
        self.PlayedCardsSize = (600, 600)
        self.AutoExpandPlayedCards = True
        self.RegionsConfigPath = _asset_path("config", "regions.json")
        self._load_regions()
        if self.AutoExpandPlayedCards and not self._regions_loaded:
            self.LPlayedCardsPos = self._resize_region_center(self.LPlayedCardsPos, self.PlayedCardsSize)
            self.RPlayedCardsPos = self._resize_region_center(self.RPlayedCardsPos, self.PlayedCardsSize)

        self.shouldExit = 0
        self.canRecord = threading.Lock()
        self.env = None

        # Pretrained DouZero model checkpoints
        self.card_play_model_path_dict = {
            "landlord": _asset_path("baselines", "douzero_WP", "landlord.ckpt"),
            "landlord_up": _asset_path("baselines", "douzero_WP", "landlord_up.ckpt"),
            "landlord_down": _asset_path("baselines", "douzero_WP", "landlord_down.ckpt"),
        }

        # YOLOv8 model for real-time card detection
        self.yolo_weights_path = _asset_path("weights", "best.pt")
        self.yolo_imgsz = 640
        self.MyCardConf = 0.6
        self.OtherCardConf = 0.6
        self.ThreeLandlordCardConf = 0.6
        self._last_infer_ms = None
        if not os.path.exists(self.yolo_weights_path):
            QMessageBox.critical(
                self,
                "Model Not Found",
                "Missing YOLOv8 weights: " + self.yolo_weights_path,
                QMessageBox.Yes,
                QMessageBox.Yes,
            )
            raise FileNotFoundError(self.yolo_weights_path)
        self.yolo_model = YOLO(self.yolo_weights_path)
        self._cls_to_card = self._build_class_map()

        # Optional live region preview window
        self.ShowRegions = True
        self.RegionPreviewInterval = 0.5
        self.PreviewScale = 0.5
        self._preview_stop = False
        if self.ShowRegions:
            t = threading.Thread(target=self._region_preview_loop, daemon=True)
            t.start()

        self.RecognitionInterval = 0.5
        self._recog_timer = QtCore.QTimer(self)
        self._recog_timer.timeout.connect(self.update_recognition)
        self._recog_timer.start(int(self.RecognitionInterval * 1000))
        self._recog_enabled = True
        self.DebugMode = False
        self._debug_window_open = False
        self.DebugMouseSize = (600, 600)
        self.UseMouseRegionForMain = False
        self.last_left_cards = ""
        self.last_right_cards = ""
        self._calibrating = False
        self._calib_target = None
        self._pending_region = None
        self._calib_preview_open = False
        self.CalibrationScale = 0.5
        self.update_recognition()

    def init_display(self):
        self.WinRate.setText("Win Rate: --%")
        self.InitCard.setText("Start")
        self.UserHandCards.setText("Hand Cards")
        self.LPlayedCard.setText("Left Played")
        self.RPlayedCard.setText("Right Played")
        self.PredictedCard.setText("AI Suggestion")
        self.ThreeLandlordCards.setText("Bottom Cards")
        for player in self.Players:
            player.setStyleSheet("background-color: rgba(255, 0, 0, 0);")

    def init_cards(self):
        self._stop_requested = False
        self.user_hand_cards_real = ""
        self.user_hand_cards_env = []
        self.other_played_cards_real = ""
        self.other_played_cards_env = []
        self.other_hand_cards = []
        self.three_landlord_cards_real = ""
        self.three_landlord_cards_env = []
        # Seat index: 0 = landlord_up, 1 = landlord, 2 = landlord_down
        self.user_position_code = None
        self.user_position = ""
        self.card_play_data_list = {}
        self.play_order = 0
        self.env = None

        self.user_hand_cards_real = self.find_my_cards(self.MyHandCardsPos)
        self.UserHandCards.setText(self.user_hand_cards_real)
        self.user_hand_cards_env = [RealCard2EnvCard[c] for c in list(self.user_hand_cards_real)]

        self.three_landlord_cards_real = self.find_three_landlord_cards(self.ThreeLandlordCardsPos)
        self.ThreeLandlordCards.setText("Bottom: " + self.three_landlord_cards_real)
        self.three_landlord_cards_env = [RealCard2EnvCard[c] for c in list(self.three_landlord_cards_real)]

        self.user_position_code = self.find_landlord(self.LandlordFlagPos)
        if self.user_position_code is None:
            items = ("Landlord Up (Farmer)", "Landlord", "Landlord Down (Farmer)")
            item, okPressed = QInputDialog.getItem(
                self,
                "Select Seat Role",
                "Landlord badge not detected. Please choose your seat:",
                items,
                0,
                False,
            )
            if okPressed and item:
                self.user_position_code = items.index(item)
            else:
                return
        self.user_position = ["landlord_up", "landlord", "landlord_down"][self.user_position_code]
        for player in self.Players:
            player.setStyleSheet("background-color: rgba(255, 0, 0, 0);")
        self.Players[self.user_position_code].setStyleSheet("background-color: rgba(255, 0, 0, 0.1);")

        for card_val in sorted(set(AllEnvCard)):
            self.other_hand_cards.extend(
                [card_val] * max(0, AllEnvCard.count(card_val) - self.user_hand_cards_env.count(card_val))
            )
        self.card_play_data_list.update(
            {
                "three_landlord_cards": self.three_landlord_cards_env,
                ["landlord_up", "landlord", "landlord_down"][(self.user_position_code + 0) % 3]: self.user_hand_cards_env,
                ["landlord_up", "landlord", "landlord_down"][(self.user_position_code + 1) % 3]: (
                    self.other_hand_cards[0:17]
                    if (self.user_position_code + 1) % 3 != 1
                    else self.other_hand_cards[17:]
                ),
                ["landlord_up", "landlord", "landlord_down"][(self.user_position_code + 2) % 3]: (
                    self.other_hand_cards[0:17]
                    if (self.user_position_code + 1) % 3 == 1
                    else self.other_hand_cards[17:]
                ),
            }
        )
        print(self.card_play_data_list)

        if len(self.card_play_data_list["three_landlord_cards"]) != 3:
            QMessageBox.critical(
                self,
                "Bottom Cards Error",
                "Exactly 3 bottom landlord cards must be recognized!",
                QMessageBox.Yes,
                QMessageBox.Yes,
            )
            self.init_display()
            return
        if (
            len(self.card_play_data_list["landlord_up"]) != 17
            or len(self.card_play_data_list["landlord_down"]) != 17
            or len(self.card_play_data_list["landlord"]) != 20
        ):
            QMessageBox.critical(
                self,
                "Hand Card Count Error",
                "Invalid initial hand counts (expected 20 for Landlord, 17 for Farmers).",
                QMessageBox.Yes,
                QMessageBox.Yes,
            )
            self.init_display()
            return

        self.play_order = 0 if self.user_position == "landlord" else 1 if self.user_position == "landlord_up" else 2

        ai_players = [
            self.user_position,
            DeepAgent(self.user_position, self.card_play_model_path_dict[self.user_position]),
        ]
        self.env = GameEnv(ai_players)
        self.in_game = True
        self.start()

    def _ui_wait_ms(self, duration_ms: int) -> bool:
        """Pump Qt events for `duration_ms` and return True if stop was requested."""
        self.counter.restart()
        while self.counter.elapsed() < duration_ms:
            if self._stop_requested or (self.env is not None and self.env.game_over):
                return True
            QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
        return self._stop_requested

    def _update_remaining_tooltip(self) -> None:
        if self.env is None or not self.user_position:
            return
        my_hand = self.env.info_sets[self.user_position].player_hand_cards or []
        rem = compute_remaining_cards(my_hand, self.env.played_cards)
        high_summary = format_remaining_summary(rem, high_only=True)
        full_summary = format_remaining_summary(rem, high_only=False)
        self.WinRate.setToolTip(f"Unseen Opponent Cards: {full_summary}")
        print(f"Unseen High Cards: {high_summary}")

    def start(self):
        self.env.card_play_init(self.card_play_data_list)
        print("Round started.\n")
        pass_img = _asset_path("pics", "pass.png")

        while not self.env.game_over and not self._stop_requested:
            if self.play_order == 0:
                self.PredictedCard.setText("...")
                action_message = self.env.step(self.user_position)
                hand_str = "".join(
                    [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards]
                )[::-1]
                self.UserHandCards.setText("Hand: " + hand_str)
                rec_text = action_message["action"] if action_message["action"] else "PASS"
                self.PredictedCard.setText(rec_text)
                self.WinRate.setText("Win Rate: " + action_message["win_rate"])
                print("\nHand:", hand_str)
                print("AI Suggestion:", rec_text, "| Win Rate:", action_message["win_rate"])
                self._update_remaining_tooltip()

                while not self._stop_requested and (
                    self.have_white(self.RPlayedCardsPos) == 1
                    or self._safe_locate(pass_img, self.RPlayedCardsPos, self.LandlordFlagConfidence)
                ):
                    if self._ui_wait_ms(100):
                        break
                self.play_order = 1

            elif self.play_order == 1:
                self.RPlayedCard.setText("...")
                while (
                    not self._stop_requested
                    and self.have_white(self.RPlayedCardsPos) == 0
                    and not self._safe_locate(pass_img, self.RPlayedCardsPos, self.LandlordFlagConfidence)
                ):
                    if self._ui_wait_ms(500):
                        break
                if self._stop_requested:
                    break
                self._ui_wait_ms(500)
                pass_flag = self._safe_locate(pass_img, self.RPlayedCardsPos, self.LandlordFlagConfidence)
                if pass_flag is None:
                    self.other_played_cards_real = self.last_right_cards or self.find_other_cards(self.RPlayedCardsPos)
                else:
                    self.other_played_cards_real = ""
                print("\nRight Player:", self.other_played_cards_real or "PASS")
                self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(self.other_played_cards_real)]
                self.env.step(self.user_position, self.other_played_cards_env)
                self.RPlayedCard.setText(self.other_played_cards_real if self.other_played_cards_real else "PASS")
                self._update_remaining_tooltip()
                self.play_order = 2

            elif self.play_order == 2:
                self.LPlayedCard.setText("...")
                while (
                    not self._stop_requested
                    and self.have_white(self.LPlayedCardsPos) == 0
                    and not self._safe_locate(pass_img, self.LPlayedCardsPos, self.LandlordFlagConfidence)
                ):
                    if self._ui_wait_ms(500):
                        break
                if self._stop_requested:
                    break
                self._ui_wait_ms(500)
                pass_flag = self._safe_locate(pass_img, self.LPlayedCardsPos, self.LandlordFlagConfidence)
                if pass_flag is None:
                    self.other_played_cards_real = self.last_left_cards or self.find_other_cards(self.LPlayedCardsPos)
                else:
                    self.other_played_cards_real = ""
                print("\nLeft Player:", self.other_played_cards_real or "PASS")
                self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(self.other_played_cards_real)]
                self.env.step(self.user_position, self.other_played_cards_env)
                self.LPlayedCard.setText(self.other_played_cards_real if self.other_played_cards_real else "PASS")
                self._update_remaining_tooltip()
                self.play_order = 0

            self._ui_wait_ms(100)

        if not self._stop_requested and self.env and self.env.winner:
            winner_str = "Farmers" if self.env.winner == "farmer" else "Landlord"
            print(f"{winner_str} won! Round finished.\n")
            QMessageBox.information(
                self, "Round Finished", f"{winner_str} won!", QMessageBox.Yes, QMessageBox.Yes
            )
        if self.env is not None:
            self.env.reset()
        self.in_game = False
        self.init_display()

    def find_landlord(self, landlord_flag_pos):
        badge_img = _asset_path("pics", "landlord_words.png")
        for idx, pos in enumerate(landlord_flag_pos):
            result = self._safe_locate(badge_img, pos, self.LandlordFlagConfidence)
            if result is not None:
                return idx
        return None

    def find_three_landlord_cards(self, pos):
        return self.detect_cards_yolo(pos, self.ThreeLandlordCardConf)

    def find_my_cards(self, pos):
        return self.detect_cards_yolo(pos, self.MyCardConf)

    def find_other_cards(self, pos):
        self._ui_wait_ms(500)
        return self.detect_cards_yolo(pos, self.OtherCardConf)

    def detect_cards_yolo(self, pos, conf):
        img = pyscreeze.screenshot(region=pos)
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        result = self.yolo_model.predict(img_np, imgsz=self.yolo_imgsz, conf=conf, verbose=False)[0]
        if hasattr(result, "speed") and isinstance(result.speed, dict):
            self._last_infer_ms = result.speed.get("inference")
        return self._yolo_result_to_cards(result)

    def detect_cards_yolo_debug(self, pos, conf, label):
        img = pyscreeze.screenshot(region=pos)
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        result = self.yolo_model.predict(img_np, imgsz=self.yolo_imgsz, conf=conf, verbose=False)[0]
        if hasattr(result, "speed") and isinstance(result.speed, dict):
            self._last_infer_ms = result.speed.get("inference")
        cards = self._yolo_result_to_cards(result)
        annotated = result.plot()
        cv2.putText(annotated, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return cards, annotated

    def _yolo_result_to_cards(self, result):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ""
        xyxy = boxes.xyxy
        cls = boxes.cls
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        if hasattr(cls, "cpu"):
            cls = cls.cpu().numpy()
        cards = []
        for x1, cls_id in zip(xyxy[:, 0], cls):
            cls_int = int(cls_id)
            card = self._cls_to_card.get(cls_int)
            if card:
                cards.append((float(x1), card))
        cards.sort(key=lambda x: x[0])
        return "".join(card for _, card in cards)

    def cards_filter(self, location, distance):
        if len(location) == 0:
            return 0
        locList = [location[0][0]]
        count = 1
        for e in location:
            is_new = True
            for have in locList:
                if abs(e[0] - have) <= distance:
                    is_new = False
                    break
            if is_new:
                count += 1
                locList.append(e[0])
        return count

    def have_white(self, pos):
        white_img = _asset_path("pics", "white.png")
        result = self._safe_locate(white_img, pos, self.WhiteConfidence)
        return 0 if result is None else 1

    def _safe_locate(self, image_path, region, confidence):
        try:
            return pyscreeze.locateOnScreen(image_path, region=region, confidence=confidence)
        except Exception:
            return None

    def _build_class_map(self):
        names = self.yolo_model.names
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        mapping = {}
        for cls_id, name in names.items():
            card = map_yolo_class_name(str(name))
            if card:
                mapping[int(cls_id)] = card
        return mapping

    def _map_name_to_card(self, name):
        return map_yolo_class_name(name)

    def update_recognition(self):
        speeds = []

        if self.UseMouseRegionForMain:
            region = self._get_mouse_region(self.DebugMouseSize)
            mouse_cards = self.detect_cards_yolo(
                (region["left"], region["top"], region["width"], region["height"]),
                self.MyCardConf,
            )
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            self.UserHandCards.setText("Mouse: " + mouse_cards if mouse_cards else "Mouse Region")
            self.LPlayedCard.setText("Left Played")
            self.RPlayedCard.setText("Right Played")
            self.ThreeLandlordCards.setText("Bottom Cards")
            self.last_left_cards = ""
            self.last_right_cards = ""
        else:
            my_cards = self.detect_cards_yolo(self.MyHandCardsPos, self.MyCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            self.UserHandCards.setText(my_cards if my_cards else "Hand Cards")

            left_cards = self.detect_cards_yolo(self.LPlayedCardsPos, self.OtherCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            self.LPlayedCard.setText(left_cards if left_cards else "Left Played")
            self.last_left_cards = left_cards

            right_cards = self.detect_cards_yolo(self.RPlayedCardsPos, self.OtherCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            self.RPlayedCard.setText(right_cards if right_cards else "Right Played")
            self.last_right_cards = right_cards

            bottom_cards = self.detect_cards_yolo(self.ThreeLandlordCardsPos, self.ThreeLandlordCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            self.ThreeLandlordCards.setText("Bottom: " + bottom_cards if bottom_cards else "Bottom Cards")

        if speeds:
            avg_ms = sum(speeds) / len(speeds)
            if self.in_game:
                base = self.WinRate.text()
                if "YOLO:" in base:
                    base = base.split("| YOLO:")[0].strip()
                self.WinRate.setText(f"{base} | YOLO: {avg_ms:.1f} ms")
            else:
                self.WinRate.setText(f"YOLO: {avg_ms:.1f} ms")

        if self.DebugMode:
            self._show_mouse_debug()

    def toggle_recognition(self):
        if self._recog_enabled:
            self._recog_timer.stop()
            self._recog_enabled = False
            self.ToggleRecognize.setText("Resume Recog")
        else:
            self._recog_timer.start(int(self.RecognitionInterval * 1000))
            self._recog_enabled = True
            self.ToggleRecognize.setText("Pause Recog")
            self.update_recognition()

    def toggle_debug_mode(self):
        self.DebugMode = not self.DebugMode
        if not self.DebugMode and self._debug_window_open:
            cv2.destroyWindow("YOLO Debug (Mouse)")
            self._debug_window_open = False
        self.ToggleDebug.setText("Close Debug" if self.DebugMode else "Debug Mode")
        if self.DebugMode:
            self.update_recognition()

    def start_calibration(self):
        if self._calibrating:
            return
        item, method, manual_text = self._choose_calib_target()
        if not item or not method:
            return
        self._calib_target = item
        if method == "manual":
            if not manual_text:
                return
            try:
                self._pending_region = parse_region_csv(manual_text)
            except ValueError:
                QMessageBox.critical(
                    self,
                    "Invalid Coordinates",
                    "Expected format 'x,y,w,h' with positive width and height.",
                    QMessageBox.Yes,
                    QMessageBox.Yes,
                )
                return
            self._apply_pending_calibration()
            return

        self._calibrating = True
        t = threading.Thread(target=self._calibration_loop, daemon=True)
        t.start()

    def _calibration_loop(self):
        points = []
        last_down = False
        while self._calibrating:
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                self._calibrating = False
                self._close_calib_preview()
                return
            down = bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
            if down and not last_down:
                points.append(self._get_cursor_pos())
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    left = min(x1, x2)
                    top = min(y1, y2)
                    width = max(1, abs(x2 - x1))
                    height = max(1, abs(y2 - y1))
                    self._pending_region = (left, top, width, height)
                    self._calibrating = False
                    self._close_calib_preview()
                    QtCore.QTimer.singleShot(0, self._apply_pending_calibration)
                    return
            last_down = down
            if len(points) == 1:
                self._show_calib_preview(points[0])
            time.sleep(0.05)

    def _apply_pending_calibration(self):
        if not self._pending_region or not self._calib_target:
            return
        region = self._pending_region
        target = self._calib_target
        if target == "My Hand Region":
            self.MyHandCardsPos = region
        elif target == "Left Played Region":
            self.LPlayedCardsPos = region
            self.last_left_cards = ""
        elif target == "Right Played Region":
            self.RPlayedCardsPos = region
            self.last_right_cards = ""
        elif target == "Bottom 3 Cards":
            self.ThreeLandlordCardsPos = region
        elif target == "Landlord Flag (Right)":
            self.LandlordFlagPos[0] = region
        elif target == "Landlord Flag (Me)":
            self.LandlordFlagPos[1] = region
        elif target == "Landlord Flag (Left)":
            self.LandlordFlagPos[2] = region
        self._pending_region = None
        self._calib_target = None
        self._save_regions()
        self.update_recognition()

    def _reset_regions_to_default(self):
        self._apply_regions_dict(DEFAULT_REGIONS_1080P)
        if self.AutoExpandPlayedCards:
            self.LPlayedCardsPos = self._resize_region_center(self.LPlayedCardsPos, self.PlayedCardsSize)
            self.RPlayedCardsPos = self._resize_region_center(self.RPlayedCardsPos, self.PlayedCardsSize)
        self._save_regions()
        self.update_recognition()

    def _export_regions_profile(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Calibration Profile", "regions_profile.json", "JSON Files (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._get_regions_dict(), f, indent=2)

    def _import_regions_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Calibration Profile", "", "JSON Files (*.json)"
        )
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_regions_dict(data)
            self._save_regions()
            self.update_recognition()
        except Exception as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))

    def _choose_calib_target(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Calibrate Screen Regions")
        layout = QtWidgets.QVBoxLayout(dialog)
        tip = QtWidgets.QLabel(
            "Click a region button below to select two corners on screen (Esc to cancel).\n"
            "Or enter 'x,y,w,h' manually first, then click the target region button."
        )
        layout.addWidget(tip)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        manual_edit = QtWidgets.QLineEdit()
        manual_edit.setPlaceholderText("Optional manual coordinates: x,y,w,h")

        result = {"target": None, "method": None, "manual": ""}

        def on_click(target):
            text = manual_edit.text().strip()
            if text:
                result["method"] = "manual"
                result["manual"] = text
            else:
                result["method"] = "mouse"
            result["target"] = target
            dialog.accept()

        buttons = [
            ("My Hand Region", 0, 0),
            ("Left Played Region", 0, 1),
            ("Right Played Region", 0, 2),
            ("Bottom 3 Cards", 1, 0),
            ("Landlord Flag (Right)", 1, 1),
            ("Landlord Flag (Me)", 1, 2),
            ("Landlord Flag (Left)", 2, 1),
        ]
        for text, r, c in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _, t=text: on_click(t))
            grid.addWidget(btn, r, c)

        layout.addWidget(manual_edit)

        profile_row = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset 1080p Defaults")
        import_btn = QtWidgets.QPushButton("Import Profile...")
        export_btn = QtWidgets.QPushButton("Export Profile...")
        reset_btn.clicked.connect(lambda: (self._reset_regions_to_default(), dialog.reject()))
        import_btn.clicked.connect(lambda: (self._import_regions_profile(), dialog.reject()))
        export_btn.clicked.connect(self._export_regions_profile)
        profile_row.addWidget(reset_btn)
        profile_row.addWidget(import_btn)
        profile_row.addWidget(export_btn)
        layout.addLayout(profile_row)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        layout.addWidget(cancel_btn)

        dialog.exec_()
        return result["target"], result["method"], result["manual"]

    def _show_mouse_debug(self):
        region = self._get_mouse_region(self.DebugMouseSize)
        img = pyscreeze.screenshot(region=(region["left"], region["top"], region["width"], region["height"]))
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        result = self.yolo_model.predict(img_np, imgsz=self.yolo_imgsz, conf=self.MyCardConf, verbose=False)[0]
        annotated = result.plot()
        cv2.putText(annotated, "MouseRegion", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if not self._debug_window_open:
            cv2.namedWindow("YOLO Debug (Mouse)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Debug (Mouse)", 600, 600)
            self._debug_window_open = True
        cv2.imshow("YOLO Debug (Mouse)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow("YOLO Debug (Mouse)")
            self._debug_window_open = False

    def _get_cursor_pos(self):
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _get_mouse_region(self, size):
        x, y = self._get_cursor_pos()
        w, h = size
        region = {"left": x - w // 2, "top": y - h // 2, "width": w, "height": h}
        return self._clamp_region(region)

    def _resize_region_center(self, region, size):
        x, y, w, h = region
        cx = x + w / 2
        cy = y + h / 2
        new_w, new_h = size
        new_region = {
            "left": int(cx - new_w / 2),
            "top": int(cy - new_h / 2),
            "width": int(new_w),
            "height": int(new_h),
        }
        clamped = self._clamp_region(new_region)
        return (clamped["left"], clamped["top"], clamped["width"], clamped["height"])

    def _load_regions(self):
        self._regions_loaded = False
        if not os.path.exists(self.RegionsConfigPath):
            return
        try:
            with open(self.RegionsConfigPath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_regions_dict(data)
            self._regions_loaded = True
        except Exception:
            self._regions_loaded = False

    def _save_regions(self):
        os.makedirs(os.path.dirname(self.RegionsConfigPath), exist_ok=True)
        data = self._get_regions_dict()
        with open(self.RegionsConfigPath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _get_regions_dict(self):
        return {
            "MyHandCardsPos": list(self.MyHandCardsPos),
            "LPlayedCardsPos": list(self.LPlayedCardsPos),
            "RPlayedCardsPos": list(self.RPlayedCardsPos),
            "ThreeLandlordCardsPos": list(self.ThreeLandlordCardsPos),
            "LandlordFlagPos": [list(r) for r in self.LandlordFlagPos],
        }

    def _apply_regions_dict(self, data):
        if "MyHandCardsPos" in data:
            self.MyHandCardsPos = tuple(data["MyHandCardsPos"])
        if "LPlayedCardsPos" in data:
            self.LPlayedCardsPos = tuple(data["LPlayedCardsPos"])
        if "RPlayedCardsPos" in data:
            self.RPlayedCardsPos = tuple(data["RPlayedCardsPos"])
        if "ThreeLandlordCardsPos" in data:
            self.ThreeLandlordCardsPos = tuple(data["ThreeLandlordCardsPos"])
        if "LandlordFlagPos" in data and len(data["LandlordFlagPos"]) == 3:
            self.LandlordFlagPos = [tuple(r) for r in data["LandlordFlagPos"]]

    def _show_calib_preview(self, p1):
        x2, y2 = self._get_cursor_pos()
        left = min(p1[0], x2)
        top = min(p1[1], y2)
        width = max(1, abs(x2 - p1[0]))
        height = max(1, abs(y2 - p1[1]))
        img = pyscreeze.screenshot()
        frame = np.array(img)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = np.ascontiguousarray(frame)
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, "Calibrating", (left, max(0, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.CalibrationScale != 1.0:
            frame = cv2.resize(
                frame, None, fx=self.CalibrationScale, fy=self.CalibrationScale, interpolation=cv2.INTER_AREA
            )
        if not self._calib_preview_open:
            cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Calibration Preview", 480, 270)
            self._calib_preview_open = True
        cv2.imshow("Calibration Preview", frame)
        cv2.waitKey(1)
        self._move_calib_preview_away_from_cursor(frame.shape[1], frame.shape[0], x2, y2)

    def _close_calib_preview(self):
        if self._calib_preview_open:
            cv2.destroyWindow("Calibration Preview")
            self._calib_preview_open = False

    def _move_calib_preview_away_from_cursor(self, win_w, win_h, cursor_x, cursor_y):
        screen_w, screen_h = pyscreeze.screenshot().size
        margin = 10
        x = (screen_w - win_w - margin) if cursor_x < screen_w / 2 else margin
        y = (screen_h - win_h - margin - 40) if cursor_y < screen_h / 2 else margin
        cv2.moveWindow("Calibration Preview", max(0, int(x)), max(0, int(y)))

    def _clamp_region(self, region):
        screen_w, screen_h = pyscreeze.screenshot().size
        left = max(0, min(region["left"], screen_w - 1))
        top = max(0, min(region["top"], screen_h - 1))
        right = min(screen_w, left + region["width"])
        bottom = min(screen_h, top + region["height"])
        width = max(1, right - left)
        height = max(1, bottom - top)
        return {"left": left, "top": top, "width": width, "height": height}

    def stop(self):
        self._stop_requested = True
        self._calibrating = False
        self.in_game = False
        if self.env is not None:
            self.env.game_over = True

    def closeEvent(self, event):
        self.stop()
        self._preview_stop = True
        if hasattr(self, "_recog_timer"):
            self._recog_timer.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        event.accept()

    def _region_preview_loop(self):
        window_name = "Region Preview (press q to close)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)
        while not self._preview_stop:
            try:
                img = pyscreeze.screenshot()
                frame = np.array(img)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frame = np.ascontiguousarray(frame)

                def draw_rect(pos, label):
                    x, y, w, h = pos
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                draw_rect(self.MyHandCardsPos, "MyHand")
                draw_rect(self.LPlayedCardsPos, "LeftPlayed")
                draw_rect(self.RPlayedCardsPos, "RightPlayed")
                draw_rect(self.ThreeLandlordCardsPos, "Bottom3")
                for i, p in enumerate(self.LandlordFlagPos):
                    draw_rect(p, f"LandlordFlag{i}")

                if self.PreviewScale != 1.0:
                    frame = cv2.resize(
                        frame, None, fx=self.PreviewScale, fy=self.PreviewScale, interpolation=cv2.INTER_AREA
                    )
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._preview_stop = True
                    cv2.destroyWindow(window_name)
                    break
            except Exception:
                pass
            time.sleep(self.RegionPreviewInterval)


if __name__ == "__main__":
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    _enable_windows_dpi_awareness()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(
        """
    QPushButton{
        text-align : center;
        background-color : white;
        font: bold;
        border-color: gray;
        border-width: 2px;
        border-radius: 10px;
        padding: 6px;
        height : 14px;
        border-style: outset;
        font : 14px;
    }
    QPushButton:hover{
        background-color : lightgray;
    }
    QPushButton:pressed{
        text-align : center;
        background-color : gray;
        font: bold;
        border-color: gray;
        border-width: 2px;
        border-radius: 10px;
        padding: 6px;
        height : 14px;
        border-style: outset;
        font : 14px;
        padding-left:9px;
        padding-top:9px;
    }
    QComboBox{
        background:transparent;
        border: 1px solid rgba(200, 200, 200, 100);
        font-weight: bold;
    }
    QComboBox:drop-down{
        border: 0px;
    }
    QComboBox QAbstractItemView:item{
        height: 30px;
    }
    QLabel{
        background:transparent;
        font-weight: bold;
    }
    """
    )
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())
