# -*- coding: utf-8 -*-

# Created by: Raf

import os
import sys
import time
import threading
import json
import numpy as np
import pyscreeze
import cv2
import ctypes

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QTime, QEventLoop
from MainWindowUI import Ui_Form

from douzero.env.game import GameEnv
from douzero.evaluation.deep_agent import DeepAgent
from ultralytics import YOLO

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

AllEnvCard = [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
              8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12,
              12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 17, 17, 17, 17, 20, 30]

AllCards = ['rD', 'bX', 'b2', 'r2', 'bA', 'rA', 'bK', 'rK', 'bQ', 'rQ', 'bJ', 'rJ', 'bT', 'rT',
            'b9', 'r9', 'b8', 'r8', 'b7', 'r7', 'b6', 'r6', 'b5', 'r5', 'b4', 'r4', 'b3', 'r3']

MODEL_CLASS_TO_REAL = {
    1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'X', 15: 'D'
}


class MyPyQT_Form(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MyPyQT_Form, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint |    # 使能最小化按钮
                            QtCore.Qt.WindowCloseButtonHint |       # 使能关闭按钮
                            QtCore.Qt.WindowStaysOnTopHint)         # 窗体总在最前端
        self.setFixedSize(self.width(), self.height())              # 固定窗体大小
        self.setWindowIcon(QIcon('pics/favicon.ico'))
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("pics/bg.png")))
        self.setPalette(window_pale)

        self.Players = [self.RPlayer, self.Player, self.LPlayer]
        self.counter = QTime()

        # 参数
        self.MyConfidence = 0.95  # 我的牌的置信度
        self.OtherConfidence = 0.9  # 别人的牌的置信度
        self.WhiteConfidence = 0.9  # 检测白块的置信度
        self.LandlordFlagConfidence = 0.9     # # 检测地主标志的置信度
        self.ThreeLandlordCardsConfidence = 0.9  # 检测地主底牌的置信度
        self.WaitTime = 1  # 等待状态稳定延时
        self.MyFilter = 40  # 我的牌检测结果过滤参数
        self.OtherFilter = 25  # 别人的牌检测结果过滤参数
        self.SleepTime = 0.1  # 循环中睡眠时间
        self.in_game = False

        # 坐标
        self.MyHandCardsPos = (414, 804, 1041, 59)  # 我的截图区域
        self.LPlayedCardsPos = (530, 470, 380, 160)  # 左边截图区域
        self.RPlayedCardsPos = (1010, 470, 380, 160)  # 右边截图区域
        self.LandlordFlagPos = [(1320, 300, 110, 140), (320, 720, 110, 140), (500, 300, 110, 140)]  # 地主标志截图区域(右-我-左)
        self.ThreeLandlordCardsPos = (817, 36, 287, 136)      # 地主底牌截图区域，resize成349x168
        self.PlayedCardsSize = (600, 600)
        self.AutoExpandPlayedCards = True
        self.RegionsConfigPath = os.path.join(os.path.dirname(__file__), "config", "regions.json")
        self._load_regions()
        if self.AutoExpandPlayedCards and not self._regions_loaded:
            self.LPlayedCardsPos = self._resize_region_center(self.LPlayedCardsPos, self.PlayedCardsSize)
            self.RPlayedCardsPos = self._resize_region_center(self.RPlayedCardsPos, self.PlayedCardsSize)

        # 信号量
        self.shouldExit = 0  # 通知上一轮记牌结束
        self.canRecord = threading.Lock()  # 开始记牌

        # 模型路径
        self.card_play_model_path_dict = {
            'landlord': "baselines/douzero_WP/landlord.ckpt",
            'landlord_up': "baselines/douzero_WP/landlord_up.ckpt",
            'landlord_down': "baselines/douzero_WP/landlord_down.ckpt"
        }
        # YOLO model for card recognition
        self.yolo_weights_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "weights", "best.pt"))
        self.yolo_imgsz = 640
        self.MyCardConf = 0.6
        self.OtherCardConf = 0.6
        self.ThreeLandlordCardConf = 0.6
        self._last_infer_ms = None
        if not os.path.exists(self.yolo_weights_path):
            QMessageBox.critical(self, "Model not found",
                                 "Missing YOLO weights: " + self.yolo_weights_path,
                                 QMessageBox.Yes, QMessageBox.Yes)
            raise FileNotFoundError(self.yolo_weights_path)
        self.yolo_model = YOLO(self.yolo_weights_path)
        self._cls_to_card = self._build_class_map()

        # Region preview window for calibration
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
        self.WinRate.setText("胜率：--%")
        self.InitCard.setText("开始")
        self.UserHandCards.setText("手牌")
        self.LPlayedCard.setText("上家出牌区域")
        self.RPlayedCard.setText("下家出牌区域")
        self.PredictedCard.setText("AI出牌区域")
        self.ThreeLandlordCards.setText("三张底牌")
        for player in self.Players:
            player.setStyleSheet('background-color: rgba(255, 0, 0, 0);')

    def init_cards(self):
        # 玩家手牌
        self.user_hand_cards_real = ""
        self.user_hand_cards_env = []
        # 其他玩家出牌
        self.other_played_cards_real = ""
        self.other_played_cards_env = []
        # 其他玩家手牌（整副牌减去玩家手牌，后续再减掉历史出牌）
        self.other_hand_cards = []
        # 三张底牌
        self.three_landlord_cards_real = ""
        self.three_landlord_cards_env = []
        # 玩家角色代码：0-地主上家, 1-地主, 2-地主下家
        self.user_position_code = None
        self.user_position = ""
        # 开局时三个玩家的手牌
        self.card_play_data_list = {}
        # 出牌顺序：0-玩家出牌, 1-玩家下家出牌, 2-玩家上家出牌
        self.play_order = 0

        self.env = None

        # 识别玩家手牌
        self.user_hand_cards_real = self.find_my_cards(self.MyHandCardsPos)
        self.UserHandCards.setText(self.user_hand_cards_real)
        self.user_hand_cards_env = [RealCard2EnvCard[c] for c in list(self.user_hand_cards_real)]
        # 识别三张底牌
        self.three_landlord_cards_real = self.find_three_landlord_cards(self.ThreeLandlordCardsPos)
        self.ThreeLandlordCards.setText("底牌：" + self.three_landlord_cards_real)
        self.three_landlord_cards_env = [RealCard2EnvCard[c] for c in list(self.three_landlord_cards_real)]
        # 识别玩家的角色
        self.user_position_code = self.find_landlord(self.LandlordFlagPos)
        if self.user_position_code is None:
            items = ("地主上家", "地主", "地主下家")
            item, okPressed = QInputDialog.getItem(self, "选择角色", "未识别到地主，请手动选择角色:", items, 0, False)
            if okPressed and item:
                self.user_position_code = items.index(item)
            else:
                return
        self.user_position = ['landlord_up', 'landlord', 'landlord_down'][self.user_position_code]
        for player in self.Players:
            player.setStyleSheet('background-color: rgba(255, 0, 0, 0);')
        self.Players[self.user_position_code].setStyleSheet('background-color: rgba(255, 0, 0, 0.1);')

        # 整副牌减去玩家手上的牌，就是其他人的手牌,再分配给另外两个角色（如何分配对AI判断没有影响）
        for i in set(AllEnvCard):
            self.other_hand_cards.extend([i] * (AllEnvCard.count(i) - self.user_hand_cards_env.count(i)))
        self.card_play_data_list.update({
            'three_landlord_cards': self.three_landlord_cards_env,
            ['landlord_up', 'landlord', 'landlord_down'][(self.user_position_code + 0) % 3]:
                self.user_hand_cards_env,
            ['landlord_up', 'landlord', 'landlord_down'][(self.user_position_code + 1) % 3]:
                self.other_hand_cards[0:17] if (self.user_position_code + 1) % 3 != 1 else self.other_hand_cards[17:],
            ['landlord_up', 'landlord', 'landlord_down'][(self.user_position_code + 2) % 3]:
                self.other_hand_cards[0:17] if (self.user_position_code + 1) % 3 == 1 else self.other_hand_cards[17:]
        })
        print(self.card_play_data_list)
        # 生成手牌结束，校验手牌数量
        if len(self.card_play_data_list["three_landlord_cards"]) != 3:
            QMessageBox.critical(self, "底牌识别出错", "底牌必须是3张！", QMessageBox.Yes, QMessageBox.Yes)
            self.init_display()
            return
        if len(self.card_play_data_list["landlord_up"]) != 17 or \
            len(self.card_play_data_list["landlord_down"]) != 17 or \
            len(self.card_play_data_list["landlord"]) != 20:
            QMessageBox.critical(self, "手牌识别出错", "初始手牌数目有误", QMessageBox.Yes, QMessageBox.Yes)
            self.init_display()
            return
        # 得到出牌顺序
        self.play_order = 0 if self.user_position == "landlord" else 1 if self.user_position == "landlord_up" else 2

        # 创建一个代表玩家的AI
        ai_players = [0, 0]
        ai_players[0] = self.user_position
        ai_players[1] = DeepAgent(self.user_position, self.card_play_model_path_dict[self.user_position])

        self.env = GameEnv(ai_players)

        self.in_game = True
        self.start()

    def start(self):
        self.env.card_play_init(self.card_play_data_list)
        print("开始出牌\n")
        while not self.env.game_over:
            # 玩家出牌时就通过智能体获取action，否则通过识别获取其他玩家出牌
            if self.play_order == 0:
                self.PredictedCard.setText("...")
                action_message = self.env.step(self.user_position)
                # 更新界面
                self.UserHandCards.setText("手牌：" + str(''.join(
                    [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards]))[::-1])

                self.PredictedCard.setText(action_message["action"] if action_message["action"] else "不出")
                self.WinRate.setText("胜率：" + action_message["win_rate"])
                print("\n手牌：", str(''.join(
                        [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards])))
                print("出牌：", action_message["action"] if action_message["action"] else "不出", "， 胜率：",
                      action_message["win_rate"])
                while self.have_white(self.RPlayedCardsPos) == 1 or \
                        self._safe_locate('pics/pass.png', self.RPlayedCardsPos, self.LandlordFlagConfidence):
                    print("等待玩家出牌")
                    self.counter.restart()
                    while self.counter.elapsed() < 100:
                        QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
                self.play_order = 1
            elif self.play_order == 1:
                self.RPlayedCard.setText("...")
                pass_flag = None
                while self.have_white(self.RPlayedCardsPos) == 0 and \
                        not self._safe_locate('pics/pass.png', self.RPlayedCardsPos, self.LandlordFlagConfidence):
                    print("等待下家出牌")
                    self.counter.restart()
                    while self.counter.elapsed() < 500:
                        QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
                self.counter.restart()
                while self.counter.elapsed() < 500:
                    QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
                # 不出
                pass_flag = self._safe_locate('pics/pass.png', self.RPlayedCardsPos, self.LandlordFlagConfidence)
                # 未找到"不出"
                if pass_flag is None:
                    # 识别下家出牌
                    self.other_played_cards_real = self.last_right_cards or self.find_other_cards(self.RPlayedCardsPos)
                # 找到"不出"
                else:
                    self.other_played_cards_real = ""
                print("\n下家出牌：", self.other_played_cards_real)
                self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(self.other_played_cards_real)]
                self.env.step(self.user_position, self.other_played_cards_env)
                # 更新界面
                self.RPlayedCard.setText(self.other_played_cards_real if self.other_played_cards_real else "不出")
                self.play_order = 2
            elif self.play_order == 2:
                self.LPlayedCard.setText("...")
                while self.have_white(self.LPlayedCardsPos) == 0 and \
                        not self._safe_locate('pics/pass.png', self.LPlayedCardsPos, self.LandlordFlagConfidence):
                    print("等待上家出牌")
                    self.counter.restart()
                    while self.counter.elapsed() < 500:
                        QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
                self.counter.restart()
                while self.counter.elapsed() < 500:
                    QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
                # 不出
                pass_flag = self._safe_locate('pics/pass.png', self.LPlayedCardsPos, self.LandlordFlagConfidence)
                # 未找到"不出"
                if pass_flag is None:
                    # 识别上家出牌
                    self.other_played_cards_real = self.last_left_cards or self.find_other_cards(self.LPlayedCardsPos)
                # 找到"不出"
                else:
                    self.other_played_cards_real = ""
                print("\n上家出牌：", self.other_played_cards_real)
                self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(self.other_played_cards_real)]
                self.env.step(self.user_position, self.other_played_cards_env)
                self.play_order = 0
                # 更新界面
                self.LPlayedCard.setText(self.other_played_cards_real if self.other_played_cards_real else "不出")
            else:
                pass

            self.counter.restart()
            while self.counter.elapsed() < 100:
                QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)

        print("{}胜，本局结束!\n".format("农民" if self.env.winner == "farmer" else "地主"))
        QMessageBox.information(self, "本局结束", "{}胜！".format("农民" if self.env.winner == "farmer" else "地主"),
                                QMessageBox.Yes, QMessageBox.Yes)
        self.env.reset()
        self.in_game = False
        self.init_display()

    def find_landlord(self, landlord_flag_pos):
        for pos in landlord_flag_pos:
            result = self._safe_locate('pics/landlord_words.png', pos, self.LandlordFlagConfidence)
            if result is not None:
                return landlord_flag_pos.index(pos)
        return None

    def find_three_landlord_cards(self, pos):
        return self.detect_cards_yolo(pos, self.ThreeLandlordCardConf)

    def find_my_cards(self, pos):
        return self.detect_cards_yolo(pos, self.MyCardConf)

    def find_other_cards(self, pos):
        self.counter.restart()
        while self.counter.elapsed() < 500:
            QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)
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

    def cards_filter(self, location, distance):  # 牌检测结果滤波
        if len(location) == 0:
            return 0
        locList = [location[0][0]]
        count = 1
        for e in location:
            flag = 1  # “是新的”标志
            for have in locList:
                if abs(e[0] - have) <= distance:
                    flag = 0
                    break
            if flag:
                count += 1
                locList.append(e[0])
        return count

    def have_white(self, pos):  # 是否有白块
        result = self._safe_locate('pics/white.png', pos, self.WhiteConfidence)
        if result is None:
            return 0
        else:
            return 1

    def _safe_locate(self, image_path, region, confidence):
        try:
            return pyscreeze.locateOnScreen(image_path, region=region, confidence=confidence)
        except pyscreeze.ImageNotFoundException:
            return None

    def _build_class_map(self):
        names = self.yolo_model.names
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        mapping = {}
        for cls_id, name in names.items():
            card = self._map_name_to_card(str(name))
            if card:
                mapping[int(cls_id)] = card
        return mapping

    def _map_name_to_card(self, name):
        if name == "-1":
            return None
        if name == "1":
            return "A"
        if name == "10":
            return "T"
        if name == "11":
            return "J"
        if name == "12":
            return "Q"
        if name == "13":
            return "K"
        if name == "14":
            return "X"
        if name == "15":
            return "D"
        if name in {"2", "3", "4", "5", "6", "7", "8", "9"}:
            return name
        return None

    def update_recognition(self):
        speeds = []

        if self.UseMouseRegionForMain:
            region = self._get_mouse_region(self.DebugMouseSize)
            mouse_cards = self.detect_cards_yolo(
                (region["left"], region["top"], region["width"], region["height"]),
                self.MyCardConf
            )
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            if mouse_cards:
                self.UserHandCards.setText("鼠标区：" + mouse_cards)
            else:
                self.UserHandCards.setText("鼠标区")
            self.LPlayedCard.setText("上家出牌区域")
            self.RPlayedCard.setText("下家出牌区域")
            self.ThreeLandlordCards.setText("三张底牌")
            self.last_left_cards = ""
            self.last_right_cards = ""
        else:
            my_cards = self.detect_cards_yolo(self.MyHandCardsPos, self.MyCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            if my_cards:
                self.UserHandCards.setText(my_cards)
            else:
                self.UserHandCards.setText("手牌")

            left_cards = self.detect_cards_yolo(self.LPlayedCardsPos, self.OtherCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            if left_cards:
                self.LPlayedCard.setText(left_cards)
            else:
                self.LPlayedCard.setText("上家出牌区域")
            self.last_left_cards = left_cards

            right_cards = self.detect_cards_yolo(self.RPlayedCardsPos, self.OtherCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            if right_cards:
                self.RPlayedCard.setText(right_cards)
            else:
                self.RPlayedCard.setText("下家出牌区域")
            self.last_right_cards = right_cards

            bottom_cards = self.detect_cards_yolo(self.ThreeLandlordCardsPos, self.ThreeLandlordCardConf)
            if self._last_infer_ms is not None:
                speeds.append(self._last_infer_ms)
            if bottom_cards:
                self.ThreeLandlordCards.setText("底牌：" + bottom_cards)
            else:
                self.ThreeLandlordCards.setText("三张底牌")

        if speeds:
            avg_ms = sum(speeds) / len(speeds)
            if self.in_game:
                base = self.WinRate.text()
                if "YOLO:" in base:
                    base = base.split("| YOLO:")[0].strip()
                self.WinRate.setText("{} | YOLO: {:.1f} ms".format(base, avg_ms))
            else:
                self.WinRate.setText("YOLO: {:.1f} ms".format(avg_ms))

        if self.DebugMode:
            self._show_mouse_debug()

    def toggle_recognition(self):
        if self._recog_enabled:
            self._recog_timer.stop()
            self._recog_enabled = False
            self.ToggleRecognize.setText("恢复识别")
        else:
            self._recog_timer.start(int(self.RecognitionInterval * 1000))
            self._recog_enabled = True
            self.ToggleRecognize.setText("暂停识别")
            self.update_recognition()

    def toggle_debug_mode(self):
        self.DebugMode = not self.DebugMode
        if not self.DebugMode and self._debug_window_open:
            cv2.destroyWindow("YOLO Debug (Mouse)")
            self._debug_window_open = False
        self.ToggleDebug.setText("关闭调试" if self.DebugMode else "调试模式")
        if self.DebugMode:
            self.update_recognition()

    def start_calibration(self):
        if self._calibrating:
            return
        item, method, manual_text = self._choose_calib_target()
        if not item or not method:
            return
        self._calib_target = item
        if method == "手动输入":
            text = manual_text
            if not text:
                return
            try:
                parts = [int(p.strip()) for p in text.split(",")]
                if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.critical(self, "输入错误", "格式应为 x,y,w,h 且 w/h > 0", QMessageBox.Yes, QMessageBox.Yes)
                return
            self._pending_region = (parts[0], parts[1], parts[2], parts[3])
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
        if target == "我的手牌区域":
            self.MyHandCardsPos = region
        elif target == "左家出牌区域":
            self.LPlayedCardsPos = region
            self.last_left_cards = ""
        elif target == "右家出牌区域":
            self.RPlayedCardsPos = region
            self.last_right_cards = ""
        elif target == "三张底牌区域":
            self.ThreeLandlordCardsPos = region
        elif target == "地主标志-右":
            self.LandlordFlagPos[0] = region
        elif target == "地主标志-我":
            self.LandlordFlagPos[1] = region
        elif target == "地主标志-左":
            self.LandlordFlagPos[2] = region
        self._pending_region = None
        self._calib_target = None
        self._save_regions()
        self.update_recognition()

    def _choose_calib_target(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("校准区域")
        layout = QtWidgets.QVBoxLayout(dialog)
        tip = QtWidgets.QLabel("点击区域按钮开始框选；\n如需手动输入，请在下方填入 x,y,w,h 后点击对应区域按钮。")
        layout.addWidget(tip)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        manual_edit = QtWidgets.QLineEdit()
        manual_edit.setPlaceholderText("手动输入: x,y,w,h")

        result = {"target": None, "method": None, "manual": ""}

        def on_click(target):
            text = manual_edit.text().strip()
            if text:
                result["method"] = "手动输入"
                result["manual"] = text
            else:
                result["method"] = "鼠标框选"
            result["target"] = target
            dialog.accept()

        buttons = [
            ("我的手牌区域", 0, 0),
            ("左家出牌区域", 0, 1),
            ("右家出牌区域", 0, 2),
            ("三张底牌区域", 1, 0),
            ("地主标志-右", 1, 1),
            ("地主标志-我", 1, 2),
            ("地主标志-左", 2, 1),
        ]
        for text, r, c in buttons:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _, t=text: on_click(t))
            grid.addWidget(btn, r, c)

        layout.addWidget(manual_edit)

        cancel_btn = QtWidgets.QPushButton("取消")
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
        new_region = {"left": int(cx - new_w / 2), "top": int(cy - new_h / 2),
                      "width": int(new_w), "height": int(new_h)}
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
            json.dump(data, f, ensure_ascii=False, indent=2)

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
        cv2.putText(frame, "Calibrating", (left, max(0, top - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.CalibrationScale != 1.0:
            frame = cv2.resize(frame, None, fx=self.CalibrationScale, fy=self.CalibrationScale,
                               interpolation=cv2.INTER_AREA)
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
        if cursor_x < screen_w / 2:
            x = screen_w - win_w - margin
        else:
            x = margin
        if cursor_y < screen_h / 2:
            y = screen_h - win_h - margin - 40
        else:
            y = margin
        x = max(0, int(x))
        y = max(0, int(y))
        cv2.moveWindow("Calibration Preview", x, y)

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
        try:
            self._preview_stop = True
            self.env.game_over = True
        except AttributeError as e:
            pass 

    def _region_preview_loop(self):
        window_name = "Region Preview (press q to close)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)
        while not self._preview_stop:
            img = pyscreeze.screenshot()
            frame = np.array(img)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            frame = np.ascontiguousarray(frame)

            def draw_rect(pos, label):
                x, y, w, h = pos
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            draw_rect(self.MyHandCardsPos, "MyHand")
            draw_rect(self.LPlayedCardsPos, "LeftPlayed")
            draw_rect(self.RPlayedCardsPos, "RightPlayed")
            draw_rect(self.ThreeLandlordCardsPos, "Bottom3")
            for i, p in enumerate(self.LandlordFlagPos):
                draw_rect(p, f"LandlordFlag{i}")

            if self.PreviewScale != 1.0:
                frame = cv2.resize(frame, None, fx=self.PreviewScale, fy=self.PreviewScale,
                                   interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._preview_stop = True
                cv2.destroyWindow(window_name)
                break
            time.sleep(self.RegionPreviewInterval)
            


if __name__ == '__main__':

    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["GIT_PYTHON_REFRESH"] = 'quiet'

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("""
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
        background-color : light gray;
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
    """)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())
