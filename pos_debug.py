# -*- coding: utf-8 -*-
"""
CLI screen-region calibration & bounding-box overlay visualizer.

Usage examples:
    python pos_debug.py --validate
    python pos_debug.py --image screenshot.png --output annotated_regions.png
    python pos_debug.py --live --show
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from card_tracker import DEFAULT_REGIONS_1080P, iter_named_regions, validate_regions_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "regions.json")


def load_regions_config(config_path: Optional[str]) -> Dict[str, Any]:
    target = config_path or DEFAULT_CONFIG_PATH
    if target and os.path.isfile(target):
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    return dict(DEFAULT_REGIONS_1080P)


def render_regions_overlay(image_bgr: np.ndarray, regions_dict: Dict[str, Any]) -> np.ndarray:
    annotated = image_bgr.copy()
    palette = [
        (0, 255, 0),
        (255, 180, 0),
        (0, 165, 255),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for idx, (label, (x, y, w, h)) in enumerate(iter_named_regions(regions_dict)):
        color = palette[idx % len(palette)]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            f"{label} ({x},{y},{w}x{h})",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return annotated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pos_debug",
        description="Inspect, validate, and render DouZero + YOLOv8 screen capture regions.",
    )
    parser.add_argument("--image", help="Input screenshot file path (.png/.jpg).")
    parser.add_argument("--live", action="store_true", help="Capture a live desktop screenshot via pyscreeze.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to regions.json calibration file (defaults to config/regions.json).",
    )
    parser.add_argument("--output", help="Save annotated region overlay image to this file path.")
    parser.add_argument("--show", action="store_true", help="Open an interactive OpenCV preview window.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configured region bounds and print JSON diagnostics.",
    )
    parser.add_argument("--width", type=int, default=1920, help="Reference screen width for --validate.")
    parser.add_argument("--height", type=int, default=1080, help="Reference screen height for --validate.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    regions = load_regions_config(args.config)

    if args.validate or (not args.image and not args.live):
        img_size = (args.width, args.height)
        if args.image and os.path.isfile(args.image):
            probe = cv2.imread(args.image)
            if probe is not None:
                img_size = (probe.shape[1], probe.shape[0])
        report = validate_regions_dict(regions, screen_size=img_size)
        print(json.dumps(report, indent=2))
        if not args.image and not args.live:
            return 0

    frame: Optional[np.ndarray] = None
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            return 1
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Failed to decode image file: {args.image}", file=sys.stderr)
            return 1
    elif args.live:
        try:
            import pyscreeze

            shot = pyscreeze.screenshot()
            arr = np.array(shot)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print(f"Error: Live screenshot capture failed: {exc}", file=sys.stderr)
            return 1

    if frame is None:
        return 0

    annotated = render_regions_overlay(frame, regions)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if not cv2.imwrite(args.output, annotated):
            print(f"Error: Failed to write output image: {args.output}", file=sys.stderr)
            return 1
        print(f"Saved annotated region overlay to: {args.output}")

    if args.show or (not args.output and (args.image or args.live)):
        cv2.namedWindow("Region Calibration Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Region Calibration Overlay", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
