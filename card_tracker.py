"""
Pure card-mapping, remaining-deck tracking, and screen-region validation utilities
for the DouZero + YOLOv8 DouDizhu assistant.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EnvCard2RealCard: Dict[int, str] = {
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
    17: "2",
    20: "X",
    30: "D",
}

RealCard2EnvCard: Dict[str, int] = {v: k for k, v in EnvCard2RealCard.items()}

AllEnvCard: List[int] = [
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    9, 9, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    12, 12, 12, 12,
    13, 13, 13, 13,
    14, 14, 14, 14,
    17, 17, 17, 17,
    20,
    30,
]

CARD_DISPLAY_ORDER: List[int] = [30, 20, 17, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]

DEFAULT_REGIONS_1080P: Dict[str, Any] = {
    "MyHandCardsPos": [414, 804, 1041, 59],
    "LPlayedCardsPos": [530, 470, 380, 160],
    "RPlayedCardsPos": [1010, 470, 380, 160],
    "ThreeLandlordCardsPos": [817, 36, 287, 136],
    "LandlordFlagPos": [
        [1320, 300, 110, 140],
        [320, 720, 110, 140],
        [500, 300, 110, 140],
    ],
}


def map_yolo_class_name(name: str) -> Optional[str]:
    """Map trained YOLOv8 class label strings ('1'..'15', '-1') to DouZero card chars."""
    clean = str(name).strip()
    if clean == "-1":
        return None
    mapping = {
        "1": "A",
        "10": "T",
        "11": "J",
        "12": "Q",
        "13": "K",
        "14": "X",
        "15": "D",
    }
    if clean in mapping:
        return mapping[clean]
    if clean in {"2", "3", "4", "5", "6", "7", "8", "9"}:
        return clean
    return None


def real_cards_to_env(cards_str: str) -> List[int]:
    """Convert a string of card symbols (e.g. '34567TJQKA2XD') into sorted DouZero env ints."""
    out: List[int] = []
    for ch in cards_str.strip().upper():
        if ch in RealCard2EnvCard:
            out.append(RealCard2EnvCard[ch])
    return out


def compute_remaining_cards(
    my_hand_env: Sequence[int],
    played_cards_by_pos: Dict[str, Sequence[int]],
) -> Dict[str, int]:
    """Compute unseen opponent card counts for every rank in display order (D, X, 2, A..3)."""
    seen: List[int] = list(my_hand_env)
    for seq in played_cards_by_pos.values():
        seen.extend(seq)

    remaining: Dict[str, int] = {}
    for env_val in CARD_DISPLAY_ORDER:
        total_in_deck = 1 if env_val in (20, 30) else 4
        left = max(0, total_in_deck - seen.count(env_val))
        remaining[EnvCard2RealCard[env_val]] = left
    return remaining


def format_remaining_summary(remaining: Dict[str, int], high_only: bool = False) -> str:
    keys = ["D", "X", "2", "A", "K"] if high_only else [EnvCard2RealCard[v] for v in CARD_DISPLAY_ORDER]
    parts = [f"{k}:{remaining.get(k, 0)}" for k in keys if remaining.get(k, 0) > 0 or high_only]
    return " ".join(parts) if parts else "None"


def parse_region_csv(text: str) -> Tuple[int, int, int, int]:
    """Parse 'x,y,w,h' into a validated 4-tuple with positive width and height."""
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError("Region must be formatted as 'x,y,w,h' with width > 0 and height > 0.")
    return (parts[0], parts[1], parts[2], parts[3])


def iter_named_regions(regions_dict: Dict[str, Any]) -> Iterable[Tuple[str, Tuple[int, int, int, int]]]:
    for key, label in (
        ("MyHandCardsPos", "MyHand"),
        ("LPlayedCardsPos", "LeftPlayed"),
        ("RPlayedCardsPos", "RightPlayed"),
        ("ThreeLandlordCardsPos", "Bottom3"),
    ):
        val = regions_dict.get(key)
        if isinstance(val, (list, tuple)) and len(val) == 4:
            yield label, (int(val[0]), int(val[1]), int(val[2]), int(val[3]))

    flags = regions_dict.get("LandlordFlagPos")
    if isinstance(flags, (list, tuple)):
        suffixes = ("Right", "Me", "Left")
        for idx, item in enumerate(flags[:3]):
            if isinstance(item, (list, tuple)) and len(item) == 4:
                tag = suffixes[idx] if idx < len(suffixes) else str(idx)
                yield f"LandlordFlag_{tag}", (int(item[0]), int(item[1]), int(item[2]), int(item[3]))


def validate_regions_dict(
    regions_dict: Dict[str, Any],
    screen_size: Tuple[int, int] = (1920, 1080),
) -> Dict[str, Any]:
    """Validate region dictionary structure and check bounds against `screen_size`."""
    sw, sh = screen_size
    warnings: List[str] = []
    entries: List[Dict[str, Any]] = []

    for label, (x, y, w, h) in iter_named_regions(regions_dict):
        in_bounds = x >= 0 and y >= 0 and w > 0 and h > 0 and (x + w) <= sw and (y + h) <= sh
        if not in_bounds:
            warnings.append(
                f"{label} ({x}, {y}, {w}, {h}) extends outside {sw}x{sh} screen bounds."
            )
        entries.append(
            {
                "name": label,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "in_bounds": in_bounds,
            }
        )

    expected_count = 7
    if len(entries) < expected_count:
        warnings.append(f"Expected {expected_count} configured regions, found {len(entries)}.")

    return {
        "valid": len(warnings) == 0 and len(entries) == expected_count,
        "screen_width": sw,
        "screen_height": sh,
        "region_count": len(entries),
        "regions": entries,
        "warnings": warnings,
    }
