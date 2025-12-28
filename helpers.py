from typing import Dict, Optional, TypeVar, List
# pylint: disable=no-name-in-module
# TODO: Hva skjer med lintingen?
from pygame.constants import K_UP, K_DOWN, K_LEFT, K_RIGHT
# pylint: enable=no-name-in-module

DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

KEY_TO_DIRECTION = {
    K_UP: DIRECTIONS["UP"],
    K_DOWN: DIRECTIONS["DOWN"],
    K_LEFT: DIRECTIONS["LEFT"],
    K_RIGHT: DIRECTIONS["RIGHT"],
}

DEFAULTS = {
    "player_pos": (40, 40),
    "direction": DIRECTIONS["RIGHT"],
}


type Coordinate = tuple[int, int]
# matches the values stored in DIRECTIONS[keyof DIRECTIONS]
type Direction = tuple[int, int]
type Tail = List[Coordinate]

Key = TypeVar("Key")
Value = TypeVar("Value")


def get_key_from_value(dic: Dict[Key, Value], value: Value) -> Optional[Key]:
    for key, dir_value in dic.items():
        if dir_value == value:
            return key
    return None


def is_opposite_direction(dir1: Direction, dir2: Direction) -> bool:
    return dir1[0] == -dir2[0] and dir1[1] == -dir2[1]


def direction_change_is_legal(old_direction: Direction, next_direction: Direction) -> bool:
    pretty_next_dir = get_key_from_value(DIRECTIONS, next_direction)

    if is_opposite_direction(old_direction, next_direction):
        print(f"Ignoring opposite direction {pretty_next_dir}")
        return False

    if old_direction == next_direction:
        print(f"Ignoring same direction {pretty_next_dir}")
        return False

    return True
