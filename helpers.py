
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


def get_key_from_value(dic, value):
    for key, dir_value in dic.items():
        if dir_value == value:
            return key
    return None
