import argparse
import random

from typing import Optional

import pygame as pg

# pylint: disable=no-name-in-module
# TODO: Hva skjer med lintingen?
from pygame.constants import MOUSEBUTTONDOWN, QUIT, KEYDOWN, K_ESCAPE,  K_SPACE, K_a
# pylint: enable=no-name-in-module

from helpers import Coordinate, DEFAULTS, DIRECTIONS, KEY_TO_DIRECTION, direction_change_is_legal, get_key_from_value
from player import Player
from ml_autopilot import MLAutopilot


TITLE = "YAPS - Yet Another PyGame Snake"
DEFAULT_TILES_HORIZONTAL = 10
DEFAULT_TILES_VERTICAL = 10
DEFAULT_TILE_SIZE = 80


class Game:
    def __init__(
        self,
        tile_size: int,
        tiles_horizontal: int,
        tiles_vertical: int,
        ml_model_path: Optional[str] = None,
        ml_hidden_size: int = 256,
    ):
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.init()
        # pylint: enable=no-member
        self.clock = pg.time.Clock()
        pg.display.set_caption(TITLE)
        self.tile_size = max(10, tile_size)
        self.tiles_horizontal = max(2, tiles_horizontal)
        self.tiles_vertical = max(2, tiles_vertical)
        self.window_width = self.tiles_horizontal * self.tile_size
        self.window_height = self.tiles_vertical * self.tile_size
        self.surface = pg.display.set_mode(
            (self.window_width, self.window_height))
        self.font = pg.font.Font(None, 24)
        self.loop = True
        self._ml_brain: Optional["MLAutopilot"] = self._load_ml_brain(
            ml_model_path, ml_hidden_size
        )

        self.player = Player(
            self.surface,
            self.window_width,
            self.window_height,
            self.tile_size,
            self.tiles_horizontal,
            self.tiles_vertical,
            ml_brain=self._ml_brain,
        )
        self._current_fruit: Optional[Coordinate] = None

        self._col_labels = [chr(ord('A') + i)
                            for i in range(self.tiles_horizontal)]
        self._row_labels = [str(self.tiles_vertical - i)
                            for i in range(self.tiles_vertical)]

    def main(self) -> None:
        while self.loop:
            self.grid_loop()
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.quit()
        # pylint: enable=no-member

    def grid_loop(self) -> None:
        delta_ms = self.clock.tick(120)

        self.surface.fill((0, 0, 0))
        for row in range(self.tiles_horizontal):
            for col in range(row % 2, self.tiles_vertical, 2):
                pg.draw.rect(
                    self.surface,
                    (40, 40, 40),
                    (row * self.tile_size, col * self.tile_size,
                     self.tile_size, self.tile_size),
                )
        if can_show_grid_labels := self.tile_size >= 40:
            self._draw_grid_labels()

        self._draw_player(self.player)

        if self.player.should_die():
            print("You died!")
            # self.loop = False

            # Make screen red and dramatic to indicate death
            self.surface.fill((255, 0, 0))
            pg.display.update()
            pg.time.delay(2000)
            self._reset_state()

        # Fruit logic
        if self._current_fruit and self.player.get_pos() == self._current_fruit:
            self.player.grow(self.player.is_autopilot_enabled())
            self._current_fruit = None

        if self._current_fruit is None:
            offset = self.tile_size // 2
            fruit_x = random.randint(0, self.tiles_horizontal - 1)
            fruit_y = random.randint(0, self.tiles_vertical - 1)
            self._current_fruit = (
                fruit_x * self.tile_size + offset,
                fruit_y * self.tile_size + offset,
            )

        if self._current_fruit:
            fruit_radius = max(5, self.tile_size // 4)
            pg.draw.circle(self.surface, (255, 0, 0),
                           self._current_fruit, fruit_radius)

        # Event handling

        for event in pg.event.get():
            if event.type == QUIT:
                self.loop = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.loop = False
                elif event.key == K_a:
                    print("Toggling autopilot")
                    self.player.toggle_autopilot()
                elif event.key == K_SPACE:
                    print("Resetting player position and direction")
                    self._reset_state()
                if (next_direction := KEY_TO_DIRECTION.get(event.key, None)) and direction_change_is_legal(self.player.get_direction(), next_direction):
                    print(
                        f"Setting direction to {get_key_from_value(DIRECTIONS, next_direction)}")
                    self.player.set_direction(next_direction)
            elif event.type == MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                self.player.set_pos(pos)
                self.player.reset_tail()
        self.player.update(delta_ms, self._current_fruit)
        pg.display.update()

    def _load_ml_brain(
        self, model_path: Optional[str], hidden_size: int
    ) -> Optional["MLAutopilot"]:
        if not model_path:
            return None
        try:
            from ml_autopilot import MLAutopilot

            brain = MLAutopilot(
                model_path=model_path,
                cols=self.tiles_horizontal,
                rows=self.tiles_vertical,
                tile_size=self.tile_size,
                hidden_size=hidden_size,
            )
            print(f"Loaded ML autopilot from {model_path}")
            return brain
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to initialize ML autopilot: {exc}")
            return None

    def _reset_state(self) -> None:
        self.player.set_pos(self.player.get_spawn_pos())
        self.player.set_direction(DEFAULTS["direction"])
        self.player.reset_tail()
        self._current_fruit = None

    def _draw_grid_labels(self) -> None:
        """Render chess-like file/rank labels along the board edges."""
        label_color = (160, 160, 160)
        margin = max(12, self.tile_size // 4)
        for idx, letter in enumerate(self._col_labels):
            x = idx * self.tile_size + self.tile_size // 2
            top_text = self.font.render(letter, True, label_color)
            bottom_text = self.font.render(letter, True, label_color)
            self.surface.blit(top_text, top_text.get_rect(center=(x, margin)))
            self.surface.blit(bottom_text, bottom_text.get_rect(
                center=(x, self.window_height - margin)))

        for idx, number in enumerate(self._row_labels):
            y = idx * self.tile_size + self.tile_size // 2
            left_text = self.font.render(number, True, label_color)
            right_text = self.font.render(number, True, label_color)
            self.surface.blit(
                left_text, left_text.get_rect(center=(margin, y)))
            self.surface.blit(right_text, right_text.get_rect(
                center=(self.window_width - margin, y)))

    def _draw_player(self, player: Player) -> None:
        # Draw tail
        tail_radius = max(5, self.tile_size // 4)
        head_radius = max(tail_radius, self.tile_size // 2)
        for coords, was_ai in player.get_tail():
            color = (100, 200, 100) if was_ai else (200, 200, 200)
            pg.draw.circle(self.surface, color, coords, tail_radius)
        # Draw head
        pg.draw.circle(self.surface, (255, 255, 255),
                       player.get_pos(), head_radius)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Pixel size for each grid tile (default: 80).",
    )
    parser.add_argument(
        "--tiles-horizontal",
        type=int,
        default=DEFAULT_TILES_HORIZONTAL,
        help="Number of tiles along the X axis (default: 10).",
    )
    parser.add_argument(
        "--tiles-vertical",
        type=int,
        default=DEFAULT_TILES_VERTICAL,
        help="Number of tiles along the Y axis (default: 10).",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        default=None,
        help="Path to a trained DQN checkpoint for the autopilot.",
    )
    parser.add_argument(
        "--ml-hidden-size",
        type=int,
        default=256,
        help="Hidden layer width that was used during training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mygame = Game(
        tile_size=args.tile_size,
        tiles_horizontal=args.tiles_horizontal,
        tiles_vertical=args.tiles_vertical,
        ml_model_path=args.ml_model,
        ml_hidden_size=args.ml_hidden_size,
    )
    mygame.main()
