import random

import pygame as pg

# pylint: disable=no-name-in-module
# TODO: Hva skjer med lintingen?
from pygame.constants import MOUSEBUTTONDOWN, QUIT, KEYDOWN, K_ESCAPE,  K_SPACE

from helpers import DEFAULTS, DIRECTIONS, KEY_TO_DIRECTION, direction_change_is_legal, get_key_from_value, is_opposite_direction
from npc import Npc
from player import Player
# pylint: enable=no-name-in-module


TITLE = "YAPS - Yet Another PyGame Snake"
TILES_HORIZONTAL = 10
TILES_VERTICAL = 10
TILE_SIZE = 80
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

USE_NPC = True


class Game:
    def __init__(self):
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.init()
        # pylint: enable=no-member
        self.clock = pg.time.Clock()
        pg.display.set_caption(TITLE)
        self.surface = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pg.font.Font(None, 24)
        self.loop = True
        actor = Npc if USE_NPC else Player
        self.player = actor(self.surface, WINDOW_WIDTH, WINDOW_HEIGHT)
        self._current_fruit = None, None
        self._col_labels = [chr(ord('A') + i) for i in range(TILES_HORIZONTAL)]
        self._row_labels = [str(TILES_VERTICAL - i)
                            for i in range(TILES_VERTICAL)]

    def main(self):
        while self.loop:
            self.grid_loop()
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.quit()
        # pylint: enable=no-member

    def grid_loop(self):
        delta_ms = self.clock.tick(120)

        self.surface.fill((0, 0, 0))
        for row in range(TILES_HORIZONTAL):
            for col in range(row % 2, TILES_HORIZONTAL, 2):
                pg.draw.rect(
                    self.surface,
                    (40, 40, 40),
                    (row * TILE_SIZE, col * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )
        self._draw_grid_labels()
        self._draw_player(self.player)

        if self.player.should_die():
            print("You died!")
            self.loop = False

            # Make screen red and dramatic to indicate death
            self.surface.fill((255, 0, 0))
            pg.display.update()
            pg.time.delay(2000)

        # Fruit logic
        if self.player.get_pos() == self._current_fruit:
            self.player.grow()
            self._current_fruit = None, None

        if self._current_fruit == (None, None):
            fruit_x = random.randint(0, TILES_HORIZONTAL - 1) * TILE_SIZE + 40
            fruit_y = random.randint(0, TILES_VERTICAL - 1) * TILE_SIZE + 40
            self._current_fruit = (fruit_x, fruit_y)

        pg.draw.circle(self.surface, (255, 0, 0), self._current_fruit, 20)

        # Event handling

        for event in pg.event.get():
            if event.type == QUIT:
                self.loop = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.loop = False
                elif event.key == K_SPACE:
                    print("Resetting player position and direction")
                    self.player.set_pos((DEFAULTS["player_pos"]))
                    self.player.set_direction(DEFAULTS["direction"])
                    self.player.set_tail([])
                if (next_direction := KEY_TO_DIRECTION.get(event.key, None)) and direction_change_is_legal(self.player.get_direction(), next_direction):
                    print(
                        f"Setting direction to {get_key_from_value(DIRECTIONS, next_direction)}")
                    self.player.set_direction(next_direction)
            elif event.type == MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                self.player.set_pos(pos)
                self.player.set_tail([])
        self.player.update(delta_ms, self._current_fruit)
        pg.display.update()

    def _draw_grid_labels(self):
        """Render chess-like file/rank labels along the board edges."""
        label_color = (160, 160, 160)
        for idx, letter in enumerate(self._col_labels):
            x = idx * TILE_SIZE + TILE_SIZE // 2
            top_text = self.font.render(letter, True, label_color)
            bottom_text = self.font.render(letter, True, label_color)
            self.surface.blit(top_text, top_text.get_rect(center=(x, 12)))
            self.surface.blit(bottom_text, bottom_text.get_rect(
                center=(x, WINDOW_HEIGHT - 12)))

        for idx, number in enumerate(self._row_labels):
            y = idx * TILE_SIZE + TILE_SIZE // 2
            left_text = self.font.render(number, True, label_color)
            right_text = self.font.render(number, True, label_color)
            self.surface.blit(left_text, left_text.get_rect(center=(12, y)))
            self.surface.blit(right_text, right_text.get_rect(
                center=(WINDOW_WIDTH - 12, y)))

    def _draw_player(self, player):
        # Draw tail
        for segment in player.get_tail():
            pg.draw.circle(self.surface, (200, 200, 200), segment, 20)
        # Draw head
        pg.draw.circle(self.surface, (255, 255, 255),
                       player.get_pos(), 40)


if __name__ == "__main__":
    mygame = Game()
    mygame.main()
