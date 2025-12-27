import random

import pygame as pg

# pylint: disable=no-name-in-module
# TODO: Hva skjer med lintingen?
from pygame.constants import MOUSEBUTTONDOWN, QUIT, KEYDOWN, K_ESCAPE,  K_SPACE

from helpers import DEFAULTS, DIRECTIONS, KEY_TO_DIRECTION, get_key_from_value
# pylint: enable=no-name-in-module


TITLE = "YAPS - Yet Another PyGame Snake"
TILES_HORIZONTAL = 10
TILES_VERTICAL = 10
TILE_SIZE = 80
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800


class Player:

    def __init__(self, surface):
        self.surface = surface
        self._pos = DEFAULTS["player_pos"]
        self._tail = []
        self._current_direction = DEFAULTS["direction"]
        # Movement cadence state so framerate and speed can be tuned independently
        self._move_delay = 150  # milliseconds between steps
        self._move_accumulator = 0

        # TODO: Wack hack...
        self._should_eat = False

    def draw(self):
        # Draw tail
        for segment in self._tail:
            pg.draw.circle(self.surface, (200, 200, 200), segment, 20)
        # Draw head
        pg.draw.circle(self.surface, (255, 255, 255), self._pos, 40)

    # TODO: Should proabably nuke the tail if dist > 1 (due to mouse click)
    def set_pos(self, target):
        self._pos = target
        if self._tail:
            self._tail = [self._pos] + self._tail[:-1]

    def get_pos(self):
        return self._pos

    def get_next_pos(self, old_pos):
        old_x, old_y = old_pos
        new_x = (80 * (old_x // 80)) + 40 + \
            self._current_direction[0] * 80
        new_y = (80 * (old_y // 80)) + 40 + \
            self._current_direction[1] * 80

        # wrap around logic
        if new_x < 40:
            new_x = WINDOW_WIDTH - 40
        elif new_x > WINDOW_WIDTH - 40:
            new_x = 40

        if new_y < 40:
            new_y = WINDOW_HEIGHT - 40
        elif new_y > WINDOW_HEIGHT - 40:
            new_y = 40

        return (new_x, new_y)

    def move(self):
        prev_pos = self._pos
        new_pos = self.get_next_pos(prev_pos)
        if self._should_eat:
            self._tail.insert(0, prev_pos)
            self._should_eat = False
        elif self._tail:
            self._tail = [prev_pos] + self._tail[:-1]
        self._pos = new_pos

    # TODO: Er dette måten å gjøre det på??
    def update(self, delta_ms):
        """Advance the player when enough time has elapsed."""
        self._move_accumulator += delta_ms
        if self._move_accumulator < self._move_delay:
            return
        # Preserve leftover time so slight jitter is averaged out
        self._move_accumulator %= self._move_delay
        self.move()

    # TODO: Refactor grow logic
    def grow(self):
        self._should_eat = True

    def should_die(self):
        return self._pos in self._tail

    def set_direction(self, direction):
        self._current_direction = direction

    def set_tail(self, tail):
        self._tail = tail


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
        self.player = Player(self.surface)
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
        self.player.draw()

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
                if (next_direction := KEY_TO_DIRECTION.get(event.key, None)):
                    print(
                        f"Setting direction to {get_key_from_value(DIRECTIONS, next_direction)}")
                    self.player.set_direction(next_direction)
            elif event.type == MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                self.player.set_pos(pos)
                self.player.set_tail([])
        self.player.update(delta_ms)
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


if __name__ == "__main__":
    mygame = Game()
    mygame.main()
