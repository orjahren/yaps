import pygame as pg
import random

# pylint: disable=no-name-in-module
# TODO: Hva skjer med lintingen?
from pygame.constants import MOUSEBUTTONDOWN, QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_SPACE
# pylint: enable=no-name-in-module


TITLE = "Grid"
TILES_HORIZONTAL = 10
TILES_VERTICAL = 10
TILE_SIZE = 80
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800


class Player:
    def __init__(self, surface):
        self.surface = surface
        self.pos = (40, 40)
        self._tail = []

    def draw(self):
        # Draw tail
        for segment in self._tail:
            pg.draw.circle(self.surface, (200, 200, 200), segment, 20)
        # Draw head
        pg.draw.circle(self.surface, (255, 255, 255), self.pos, 40)

    def move(self, target):
        x = (80 * (target[0] // 80)) + 40
        y = (80 * (target[1] // 80)) + 40

        self.pos = (x, y)

    def grow(self):
        self._tail.append(self.pos)


class Game:
    def __init__(self):
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.init()
        # pylint: enable=no-member
        self.clock = pg.time.Clock()
        pg.display.set_caption(TITLE)
        self.surface = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.loop = True
        self.player = Player(self.surface)
        self._current_fruit = None, None

    def main(self):
        while self.loop:
            self.grid_loop()
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.quit()
        # pylint: enable=no-member

    def spawn_fruit(self):
        pass

    def grid_loop(self):
        self.surface.fill((0, 0, 0))
        for row in range(TILES_HORIZONTAL):
            for col in range(row % 2, TILES_HORIZONTAL, 2):
                pg.draw.rect(
                    self.surface,
                    (40, 40, 40),
                    (row * TILE_SIZE, col * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )
        self.player.draw()

        # Fruit logic
        if self.player.pos == self._current_fruit:
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
                    self.player.move((40, 40))
                elif event.key == K_LEFT:
                    x, y = self.player.pos
                    self.player.move((x - TILE_SIZE, y))
                elif event.key == K_RIGHT:
                    x, y = self.player.pos
                    self.player.move((x + TILE_SIZE, y))
                elif event.key == K_UP:
                    x, y = self.player.pos
                    self.player.move((x, y - TILE_SIZE))
                elif event.key == K_DOWN:
                    x, y = self.player.pos
                    self.player.move((x, y + TILE_SIZE))
            elif event.type == MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                self.player.move(pos)
        pg.display.update()


if __name__ == "__main__":
    mygame = Game()
    mygame.main()
