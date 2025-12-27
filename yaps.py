import pygame as pg

from pygame.event import MOUSEBUTTONDOWN, QUIT, KEYDOWN, K_ESCAPE


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

    def draw(self):
        pg.draw.circle(self.surface, (255, 255, 255), self.pos, 40)

    def move(self, target):
        x = (80 * (target[0] // 80)) + 40
        y = (80 * (target[1] // 80)) + 40

        self.pos = (x, y)


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

    def main(self):
        while self.loop:
            self.grid_loop()
        # pylint: disable=no-member
        # TODO: Hva skjer med lintingen?
        pg.quit()
        # pylint: enable=no-member

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

        for event in pg.event.get():
            if event.type == QUIT:
                self.loop = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.loop = False
            elif event.type == MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                self.player.move(pos)
        pg.display.update()


if __name__ == "__main__":
    mygame = Game()
    mygame.main()
