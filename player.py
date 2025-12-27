from helpers import DEFAULTS


class Player:

    def __init__(self, surface, width, height):
        self.surface = surface
        self.width = width
        self.height = height
        self._pos = DEFAULTS["player_pos"]
        self._tail = []
        self._current_direction = DEFAULTS["direction"]
        # Movement cadence state so framerate and speed can be tuned independently
        self._move_delay = 150  # milliseconds between steps
        self._move_accumulator = 0

        # TODO: Wack hack...
        self._should_eat = False

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
            new_x = self.width - 40
        elif new_x > self.width - 40:
            new_x = 40

        if new_y < 40:
            new_y = self.height - 40
        elif new_y > self.height - 40:
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

    def get_direction(self):
        return self._current_direction

    def set_tail(self, tail):
        self._tail = tail

    def get_tail(self):
        return self._tail
