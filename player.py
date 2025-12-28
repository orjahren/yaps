from collections import deque
from pygame import Surface
from helpers import DEFAULTS, DIRECTIONS, Coordinate, Direction, Tail, direction_change_is_legal, get_key_from_value


class Player:

    def __init__(self, surface: Surface, width: int, height: int):
        self.surface: Surface = surface
        self.width = width
        self.height = height
        self._pos = DEFAULTS["player_pos"]
        self._tail: Tail = []
        self._current_direction = DEFAULTS["direction"]
        self._autopilot_enabled = DEFAULTS["autopilot_enabled"]
        # Movement cadence state so framerate and speed can be tuned independently
        self._move_delay = 150  # milliseconds between steps
        self._move_accumulator = 0
        self._pending_tail_owners: deque[bool] = deque()

        # TODO: Wack hack...
        self._should_eat = False

    def set_pos(self, target: Coordinate) -> None:
        self._pos = target
        if self._tail:
            self._tail = [(self._pos, self._autopilot_enabled)
                          ] + self._tail[:-1]

    def get_pos(self):
        return self._pos

    def get_next_pos(self, old_pos: Coordinate) -> Coordinate:
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

    def move(self) -> None:
        prev_pos = self._pos
        new_pos = self.get_next_pos(prev_pos)

        if self._pending_tail_owners:
            while self._pending_tail_owners:
                owner = self._pending_tail_owners.popleft()
                self._tail.insert(0, (prev_pos, owner))
        elif self._tail:
            previous_positions = [prev_pos] + \
                [pos for pos, _ in self._tail[:-1]]
            self._tail = [
                (position, owner)
                for position, (_, owner) in zip(previous_positions, self._tail)
            ]

        self._pos = new_pos

    # TODO: Er dette måten å gjøre det på??
    def update(self, delta_ms: int, fruit_coords: Coordinate) -> None:
        """Advance the player when enough time has elapsed."""
        self._move_accumulator += delta_ms
        if self._move_accumulator < self._move_delay:
            return
        # Preserve leftover time so slight jitter is averaged out
        self._move_accumulator %= self._move_delay

        if self._autopilot_enabled:
            next_direction = self._get_next_direction(fruit_coords)
            if direction_change_is_legal(self.get_direction(), next_direction):
                pretty_dir = get_key_from_value(DIRECTIONS, next_direction)
                print(f"Autopilot changing direction to {pretty_dir}")
                self.set_direction(next_direction)

        self.move()

    def grow(self, by_autopilot: bool) -> None:
        self._pending_tail_owners.append(by_autopilot)

    def should_die(self) -> bool:
        return self._pos in (pos for pos, _ in self._tail)

    def set_direction(self, direction: Direction) -> None:
        self._current_direction = direction

    def get_direction(self) -> Direction:
        return self._current_direction

    def set_tail(self, tail: Tail) -> None:
        self._tail = tail

    def get_tail(self) -> Tail:
        return self._tail

    def toggle_autopilot(self, override: bool = None) -> None:  # type: ignore
        if override is not None:
            self._autopilot_enabled = override
        else:
            self._autopilot_enabled = not self._autopilot_enabled

    def is_autopilot_enabled(self) -> bool:
        return self._autopilot_enabled

    def _get_next_direction(self, fruit_coords: Coordinate) -> Direction:

        if fruit_coords:
            fruit_x, fruit_y = fruit_coords
            npc_x, npc_y = self.get_pos()

            if fruit_x > npc_x:
                return DIRECTIONS["RIGHT"]
            elif fruit_x < npc_x:
                return DIRECTIONS["LEFT"]
            elif fruit_y > npc_y:
                return DIRECTIONS["DOWN"]
            elif fruit_y < npc_y:
                return DIRECTIONS["UP"]

            print("NPC is already at the fruit position.")
            return self.get_direction()
