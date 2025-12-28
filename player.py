from collections import deque
from itertools import islice
from pygame import Surface
from helpers import DEFAULTS, DIRECTIONS, Coordinate, Direction, Tail, direction_change_is_legal, get_key_from_value


class Player:

    def __init__(self, surface: Surface, width: int, height: int, tile_size: int):
        self.surface: Surface = surface
        self.width = width
        self.height = height
        self._tile_size = tile_size
        self._half_tile = max(1, tile_size // 2)
        self._spawn_pos = (self._half_tile, self._half_tile)
        self._pos = self._spawn_pos
        self._tail: Tail = deque()
        self._current_direction = DEFAULTS["direction"]
        self._autopilot_enabled = DEFAULTS["autopilot_enabled"]
        # Movement cadence state so framerate and speed can be tuned independently
        self._move_delay = 150  # milliseconds between steps
        self._move_accumulator = 0
        self._pending_tail_owners: deque[bool] = deque()

    def set_pos(self, target: Coordinate) -> None:
        snapped_target = self._snap_to_grid(target)
        self._pos = snapped_target
        if self._tail:
            self._tail.pop()
        self._tail.appendleft((self._pos, self._autopilot_enabled))

    def get_pos(self):
        return self._pos

    def get_next_pos(self, old_pos: Coordinate) -> Coordinate:
        step = self._tile_size
        offset = self._half_tile
        old_x, old_y = old_pos
        snapped_x = ((old_x - offset) // step) * step + offset
        snapped_y = ((old_y - offset) // step) * step + offset

        new_x = snapped_x + self._current_direction[0] * step
        new_y = snapped_y + self._current_direction[1] * step

        # wrap around logic
        if new_x < offset:
            new_x = self.width - offset
        elif new_x > self.width - offset:
            new_x = offset

        if new_y < offset:
            new_y = self.height - offset
        elif new_y > self.height - offset:
            new_y = offset

        return (new_x, new_y)

    def move(self) -> None:
        prev_pos = self._pos
        new_pos = self.get_next_pos(prev_pos)

        if self._pending_tail_owners:
            while self._pending_tail_owners:
                owner = self._pending_tail_owners.popleft()
                self._tail.appendleft((prev_pos, owner))
        elif self._tail:
            owners = [owner for _, owner in self._tail]
            positions = [prev_pos]
            positions.extend(
                pos for pos, _ in islice(self._tail, 0, len(self._tail) - 1)
            )

            self._tail.clear()
            self._tail.extend(zip(positions, owners))

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

    def reset_tail(self) -> None:
        self._tail.clear()
        self._pending_tail_owners.clear()

    def toggle_autopilot(self, override: bool = None) -> None:  # type: ignore
        if override is not None:
            self._autopilot_enabled = override
        else:
            self._autopilot_enabled = not self._autopilot_enabled

    def is_autopilot_enabled(self) -> bool:
        return self._autopilot_enabled

    def get_spawn_pos(self) -> Coordinate:
        return self._spawn_pos

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

    def _snap_to_grid(self, coord: Coordinate) -> Coordinate:
        step = self._tile_size
        offset = self._half_tile
        clamped_x = min(max(coord[0], offset), self.width - offset)
        clamped_y = min(max(coord[1], offset), self.height - offset)
        snapped_x = ((clamped_x - offset) // step) * step + offset
        snapped_y = ((clamped_y - offset) // step) * step + offset
        return (snapped_x, snapped_y)
