"""Pathfinding helpers for the Snake autopilot."""

from collections import deque
from typing import Dict, Optional, Tuple

from helpers import DIRECTIONS, Coordinate, Direction, Tail

GridCoord = Tuple[int, int]


def compute_autopilot_direction(
    head: Coordinate,
    fruit: Optional[Coordinate],
    tail: Tail,
    tile_size: int,
    cols: int,
    rows: int,
) -> Optional[Direction]:
    """Return the next direction the autopilot should take."""
    if fruit is None:
        return None

    bfs_direction = _direction_via_bfs(
        head, fruit, tail, tile_size, cols, rows)
    if bfs_direction is not None:
        return bfs_direction

    return _greedy_direction(head, fruit, tile_size, cols, rows)


def _direction_via_bfs(
    head: Coordinate,
    fruit: Coordinate,
    tail: Tail,
    tile_size: int,
    cols: int,
    rows: int,
) -> Optional[Direction]:
    start = _to_grid(head, tile_size, cols, rows)
    goal = _to_grid(fruit, tile_size, cols, rows)

    if start == goal:
        return None

    occupied = _tail_cells(tail, tile_size, cols, rows)
    queue = deque([start])
    parents: Dict[GridCoord, Optional[GridCoord]] = {start: None}
    moves: Dict[GridCoord, Direction] = {}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for step in DIRECTIONS.values():
            neighbor = _wrap_add(current, step, cols, rows)
            if neighbor in parents:
                continue
            if neighbor in occupied and neighbor != goal:
                continue
            parents[neighbor] = current
            moves[neighbor] = step
            queue.append(neighbor)

    if goal not in parents:
        return None

    node = goal
    direction: Optional[Direction] = None
    while True:
        parent = parents[node]
        if parent is None:
            break
        direction = moves[node]
        node = parent

    return direction


def _wrap_add(position: GridCoord, delta: Direction, cols: int, rows: int) -> GridCoord:
    col = (position[0] + delta[0]) % cols
    row = (position[1] + delta[1]) % rows
    return (col, row)


def _tail_cells(tail: Tail, tile_size: int, cols: int, rows: int) -> set[GridCoord]:
    return {
        _to_grid(segment_pos, tile_size, cols, rows)
        for segment_pos, _ in tail
    }


def _to_grid(coord: Coordinate, tile_size: int, cols: int, rows: int) -> GridCoord:
    offset = tile_size // 2
    col = ((coord[0] - offset) // tile_size) % cols
    row = ((coord[1] - offset) // tile_size) % rows
    return (int(col), int(row))


def _greedy_direction(
    head: Coordinate,
    fruit: Coordinate,
    tile_size: int,
    cols: int,
    rows: int,
) -> Optional[Direction]:
    head_grid = _to_grid(head, tile_size, cols, rows)
    fruit_grid = _to_grid(fruit, tile_size, cols, rows)

    step_x = _wrapped_step(head_grid[0], fruit_grid[0], cols)
    if step_x > 0:
        return DIRECTIONS["RIGHT"]
    if step_x < 0:
        return DIRECTIONS["LEFT"]

    step_y = _wrapped_step(head_grid[1], fruit_grid[1], rows)
    if step_y > 0:
        return DIRECTIONS["DOWN"]
    if step_y < 0:
        return DIRECTIONS["UP"]

    return None


def _wrapped_step(current: int, target: int, limit: int) -> int:
    if limit <= 1 or current == target:
        return 0
    forward = (target - current) % limit
    backward = (current - target) % limit
    if forward == 0:
        return 0
    if forward <= backward:
        return 1
    return -1
