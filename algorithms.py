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

    occupied = _tail_cells(tail, tile_size, cols, rows)

    bfs_direction = _direction_via_bfs(
        head, fruit, occupied, tile_size, cols, rows)
    if bfs_direction is not None:
        return bfs_direction

    return _greedy_direction(head, fruit, occupied, tile_size, cols, rows)


def _direction_via_bfs(
    head: Coordinate,
    fruit: Coordinate,
    occupied: set[GridCoord],
    tile_size: int,
    cols: int,
    rows: int,
) -> Optional[Direction]:
    start = _to_grid(head, tile_size, cols, rows)
    goal = _to_grid(fruit, tile_size, cols, rows)

    if start == goal:
        return None

    queue = deque([start])
    parents: Dict[GridCoord, Optional[GridCoord]] = {start: None}
    moves: Dict[GridCoord, Direction] = {}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for step in DIRECTIONS.values():
            neighbor = _neighbor(current, step, cols, rows)
            if neighbor is None or neighbor in parents:
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


def _neighbor(position: GridCoord, delta: Direction, cols: int, rows: int) -> Optional[GridCoord]:
    col = position[0] + delta[0]
    row = position[1] + delta[1]
    if 0 <= col < cols and 0 <= row < rows:
        return (col, row)
    return None


def _tail_cells(tail: Tail, tile_size: int, cols: int, rows: int) -> set[GridCoord]:
    return {
        _to_grid(segment_pos, tile_size, cols, rows)
        for segment_pos, _ in tail
    }


def _to_grid(coord: Coordinate, tile_size: int, cols: int, rows: int) -> GridCoord:
    offset = tile_size // 2
    col = int((coord[0] - offset) // tile_size)
    row = int((coord[1] - offset) // tile_size)
    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))
    return (col, row)


def _greedy_direction(
    head: Coordinate,
    fruit: Coordinate,
    occupied: set[GridCoord],
    tile_size: int,
    cols: int,
    rows: int,
) -> Optional[Direction]:
    head_grid = _to_grid(head, tile_size, cols, rows)
    fruit_grid = _to_grid(fruit, tile_size, cols, rows)

    step_x = _linear_step(head_grid[0], fruit_grid[0])
    if step_x != 0:
        direction = DIRECTIONS["RIGHT"] if step_x > 0 else DIRECTIONS["LEFT"]
        target = _neighbor(head_grid, direction, cols, rows)
        if target is not None and target not in occupied:
            return direction

    step_y = _linear_step(head_grid[1], fruit_grid[1])
    if step_y != 0:
        direction = DIRECTIONS["DOWN"] if step_y > 0 else DIRECTIONS["UP"]
        target = _neighbor(head_grid, direction, cols, rows)
        if target is not None and target not in occupied:
            return direction

    return None


def _linear_step(current: int, target: int) -> int:
    if target > current:
        return 1
    if target < current:
        return -1
    return 0
