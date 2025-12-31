"""Runtime inference helper for the DQN-based Snake autopilot."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch

from helpers import Coordinate, Direction, Tail, DIRECTIONS
from ml import DIRECTION_ORDER, LinearQNet, SnakeEnv, DEVICE


class MLAutopilot:
    """Wrap a trained DQN checkpoint and expose direction suggestions."""

    def __init__(
        self,
        model_path: str,
        cols: int,
        rows: int,
        tile_size: int,
        hidden_size: int = 256,
    ) -> None:
        self._path = Path(model_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Model weights not found: {self._path}")

        self._cols = cols
        self._rows = rows
        self._tile_size = tile_size
        self._half_tile = max(1, tile_size // 2)
        self._direction_lookup = {
            DIRECTIONS["RIGHT"]: 0,
            DIRECTIONS["DOWN"]: 1,
            DIRECTIONS["LEFT"]: 2,
            DIRECTIONS["UP"]: 3,
        }
        self._directions = [
            DIRECTIONS["RIGHT"],
            DIRECTIONS["DOWN"],
            DIRECTIONS["LEFT"],
            DIRECTIONS["UP"],
        ]

        self._model = LinearQNet(
            SnakeEnv.state_size, hidden_size, SnakeEnv.action_size
        ).to(DEVICE)
        state_dict = torch.load(self._path, map_location=DEVICE)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def suggest_direction(
        self,
        head: Coordinate,
        fruit: Optional[Coordinate],
        direction: Direction,
        tail: Tail,
        pending_growth: bool,
    ) -> Optional[Direction]:
        if fruit is None:
            return None
        state = self._build_state(head, fruit, direction, tail, pending_growth)
        if state is None:
            return None
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self._model(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return self._action_to_direction(direction, action)

    def _build_state(
        self,
        head: Coordinate,
        fruit: Coordinate,
        direction: Direction,
        tail: Tail,
        pending_growth: bool,
    ) -> Optional[List[float]]:
        head_cell = self._to_grid(head)
        fruit_cell = self._to_grid(fruit)
        if head_cell is None or fruit_cell is None:
            return None

        body_cells: List[tuple[int, int]] = [head_cell]
        for pos, _ in tail:
            cell = self._to_grid(pos)
            if cell is not None:
                body_cells.append(cell)
        if not body_cells:
            body_cells.append(head_cell)

        dir_idx = self._direction_lookup.get(direction)
        if dir_idx is None:
            return None
        left_idx = (dir_idx - 1) % len(self._directions)
        right_idx = (dir_idx + 1) % len(self._directions)

        dangers = [
            self._danger(dir_idx, head_cell, body_cells, pending_growth),
            self._danger(left_idx, head_cell, body_cells, pending_growth),
            self._danger(right_idx, head_cell, body_cells, pending_growth),
        ]

        direction_flags = [int(dir_idx == idx) for idx in range(4)]
        fruit_flags = [
            int(fruit_cell[0] > head_cell[0]),
            int(fruit_cell[0] < head_cell[0]),
            int(fruit_cell[1] > head_cell[1]),
            int(fruit_cell[1] < head_cell[1]),
        ]

        return [*(int(flag) for flag in dangers), *direction_flags, *fruit_flags]

    def _action_to_direction(self, current: Direction, action: int) -> Direction:
        idx = self._direction_lookup.get(current, 0)
        if action == 1:
            idx = (idx - 1) % len(self._directions)
        elif action == 2:
            idx = (idx + 1) % len(self._directions)
        return self._directions[idx]

    def _danger(
        self,
        direction_idx: int,
        head_cell: tuple[int, int],
        body_cells: Sequence[tuple[int, int]],
        pending_growth: bool,
    ) -> bool:
        dx, dy = DIRECTION_ORDER[direction_idx]
        next_cell = (head_cell[0] + dx, head_cell[1] + dy)
        if not self._is_within_bounds(next_cell):
            return True
        if len(body_cells) == 1:
            return False
        occupied = set(body_cells[1:])
        tail_end = body_cells[-1]
        if not pending_growth:
            occupied.discard(tail_end)
        return next_cell in occupied

    def _to_grid(self, coord: Coordinate) -> Optional[tuple[int, int]]:
        x, y = coord
        col = int((x - self._half_tile) // self._tile_size)
        row = int((y - self._half_tile) // self._tile_size)
        if not (0 <= col < self._cols and 0 <= row < self._rows):
            return None
        return (col, row)

    def _is_within_bounds(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        return 0 <= col < self._cols and 0 <= row < self._rows
