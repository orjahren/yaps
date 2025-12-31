"""Train a Deep Q-Network agent for the YAPS Snake game."""
from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn, optim

Vector = Tuple[int, int]
DIRECTION_ORDER: Tuple[Vector, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTION_LABELS = ("straight", "turn_left", "turn_right")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _wrap_index(value: int, modulo: int) -> int:
    return (value + modulo) % modulo


class SnakeEnv:
    """Lightweight Snake simulator that mirrors the PyGame rules."""

    state_size = 11
    action_size = 3

    def __init__(self, cols: int = 10, rows: int = 10, seed: Optional[int] = None):
        if cols < 4 or rows < 4:
            raise ValueError("Need at least a 4x4 board for training.")
        self.cols = cols
        self.rows = rows
        self.random = random.Random(seed)
        self.max_idle_steps = cols * rows * 2
        self.apple_reward = 10.0
        self.death_penalty = -10.0
        self.step_penalty = -0.02
        self.timeout_penalty = -5.0
        self._snake: Deque[Vector] = deque()
        self._fruit: Optional[Vector] = None
        self._direction_idx = 0
        self.score = 0
        self.steps_since_reward = 0
        self.reset()

    def reset(self) -> List[float]:
        start_x = max(2, self.cols // 2)
        start_y = self.rows // 2
        self._direction_idx = 0
        self._snake = deque([
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ])
        self.score = 0
        self.steps_since_reward = 0
        self._place_fruit()
        return self._state()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, int]]:
        if action not in (0, 1, 2):
            raise ValueError(f"Action must be in [0, 2], got {action}.")

        if action == 1:
            self._direction_idx = _wrap_index(
                self._direction_idx - 1, len(DIRECTION_ORDER))
        elif action == 2:
            self._direction_idx = _wrap_index(
                self._direction_idx + 1, len(DIRECTION_ORDER))

        next_pos = self._next_position(self._snake[0], self._direction_idx)
        reward = self.step_penalty
        done = False

        if self._collision(next_pos):
            return self._state(), self.death_penalty, True, self._info()

        self._snake.appendleft(next_pos)
        ate = self._fruit is not None and next_pos == self._fruit
        if ate:
            reward += self.apple_reward
            self.score += 1
            self.steps_since_reward = 0
            self._place_fruit()
        else:
            self._snake.pop()
            self.steps_since_reward += 1

        if self.steps_since_reward > self.max_idle_steps:
            reward += self.timeout_penalty
            done = True

        if len(self._snake) == self.cols * self.rows:
            reward += self.apple_reward
            done = True

        return self._state(), reward, done, self._info()

    def _collision(self, pos: Vector) -> bool:
        x, y = pos
        if x < 0 or y < 0 or x >= self.cols or y >= self.rows:
            return True
        tail_end = self._snake[-1]
        return pos in self._snake and pos != tail_end

    def _next_position(self, origin: Vector, direction_idx: int) -> Vector:
        dx, dy = DIRECTION_ORDER[direction_idx]
        return origin[0] + dx, origin[1] + dy

    def _place_fruit(self) -> None:
        open_cells = [
            (x, y)
            for x in range(self.cols)
            for y in range(self.rows)
            if (x, y) not in self._snake
        ]
        self._fruit = self.random.choice(open_cells) if open_cells else None

    def _state(self) -> List[float]:
        head = self._snake[0]
        left_idx = _wrap_index(self._direction_idx - 1, len(DIRECTION_ORDER))
        right_idx = _wrap_index(self._direction_idx + 1, len(DIRECTION_ORDER))
        fruit = self._fruit or head

        dangers = [
            self._danger(self._direction_idx),
            self._danger(left_idx),
            self._danger(right_idx),
        ]

        direction_flags = [
            int(self._direction_idx == 0),
            int(self._direction_idx == 1),
            int(self._direction_idx == 2),
            int(self._direction_idx == 3),
        ]

        fruit_flags = [
            int(fruit[0] > head[0]),
            int(fruit[0] < head[0]),
            int(fruit[1] > head[1]),
            int(fruit[1] < head[1]),
        ]

        return [*(int(flag) for flag in dangers), *direction_flags, *fruit_flags]

    def _danger(self, direction_idx: int) -> bool:
        next_pos = self._next_position(self._snake[0], direction_idx)
        return self._collision(next_pos)

    def _info(self) -> Dict[str, int]:
        return {"score": self.score, "length": len(self._snake)}


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: Deque[Tuple[List[float], int, float,
                                 List[float], bool]] = deque(maxlen=capacity)

    def push(self, transition: Tuple[List[float], int, float, List[float], bool]) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)


class LinearQNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # type: ignore[override]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        gamma: float = 0.95,
        lr: float = 1e-3,
        batch_size: int = 256,
        replay_capacity: int = 50_000,
        target_sync: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 10_000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(replay_capacity)
        self.policy_net = LinearQNet(
            state_size, hidden_size, action_size).to(DEVICE)
        self.target_net = LinearQNet(
            state_size, hidden_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.target_sync = target_sync
        self.learn_steps = 0
        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self._sync_target()

    def act(self, state: Sequence[float], explore: bool = True) -> int:
        epsilon = self.current_epsilon() if explore else 0.0
        if explore:
            self.steps_done += 1
        if explore and random.random() < epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: List[float], action: int, reward: float, next_state: List[float], done: bool) -> None:
        self.replay.push((state, action, reward, next_state, done))

    def learn(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size)
        state_batch = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(
            actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(
            rewards, dtype=torch.float32, device=DEVICE)
        next_state_batch = torch.tensor(
            next_states, dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        q_values = self.policy_net(state_batch).gather(
            1, action_batch).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(dim=1)[0]
        targets = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=2.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self._sync_target()
        return float(loss.item())

    def current_epsilon(self) -> float:
        decay = math.exp(-max(0, self.steps_done) / max(1, self.eps_decay))
        return self.eps_end + (self.eps_start - self.eps_end) * decay

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(state_dict)
        self._sync_target()


@dataclass
class EpisodeResult:
    episode: int
    score: int
    length: int
    reward: float


def run_training(args: argparse.Namespace) -> List[EpisodeResult]:
    env = SnakeEnv(cols=args.cols, rows=args.rows, seed=args.seed)
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_sync=args.target_sync,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
    )
    model_path = Path(args.model_path)
    history: List[EpisodeResult] = []
    rolling_rewards: Deque[float] = deque(maxlen=max(1, args.log_window))
    best_score = -1

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
        info = info or {}  # type: ignore TODO: Dette er wack
        rolling_rewards.append(total_reward)
        result = EpisodeResult(
            episode, info["score"], info["length"], total_reward)
        history.append(result)
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save(model_path)
        if episode % args.log_every == 0 or episode == 1:
            avg_reward = sum(rolling_rewards) / len(rolling_rewards)
            print(
                f"Episode {episode:4d} | score={info['score']:2d} | "
                f"length={info['length']:3d} | reward={total_reward:7.2f} | "
                f"avg_reward={avg_reward:7.2f} | epsilon={agent.current_epsilon():.3f}"
            )

    print(f"Best score {best_score} saved to {model_path}")
    agent.load(model_path)
    if args.eval_games > 0:
        evaluate(agent, env, args.eval_games)
    return history


def evaluate(agent: DQNAgent, env: SnakeEnv, episodes: int) -> None:
    scores: List[int] = []
    for idx in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, explore=False)
            state, _, done, info = env.step(action)
        info = info or {}  # type: ignore TODO: Dette er wack
        scores.append(info["score"])
        print(
            f"Eval game {idx + 1}: score={info['score']} length={info['length']}")
    mean_score = sum(scores) / len(scores)
    print(f"Average evaluation score over {episodes} games: {mean_score:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for YAPS Snake.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes.")
    parser.add_argument("--cols", type=int, default=10,
                        help="Board width in tiles.")
    parser.add_argument("--rows", type=int, default=10,
                        help="Board height in tiles.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed.")
    parser.add_argument("--hidden-size", type=int,
                        default=256, help="Hidden layer width.")
    parser.add_argument("--batch-size", type=int,
                        default=256, help="Batch size for SGD.")
    parser.add_argument("--replay-capacity", type=int,
                        default=50_000, help="Replay buffer capacity.")
    parser.add_argument("--target-sync", type=int, default=1000,
                        help="Steps between target net syncs.")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--eps-start", type=float,
                        default=1.0, help="Initial epsilon.")
    parser.add_argument("--eps-end", type=float,
                        default=0.05, help="Final epsilon.")
    parser.add_argument("--eps-decay", type=int,
                        default=5000, help="Epsilon decay speed.")
    parser.add_argument("--log-every", type=int, default=25,
                        help="How often to print stats.")
    parser.add_argument("--log-window", type=int, default=50,
                        help="Episodes used for rolling reward average.")
    parser.add_argument("--eval-games", type=int, default=5,
                        help="Run greedy evaluation after training.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/snake_dqn.pt",
        help="Where to store the trained weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
