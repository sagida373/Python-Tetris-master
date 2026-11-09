import argparse
import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
]

ROTATIONS = []

for shape in SHAPES:
    variants = []

    for k in range(4):
        rotated = np.rot90(
            np.array(shape, dtype=np.uint8), -k
        ).copy()

        if not any(
            np.array_equal(rotated, old)
            for old in variants
        ):
            variants.append(rotated)

    ROTATIONS.append(variants)


def features(board):
    occupied = board != 0

    heights = np.where(
        occupied.any(axis=0),
        20 - occupied.argmax(axis=0),
        0
    )

    holes = int(
        (
            (np.cumsum(occupied, axis=0) > 0)
            & ~occupied
        ).sum()
    )

    bumpiness = int(np.abs(np.diff(heights)).sum())

    return heights, holes, bumpiness


class TetrisEnv:
    def __init__(self, max_pieces=1000):
        self.max_pieces = max_pieces
        self.rng = np.random.default_rng()
        self.ended = True

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board = np.zeros((20, 10), dtype=np.uint8)
        self.piece = int(self.rng.integers(7))

        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.ended = False

        self._placements()

        return self.state(), self.info()

    def _fits(self, shape, x, y):
        h, w = shape.shape

        return (
            0 <= x <= 10 - w
            and 0 <= y <= 20 - h
            and not np.any(
                self.board[y:y + h, x:x + w] * shape
            )
        )

    def _placements(self):
        self.placements = {}

        for rotation, shape in enumerate(
            ROTATIONS[self.piece]
        ):
            for x in range(11 - shape.shape[1]):
                if self._fits(shape, x, 0):
                    y = 0

                    while self._fits(shape, x, y + 1):
                        y += 1

                    action = rotation * 10 + x
                    self.placements[action] = (shape, x, y)

        self.mask = np.zeros(40, dtype=bool)
        self.mask[list(self.placements)] = True

    def state(self):
        heights, holes, bump = features(self.board)

        extra = np.r_[
            heights / 20,
            holes / 200,
            bump / 180,
            heights.sum() / 200,
            heights.max() / 20
        ]

        return np.r_[
            (self.board != 0).ravel(),
            np.eye(7)[self.piece],
            extra
        ].astype(np.float32)

    def info(self):
        return {
            "mask": self.mask.copy(),
            "score": self.score,
            "lines": self.lines,
            "pieces": self.pieces
        }

    def step(self, action):
        if self.ended:
            raise RuntimeError(
                "Call reset() after the episode ends."
            )

        if action not in self.placements:
            raise ValueError(f"Invalid action: {action}")

        old_heights, old_holes, old_bump = features(
            self.board
        )

        shape, x, y = self.placements[action]
        h, w = shape.shape

        self.board[y:y + h, x:x + w] |= shape

        full = self.board.all(axis=1)
        cleared = int(full.sum())

        self.board = np.vstack((
            np.zeros((cleared, 10), dtype=np.uint8),
            self.board[~full]
        ))

        self.lines += cleared
        self.score += [0, 100, 300, 700, 1500][cleared]
        self.pieces += 1

        heights, holes, bump = features(self.board)

        reward = (
            1
            + 10 * cleared ** 2
            - 0.35 * (heights.sum() - old_heights.sum())
            - 0.7 * (holes - old_holes)
            - 0.15 * (bump - old_bump)
        )

        self.piece = int(self.rng.integers(7))
        self._placements()

        terminated = not bool(self.mask.any())

        truncated = (
            self.pieces >= self.max_pieces
            and not terminated
        )

        if terminated:
            reward -= 10

        self.ended = terminated or truncated

        return (
            self.state(),
            float(reward),
            terminated,
            truncated,
            self.info()
        )


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(221, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 40)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def choose_action(model, state, mask, rng, epsilon=0.0):
    legal = np.flatnonzero(mask)

    if len(legal) == 0:
        raise ValueError("No valid actions.")

    if model is None or rng.random() < epsilon:
        return int(rng.choice(legal))

    state_tensor = torch.from_numpy(state).unsqueeze(0)
    q_values = model(state_tensor)[0]

    q_values = q_values.masked_fill(
        ~torch.from_numpy(mask),
        -torch.inf
    )

    return int(q_values.argmax().item())


def train(name, lr, steps, folder, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    rng = np.random.default_rng(seed)

    model = QNetwork()
    target = QNetwork()

    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    replay = deque(maxlen=30000)

    env = TetrisEnv()
    state, info = env.reset(seed=seed)

    history = []
    losses = []
    episode_reward = 0.0

    start = time.perf_counter()

    config = {
        "algorithm": "Double DQN",
        "lr": lr,
        "seed": seed,
        "budget": steps,
        "gamma": 0.99,
        "batch_size": 64,
        "replay_size": 30000,
        "learning_starts": 1000,
        "train_every": 4,
        "target_update": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_fraction": 0.7,
        "max_pieces": 1000,
        "network": [221, 128, 128, 40],
        "env_version": 1,
        "reward": (
            "1+10*lines^2"
            "-0.35*delta_height"
            "-0.7*delta_holes"
            "-0.15*delta_bump"
            "-10*terminal"
        )
    }

    def save(completed):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "steps_completed": completed
            },
            folder / f"{name}.pt"
        )

        (folder / f"{name}_history.json").write_text(
            json.dumps(history),
            encoding="utf-8"
        )

        (folder / f"{name}_loss.json").write_text(
            json.dumps(losses),
            encoding="utf-8"
        )

    print(
        f"Training {name}: lr={lr}, budget={steps:,}",
        flush=True
    )

    for step in range(1, steps + 1):
        epsilon = 1 - 0.95 * min(
            1.0,
            (step - 1) / (0.7 * steps)
        )

        action = choose_action(
            model,
            state,
            info["mask"],
            rng,
            epsilon
        )

        nxt, reward, terminated, truncated, next_info = (
            env.step(action)
        )

        replay.append((
            state,
            action,
            reward,
            nxt,
            terminated,
            next_info["mask"]
        ))

        state = nxt
        info = next_info
        episode_reward += reward

        if step >= 1000 and step % 4 == 0:
            batch = random.sample(replay, 64)

            states, actions, rewards, next_states, terminals, masks = (
                zip(*batch)
            )

            states = torch.from_numpy(
                np.stack(states)
            )

            next_states = torch.from_numpy(
                np.stack(next_states)
            )

            actions = torch.tensor(
                actions,
                dtype=torch.long
            ).unsqueeze(1)

            rewards = torch.tensor(
                rewards,
                dtype=torch.float32
            )

            terminals = torch.tensor(
                terminals,
                dtype=torch.bool
            )

            masks = torch.from_numpy(
                np.stack(masks)
            )

            predictions = model(states).gather(
                1, actions
            ).squeeze(1)

            with torch.no_grad():
                next_actions = (
                    model(next_states)
                    .masked_fill(~masks, -torch.inf)
                    .argmax(dim=1, keepdim=True)
                )

                next_q = target(next_states).gather(
                    1, next_actions
                ).squeeze(1)

                next_q = torch.where(
                    terminals | ~masks.any(dim=1),
                    0.0,
                    next_q
                )

                expected = rewards + 0.99 * next_q

            loss = nn.functional.smooth_l1_loss(
                predictions,
                expected
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite training loss."
                )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0
            )

            optimizer.step()

            if step % 100 == 0:
                losses.append({
                    "step": step,
                    "loss": float(loss.item())
                })

        if step % 1000 == 0:
            target.load_state_dict(
                model.state_dict()
            )

        if terminated or truncated:
            history.append({
                "step": step,
                "reward": episode_reward,
                "score": info["score"],
                "lines": info["lines"],
                "pieces": info["pieces"],
                "truncated": truncated
            })

            state, info = env.reset()
            episode_reward = 0.0

        if step % 5000 == 0:
            recent = history[-20:]

            mean_lines = (
                np.mean([row["lines"] for row in recent])
                if recent else 0.0
            )

            elapsed = time.perf_counter() - start

            print(
                f"{name}: {step:,}/{steps:,} | "
                f"eps={epsilon:.2f} | "
                f"lines(last 20)={mean_lines:.2f} | "
                f"{elapsed / 60:.1f} min",
                flush=True
            )

            save(step)

    config["training_seconds"] = (
        time.perf_counter() - start
    )

    save(steps)
    model.eval()

    return model, history


def evaluate(model, name, episodes):
    rows = []

    for seed in range(10000, 10000 + episodes):
        env = TetrisEnv()
        state, info = env.reset(seed=seed)

        rng = np.random.default_rng(seed + 1000)
        total_reward = 0.0

        while True:
            action = choose_action(
                model,
                state,
                info["mask"],
                rng,
                epsilon=0.0
            )

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            if terminated or truncated:
                rows.append({
                    "agent": name,
                    "seed": seed,
                    "reward": total_reward,
                    "score": info["score"],
                    "lines": info["lines"],
                    "pieces": info["pieces"],
                    "truncated": truncated
                })

                break

    return rows


def write_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_curves(histories, folder):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4)
    )

    for name, history in histories.items():
        if not history:
            continue

        steps = np.array([
            row["step"] for row in history
        ])

        for ax, metric in zip(
            axes,
            ["reward", "lines"]
        ):
            values = np.array([
                row[metric] for row in history
            ])

            window = min(20, len(values))

            smoothed = np.convolve(
                values,
                np.ones(window) / window,
                mode="valid"
            )

            line, = ax.plot(
                steps[window - 1:],
                smoothed,
                label=name
            )

            ax.plot(
                steps,
                values,
                alpha=0.15,
                color=line.get_color()
            )

            ax.set_xlabel(
                "Environment steps (placed pieces)"
            )

            ax.set_ylabel(
                metric.title() + " per episode"
            )

            ax.grid(alpha=0.2)
            ax.legend()

    fig.suptitle(
        "Training: completed episodes; "
        "moving average up to 20 episodes"
    )

    fig.tight_layout()

    fig.savefig(
        folder / "training_curves.png",
        dpi=160
    )

    plt.close(fig)


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["config"]["env_version"] != 1:
        raise ValueError(
            "Different environment version."
        )

    model = QNetwork()

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    if args.smoke:
        args.steps = 2000
        args.episodes = 10

    if args.steps < 1000 or args.episodes < 10:
        parser.error(
            "Use >=1000 steps and >=10 test episodes."
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent / "results"

    prefix = (
        "smoke_"
        if args.smoke
        else "experiment_"
    )

    folder = root / (prefix + stamp)
    folder.mkdir(parents=True, exist_ok=False)

    histories = {}
    all_rows = []
    summary = []

    experiments = [
        ("ddqn_a", 0.001),
        ("ddqn_b", 0.0003)
    ]

    for name, lr in experiments:
        _, histories[name] = train(
            name=name,
            lr=lr,
            steps=args.steps,
            folder=folder
        )

        saved_model = load_model(
            folder / f"{name}.pt"
        )

        all_rows.extend(
            evaluate(
                saved_model,
                name,
                args.episodes
            )
        )

    all_rows.extend(
        evaluate(
            None,
            "random",
            args.episodes
        )
    )

    for name in ["ddqn_a", "ddqn_b", "random"]:
        rows = [
            row for row in all_rows
            if row["agent"] == name
        ]

        result = {
            "agent": name,
            "episodes": len(rows)
        }

        for metric in [
            "reward",
            "score",
            "lines",
            "pieces"
        ]:
            values = [
                row[metric] for row in rows
            ]

            result[metric + "_mean"] = float(
                np.mean(values)
            )

            result[metric + "_std"] = float(
                np.std(values, ddof=1)
            )

        result["truncated_episodes"] = sum(
            row["truncated"] for row in rows
        )

        summary.append(result)

        print(
            f"{name}: "
            f"reward={result['reward_mean']:.2f} "
            f"+/- {result['reward_std']:.2f}, "
            f"lines={result['lines_mean']:.2f}, "
            f"score={result['score_mean']:.2f}"
        )

    write_csv(
        folder / "test_episodes.csv",
        all_rows
    )

    write_csv(
        folder / "comparison.csv",
        summary
    )

    plot_curves(histories, folder)

    print(f"Results saved to: {folder}")

    if args.smoke:
        print(
            "SMOKE TEST ONLY. "
            "These are not final training results."
        )


if __name__ == "__main__":
    main()