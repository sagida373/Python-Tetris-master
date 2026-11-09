import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_agent import (
    TetrisEnv,
    QNetwork,
    choose_action,
    plot_curves,
    write_csv
)


def pack_replay(replay):
    states, actions, rewards, next_states, terminals, masks = zip(*replay)

    return {
        "states": torch.from_numpy(np.stack(states)),
        "actions": torch.tensor(actions),
        "rewards": torch.tensor(rewards, dtype=torch.float64),
        "next_states": torch.from_numpy(np.stack(next_states)),
        "terminals": torch.tensor(terminals),
        "masks": torch.from_numpy(np.stack(masks))
    }


def unpack_replay(data):
    keys = [
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminals",
        "masks"
    ]

    arrays = [data[key].numpy() for key in keys]

    return deque(
        (
            (
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                bool(terminal),
                mask.copy()
            )
            for state, action, reward, next_state, terminal, mask
            in zip(*arrays)
        ),
        maxlen=30000
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--agent", choices=["a", "b"], default="b")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--episodes", type=int, default=20)

    args = parser.parse_args()

    if args.steps < 1000 or args.episodes < 10:
        parser.error(
            "Use at least 1000 additional steps and 10 test episodes."
        )

    root = Path(__file__).resolve().parent

    if args.model is None:
        models = sorted(
            (root / "results").glob(
                f"experiment_*/ddqn_{args.agent}.pt"
            )
        )

        if not models:
            parser.error(
                "No saved model found in results/experiment_*."
            )

        source = models[-1]

        full_path = source.with_name(
            source.stem + ".resume.pt"
        )

        if full_path.is_file():
            source = full_path
    else:
        source = args.model.resolve()

    print(f"Loading: {source}", flush=True)

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=True
    )

    config = dict(checkpoint["config"])

    if config["env_version"] != 1:
        parser.error("Incompatible environment version.")

    name = (
        source.name
        .removesuffix(".resume.pt")
        .removesuffix(".pt")
    )

    start_step = int(checkpoint["steps_completed"])
    end_step = start_step + args.steps

    torch.set_num_threads(1)

    seed = int(config["seed"]) + 1

    random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed)

    model = QNetwork()
    target = QNetwork()

    model.load_state_dict(checkpoint["state_dict"])
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    env = TetrisEnv(
        max_pieces=config.get("max_pieces", 1000)
    )

    state, info = env.reset(seed=seed)

    replay = deque(maxlen=30000)

    episode_reward = 0.0
    history = []
    losses = []

    full_resume = "resume_state" in checkpoint

    if full_resume:
        saved = checkpoint["resume_state"]

        target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])

        replay = unpack_replay(saved["replay"])

        env.board = saved["board"].numpy().copy()

        for key in ["piece", "score", "lines", "pieces"]:
            setattr(env, key, int(saved[key]))

        env.ended = False

        env.rng.bit_generator.state = json.loads(
            saved["env_rng"]
        )

        env._placements()

        state = env.state()
        info = env.info()

        episode_reward = saved["episode_reward"]
        history = saved["history"]
        losses = saved["losses"]

        rng.bit_generator.state = json.loads(
            saved["action_rng"]
        )

        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])

        schedule_start = saved["schedule_start"]
        schedule_length = saved["schedule_length"]
        epsilon_start = saved["epsilon_start"]

        print(
            "Full training state restored.",
            flush=True
        )

    else:
        for suffix, destination in [
            ("_history.json", history),
            ("_loss.json", losses)
        ]:
            path = source.with_name(name + suffix)

            if path.is_file():
                destination.extend(
                    json.loads(
                        path.read_text(encoding="utf-8")
                    )
                )

        schedule_start = start_step
        schedule_length = max(
            1,
            int(args.steps * 0.7)
        )

        epsilon_start = 0.10

        print(
            "Weights restored. Replay and optimizer "
            "start fresh; epsilon=0.10.",
            flush=True
        )

    folder = root / "results" / (
        "experiment_"
        + time.strftime("%Y%m%d_%H%M%S")
        + "_continued"
    )

    folder.mkdir(parents=True, exist_ok=False)

    config.update(
        budget=end_step,
        total_steps=end_step,
        continuation_source=str(source),
        additional_steps=args.steps,
        restored_full_state=full_resume
    )

    started = time.perf_counter()

    def save(step):
        base = {
            "state_dict": model.state_dict(),
            "config": dict(config),
            "steps_completed": step
        }

        # Небольшой файл для play_agent.py.
        torch.save(
            base,
            folder / f"{name}.pt"
        )

        # Полное состояние для следующего продолжения.
        resume_state = {
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay": pack_replay(replay),
            "board": torch.from_numpy(env.board.copy()),
            "piece": env.piece,
            "score": env.score,
            "lines": env.lines,
            "pieces": env.pieces,
            "env_rng": json.dumps(
                env.rng.bit_generator.state
            ),
            "action_rng": json.dumps(
                rng.bit_generator.state
            ),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "episode_reward": episode_reward,
            "history": history,
            "losses": losses,
            "schedule_start": schedule_start,
            "schedule_length": schedule_length,
            "epsilon_start": epsilon_start
        }

        torch.save(
            dict(base, resume_state=resume_state),
            folder / f"{name}.resume.pt"
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
        f"{name}: {start_step:,} -> {end_step:,} steps",
        flush=True
    )

    for step in range(start_step + 1, end_step + 1):
        progress = min(
            1.0,
            (step - 1 - schedule_start) / schedule_length
        )

        epsilon = (
            epsilon_start
            + (0.05 - epsilon_start) * progress
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

        if len(replay) >= 1000 and step % 4 == 0:
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
                1,
                actions
            ).squeeze(1)

            with torch.no_grad():
                next_actions = (
                    model(next_states)
                    .masked_fill(~masks, -torch.inf)
                    .argmax(dim=1, keepdim=True)
                )

                next_q = target(next_states).gather(
                    1,
                    next_actions
                ).squeeze(1)

                next_q = torch.where(
                    terminals | ~masks.any(dim=1),
                    0.0,
                    next_q
                )

                expected = (
                    rewards
                    + config["gamma"] * next_q
                )

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

        if (step - start_step) % 5000 == 0:
            recent = history[-20:]

            mean_lines = (
                np.mean([
                    row["lines"]
                    for row in recent
                ])
                if recent else 0.0
            )

            elapsed = time.perf_counter() - started

            print(
                f"{step:,}/{end_step:,} | "
                f"eps={epsilon:.3f} | "
                f"lines={mean_lines:.2f} | "
                f"{elapsed / 60:.1f} min",
                flush=True
            )

            save(step)

    config["additional_training_seconds"] = (
        time.perf_counter() - started
    )

    save(end_step)
    model.eval()

    # Новые seeds для проверки после дополнительного обучения.
    rows = []

    for seed in range(30000, 30000 + args.episodes):
        test_env = TetrisEnv()

        state, info = test_env.reset(seed=seed)
        test_rng = np.random.default_rng(seed)

        total_reward = 0.0

        while True:
            action = choose_action(
                model,
                state,
                info["mask"],
                test_rng,
                epsilon=0.0
            )

            state, reward, terminated, truncated, info = (
                test_env.step(action)
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

    summary = {
        "agent": name,
        "episodes": len(rows),
        "training_steps": end_step
    }

    for metric in ["reward", "score", "lines", "pieces"]:
        values = [
            row[metric]
            for row in rows
        ]

        summary[metric + "_mean"] = float(
            np.mean(values)
        )

        summary[metric + "_std"] = float(
            np.std(values, ddof=1)
        )

    write_csv(
        folder / "test_episodes.csv",
        rows
    )

    write_csv(
        folder / "comparison.csv",
        [summary]
    )

    plot_curves(
        {name: history},
        folder
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved to: {folder}")


if __name__ == "__main__":
    main()