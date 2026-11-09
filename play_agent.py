import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pygame

from train_agent import (
    TetrisEnv,
    ROTATIONS,
    choose_action,
    load_model
)


ROOT = Path(__file__).resolve().parent

RES = (750, 940)
WINDOW_SIZE = (600, 752)
GAME_RES = (450, 900)

TILE = 45
FPS = 60

COLORS = [
    (70, 210, 230),
    (245, 210, 75),
    (180, 115, 240),
    (100, 220, 135),
    (245, 105, 115),
    (100, 145, 250),
    (245, 165, 80),
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        choices=["a", "b"],
        default="a"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20000
    )

    args = parser.parse_args()

    if args.model is None:
        models = sorted(
            (ROOT / "results").glob(
                f"experiment_*/ddqn_{args.agent}.pt"
            )
        )

        if not models:
            parser.error(
                "No model found. Run train_agent.py first "
                "or use --model PATH."
            )

        model_path = models[-1]

    else:
        model_path = args.model

    if not model_path.is_file():
        parser.error(
            f"Model file not found: {model_path}"
        )

    model = load_model(model_path)

    print(f"Loaded: {model_path.resolve()}")
    print("P: pause | R: new game | Up/Down: speed | Esc: exit")

    pygame.init()

    window = pygame.display.set_mode(WINDOW_SIZE)
    sc = pygame.Surface(RES).convert()
    game_sc = pygame.Surface(GAME_RES).convert()

    clock = pygame.time.Clock()

    main_font = pygame.font.Font(
        str(ROOT / "font" / "font.ttf"),
        65
    )

    font = pygame.font.Font(
        str(ROOT / "font" / "font.ttf"),
        45
    )

    title_tetris = main_font.render(
        "TETRIS",
        True,
        pygame.Color("darkorange")
    )

    title_score = font.render(
        "score:",
        True,
        pygame.Color("green")
    )

    title_record = font.render(
        "record:",
        True,
        pygame.Color("purple")
    )

    bg_path = ROOT / "img" / "bg.jpg"

    if not bg_path.is_file():
        bg_path = ROOT / "img" / "bg.png"

    bg = pygame.image.load(str(bg_path)).convert()

    game_bg = pygame.image.load(
        str(ROOT / "img" / "bg2.jpg")
    ).convert()

    record_path = ROOT / "record"

    try:
        record = int(
            record_path.read_text().strip() or "0"
        )

    except (FileNotFoundError, ValueError):
        record = 0

    grid = [
        pygame.Rect(
            x * TILE,
            y * TILE,
            TILE,
            TILE
        )
        for x in range(10)
        for y in range(20)
    ]

    env = TetrisEnv(max_pieces=1000)

    seed = args.seed
    state, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    colored_board = np.zeros(
        (20, 10),
        dtype=np.uint8
    )

    paused = False
    running = True

    action = None
    fall_y = 0.0
    speed = 1.0
    end_message = ""

    def draw_tile(x, y, color):
        rect = pygame.Rect(
            x * TILE,
            y * TILE,
            TILE - 2,
            TILE - 2
        )

        pygame.draw.rect(
            game_sc,
            color,
            rect
        )

    while running:
        dt = min(
            clock.tick(FPS) / 1000,
            0.05
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_p:
                    paused = not paused

                elif event.key == pygame.K_UP:
                    speed = min(16.0, speed * 2)

                elif event.key == pygame.K_DOWN:
                    speed = max(0.25, speed / 2)

                elif event.key == pygame.K_r:
                    seed += 1

                    state, info = env.reset(seed=seed)
                    rng = np.random.default_rng(seed)

                    colored_board.fill(0)

                    action = None
                    fall_y = 0.0
                    paused = False
                    end_message = ""

        if not running:
            break

        if not paused and not env.ended:
            if action is None:
                action = choose_action(
                    model,
                    state,
                    info["mask"],
                    rng,
                    epsilon=0.0
                )

                fall_y = 0.0

            shape, x, landing_y = (
                env.placements[action]
            )

            fall_y += 25 * speed * dt

            if fall_y >= landing_y + 1:
                h, w = shape.shape

                region = colored_board[
                    landing_y:landing_y + h,
                    x:x + w
                ]

                region[shape.astype(bool)] = (
                    env.piece + 1
                )

                full = (
                    colored_board != 0
                ).all(axis=1)

                colored_board = np.vstack((
                    np.zeros(
                        (int(full.sum()), 10),
                        dtype=np.uint8
                    ),
                    colored_board[~full]
                ))

                state, reward, terminated, truncated, info = (
                    env.step(action)
                )

                action = None

                if terminated:
                    end_message = "Game over — R: restart"

                elif truncated:
                    end_message = "1000 pieces — R: restart"

        sc.blit(bg, (0, 0))
        game_sc.fill((0, 0, 0))
        game_sc.blit(game_bg, (0, 0))

        for rect in grid:
            pygame.draw.rect(
                game_sc,
                (40, 40, 40),
                rect,
                1
            )

        for y in range(20):
            for x in range(10):
                color_id = int(
                    colored_board[y, x]
                )

                if color_id:
                    draw_tile(
                        x,
                        y,
                        COLORS[color_id - 1]
                    )

        if action is not None:
            shape, x, landing_y = (
                env.placements[action]
            )

            for yy, xx in np.argwhere(shape):
                draw_tile(
                    x + int(xx),
                    min(
                        int(fall_y),
                        landing_y
                    ) + int(yy),
                    COLORS[env.piece]
                )

        sc.blit(game_sc, (20, 20))

        preview_rng = deepcopy(env.rng)
        next_piece = int(preview_rng.integers(7))

        preview_shape = ROTATIONS[next_piece][0]

        preview_x = (
            605
            - preview_shape.shape[1] * TILE // 2
        )

        for yy, xx in np.argwhere(preview_shape):
            rect = pygame.Rect(
                preview_x + int(xx) * TILE,
                185 + int(yy) * TILE,
                TILE - 2,
                TILE - 2
            )

            pygame.draw.rect(
                sc,
                COLORS[next_piece],
                rect
            )

        sc.blit(
            title_tetris,
            (485, 20)
        )

        sc.blit(
            title_score,
            (535, 780)
        )

        sc.blit(
            font.render(
                str(env.score),
                True,
                pygame.Color("white")
            ),
            (550, 840)
        )

        sc.blit(
            title_record,
            (525, 650)
        )

        sc.blit(
            font.render(
                str(max(record, env.score)),
                True,
                pygame.Color("gold")
            ),
            (550, 710)
        )

        if env.score > record:
            record = env.score

            record_path.write_text(
                str(record)
            )

        status = end_message or (
            "PAUSED" if paused else "Playing"
        )

        pygame.display.set_caption(
            f"Tetris — {model_path.stem} — {status}"
        )

        window.blit(
            pygame.transform.smoothscale(
                sc,
                WINDOW_SIZE
            ),
            (0, 0)
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()