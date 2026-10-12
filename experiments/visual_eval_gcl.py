"""Visual consistency check for a GCL cell's FINAL result: plays the
EXPERT policy (the NanoGoal-RL model that generated the demonstrations)
and the RECOVERED policy (trained only on GCL's recovered reward, from
models/) on the SAME world seeds, and saves one side-by-side GIF per seed
(left = expert, right = recovered).

Adapted from NanoGoal-RL's own visual_eval.py (same rendering mechanism:
frames are read from env._last_frame after each step, so it works
headless), with three differences:
  - loads the recovered policy from models/phase0b_gcl_{cell}_policy.zip
    instead of a single NanoGoal-RL checkpoint;
  - runs expert and recovered policy on the same seed and stitches the
    two episodes together (the shorter one freezes on its last frame so
    the GIF lasts as long as the longer episode);
  - uses render_mode="rgb_array" (no live window), so it behaves the same
    on Windows, macOS, Linux and over SSH.

Purely local and read-only on models/: no git operations.

Usage:
    python -m experiments.visual_eval_gcl --cell easy_easy
    python -m experiments.visual_eval_gcl --cell easy_easy --seed_value 1520 4242
    python -m experiments.visual_eval_gcl --cell easy_easy --n-seeds 6 --fps 30

Without --seed_value, seeds are drawn (reproducibly, from --seed) from the
cell's held-out TEST seed pool, according to the cell's seed_mode. Output:
experiments/results/visual_eval/phase0b_gcl_{cell}_seed{S}.gif
"""
import argparse
import os
from pathlib import Path

# Headless-safe: if no display is available, pygame falls back to its
# off-screen driver. setdefault so a real display is still used when set.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from sim.nanogoal_adapter import (
    SEED_MODE_PROPORTIONS,
    create_env,
    load_policy,
    load_test_seeds,
    sample_seed_from_mode,
)

# Absolute, resolved up front: NanoEnv loads its sprites with a path
# relative to the CURRENT directory ("assets/agent.png"), so main() chdir's
# into NanoGoal-RL while rendering -- every path of ours must survive that.
NANOGOAL_PATH = str(Path("external/NanoGoal-RL").resolve())
OUTPUT_DIR = Path("experiments/results/visual_eval").resolve()
MODELS_DIR = Path("models").resolve()
LABEL_HEIGHT = 28


def make_render_env(nanogoal_path: str):
    # create_env() puts NanoGoal-RL on sys.path (and handles the eval/
    # name-collision); it builds a render-less env, so we only use it for
    # that side effect and then instantiate NanoEnv with a render mode.
    create_env(nanogoal_path)
    import importlib

    env_module = importlib.import_module("env")
    return env_module.NanoEnv(render_mode="rgb_array")


def play_episode(env, policy, seed: int) -> dict:
    """One deterministic episode; returns the frames plus true-reward stats."""
    obs, _ = env.reset(seed=seed)
    frames = [env._last_frame]
    total_reward = 0.0
    terminated = truncated = False
    info = {"is_success": False}
    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env._last_frame)
        total_reward += float(reward)
    return {
        "frames": frames,
        "steps": len(frames) - 1,
        "return": total_reward,
        "success": bool(info["is_success"]),
    }


def labelled(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame)
    canvas = Image.new("RGB", (img.width, img.height + LABEL_HEIGHT), "white")
    canvas.paste(img, (0, LABEL_HEIGHT))
    ImageDraw.Draw(canvas).text((6, 8), text, fill="black")
    return canvas


def side_by_side(expert: dict, recovered: dict, stride: int) -> list[Image.Image]:
    n = max(len(expert["frames"]), len(recovered["frames"]))
    out = []
    for i in range(0, n, stride):
        left = expert["frames"][min(i, len(expert["frames"]) - 1)]
        right = recovered["frames"][min(i, len(recovered["frames"]) - 1)]
        left_img = labelled(
            left,
            f"EXPERT  success={expert['success']}  steps={expert['steps']}  "
            f"return={expert['return']:.1f}",
        )
        right_img = labelled(
            right,
            f"GCL-RECOVERED  success={recovered['success']}  "
            f"steps={recovered['steps']}  return={recovered['return']:.1f}",
        )
        combined = Image.new("RGB", (left_img.width + right_img.width, left_img.height))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (left_img.width, 0))
        # Explicit FASTOCTREE quantization, as in NanoGoal-RL's visual_eval.py:
        # Pillow's default GIF palette conversion raised "quantization
        # error" on some Windows/Pillow builds (these frames have few flat
        # colors, which its default adaptive quantizer can choke on).
        out.append(combined.quantize(colors=256, method=Image.Quantize.FASTOCTREE))
    return out


def pick_seeds(cell: str, n_seeds: int, rng_seed: int) -> list[int]:
    _, seed_mode = cell.split("_", 1)
    test_seeds = load_test_seeds(NANOGOAL_PATH)
    rng = np.random.default_rng(rng_seed)
    # sample_seed_from_mode respects the cell's easy / easy_medium mix
    return [int(sample_seed_from_mode(test_seeds, seed_mode, rng)) for _ in range(n_seeds)]


def main(cell: str, seed_values: list[int] | None, n_seeds: int, rng_seed: int,
         fps: int, stride: int) -> None:
    torch.set_num_threads(1)  # same determinism setting as NanoGoal-RL's eval
    model_difficulty, seed_mode = cell.split("_", 1)
    if seed_mode not in SEED_MODE_PROPORTIONS:
        raise ValueError(f"Unknown seed_mode '{seed_mode}' in cell '{cell}'")

    env = make_render_env(NANOGOAL_PATH)
    expert_policy = load_policy(NANOGOAL_PATH, model_difficulty, env)
    recovered_policy = PPO.load(str(MODELS_DIR / f"phase0b_gcl_{cell}_policy"), env=env, device="cpu")

    seeds = seed_values if seed_values else pick_seeds(cell, n_seeds, rng_seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cell '{cell}': {len(seeds)} seed(s): {seeds}\n")
    print(f"{'seed':>8} | {'expert':^28} | {'recovered':^28}")
    n_expert_ok = n_recovered_ok = 0
    original_cwd = os.getcwd()
    os.chdir(NANOGOAL_PATH)  # see NANOGOAL_PATH comment above
    try:
        for seed in seeds:
            expert = play_episode(env, expert_policy, seed)
            recovered = play_episode(env, recovered_policy, seed)
            n_expert_ok += expert["success"]
            n_recovered_ok += recovered["success"]
            print(
                f"{seed:>8} | ok={str(expert['success']):<5} steps={expert['steps']:<4} "
                f"R={expert['return']:>8.1f} | ok={str(recovered['success']):<5} "
                f"steps={recovered['steps']:<4} R={recovered['return']:>8.1f}"
            )

            frames = side_by_side(expert, recovered, stride)
            path = OUTPUT_DIR / f"phase0b_gcl_{cell}_seed{seed}.gif"
            frames[0].save(
                path, save_all=True, append_images=frames[1:],
                duration=int(1000 / fps), loop=0,
            )
    finally:
        os.chdir(original_cwd)

    env.close()
    print(
        f"\nSuccess on these seeds: expert {n_expert_ok}/{len(seeds)}, "
        f"recovered {n_recovered_ok}/{len(seeds)}. GIFs in experiments/results/visual_eval/"
    )
    print("Too few seeds for statistics -- see evaluate_final_gcl_model.py for the "
          "30-seed numbers; this is a visual sanity check only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cell", required=True, help="e.g. easy_easy")
    parser.add_argument("--seed_value", type=int, nargs="+", default=None,
                        help="exact world seed(s); overrides --n-seeds")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="how many held-out seeds to draw if --seed_value is not given")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for drawing held-out seeds")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="keep every k-th frame (>1 gives smaller GIFs)")
    args = parser.parse_args()
    main(args.cell, args.seed_value, args.n_seeds, args.seed, args.fps, args.frame_stride)
