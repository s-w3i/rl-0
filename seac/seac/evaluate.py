from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from csv import DictWriter
from pathlib import Path
import multiprocessing as mp
import shutil
import subprocess

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces as gym_spaces

import robotic_warehouse  # noqa: F401
import lbforaging  # noqa: F401

from robotic_warehouse import load_env_config
from a2c import A2C
from wrappers import FlattenObservation, RecordEpisodeStatistics, TimeLimit


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env", default="rware-small-4ag-v1")
    parser.add_argument("--path", default="pretrained/rware-small-4ag")
    parser.add_argument("--time_limit", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--render", action="store_true", help="Render each evaluation step.")
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--record-video", action="store_true", help="Save an MP4 for each episode.")
    parser.add_argument("--export-csv", action="store_true", help="Export a CSV summary for each episode.")
    parser.add_argument("--output-dir", default="seac_eval", help="Directory for evaluation outputs.")
    return parser.parse_args()


def _gymnasium_env_name(env_id):
    env_id = (env_id or "").strip()
    if env_id.startswith("rware-") and env_id.endswith("-v1"):
        return env_id[:-3] + "-v2"
    return env_id


def make_env(env_name, env_config, render_mode=None):
    if env_config:
        config_env_id, config_kwargs = load_env_config(env_config)
        env = gym.make(
            _gymnasium_env_name(config_env_id),
            disable_env_checker=True,
            render_mode=render_mode,
            **config_kwargs,
        )
    else:
        env = gym.make(
            _gymnasium_env_name(env_name),
            disable_env_checker=True,
            render_mode=render_mode,
        )
    if isinstance(env.observation_space, gym_spaces.Tuple) and any(
        isinstance(space, (gym_spaces.Dict, gym_spaces.Tuple))
        for space in env.observation_space.spaces
    ):
        env = FlattenObservation(env)
    return env


def _load_agents(env_name, env_config, model_path, device):
    env = make_env(env_name, env_config)
    agents = [
        A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, device)
        for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
    ]
    env.close()
    for agent in agents:
        agent.restore(model_path + f"/agent{agent.agent_id}")
        agent.model.eval()
    return agents


def _capture_frame(env):
    frame = env.render()
    if frame is None and hasattr(env, "unwrapped"):
        frame = env.unwrapped.render()
    return frame


def _prepare_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected an HxWxC video frame, got shape {frame.shape}.")
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB frames, got shape {frame.shape}.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    height, width, _ = frame.shape
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.ascontiguousarray(frame)


def _episode_report_row(ep, info):
    rewards = info.get("episode_reward")
    if rewards is None:
        rewards = []
    row = {
        "episode": ep,
        "total_reward": float(sum(rewards)),
        "episode_length": int(info.get("episode_length", 0)),
        "episode_time": float(info.get("episode_time", 0.0)),
    }
    for idx, reward in enumerate(rewards):
        row[f"agent{idx}_reward"] = float(reward)
    return row


def _write_csv_report(output_dir, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = output_dir / "evaluation_report.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_video(video_path, frames, fps):
    if not frames:
        return
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("Recording video requires ffmpeg to be installed.")

    prepared_frames = [_prepare_video_frame(frame) for frame in frames]
    height, width, _ = prepared_frames[0].shape
    for frame in prepared_frames[1:]:
        if frame.shape[:2] != (height, width):
            raise ValueError("All recorded frames must share the same resolution.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        for frame in prepared_frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        assert proc.stderr is not None
        stderr = proc.stderr.read()
        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())


def _render_mode(render, record_video):
    if record_video:
        return "rgb_array"
    if render:
        return "human"
    return None


def _run_episode(agents, episode_num, worker_config):
    render_mode = _render_mode(worker_config["render"], worker_config["record_video"])
    env = make_env(worker_config["env"], worker_config["env_config"], render_mode=render_mode)
    env = TimeLimit(env, worker_config["time_limit"])
    env = RecordEpisodeStatistics(env)

    obs, _ = env.reset()
    done = False
    frames = []
    if worker_config["record_video"]:
        first_frame = _capture_frame(env)
        if first_frame is not None:
            frames.append(first_frame)

    while not done:
        obs = [torch.from_numpy(o).float().to(worker_config["device"]) for o in obs]
        _, actions, _, _ = zip(
            *[agent.model.act(obs[agent.agent_id], None, None) for agent in agents]
        )
        actions = [
            a.squeeze(0).cpu().numpy().astype("int64") if a.numel() > 1 else int(a.item())
            for a in actions
        ]
        if worker_config["render"]:
            env.render()
        obs, _, terminated, truncated, info = env.step(actions)
        if worker_config["record_video"]:
            frame = _capture_frame(env)
            if frame is not None:
                frames.append(frame)
        done = bool(terminated or truncated)

    row = _episode_report_row(episode_num, info)
    if worker_config["record_video"] and frames:
        output_dir = Path(worker_config["output_dir"])
        video_path = output_dir / f"episode_{episode_num:05d}.mp4"
        _write_video(video_path, frames, fps=env.metadata.get("render_fps", 10))
    env.close()
    return row


def _evaluate_worker(worker_id, episode_ids, worker_config):
    del worker_id
    torch.set_num_threads(1)
    agents = _load_agents(
        worker_config["env"],
        worker_config["env_config"],
        worker_config["path"],
        worker_config["device"],
    )
    return [_run_episode(agents, episode_num, worker_config) for episode_num in episode_ids]


def _distribute_episode_ids(episodes, workers):
    assignments = [[] for _ in range(workers)]
    for idx, episode_num in enumerate(range(1, episodes + 1)):
        assignments[idx % workers].append(episode_num)
    return [assignment for assignment in assignments if assignment]


def _print_report_rows(rows, workers):
    for row in rows:
        print("--- Episode Finished ---")
        print(f"Episode {row['episode']} rewards: {row['total_reward']}")
        print(row)
        print(" --- ")
    if rows:
        mean_reward = float(np.mean([row["total_reward"] for row in rows]))
        print(
            f"Completed {len(rows)} evaluation episodes using {workers} worker(s). "
            f"Mean reward: {mean_reward}"
        )


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    workers = max(1, min(args.workers, args.episodes))
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.render and args.record_video:
        raise ValueError("Use either --render or --record-video, not both.")
    if args.render and workers > 1:
        raise ValueError("Parallel evaluation workers are incompatible with --render.")
    if args.record_video and shutil.which("ffmpeg") is None:
        raise RuntimeError("Recording video requires ffmpeg to be installed.")
    if args.record_video or args.export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    worker_config = {
        "device": device,
        "env": args.env,
        "env_config": args.env_config,
        "output_dir": str(output_dir),
        "path": args.path,
        "record_video": args.record_video,
        "render": args.render,
        "time_limit": args.time_limit,
    }

    assignments = _distribute_episode_ids(args.episodes, workers)
    report_rows = []

    if workers == 1:
        report_rows = _evaluate_worker(0, assignments[0], worker_config)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(_evaluate_worker, worker_id, episode_ids, worker_config)
                for worker_id, episode_ids in enumerate(assignments)
            ]
            for future in as_completed(futures):
                report_rows.extend(future.result())

    report_rows.sort(key=lambda row: row["episode"])
    _print_report_rows(report_rows, workers)

    if args.export_csv:
        _write_csv_report(output_dir, report_rows)


if __name__ == "__main__":
    main()
