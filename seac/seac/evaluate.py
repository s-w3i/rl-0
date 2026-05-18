from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import shutil
import tarfile

import gymnasium as gym
import torch
from gymnasium import spaces as gym_spaces

import robotic_warehouse  # noqa: F401
import lbforaging  # noqa: F401

from planner import (
    build_planner_context_pair,
    maybe_add_planner_hints,
    planner_context_risk_score,
)
from robotic_warehouse import load_env_config, load_env_training_overrides
from a2c import A2C, RGSEAC
from wrappers import FlattenObservation, Monitor, RecordEpisodeStatistics, TimeLimit


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--env", default="rware-small-4ag-v1")
    parser.add_argument("--path", default="pretrained/rware-small-4ag")
    parser.add_argument("--time_limit", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--render", action="store_true", help="Render each evaluation step.")
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--recurrent-policy", action="store_true")
    parser.add_argument("--relevance-gated", action="store_true")
    parser.add_argument(
        "--relevance-gate-mode",
        default="planner_context",
        choices=[
            "planner_context",
            "latent_learned",
            "learned",
            "constant_one",
            "constant_target",
        ],
    )
    parser.add_argument("--relevance-gate-hidden-dim", type=int, default=64)
    parser.add_argument("--relevance-gate-min-weight", type=float, default=0.25)
    parser.add_argument("--log-gate-stats", action="store_true")
    parser.add_argument("--use-global-planner", action="store_true")
    parser.add_argument("--planner-type", default="astar")
    parser.add_argument("--planner-recompute-interval", type=int, default=1)
    parser.add_argument("--planner-prefix-len", type=int, default=4)
    parser.add_argument(
        "--planner-feature-mode", default="local", choices=["local", "basic"]
    )
    parser.add_argument("--eval-dir", default="seac_eval")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--no-record-video", action="store_true")
    parser.add_argument(
        "--no-env-reward-shaping",
        action="store_true",
        help="Do not apply env-config training_overrides during evaluation.",
    )
    return parser.parse_args()


def _gymnasium_env_name(env_id):
    env_id = (env_id or "").strip()
    if env_id.startswith("rware-") and env_id.endswith("-v1"):
        return env_id[:-3] + "-v2"
    return env_id


def make_env(env_name, env_config, video_compatible=False):
    if env_config:
        config_env_id, config_kwargs = load_env_config(env_config)
        if video_compatible:
            config_kwargs = dict(config_kwargs)
            config_kwargs["render_mode"] = "rgb_array"
        env = gym.make(_gymnasium_env_name(config_env_id), disable_env_checker=True, **config_kwargs)
    else:
        kwargs = {"render_mode": "rgb_array"} if video_compatible else {}
        env = gym.make(_gymnasium_env_name(env_name), disable_env_checker=True, **kwargs)
    env.reset()
    if isinstance(env.observation_space, gym_spaces.Tuple) and any(
        isinstance(space, (gym_spaces.Dict, gym_spaces.Tuple))
        for space in env.observation_space.spaces
    ):
        env = FlattenObservation(env)
        env.reset()
    return env


def _canonical_gate_mode(mode):
    if mode == "learned":
        return "latent_learned"
    return mode


def _planner_reward_shaping_kwargs(env_config, enabled=True):
    if not env_config or not enabled:
        return {}
    overrides = load_env_training_overrides(env_config)
    mapping = {
        "algorithm.planner_blocked_penalty": "blocked_penalty",
        "algorithm.planner_agent_blocked_penalty": "agent_blocked_penalty",
        "algorithm.planner_swap_penalty": "swap_penalty",
        "algorithm.planner_blocked_wait_bonus": "blocked_wait_bonus",
        "algorithm.planner_unblocked_deviation_penalty": "unblocked_deviation_penalty",
        "algorithm.planner_follow_bonus": "planner_follow_bonus",
        "algorithm.planner_conflict_clear_bonus": "conflict_clear_bonus",
        "algorithm.planner_persistent_agent_blocked_penalty": "persistent_agent_blocked_penalty",
        "algorithm.planner_persistent_agent_block_threshold": "persistent_agent_block_threshold",
    }
    return {
        planner_key: float(overrides[override_key])
        for override_key, planner_key in mapping.items()
        if override_key in overrides
    }


def _restorable_model_path(model_path, eval_dir):
    path = Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".tar.xz"):
        extract_dir = Path(eval_dir) / "_checkpoint"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, mode="r:xz") as archive:
            _safe_extract(archive, extract_dir)
        children = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(children) == 1 and (children[0] / "agent0").exists():
            return str(children[0])
        return str(extract_dir)
    return str(path)


def _safe_extract(archive, destination):
    destination = Path(destination).resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe checkpoint archive member: {member.name}")
    archive.extractall(destination)


def main():
    args = parse_args()
    if args.record_video and args.no_record_video:
        raise ValueError("Use only one of --record-video or --no-record-video.")
    should_record_video = bool(args.record_video and not args.no_record_video)
    if should_record_video:
        try:
            import moviepy  # noqa: F401
            import imageio_ffmpeg  # noqa: F401
        except ModuleNotFoundError as exc:
            print(
                "Video recording disabled for this run: missing dependency "
                f"'{exc.name}'."
            )
            print(
                "Install with: /home/usern/rl-0/.venv_rl0/bin/python -m pip install moviepy imageio-ffmpeg"
            )
            should_record_video = False
    if args.relevance_gated and not args.recurrent_policy:
        raise ValueError("RGSEAC evaluation requires --recurrent-policy.")
    if (
        args.relevance_gated
        and _canonical_gate_mode(args.relevance_gate_mode) == "planner_context"
        and not args.use_global_planner
    ):
        raise ValueError("planner_context RGSEAC evaluation requires --use-global-planner.")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    eval_dir = Path(args.eval_dir).expanduser()
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_path = _restorable_model_path(args.path, eval_dir)
    shaping_kwargs = _planner_reward_shaping_kwargs(
        args.env_config, enabled=not args.no_env_reward_shaping
    )
    if shaping_kwargs:
        print(f"Applied env reward shaping during evaluation: {shaping_kwargs}")

    env = make_env(args.env, args.env_config, video_compatible=should_record_video)
    env = maybe_add_planner_hints(
        env,
        use_global_planner=args.use_global_planner,
        planner_type=args.planner_type,
        planner_recompute_interval=args.planner_recompute_interval,
        planner_prefix_len=args.planner_prefix_len,
        planner_feature_mode=args.planner_feature_mode,
        **shaping_kwargs,
    )
    agent_cls = RGSEAC if args.relevance_gated else A2C
    agents = [
        agent_cls(
            i,
            osp,
            asp,
            0.1,
            0.1,
            args.recurrent_policy,
            1,
            1,
            device,
            args.use_global_planner,
            args.planner_prefix_len,
            args.relevance_gated,
            _canonical_gate_mode(args.relevance_gate_mode),
            args.relevance_gate_hidden_dim,
            args.relevance_gate_min_weight,
            args.planner_feature_mode,
        )
        for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
    ]
    env.close()
    for agent in agents:
        try:
            agent.restore(str(Path(model_path) / f"agent{agent.agent_id}"))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint for agent{agent.agent_id} from '{model_path}'. "
                "This usually means the checkpoint does not match the current evaluation "
                "setup, for example when planner-guided evaluation flags are used with a "
                "baseline non-planner checkpoint, or vice versa."
            ) from exc

    episode_rows = []
    aggregate_keys = (
        "delivery_count",
        "task_completed",
        "step_blocked_total",
        "step_vertex_conflicts",
        "step_swap_attempts",
        "step_moved_total",
        "step_rotation_total",
        "step_noop_total",
        "step_forward_total",
        "episode_moved_total",
        "episode_rotation_total",
        "episode_noop_total",
        "episode_forward_total",
        "episode_toggle_load_total",
        "episode_blocked_total",
        "episode_blocked_static_total",
        "episode_blocked_agent_total",
        "episode_vertex_conflict_total",
        "episode_swap_attempt_total",
        "episode_conflict_clear_total",
        "episode_max_consecutive_blocked",
        "episode_max_consecutive_agent_blocked",
        "episode_max_consecutive_no_progress",
        "episode_persistent_block_events",
        "episode_persistent_agent_block_events",
        "episode_persistent_no_progress_events",
        "conflict_unresolved",
        "steps_since_task_progress",
        "path_adherence_rate",
        "blocked_planned_step_frequency",
        "planner_follow_rate",
        "blocked_wait_rate",
        "blocked_rotate_rate",
        "blocked_other_rate",
        "unblocked_deviation_rate",
        "local_deviation_count",
        "conflict_shaping_sum",
        "conflict_shaping_mean",
    )

    for ep in range(args.episodes):
        env = make_env(args.env, args.env_config, video_compatible=should_record_video)
        env = maybe_add_planner_hints(
            env,
            use_global_planner=args.use_global_planner,
            planner_type=args.planner_type,
            planner_recompute_interval=args.planner_recompute_interval,
            planner_prefix_len=args.planner_prefix_len,
            planner_feature_mode=args.planner_feature_mode,
            **shaping_kwargs,
        )
        ep_dir = eval_dir / f"episode_{ep + 1:03d}"
        env = Monitor(env, str(ep_dir), mode="evaluation")
        if should_record_video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(ep_dir / "video"),
                episode_trigger=lambda episode_id: True,
            )
        env = TimeLimit(env, args.time_limit)
        env = RecordEpisodeStatistics(env)

        obs, _ = env.reset()
        done = False
        recurrent_hidden_states = [
            torch.zeros(
                1, agent.model.recurrent_hidden_state_size, device=device
            )
            for agent in agents
        ]
        masks = torch.ones(1, 1, device=device)
        episode_gates = []
        episode_gate_risks = []

        while not done:
            obs = [torch.from_numpy(o).float().to(device).unsqueeze(0) for o in obs]
            if args.log_gate_stats and args.relevance_gated:
                for agent in agents:
                    if agent.model.relevance_gate is None or agent.relevance_gate_mode not in {
                        "planner_context",
                        "latent_learned",
                    }:
                        continue
                    for other in agents:
                        if agent.agent_id == other.agent_id:
                            continue
                        if agent.relevance_gate_mode == "planner_context":
                            gate_input = build_planner_context_pair(
                                obs[agent.agent_id],
                                obs[other.agent_id],
                                args.planner_prefix_len,
                                planner_feature_mode=args.planner_feature_mode,
                            )
                            episode_gate_risks.append(
                                planner_context_risk_score(gate_input)
                                .detach()
                                .view(-1)
                                .cpu()
                            )
                        else:
                            target_feature = agent.model.get_relevance_features(
                                obs[agent.agent_id],
                                recurrent_hidden_states[agent.agent_id],
                                masks,
                            ).detach()
                            source_feature = agent.model.get_relevance_features(
                                obs[other.agent_id],
                                recurrent_hidden_states[agent.agent_id],
                                masks,
                            ).detach()
                            gate_input = torch.cat(
                                [
                                    target_feature,
                                    source_feature,
                                    (target_feature - source_feature).abs(),
                                    target_feature * source_feature,
                                ],
                                dim=-1,
                            )
                        episode_gates.append(
                            agent.model.relevance_gate(gate_input).detach().view(-1).cpu()
                        )

            _, actions, _, next_hidden_states = zip(
                *[
                    agent.model.act(
                        obs[agent.agent_id],
                        recurrent_hidden_states[agent.agent_id],
                        masks,
                    )
                    for agent in agents
                ]
            )
            actions = [
                a.squeeze(0).cpu().numpy().astype("int64") if a.numel() > 1 else int(a.item())
                for a in actions
            ]
            recurrent_hidden_states = list(next_hidden_states)
            if args.render:
                env.render()
            obs, _, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            masks = torch.tensor(
                [[0.0] if done else [1.0]], dtype=torch.float32, device=device
            )

        print("--- Episode Finished ---")
        print(f"Episode rewards: {sum(info['episode_reward'])}")
        for key in (
            "delivery_count",
            "task_completed",
            "step_blocked_total",
            "step_vertex_conflicts",
            "step_swap_attempts",
            "step_moved_total",
            "step_rotation_total",
            "step_noop_total",
            "step_forward_total",
            "episode_moved_total",
            "episode_rotation_total",
            "episode_noop_total",
            "episode_forward_total",
            "episode_toggle_load_total",
            "episode_blocked_total",
            "episode_blocked_static_total",
            "episode_blocked_agent_total",
            "episode_vertex_conflict_total",
            "episode_swap_attempt_total",
            "episode_conflict_clear_total",
            "episode_max_consecutive_blocked",
            "episode_max_consecutive_agent_blocked",
            "episode_max_consecutive_no_progress",
            "episode_persistent_block_events",
            "episode_persistent_agent_block_events",
            "episode_persistent_no_progress_events",
            "conflict_unresolved",
            "steps_since_task_progress",
            "path_adherence_rate",
            "blocked_planned_step_frequency",
            "planner_follow_rate",
            "blocked_wait_rate",
            "blocked_rotate_rate",
            "blocked_other_rate",
            "unblocked_deviation_rate",
            "local_deviation_count",
        ):
            if key in info:
                print(f"{key}: {info[key]}")
        if episode_gates:
            stacked_gates = torch.cat(episode_gates)
            print(f"gate_mean: {stacked_gates.mean().item():.6f}")
            print(f"gate_var: {stacked_gates.var(unbiased=False).item():.6f}")
            if episode_gate_risks:
                stacked_risks = torch.cat(episode_gate_risks)
                high_risk = stacked_risks >= 0.5
                low_risk = stacked_risks < 0.1
                if high_risk.any():
                    print(
                        "gate_high_risk_mean: "
                        f"{stacked_gates[high_risk].mean().item():.6f}"
                    )
                if low_risk.any():
                    print(
                        "gate_low_risk_mean: "
                        f"{stacked_gates[low_risk].mean().item():.6f}"
                    )
        row = {
            "episode": ep + 1,
            "episode_reward_sum": float(sum(info.get("episode_reward", []))),
            "episode_length": int(info.get("episode_length", 0)),
            "episode_time": float(info.get("episode_time", 0.0)),
        }
        for key in aggregate_keys:
            if key in info:
                try:
                    row[key] = float(info[key])
                except (TypeError, ValueError):
                    continue
        agent_delivery_count = info.get("agent_delivery_count")
        if agent_delivery_count is not None:
            deliveries = [float(v) for v in agent_delivery_count]
            if deliveries:
                row["episode_min_agent_delivery"] = min(deliveries)
                row["episode_delivery_imbalance"] = max(deliveries) - min(deliveries)
                row["episode_has_starvation"] = float(min(deliveries) == 0)
        agent_task_completed = info.get("agent_task_completed")
        if agent_task_completed is not None:
            tasks = [float(v) for v in agent_task_completed]
            if tasks:
                row["episode_min_agent_task_completed"] = min(tasks)
                row["episode_task_imbalance"] = max(tasks) - min(tasks)
        row["episode_has_persistent_agent_block"] = float(
            info.get("episode_persistent_agent_block_events", 0) > 0
            or info.get("episode_max_consecutive_agent_blocked", 0) >= 5
        )
        row["episode_has_persistent_no_progress"] = float(
            info.get("episode_persistent_no_progress_events", 0) > 0
            or info.get("episode_max_consecutive_no_progress", 0) >= 25
        )
        if episode_gates:
            row["gate_mean"] = float(stacked_gates.mean().item())
            row["gate_var"] = float(stacked_gates.var(unbiased=False).item())
            if episode_gate_risks:
                row["gate_risk_mean"] = float(stacked_risks.mean().item())
                row["gate_high_risk_frac"] = float(
                    (stacked_risks >= 0.5).float().mean().item()
                )
                if high_risk.any():
                    row["gate_high_risk_mean"] = float(
                        stacked_gates[high_risk].mean().item()
                    )
                if low_risk.any():
                    row["gate_low_risk_mean"] = float(
                        stacked_gates[low_risk].mean().item()
                    )
        episode_rows.append(row)
        print(info)
        print(" --- ")
        env.close()

    if episode_rows:
        csv_path = eval_dir / "evaluation_episodes.csv"
        columns = sorted({k for row in episode_rows for k in row.keys()})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in episode_rows:
                writer.writerow(row)

        summary = {
            "episodes": len(episode_rows),
            "record_video": should_record_video,
            "env": args.env,
            "env_config": args.env_config,
            "model_path": args.path,
            "env_reward_shaping_applied": shaping_kwargs,
            "metrics_mean": {},
        }
        for key in columns:
            values = [row[key] for row in episode_rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            summary["metrics_mean"][key] = float(sum(values) / len(values))
        summary_path = eval_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved evaluation episode log: {csv_path}")
        print(f"Saved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
