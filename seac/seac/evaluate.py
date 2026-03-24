from argparse import ArgumentParser

import gymnasium as gym
import torch
from gymnasium import spaces as gym_spaces

import robotic_warehouse  # noqa: F401
import lbforaging  # noqa: F401

from robotic_warehouse import load_env_config
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
        default="learned",
        choices=["learned", "constant_one", "constant_target"],
    )
    parser.add_argument("--relevance-gate-hidden-dim", type=int, default=64)
    parser.add_argument("--relevance-gate-min-weight", type=float, default=0.25)
    parser.add_argument("--log-gate-stats", action="store_true")
    return parser.parse_args()


def _gymnasium_env_name(env_id):
    env_id = (env_id or "").strip()
    if env_id.startswith("rware-") and env_id.endswith("-v1"):
        return env_id[:-3] + "-v2"
    return env_id


def make_env(env_name, env_config):
    if env_config:
        config_env_id, config_kwargs = load_env_config(env_config)
        env = gym.make(_gymnasium_env_name(config_env_id), disable_env_checker=True, **config_kwargs)
    else:
        env = gym.make(_gymnasium_env_name(env_name), disable_env_checker=True)
    if isinstance(env.observation_space, gym_spaces.Tuple) and any(
        isinstance(space, (gym_spaces.Dict, gym_spaces.Tuple))
        for space in env.observation_space.spaces
    ):
        env = FlattenObservation(env)
    return env


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    env = make_env(args.env, args.env_config)
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
            args.relevance_gated,
            args.relevance_gate_mode,
            args.relevance_gate_hidden_dim,
            args.relevance_gate_min_weight,
        )
        for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
    ]
    for agent in agents:
        agent.restore(args.path + f"/agent{agent.agent_id}")

    for ep in range(args.episodes):
        env = make_env(args.env, args.env_config)
        env = Monitor(env, f"seac_eval/{ep + 1}", mode="evaluation")
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

        while not done:
            obs = [torch.from_numpy(o).float().to(device).unsqueeze(0) for o in obs]
            if args.log_gate_stats and args.relevance_gated:
                features = [
                    agent.model.get_relevance_features(
                        obs[agent.agent_id],
                        recurrent_hidden_states[agent.agent_id],
                        masks,
                    ).detach()
                    for agent in agents
                ]
                for agent in agents:
                    if agent.model.relevance_gate is None:
                        continue
                    for other in agents:
                        if agent.agent_id == other.agent_id:
                            continue
                        episode_gates.append(
                            agent.model.relevance_gate(
                                features[agent.agent_id], features[other.agent_id]
                            )
                            .detach()
                            .view(-1)
                            .cpu()
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
            "conflict_unresolved",
            "steps_since_task_progress",
        ):
            if key in info:
                print(f"{key}: {info[key]}")
        if episode_gates:
            stacked_gates = torch.cat(episode_gates)
            print(f"gate_mean: {stacked_gates.mean().item():.6f}")
            print(f"gate_var: {stacked_gates.var(unbiased=False).item():.6f}")
        print(info)
        print(" --- ")
        env.close()


if __name__ == "__main__":
    main()
