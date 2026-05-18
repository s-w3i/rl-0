import glob
import logging
import os
import shutil
import re
import tempfile
import time
import copy
from contextlib import contextmanager
from collections import deque
from os import path
from pathlib import Path

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (  # noqa
    FileStorageObserver,
    MongoObserver,
    QueuedMongoObserver,
    QueueObserver,
)
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import A2C, RGSEAC, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones

import robotic_warehouse # noqa
from robotic_warehouse import load_env_training_overrides
import lbforaging # noqa

ex = Experiment(ingredients=[algorithm])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."


class ResumeFileStorageObserver(FileStorageObserver):
    def _make_run_dir(self, _id):
        if _id is None:
            return super()._make_run_dir(_id)
        os.makedirs(self.basedir, exist_ok=True)
        self.dir = os.path.join(self.basedir, str(_id))
        os.makedirs(self.dir, exist_ok=True)


ex.observers.append(ResumeFileStorageObserver("./results/sacred"))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


@ex.config
def config():
    env_name = None
    env_config = None
    time_limit = None
    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
    )
    dummy_vecenv = False

    num_env_steps = 100e6

    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"

    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = int(1e6)
    episodes_per_eval = 8
    resume_checkpoint = None
    resume_start_update = None


for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    ex.add_named_config(name, conf)

def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        values = []
        for d in info:
            if key not in d:
                continue
            arr = np.asarray(d[key])
            if arr.dtype.kind not in {"b", "i", "u", "f"}:
                continue
            values.append(arr.sum())
        if not values:
            continue
        mean = np.mean(values)
        new_info[key] = mean
    return new_info


def _planner_kwargs(algorithm):
    return {
        "use_global_planner": algorithm["use_global_planner"],
        "planner_type": algorithm["planner_type"],
        "planner_recompute_interval": algorithm["planner_recompute_interval"],
        "planner_prefix_len": algorithm["planner_prefix_len"],
        "planner_feature_mode": algorithm["planner_feature_mode"],
        "blocked_penalty": algorithm["planner_blocked_penalty"],
        "agent_blocked_penalty": algorithm["planner_agent_blocked_penalty"],
        "swap_penalty": algorithm["planner_swap_penalty"],
        "blocked_wait_bonus": algorithm["planner_blocked_wait_bonus"],
        "unblocked_deviation_penalty": algorithm[
            "planner_unblocked_deviation_penalty"
        ],
        "planner_follow_bonus": algorithm["planner_follow_bonus"],
        "conflict_clear_bonus": algorithm["planner_conflict_clear_bonus"],
        "persistent_agent_blocked_penalty": algorithm[
            "planner_persistent_agent_blocked_penalty"
        ],
        "persistent_agent_block_threshold": algorithm[
            "planner_persistent_agent_block_threshold"
        ],
        "planner_action_follow_bonus": algorithm["planner_action_follow_bonus"],
        "planner_turn_to_plan_bonus": algorithm["planner_turn_to_plan_bonus"],
        "task_progress_bonus": algorithm["planner_task_progress_bonus"],
        "task_regress_penalty": algorithm["planner_task_regress_penalty"],
        "pickup_bonus": algorithm["planner_pickup_bonus"],
        "delivery_bonus": algorithm["planner_delivery_bonus"],
        "no_progress_penalty": algorithm["planner_no_progress_penalty"],
        "no_progress_threshold": algorithm["planner_no_progress_threshold"],
        "rotation_penalty": algorithm["planner_rotation_penalty"],
        "idle_penalty": algorithm["planner_idle_penalty"],
        "toggle_penalty": algorithm["planner_toggle_penalty"],
    }


def _coerce_override_value(current_value, override_value):
    if isinstance(current_value, bool):
        if isinstance(override_value, bool):
            return override_value
        return str(override_value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(override_value)
    if isinstance(current_value, float):
        return float(override_value)
    return override_value


def _apply_env_training_overrides(algorithm, env_config, _log, _run):
    effective = copy.deepcopy(algorithm)
    if not env_config:
        return effective
    overrides = load_env_training_overrides(env_config)
    allowed_override_keys = {
        "planner_blocked_penalty",
        "planner_agent_blocked_penalty",
        "planner_swap_penalty",
        "planner_blocked_wait_bonus",
        "planner_unblocked_deviation_penalty",
        "planner_follow_bonus",
        "planner_conflict_clear_bonus",
        "planner_persistent_agent_blocked_penalty",
        "planner_persistent_agent_block_threshold",
        "planner_action_follow_bonus",
        "planner_turn_to_plan_bonus",
        "planner_task_progress_bonus",
        "planner_task_regress_penalty",
        "planner_pickup_bonus",
        "planner_delivery_bonus",
        "planner_no_progress_penalty",
        "planner_no_progress_threshold",
        "planner_rotation_penalty",
        "planner_idle_penalty",
        "planner_toggle_penalty",
    }
    applied = {}
    for full_key, value in overrides.items():
        parts = str(full_key).split(".")
        if len(parts) != 2 or parts[0] != "algorithm":
            _log.warning(
                f"Ignoring unsupported env training override {full_key!r}; "
                "only algorithm.* overrides are supported."
            )
            continue
        key = parts[1]
        if key not in allowed_override_keys:
            _log.warning(
                f"Ignoring env training override {full_key!r}; env JSON overrides "
                "are limited to reward-shaping algorithm keys."
            )
            continue
        if key not in effective:
            _log.warning(
                f"Ignoring unknown env training override {full_key!r}; "
                "no matching algorithm config key exists."
            )
            continue
        effective[key] = _coerce_override_value(effective[key], value)
        applied[full_key] = effective[key]
    if applied:
        _run.info["env_training_overrides_applied"] = applied
        _log.info(f"Applied env training overrides from {env_config}: {applied}")
    return effective


def _checkpoint_update(checkpoint_path):
    name = Path(checkpoint_path).name
    match = re.search(r"u(\d+)(?:\.tar\.xz)?$", name)
    if not match:
        return None
    return int(match.group(1))


def _max_checkpoint_update(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    updates = []
    for candidate in checkpoint_dir.iterdir():
        update = _checkpoint_update(candidate)
        if update is not None:
            updates.append(update)
    if not updates:
        return None
    return max(updates)


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _agent_checkpoint_root(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "agent0" / "models.pt").exists():
        return checkpoint_dir
    candidates = [
        candidate
        for candidate in checkpoint_dir.iterdir()
        if candidate.is_dir() and (candidate / "agent0" / "models.pt").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    return checkpoint_dir


@contextmanager
def _restorable_checkpoint_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser()
    if checkpoint_path.is_dir():
        yield _agent_checkpoint_root(checkpoint_path)
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.suffixes[-2:] != [".tar", ".xz"]:
        raise ValueError(
            "Resume checkpoint must be a checkpoint directory or .tar.xz archive."
        )

    with tempfile.TemporaryDirectory(prefix="seac_resume_") as tmpdir:
        shutil.unpack_archive(str(checkpoint_path), tmpdir)
        yield _agent_checkpoint_root(tmpdir)


def _safe_cleanup_run_dir(directory, resume_checkpoint):
    if resume_checkpoint:
        os.makedirs(directory, exist_ok=True)
        return
    utils.cleanup_log_dir(directory)


@ex.capture
def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    env_config,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
):
    device = algorithm["device"]

    eval_envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        monitor_dir=monitor_dir,
        env_config=env_config,
        planner_kwargs=_planner_kwargs(algorithm),
    )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = [
        torch.zeros(
            episodes_per_eval, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )

        # Obser reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)

        n_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        masks = n_masks
        all_infos.extend([i for i in infos if i and "episode_reward" in i])

    eval_envs.close()
    info = _squash_info(all_infos)
    _log.info(
        f"Evaluation using {len(all_infos)} episodes: mean reward {info['episode_reward']:.5f}\n"
    )
    for key in (
        "path_adherence_rate",
        "blocked_planned_step_frequency",
        "planner_follow_rate",
        "blocked_wait_rate",
        "blocked_rotate_rate",
        "blocked_other_rate",
        "unblocked_deviation_rate",
        "local_deviation_count",
        "episode_moved_total",
        "episode_rotation_total",
        "episode_noop_total",
        "episode_forward_total",
        "episode_blocked_total",
        "episode_blocked_agent_total",
        "episode_blocked_static_total",
        "episode_vertex_conflict_total",
        "episode_conflict_clear_total",
        "episode_max_consecutive_agent_blocked",
        "episode_max_consecutive_no_progress",
        "episode_persistent_agent_block_events",
        "episode_persistent_no_progress_events",
        "planner_action_shaping_sum",
        "planner_task_shaping_sum",
        "planner_loop_shaping_sum",
    ):
        if key in info:
            _log.info(f"Evaluation {key}: {info[key]:.5f}")


@ex.automain
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    env_config,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
    resume_checkpoint,
    resume_start_update,
):
    algorithm = _apply_env_training_overrides(algorithm, env_config, _log, _run)
    if algorithm["relevance_gated_seac"] and not algorithm["recurrent_policy"]:
        raise ValueError("RGSEAC requires algorithm.recurrent_policy=True.")
    if (
        algorithm["relevance_gated_seac"]
        and algorithm["relevance_gate_mode"] == "planner_context"
        and not algorithm["use_global_planner"]
    ):
        raise ValueError("planner_context RGSEAC requires algorithm.use_global_planner=True.")

    if loss_dir:
        loss_dir = path.expanduser(loss_dir.format(id=str(_run._id)))
        _safe_cleanup_run_dir(loss_dir, resume_checkpoint)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(eval_dir.format(id=str(_run._id)))
    save_dir = path.expanduser(save_dir.format(id=str(_run._id)))

    _safe_cleanup_run_dir(eval_dir, resume_checkpoint)
    _safe_cleanup_run_dir(save_dir, resume_checkpoint)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
        env_config=env_config,
        planner_kwargs=_planner_kwargs(algorithm),
    )

    agent_cls = RGSEAC if algorithm["relevance_gated_seac"] else A2C
    agents = [
        agent_cls(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    inferred_resume_update = None
    if resume_checkpoint:
        with _restorable_checkpoint_dir(resume_checkpoint) as checkpoint_dir:
            for agent in agents:
                agent_path = checkpoint_dir / f"agent{agent.agent_id}"
                if not agent_path.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint for agent{agent.agent_id}: {agent_path}"
                    )
                agent.restore(str(agent_path))
                _optimizer_to_device(agent.optimizer, algorithm["device"])
        inferred_resume_update = _checkpoint_update(resume_checkpoint)
        _log.info(
            f"Resumed agent model/optimizer state from {resume_checkpoint}"
        )

    obs = envs.reset()

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(algorithm["device"])

    start = time.time()
    num_updates = (
        int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    )
    start_update = int(
        resume_start_update
        if resume_start_update is not None
        else inferred_resume_update
        if inferred_resume_update is not None
        else 0
    )
    if resume_checkpoint:
        max_existing_update = _max_checkpoint_update(save_dir)
        if max_existing_update is not None and max_existing_update > start_update:
            raise ValueError(
                f"Refusing to resume run {_run._id} from u{start_update} because "
                f"{save_dir} already contains newer checkpoint u{max_existing_update}. "
                "Resume from the latest checkpoint or use a different run id."
            )
    if start_update >= num_updates:
        _log.info(
            f"Resume checkpoint update {start_update} is already at or beyond target "
            f"num_updates {num_updates}; no training updates to run."
        )
        envs.close()
        if writer:
            writer.close()
        return

    all_infos = deque(maxlen=10)
    loss_infos = deque(maxlen=max(1, log_interval * max(1, len(agents))))

    for j in range(start_update + 1, num_updates + 1):

        for step in range(algorithm["num_steps"]):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=algorithm["device"],
            )

            bad_masks = torch.tensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ],
                dtype=torch.float32,
                device=algorithm["device"],
            )
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    n_action_log_prob[i],
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )

            for info in infos:
                if info and "episode_reward" in info:
                    all_infos.append(info)

        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        for agent in agents:
            agent.compute_returns()

        for agent in agents:
            loss = agent.update(agents)
            loss_infos.append(loss)
            for k, v in loss.items():
                if writer:
                    writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

        for agent in agents:
            agent.storage.after_update()

        if j % log_interval == 0 and len(all_infos) > 0:
            squashed = _squash_info(all_infos)

            total_num_steps = (
                (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
            )
            end = time.time()
            _log.info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            if "episode_reward" in squashed:
                mean_reward = np.asarray(squashed["episode_reward"]).sum()
                _log.info(
                    f"Last {len(all_infos)} training episodes mean reward {mean_reward:.3f}"
                )

            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
            if loss_infos:
                loss_keys = sorted({key for loss in loss_infos for key in loss})
                loss_means = {}
                for key in loss_keys:
                    values = [float(loss[key]) for loss in loss_infos if key in loss]
                    if values:
                        mean_value = float(np.mean(values))
                        loss_means[key] = mean_value
                        _run.log_scalar(f"loss_{key}", mean_value, j)
                if "gate_mean" in loss_means:
                    _log.info(
                        "Loss gate mean %.4f, var %.6f, target %.4f",
                        loss_means.get("gate_mean", 0.0),
                        loss_means.get("gate_var", 0.0),
                        loss_means.get("gate_context_target_mean", 0.0),
                    )
                loss_infos.clear()
            all_infos.clear()

        if save_interval is not None and (
            j > 0 and j % save_interval == 0 or j == num_updates
        ):
            cur_save_dir = path.join(save_dir, f"u{j}")
            for agent in agents:
                save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                os.makedirs(save_at, exist_ok=True)
                agent.save(save_at)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_save_dir)
            _run.add_artifact(archive_name)

        if eval_interval is not None and (
            j > 0 and j % eval_interval == 0 or j == num_updates
        ):
            evaluate(
                agents, os.path.join(eval_dir, f"u{j}"), algorithm=algorithm
            )
            videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
            for i, v in enumerate(videos):
                _run.add_artifact(v, f"u{j}.{i}.mp4")
    envs.close()
    if writer:
        writer.close()
