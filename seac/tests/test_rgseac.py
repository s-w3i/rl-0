from pathlib import Path
import sys
from types import MethodType

import gymnasium as gym
import pytest
import torch


SEAC_SRC = Path(__file__).resolve().parents[1] / "seac"
if str(SEAC_SRC) not in sys.path:
    sys.path.insert(0, str(SEAC_SRC))

import robotic_warehouse  # noqa: E402,F401
from a2c import A2C, RGSEAC  # noqa: E402
from envs import make_env  # noqa: E402
from evaluate import make_env as make_eval_env  # noqa: E402
from model import Policy, RelevanceGate  # noqa: E402
from planner import (  # noqa: E402
    build_planner_context_pair,
    extract_planner_hints,
    maybe_add_planner_hints,
    planner_hint_dim,
    planner_pair_feature_dim,
)
from rware.warehouse import Action, Direction  # noqa: E402


def _make_agent(
    agent_cls,
    agent_id,
    action_space,
    relevance_gated=False,
    gate_mode="latent_learned",
    use_global_planner=False,
    planner_prefix_len=4,
):
    obs_dim = 4 + (planner_hint_dim(planner_prefix_len) if use_global_planner else 0)
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=float)
    return agent_cls(
        agent_id,
        obs_space,
        action_space,
        3e-4,
        1e-5,
        True,
        3,
        2,
        "cpu",
        use_global_planner,
        planner_prefix_len,
        relevance_gated,
        gate_mode,
        32,
        0.25,
    )


def _populate_storage(agent, seed):
    torch.manual_seed(seed)
    obs_dim = agent.obs_space.shape[0]
    obs = torch.rand(2, obs_dim)
    agent.storage.obs[0].copy_(obs)
    recurrent_hidden_states = torch.zeros(
        2, agent.model.recurrent_hidden_state_size
    )
    agent.storage.recurrent_hidden_states[0].copy_(recurrent_hidden_states)
    masks = torch.ones(2, 1)
    bad_masks = torch.ones(2, 1)

    for _ in range(agent.storage.num_steps):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = agent.model.act(
                obs, recurrent_hidden_states, masks
            )
        next_obs = torch.rand(2, obs_dim)
        rewards = torch.randn(2, 1)
        agent.storage.insert(
            next_obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            rewards,
            masks,
            bad_masks,
        )
        obs = next_obs

    agent.compute_returns(
        use_gae=False,
        gamma=0.99,
        gae_lambda=0.95,
        use_proper_time_limits=True,
    )


def _copy_storage(src, dst):
    dst.obs.copy_(src.obs)
    dst.recurrent_hidden_states.copy_(src.recurrent_hidden_states)
    dst.rewards.copy_(src.rewards)
    dst.value_preds.copy_(src.value_preds)
    dst.returns.copy_(src.returns)
    dst.action_log_probs.copy_(src.action_log_probs)
    dst.actions.copy_(src.actions)
    dst.masks.copy_(src.masks)
    dst.bad_masks.copy_(src.bad_masks)
    dst.step = src.step


def test_policy_can_return_features_for_discrete_and_multidiscrete():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    recurrent_hidden_states = torch.zeros(2, 64)
    masks = torch.ones(2, 1)
    inputs = torch.randn(2, 4)

    discrete_policy = Policy(
        obs_space,
        gym.spaces.Discrete(3),
        base_kwargs={"recurrent": True},
        enable_relevance_gate=True,
        relevance_gate_hidden_dim=32,
        relevance_gate_min_weight=0.25,
    )
    discrete_actions = torch.tensor([[0], [1]])
    _, _, _, _, discrete_features = discrete_policy.evaluate_actions(
        inputs,
        recurrent_hidden_states,
        masks,
        discrete_actions,
        return_features=True,
    )
    assert discrete_features.shape == (2, discrete_policy.base.output_size)

    multi_policy = Policy(
        obs_space,
        gym.spaces.MultiDiscrete([5, 2, 2]),
        base_kwargs={"recurrent": True},
        enable_relevance_gate=True,
        relevance_gate_hidden_dim=32,
        relevance_gate_min_weight=0.25,
    )
    multi_actions = torch.tensor([[1, 0, 1], [3, 1, 0]])
    _, _, _, _, multi_features = multi_policy.evaluate_actions(
        inputs,
        recurrent_hidden_states,
        masks,
        multi_actions,
        return_features=True,
    )
    assert multi_features.shape == (2, multi_policy.base.output_size)


def test_relevance_gate_bounds():
    gate = RelevanceGate(planner_pair_feature_dim(), hidden_dim=32, min_weight=0.25)
    pair_features = torch.randn(3, 2, planner_pair_feature_dim())
    output = gate(pair_features)
    assert output.shape == (3, 2, 1)
    assert torch.all(output >= 0.25 - 1e-6)
    assert torch.all(output <= 1.0 + 1e-6)


def test_rgseac_constant_one_matches_recurrent_seac():
    torch.manual_seed(0)
    action_space = gym.spaces.Discrete(3)

    baseline_agents = [_make_agent(A2C, idx, action_space) for idx in range(2)]
    rg_agents = [
        _make_agent(RGSEAC, idx, action_space, relevance_gated=True, gate_mode="constant_one")
        for idx in range(2)
    ]

    for base_agent, rg_agent in zip(baseline_agents, rg_agents):
        rg_agent.model.load_state_dict(base_agent.model.state_dict())
        rg_agent.optimizer.load_state_dict(base_agent.optimizer.state_dict())
        _populate_storage(base_agent, seed=1234 + base_agent.agent_id)
        _copy_storage(base_agent.storage, rg_agent.storage)

    baseline_loss = baseline_agents[0].update(
        baseline_agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        normalize_shared_loss=False,
    )
    rg_loss = rg_agents[0].update(
        rg_agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        relevance_gate_mode="constant_one",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=0.0,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    for key in (
        "policy_loss",
        "value_loss",
        "dist_entropy",
        "importance_sampling",
        "seac_policy_loss",
        "seac_value_loss",
    ):
        assert baseline_loss[key] == pytest.approx(rg_loss[key], rel=1e-6, abs=1e-6)

    for base_param, rg_param in zip(
        baseline_agents[0].model.parameters(), rg_agents[0].model.parameters()
    ):
        assert torch.allclose(base_param, rg_param, atol=1e-6, rtol=1e-6)


def test_planner_guided_rgseac_constant_one_matches_recurrent_seac():
    torch.manual_seed(0)
    action_space = gym.spaces.Discrete(3)

    baseline_agents = [
        _make_agent(
            A2C,
            idx,
            action_space,
            use_global_planner=True,
            planner_prefix_len=3,
        )
        for idx in range(2)
    ]
    rg_agents = [
        _make_agent(
            RGSEAC,
            idx,
            action_space,
            relevance_gated=True,
            gate_mode="constant_one",
            use_global_planner=True,
            planner_prefix_len=3,
        )
        for idx in range(2)
    ]

    for base_agent, rg_agent in zip(baseline_agents, rg_agents):
        rg_agent.model.load_state_dict(base_agent.model.state_dict())
        rg_agent.optimizer.load_state_dict(base_agent.optimizer.state_dict())
        _populate_storage(base_agent, seed=2234 + base_agent.agent_id)
        _copy_storage(base_agent.storage, rg_agent.storage)

    baseline_loss = baseline_agents[0].update(
        baseline_agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        normalize_shared_loss=False,
    )
    rg_loss = rg_agents[0].update(
        rg_agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        relevance_gate_mode="constant_one",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=0.0,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    for key in (
        "policy_loss",
        "value_loss",
        "dist_entropy",
        "importance_sampling",
        "seac_policy_loss",
        "seac_value_loss",
    ):
        assert baseline_loss[key] == pytest.approx(rg_loss[key], rel=1e-6, abs=1e-6)

    for base_param, rg_param in zip(
        baseline_agents[0].model.parameters(), rg_agents[0].model.parameters()
    ):
        assert torch.allclose(base_param, rg_param, atol=1e-6, rtol=1e-6)


def test_rgseac_latent_gate_preserves_time_env_shape():
    class CaptureGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pair_shape = None

        def forward(self, pair_features):
            self.pair_shape = tuple(pair_features.shape)
            return torch.ones(*pair_features.shape[:-1], 1)

    action_space = gym.spaces.Discrete(3)
    agents = [
        _make_agent(RGSEAC, idx, action_space, relevance_gated=True, gate_mode="latent_learned")
        for idx in range(2)
    ]
    for idx, agent in enumerate(agents):
        _populate_storage(agent, seed=5678 + idx)

    capture_gate = CaptureGate()
    agents[0].model.relevance_gate = capture_gate
    agents[0].update(
        agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        relevance_gate_mode="latent_learned",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=1e-3,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    assert capture_gate.pair_shape == (3, 2, agents[0].model.base.output_size * 4)


def test_rgseac_planner_context_gate_preserves_time_env_shape():
    class CaptureGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pair_shape = None

        def forward(self, pair_features):
            self.pair_shape = tuple(pair_features.shape)
            return torch.ones(*pair_features.shape[:-1], 1)

    action_space = gym.spaces.Discrete(3)
    agents = [
        _make_agent(
            RGSEAC,
            idx,
            action_space,
            relevance_gated=True,
            gate_mode="planner_context",
            use_global_planner=True,
            planner_prefix_len=3,
        )
        for idx in range(2)
    ]
    for idx, agent in enumerate(agents):
        _populate_storage(agent, seed=7000 + idx)

    capture_gate = CaptureGate()
    agents[0].model.relevance_gate = capture_gate
    agents[0].update(
        agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        relevance_gate_mode="planner_context",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=1e-3,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    assert capture_gate.pair_shape == (3, 2, planner_pair_feature_dim())


def test_rgseac_checkpoint_round_trip(tmp_path):
    agent = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="latent_learned",
    )
    save_dir = tmp_path / "agent0"
    save_dir.mkdir()
    agent.save(save_dir)

    restored = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="latent_learned",
    )
    restored.restore(save_dir)

    checkpoint = torch.load(save_dir / "models.pt", map_location="cpu")
    assert set(checkpoint.keys()) == {"model_state_dict", "optimizer_state_dict"}

    for key, value in agent.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_restore_supports_legacy_object_checkpoint(tmp_path):
    agent = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="latent_learned",
    )
    save_dir = tmp_path / "agent0"
    save_dir.mkdir()
    torch.save(
        {"model": agent.model, "optimizer": agent.optimizer},
        save_dir / "models.pt",
    )

    restored = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="latent_learned",
    )
    restored.restore(save_dir)

    for key, value in agent.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_rgseac_requires_recurrent_policy():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    with pytest.raises(ValueError, match="recurrent_policy=True"):
        RGSEAC(
            0,
            obs_space,
            gym.spaces.Discrete(3),
            3e-4,
            1e-5,
            False,
            3,
            2,
            "cpu",
            False,
            4,
            True,
            "latent_learned",
            32,
            0.25,
        )


def test_planner_context_requires_global_planner():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4 + planner_hint_dim(3),), dtype=float)
    with pytest.raises(ValueError, match="use_global_planner=True"):
        RGSEAC(
            0,
            obs_space,
            gym.spaces.Discrete(3),
            3e-4,
            1e-5,
            True,
            3,
            2,
            "cpu",
            False,
            3,
            True,
            "planner_context",
            32,
            0.25,
        )


def test_plain_a2c_allows_planner_context_default_when_rgseac_disabled():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
    agent = A2C(
        0,
        obs_space,
        gym.spaces.Discrete(3),
        3e-4,
        1e-5,
        False,
        3,
        2,
        "cpu",
        False,
        4,
        False,
        "planner_context",
        32,
        0.25,
    )
    assert agent.relevance_gated_seac is False
    assert agent.relevance_gate_mode == "planner_context"


def test_cross_agent_recurrent_transfer_uses_target_initial_hidden_state():
    action_space = gym.spaces.Discrete(3)
    agents = [_make_agent(A2C, idx, action_space) for idx in range(2)]
    for idx, agent in enumerate(agents):
        _populate_storage(agent, seed=9800 + idx)

    agents[0].storage.recurrent_hidden_states[0].fill_(0.5)
    agents[1].storage.recurrent_hidden_states[0].fill_(3.0)

    captured_hxs = []
    original_evaluate_actions = agents[0].model.evaluate_actions

    def wrapped_evaluate_actions(
        model_self, inputs, rnn_hxs, masks, action, return_features=False
    ):
        captured_hxs.append(rnn_hxs.detach().clone())
        return original_evaluate_actions(
            inputs, rnn_hxs, masks, action, return_features=return_features
        )

    agents[0].model.evaluate_actions = MethodType(
        wrapped_evaluate_actions, agents[0].model
    )

    agents[0].update(
        agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        normalize_shared_loss=False,
    )

    hidden_size = agents[0].model.recurrent_hidden_state_size
    expected = agents[0].storage.recurrent_hidden_states[0].view(-1, hidden_size)
    source_initial = agents[1].storage.recurrent_hidden_states[0].view(-1, hidden_size)

    assert len(captured_hxs) >= 2
    assert torch.allclose(captured_hxs[1], expected)
    assert not torch.allclose(captured_hxs[1], source_initial)


def test_rgseac_single_agent_excludes_self_transfer():
    agent = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="constant_one",
    )
    _populate_storage(agent, seed=9012)
    loss = agent.update(
        [agent],
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        relevance_gate_mode="constant_one",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=0.0,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    assert loss["seac_policy_loss"] == pytest.approx(0.0)
    assert loss["seac_value_loss"] == pytest.approx(0.0)


def test_planner_hints_are_appended_and_metrics_reported():
    prefix_len = 3
    env = make_env(
        "rware-small-4ag-v1",
        0,
        0,
        None,
        (),
        None,
        planner_kwargs={
            "use_global_planner": True,
            "planner_type": "astar",
            "planner_recompute_interval": 2,
            "planner_prefix_len": prefix_len,
            "planner_feature_mode": "basic",
        },
    )
    obs, info = env.reset()

    assert env.observation_space[0].shape[0] == 71 + planner_hint_dim(prefix_len)
    assert len(obs[0]) == 71 + planner_hint_dim(prefix_len)
    assert "path_adherence_rate" in info
    assert "blocked_planned_step_frequency" in info
    assert "planner_follow_rate" in info
    assert "blocked_wait_rate" in info
    assert "blocked_rotate_rate" in info
    assert "blocked_other_rate" in info
    assert "unblocked_deviation_rate" in info
    assert "local_deviation_count" in info

    obs, _, _, _, info = env.step([0, 0, 0, 0])
    assert len(obs[0]) == 71 + planner_hint_dim(prefix_len)
    assert "planner_step_on_path" in info
    assert "planner_step_blocked_planned" in info
    assert "planner_step_followed" in info
    assert "planner_step_blocked_wait" in info
    assert "planner_step_blocked_rotate" in info
    assert "planner_step_blocked_other" in info
    assert "planner_step_unblocked_deviation" in info
    assert "episode_moved_total" in info
    assert "episode_rotation_total" in info
    assert "episode_noop_total" in info
    assert "episode_max_consecutive_agent_blocked" in info
    assert "episode_max_consecutive_no_progress" in info
    assert "episode_persistent_agent_block_events" in info
    assert "episode_persistent_no_progress_events" in info
    assert "agent_consecutive_agent_blocked" in info
    assert "agent_consecutive_no_progress" in info


def test_local_planner_feature_mode_does_not_append_absolute_positions():
    prefix_len = 3
    env = make_env(
        "rware-small-4ag-v1",
        0,
        0,
        None,
        (),
        None,
        planner_kwargs={
            "use_global_planner": True,
            "planner_type": "astar",
            "planner_recompute_interval": 2,
            "planner_prefix_len": prefix_len,
            "planner_feature_mode": "local",
        },
    )
    obs, _ = env.reset()
    hints = extract_planner_hints(torch.as_tensor(obs[0]).unsqueeze(0), prefix_len)

    assert torch.allclose(hints["position"], torch.zeros_like(hints["position"]))
    assert torch.all(hints["next_cell"] >= -1.0)
    assert torch.all(hints["next_cell"] <= 1.0)
    assert torch.all(hints["prefix_cells"] >= -float(prefix_len))
    assert torch.all(hints["prefix_cells"] <= float(prefix_len))


def test_local_planner_context_pair_uses_raw_positions_for_gate_context():
    prefix_len = 1
    obs_dim = 4 + planner_hint_dim(prefix_len)
    target = torch.zeros(1, obs_dim)
    source = torch.zeros(1, obs_dim)
    target[:, 0:2] = torch.tensor([[1.0, 2.0]])
    source[:, 0:2] = torch.tensor([[2.0, 2.0]])

    target_hint = target[:, -planner_hint_dim(prefix_len) :]
    source_hint = source[:, -planner_hint_dim(prefix_len) :]
    target_slices = extract_planner_hints(target, prefix_len)
    source_slices = extract_planner_hints(source, prefix_len)
    target_slices["has_plan"].fill_(1.0)
    source_slices["has_plan"].fill_(1.0)
    target_slices["next_cell"].copy_(torch.tensor([[1.0, 0.0]]))
    source_slices["next_cell"].copy_(torch.tensor([[-1.0, 0.0]]))
    target_slices["prefix_mask"].fill_(1.0)
    source_slices["prefix_mask"].fill_(1.0)
    target_slices["prefix_cells"].copy_(torch.tensor([[1.0, 0.0]]))
    source_slices["prefix_cells"].copy_(torch.tensor([[0.0, 0.0]]))

    pair = build_planner_context_pair(
        target,
        source,
        prefix_len,
        planner_feature_mode="local",
    )

    assert pair.shape == (1, planner_pair_feature_dim())
    assert pair[0, 0].item() == pytest.approx(1.0)
    assert pair[0, 1].item() == pytest.approx(0.0)
    assert pair[0, 2].item() == pytest.approx(1.0)
    assert pair[0, 15].item() == pytest.approx(0.0)
    assert pair[0, 16].item() == pytest.approx(1.0)
    assert pair[0, 17].item() == pytest.approx(1.0)


def test_rotation_does_not_resolve_positional_conflict():
    env = gym.make("rware-tiny-2ag-v2", disable_env_checker=True)
    env.reset(seed=123)
    warehouse = env.unwrapped

    warehouse.agents[0].x = 0
    warehouse.agents[0].y = 0
    warehouse.agents[0].dir = Direction.RIGHT
    warehouse.agents[0].carrying_shelf = None
    warehouse.agents[1].x = 1
    warehouse.agents[1].y = 0
    warehouse.agents[1].dir = Direction.UP
    warehouse.agents[1].carrying_shelf = None
    warehouse._recalc_grid()

    _, _, _, _, info = env.step([Action.FORWARD.value, Action.NOOP.value])
    assert info["step_conflict_detected"] == 1
    assert info["step_conflict_resolved"] == 0
    assert info["conflict_unresolved"] == 1
    assert info["active_conflict_episodes"] == 1

    _, _, _, _, info = env.step([Action.RIGHT.value, Action.LEFT.value])
    assert info["step_moved_total"] == 0
    assert info["step_conflict_detected"] == 0
    assert info["step_conflict_resolved"] == 0
    assert info["conflict_unresolved"] == 1
    assert info["active_conflict_episodes"] == 1


def test_persistent_agent_block_metrics_are_reported():
    env = gym.make("rware-tiny-2ag-v2", disable_env_checker=True)
    env.reset(seed=123)
    warehouse = env.unwrapped

    warehouse.agents[0].x = 0
    warehouse.agents[0].y = 0
    warehouse.agents[0].dir = Direction.RIGHT
    warehouse.agents[0].carrying_shelf = None
    warehouse.agents[1].x = 1
    warehouse.agents[1].y = 0
    warehouse.agents[1].dir = Direction.UP
    warehouse.agents[1].carrying_shelf = None
    warehouse._recalc_grid()

    info = {}
    for _ in range(5):
        _, _, _, _, info = env.step([Action.FORWARD.value, Action.NOOP.value])

    assert info["agent_consecutive_agent_blocked"][0] == 5
    assert info["step_persistent_agent_block_by_agent"][0] == 1
    assert info["episode_max_consecutive_agent_blocked"] == 5
    assert info["episode_persistent_agent_block_events"] >= 1
    assert info["agent_consecutive_agent_blocked"][1] == 0


def test_delivery_resets_no_progress_counter():
    env = gym.make("rware-tiny-2ag-v2", disable_env_checker=True)
    env.reset(seed=321)
    warehouse = env.unwrapped
    shelf = warehouse.request_queue[0]
    goal_x, goal_y = warehouse.goals[0]

    warehouse.agents[0].x = goal_x
    warehouse.agents[0].y = goal_y
    warehouse.agents[0].carrying_shelf = None
    warehouse.agents[1].x = 0
    warehouse.agents[1].y = 0
    warehouse.agents[1].carrying_shelf = None
    shelf.x = goal_x
    shelf.y = goal_y
    warehouse._consecutive_no_progress_by_agent = [24, 0]
    warehouse._recalc_grid()

    _, _, _, _, info = env.step([Action.NOOP.value, Action.NOOP.value])

    assert info["agent_delivery_count"][0] == 1
    assert info["step_progress_by_agent"][0] == 1
    assert info["agent_consecutive_no_progress"][0] == 0
    assert info["step_persistent_no_progress_by_agent"][0] == 0


def test_persistent_agent_block_penalty_escalates_shaping():
    env = gym.make("rware-tiny-2ag-v2", disable_env_checker=True)
    env.reset(seed=123)
    env = maybe_add_planner_hints(
        env,
        use_global_planner=True,
        planner_type="astar",
        planner_recompute_interval=1,
        planner_prefix_len=2,
        planner_feature_mode="basic",
        agent_blocked_penalty=0.02,
        persistent_agent_blocked_penalty=0.01,
        persistent_agent_block_threshold=3,
    )
    env.reset(seed=123)
    warehouse = env.unwrapped

    warehouse.agents[0].x = 0
    warehouse.agents[0].y = 0
    warehouse.agents[0].dir = Direction.RIGHT
    warehouse.agents[0].carrying_shelf = None
    warehouse.agents[1].x = 1
    warehouse.agents[1].y = 0
    warehouse.agents[1].dir = Direction.UP
    warehouse.agents[1].carrying_shelf = None
    warehouse._recalc_grid()

    info = {}
    for _ in range(3):
        _, _, _, _, info = env.step([Action.FORWARD.value, Action.NOOP.value])

    assert info["agent_consecutive_agent_blocked"][0] == 3
    assert info["conflict_shaping_by_agent"][0] == pytest.approx(-0.03)
    assert info["conflict_shaping_by_agent"][1] == pytest.approx(0.0)


def test_standalone_evaluate_env_supports_planner_wrapper():
    env = make_eval_env("rware-small-4ag-v1", None)
    env = maybe_add_planner_hints(
        env,
        use_global_planner=True,
        planner_type="astar",
        planner_recompute_interval=2,
        planner_prefix_len=3,
        planner_feature_mode="basic",
    )
    obs, info = env.reset()
    assert len(obs[0]) == 71 + planner_hint_dim(3)
    assert "planner_follow_rate" in info


def test_rgseac_planner_context_update_does_not_accumulate_gradients_on_other_agents():
    action_space = gym.spaces.Discrete(3)
    agents = [
        _make_agent(
            RGSEAC,
            idx,
            action_space,
            relevance_gated=True,
            gate_mode="planner_context",
            use_global_planner=True,
            planner_prefix_len=3,
        )
        for idx in range(2)
    ]
    for idx, agent in enumerate(agents):
        _populate_storage(agent, seed=4321 + idx)
        for param in agent.model.parameters():
            param.grad = None

    agents[0].update(
        agents,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        seac_coef=1.0,
        max_grad_norm=0.5,
        device="cpu",
        relevance_gate_mode="planner_context",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=1e-3,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    assert any(param.grad is not None for param in agents[0].model.parameters())
    assert all(param.grad is None for param in agents[1].model.parameters())
