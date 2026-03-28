from pathlib import Path
import sys

import gymnasium as gym
import pytest
import torch


SEAC_SRC = Path(__file__).resolve().parents[1] / "seac"
if str(SEAC_SRC) not in sys.path:
    sys.path.insert(0, str(SEAC_SRC))

from a2c import A2C, RGSEAC  # noqa: E402
from model import Policy, RelevanceGate  # noqa: E402


def _make_agent(agent_cls, agent_id, action_space, relevance_gated=False, gate_mode="learned"):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float)
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
        relevance_gated,
        gate_mode,
        32,
        0.25,
    )


def _populate_storage(agent, seed):
    torch.manual_seed(seed)
    obs_dim = agent.obs_space.shape[0]
    obs = torch.randn(2, obs_dim)
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
        next_obs = torch.randn(2, obs_dim)
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
    gate = RelevanceGate(64, hidden_dim=32, min_weight=0.25)
    target = torch.randn(3, 2, 64)
    source = torch.randn(3, 2, 64)
    output = gate(target, source)
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


def test_rgseac_learned_gate_preserves_time_env_shape():
    class CaptureGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target_shape = None
            self.source_shape = None

        def forward(self, target, source):
            self.target_shape = tuple(target.shape)
            self.source_shape = tuple(source.shape)
            return torch.ones(*target.shape[:-1], 1)

    action_space = gym.spaces.Discrete(3)
    agents = [
        _make_agent(RGSEAC, idx, action_space, relevance_gated=True, gate_mode="learned")
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
        relevance_gate_mode="learned",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=1e-3,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    assert capture_gate.target_shape == (3, 2, agents[0].model.base.output_size)
    assert capture_gate.source_shape == (3, 2, agents[0].model.base.output_size)


def test_rgseac_checkpoint_round_trip(tmp_path):
    agent = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="learned",
    )
    save_dir = tmp_path / "agent0"
    save_dir.mkdir()
    agent.save(save_dir)

    restored = _make_agent(
        RGSEAC,
        0,
        gym.spaces.Discrete(3),
        relevance_gated=True,
        gate_mode="learned",
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
        gate_mode="learned",
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
        gate_mode="learned",
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
            True,
            "learned",
            32,
            0.25,
        )


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


def test_rgseac_learned_update_does_not_accumulate_gradients_on_other_agents():
    action_space = gym.spaces.Discrete(3)
    agents = [
        _make_agent(RGSEAC, idx, action_space, relevance_gated=True, gate_mode="learned")
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
        relevance_gate_mode="learned",
        relevance_gate_target_mean=0.60,
        relevance_gate_reg_coef=1e-3,
        relevance_gate_min_weight=0.25,
        normalize_shared_loss=False,
    )

    assert any(param.grad is not None for param in agents[0].model.parameters())
    assert all(param.grad is None for param in agents[1].model.parameters())
