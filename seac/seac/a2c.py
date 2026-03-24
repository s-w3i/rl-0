import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import gymnasium
from model import Policy, FCNetwork
from storage import RolloutStorage
from sacred import Ingredient


def flatdim(space):
    return gymnasium.spaces.utils.flatdim(space)

algorithm = Ingredient("algorithm")


@algorithm.config
def config():
    lr = 3e-4
    adam_eps = 0.001
    gamma = 0.99
    use_gae = False
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    use_proper_time_limits = True
    recurrent_policy = False
    use_linear_lr_decay = False

    seac_coef = 1.0

    num_processes = 4
    num_steps = 5

    device = "cpu"
    relevance_gated_seac = False
    relevance_gate_mode = "learned"
    relevance_gate_hidden_dim = 64
    relevance_gate_min_weight = 0.25
    relevance_gate_target_mean = 0.60
    relevance_gate_reg_coef = 1e-3


class A2C:
    @algorithm.capture()
    def __init__(
        self,
        agent_id,
        obs_space,
        action_space,
        lr,
        adam_eps,
        recurrent_policy,
        num_steps,
        num_processes,
        device,
        relevance_gated_seac,
        relevance_gate_mode,
        relevance_gate_hidden_dim,
        relevance_gate_min_weight,
    ):
        self.agent_id = agent_id
        self.obs_size = flatdim(obs_space)
        self.action_size = flatdim(action_space)
        self.obs_space = obs_space
        self.action_space = action_space
        self.relevance_gated_seac = relevance_gated_seac
        self.relevance_gate_mode = relevance_gate_mode

        self.model = Policy(
            obs_space,
            action_space,
            base_kwargs={"recurrent": recurrent_policy},
            enable_relevance_gate=(
                relevance_gated_seac and relevance_gate_mode == "learned"
            ),
            relevance_gate_hidden_dim=relevance_gate_hidden_dim,
            relevance_gate_min_weight=relevance_gate_min_weight,
        )

        self.storage = RolloutStorage(
            obs_space,
            action_space,
            self.model.recurrent_hidden_state_size,
            num_steps,
            num_processes,
        )

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr, eps=adam_eps)

        # self.intr_stats = RunningStats()
        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def _resolve_storages(self, storages_or_agents):
        if not storages_or_agents:
            return []
        first = storages_or_agents[0]
        if hasattr(first, "storage"):
            return [agent.storage for agent in storages_or_agents]
        return storages_or_agents

    def _resolve_agents(self, storages_or_agents):
        if not storages_or_agents:
            return []
        first = storages_or_agents[0]
        if hasattr(first, "storage") and hasattr(first, "model"):
            return storages_or_agents
        return None

    def _rollout_shapes(self):
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()
        return obs_shape, action_shape, num_steps, num_processes

    def _evaluate_storage(self, storage, return_features=False):
        obs_shape, action_shape, _, _ = self._rollout_shapes()
        return self.model.evaluate_actions(
            storage.obs[:-1].view(-1, *obs_shape),
            storage.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size
            ),
            storage.masks[:-1].view(-1, 1),
            storage.actions.view(-1, action_shape),
            return_features=return_features,
        )

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @algorithm.capture
    def compute_returns(self, use_gae, gamma, gae_lambda, use_proper_time_limits):
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()

        self.storage.compute_returns(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
        )

    @algorithm.capture
    def update(
        self,
        storages,
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
    ):
        storages = self._resolve_storages(storages)

        obs_shape, action_shape, num_steps, num_processes = self._rollout_shapes()

        values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size
            ),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.storage.returns[:-1] - values

        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()


        # calculate prediction loss for the OTHER actor
        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
        seac_policy_loss = torch.zeros(1, device=values.device)
        seac_value_loss = torch.zeros(1, device=values.device)
        importance_sampling = torch.ones(1, device=values.device)
        for oid in other_agent_ids:

            other_values, logp, _, _ = self.model.evaluate_actions(
                storages[oid].obs[:-1].view(-1, *obs_shape),
                storages[oid]
                .recurrent_hidden_states[0]
                .view(-1, self.model.recurrent_hidden_state_size),
                storages[oid].masks[:-1].view(-1, 1),
                storages[oid].actions.view(-1, action_shape),
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            other_advantage = (
                storages[oid].returns[:-1] - other_values
            )  # or storages[oid].rewards

            importance_sampling = (
                logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            ).detach()
            # importance_sampling = 1.0
            seac_value_loss += (
                importance_sampling * other_advantage.pow(2)
            ).mean()
            seac_policy_loss += (
                -importance_sampling * logp * other_advantage.detach()
            ).mean()

        self.optimizer.zero_grad()
        (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * dist_entropy
            + seac_coef * seac_policy_loss
            + seac_coef * value_loss_coef * seac_value_loss
        ).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "importance_sampling": importance_sampling.mean().item(),
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),
            "seac_value_loss": seac_coef
            * value_loss_coef
            * seac_value_loss.item(),
        }


class RGSEAC(A2C):
    def _constant_gate(self, reference, mode, target_mean):
        if mode == "constant_one":
            return torch.ones_like(reference)
        if mode == "constant_target":
            return torch.full_like(reference, float(target_mean))
        raise ValueError(f"Unsupported relevance gate mode: {mode}")

    def _compute_gate(self, target_features, source_features):
        if self.model.relevance_gate is None:
            raise RuntimeError("Learned RGSEAC requires an instantiated relevance gate.")
        return self.model.relevance_gate(target_features.detach(), source_features.detach())

    def _self_features(self, agent):
        obs_shape, _, num_steps, num_processes = self._rollout_shapes()
        storage = agent.storage
        gate_features = agent.model.get_relevance_features(
            storage.obs[:-1].view(-1, *obs_shape),
            storage.recurrent_hidden_states[0].view(
                -1, agent.model.recurrent_hidden_state_size
            ),
            storage.masks[:-1].view(-1, 1),
        )
        return gate_features.view(num_steps, num_processes, -1).detach()

    @algorithm.capture
    def update(
        self,
        storages,
        value_loss_coef,
        entropy_coef,
        seac_coef,
        max_grad_norm,
        device,
        relevance_gate_mode,
        relevance_gate_target_mean,
        relevance_gate_reg_coef,
        relevance_gate_min_weight,
    ):
        agents = self._resolve_agents(storages)
        if agents is None:
            raise ValueError("RGSEAC update requires peer agents, not storages alone.")
        storages = self._resolve_storages(storages)

        _, _, num_steps, num_processes = self._rollout_shapes()
        values, action_log_probs, dist_entropy, _, _ = self._evaluate_storage(
            self.storage, return_features=True
        )
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.storage.returns[:-1] - values
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        value_loss = advantages.pow(2).mean()

        other_agent_ids = [x for x in range(len(storages)) if x != self.agent_id]
        seac_policy_loss = torch.zeros(1, device=values.device)
        seac_value_loss = torch.zeros(1, device=values.device)
        gate_values = []
        self_features = None
        if relevance_gate_mode == "learned":
            self_features = [self._self_features(agent) for agent in agents]

        mean_importance = torch.ones(1, device=values.device)
        for oid in other_agent_ids:
            other_values, logp, _, _, _ = self._evaluate_storage(
                storages[oid], return_features=True
            )
            other_values = other_values.view(num_steps, num_processes, 1)
            logp = logp.view(num_steps, num_processes, 1)
            other_advantage = storages[oid].returns[:-1] - other_values

            importance_sampling = (
                logp.exp() / (storages[oid].action_log_probs.exp() + 1e-7)
            ).detach()
            if relevance_gate_mode == "learned":
                gate = self._compute_gate(
                    self_features[self.agent_id], self_features[oid]
                )
            else:
                gate = self._constant_gate(
                    importance_sampling, relevance_gate_mode, relevance_gate_target_mean
                )
            mean_importance = importance_sampling.mean()
            gate_values.append(gate)
            weighted_importance = importance_sampling * gate
            seac_value_loss += (
                weighted_importance * other_advantage.pow(2)
            ).mean()
            seac_policy_loss += (
                -weighted_importance * logp * other_advantage.detach()
            ).mean()

        if gate_values:
            stacked_gates = torch.stack(gate_values)
            gate_mean = stacked_gates.mean()
            gate_var = stacked_gates.var(unbiased=False)
            gate_min = (
                getattr(self.model.relevance_gate, "min_weight", relevance_gate_min_weight)
                if self.model.relevance_gate
                else float(relevance_gate_min_weight)
            )
            gate_near_min = (stacked_gates <= gate_min + 1e-3).float().mean()
            gate_near_one = (stacked_gates >= 1.0 - 1e-3).float().mean()
            gate_reg_loss = (gate_mean - relevance_gate_target_mean).pow(2)
        else:
            gate_mean = torch.zeros(1, device=values.device)
            gate_var = torch.zeros(1, device=values.device)
            gate_near_min = torch.zeros(1, device=values.device)
            gate_near_one = torch.zeros(1, device=values.device)
            gate_reg_loss = torch.zeros(1, device=values.device)

        self.optimizer.zero_grad()
        (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * dist_entropy
            + seac_coef * seac_policy_loss
            + seac_coef * value_loss_coef * seac_value_loss
            + relevance_gate_reg_coef * gate_reg_loss
        ).backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss_coef * value_loss.item(),
            "dist_entropy": entropy_coef * dist_entropy.item(),
            "importance_sampling": mean_importance.item(),
            "seac_policy_loss": seac_coef * seac_policy_loss.item(),
            "seac_value_loss": seac_coef
            * value_loss_coef
            * seac_value_loss.item(),
            "gate_mean": float(gate_mean.item()),
            "gate_var": float(gate_var.item()),
            "gate_near_min_frac": float(gate_near_min.item()),
            "gate_near_one_frac": float(gate_near_one.item()),
            "gate_reg_loss": float(
                (relevance_gate_reg_coef * gate_reg_loss).item()
            ),
        }
