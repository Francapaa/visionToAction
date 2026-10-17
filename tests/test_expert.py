"""F2 — Experto scripted: protocolo de éxito (smoke 3 seeds; slow 30 por tarea)."""

from __future__ import annotations

import pytest

from vta.data.collector import collect_episode
from vta.envs.env import PickPlaceEnv

TRAIN_TASKS = ("red_a", "red_b", "blue_a", "blue_b")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    env = PickPlaceEnv(scene_dir=tmp_path_factory.mktemp("scenes"))
    try:
        yield env
    finally:
        env.close()


@pytest.mark.smoke
def test_expert_success_smoke(env, tmp_path) -> None:
    results = [collect_episode(env, "red_a", seed, tmp_path) for seed in (0, 1, 2)]
    assert all(r.success for r in results), [r.metrics for r in results]
    for result in results:
        assert result.steps <= 500
        assert result.phases[0][0] == "pre_grasp"
        assert result.phases[-1][0] == "done"


@pytest.mark.slow
def test_expert_acceptance_30_seeds(env, tmp_path) -> None:
    """Criterio de aceptación F2: ≥90% de éxito en 30 seeds por tarea (§5.4)."""
    rates = {}
    for task in TRAIN_TASKS:
        outcomes = [collect_episode(env, task, seed, tmp_path).success for seed in range(30)]
        rates[task] = sum(outcomes) / len(outcomes)
    assert all(rate >= 0.90 for rate in rates.values()), rates
