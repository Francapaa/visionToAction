"""F2 — Formato de dataset: episodio HDF5 + índice JSONL (AGENTS §5.5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vta.data.collector import collect_episode
from vta.data.dataset import read_episode
from vta.envs.env import PickPlaceEnv


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    env = PickPlaceEnv(scene_dir=tmp_path_factory.mktemp("scenes"))
    try:
        yield env
    finally:
        env.close()


@pytest.mark.smoke
def test_collect_episode_roundtrip(env, tmp_path) -> None:
    result = collect_episode(env, "blue_b", 1, tmp_path)
    assert result.success, result.metrics

    episode = read_episode(result.path)
    t = result.steps
    assert episode["image"].shape == (t, 84, 84, 3)
    assert episode["image"].dtype == np.uint8
    assert episode["proprio"].shape == (t, 22)
    assert episode["proprio"].dtype == np.float32
    assert episode["actions"].shape == (t, 7)
    assert episode["actions"].dtype == np.float32
    assert np.all(np.abs(episode["actions"]) <= 1.0)
    assert episode["cup_pos"].shape == (t, 3)
    assert episode["cup_quat"].shape == (t, 4)
    assert len(episode["contacts"]) == t

    attrs = episode["attrs"]
    assert attrs["task_id"] == "blue_b"
    assert int(attrs["seed"]) == 1
    assert attrs["cup_color"] == "blue"
    assert attrs["target_box"] == "b"
    assert int(attrs["expert_success"]) == 1
    assert int(attrs["n_steps"]) == t
    assert np.allclose(np.asarray(attrs["spawn"], dtype=np.float64), env.scene_params.cup_xyz)
    lighting = json.loads(attrs["lighting"])
    assert {"key_pos", "key_diffuse", "fill_diffuse"} <= set(lighting)
    assert "success" in json.loads(attrs["metrics"])

    index_records = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(index_records) == 1
    record = index_records[0]
    assert record["task"] == "blue_b"
    assert record["seed"] == 1
    assert record["success"] is True
    assert record["file"] == "train/blue_b/demo_000001.h5"
    assert record["steps"] == t
