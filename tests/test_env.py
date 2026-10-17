"""F1 — tests del env: obs, determinismo, acción y contactos."""

from __future__ import annotations

import numpy as np
import pytest

from vta.envs.env import PickPlaceEnv

pytestmark = pytest.mark.smoke


@pytest.fixture()
def env(tmp_path):
    environment = PickPlaceEnv(scene_dir=tmp_path)
    yield environment
    environment.close()


def test_reset_returns_valid_observation(env) -> None:
    obs, info = env.reset(seed=0)
    assert obs["image"].shape == (84, 84, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["proprio"].shape == (22,)
    assert obs["proprio"].dtype == np.float32
    assert np.isfinite(obs["proprio"]).all()
    assert env.observation_space.contains(obs)
    assert info["seed"] == 0
    assert info["scene"]["cup_color"] in ("red", "blue", "green", "yellow")
    assert len(info["scene"]["cup_xyz"]) == 3


def test_action_space_bounds(env) -> None:
    assert env.action_space.shape == (7,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_reset_same_seed_is_deterministic(env) -> None:
    obs_a, info_a = env.reset(seed=7)
    obs_b, info_b = env.reset(seed=7)
    assert info_a["scene"] == info_b["scene"]
    assert np.array_equal(obs_a["image"], obs_b["image"])
    assert np.array_equal(obs_a["proprio"], obs_b["proprio"])


def test_different_seeds_give_different_scenes(env) -> None:
    obs_a, info_a = env.reset(seed=8)
    obs_b, info_b = env.reset(seed=9)
    assert info_a["scene"]["cup_xyz"] != info_b["scene"]["cup_xyz"]
    assert not np.array_equal(obs_a["image"], obs_b["image"])


def test_step_holds_pose_and_accepts_out_of_range_actions(env) -> None:
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert reward == 0.0 and terminated is False and truncated is False and info == {}

    # la acción se clipea, no explota (aún sin controlador)
    obs, *_ = env.step(np.full(7, 5.0, dtype=np.float32))
    assert env.observation_space.contains(obs)

    # la taza sigue apoyada ~1 mm bajo su spawn
    pos, _ = env.get_body_pose("cup")
    assert abs(pos[2] - (env.scene_params.cup_xyz[2] - 0.001)) < 0.005


def test_contacts_resolved_with_names(env) -> None:
    env.reset(seed=2)
    contacts = env.get_contacts()
    assert contacts, "debería existir al menos el contacto taza↔mesa"
    cup_table = [
        c for c in contacts if {c["geom1"], c["geom2"]} == {"cup_collision", "table_top"}
    ]
    assert cup_table, f"contactos observados: {contacts}"
    assert cup_table[0]["normal_force"] > 0
    assert {cup_table[0]["body1"], cup_table[0]["body2"]} == {"cup", "world"}


def test_body_pose_matches_spawn(env) -> None:
    _, info = env.reset(seed=4)
    pos, quat = env.get_body_pose("cup")
    scene = info["scene"]
    # la taza asienta con deriva micrométrica al apoyar (contacto cilindro-plano)
    assert np.allclose(pos[:2], scene["cup_xyz"][:2], atol=2e-4)
    yaw = scene["cup_yaw"]
    expected_quat = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    assert np.allclose(quat, expected_quat, atol=1e-2)


def test_reset_with_explicit_params(env) -> None:
    from vta.envs import scene_builder as sb

    params = sb.default_scene(color="green", target_box="b")
    _, info = env.reset(seed=0, options={"params": params})
    assert info["scene"]["cup_color"] == "green"
    assert info["scene"]["target_box"] == "b"
    assert np.allclose(info["scene"]["cup_xyz"], params.cup_xyz)
