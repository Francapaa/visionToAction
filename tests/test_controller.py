"""F2 — tests del controlador de espacio de tarea (mink IK)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from vta.controller.ik import MAX_DELTA_POS, TaskSpaceController, gripper_action_to_ctrl, rotation_error_world
from vta.envs import scene_builder as sb

pytestmark = pytest.mark.smoke

CONFIG = sb.SceneConfig.from_yaml()
DECIMATION = 25


@pytest.fixture()
def scene(tmp_path):
    params = sb.sample_scene(0, config=CONFIG)
    model, data = sb.load_scene(params, config=CONFIG, out_dir=tmp_path, settle_steps=150)
    return model, data, params


def test_tcp_offset_is_forward_in_hand_frame(scene) -> None:
    model, data, _ = scene
    controller = TaskSpaceController(model, data)
    assert abs(controller.tcp_local[0]) < 1e-9
    assert abs(controller.tcp_local[1]) < 1e-9
    assert 0.09 < controller.tcp_local[2] < 0.12


def test_gripper_mapping() -> None:
    assert gripper_action_to_ctrl(-1.0) == 255.0
    assert gripper_action_to_ctrl(0.0) == 127.5
    assert gripper_action_to_ctrl(1.0) == 0.0


def test_ik_converges_to_nearby_target(scene) -> None:
    model, data, _ = scene
    controller = TaskSpaceController(model, data)
    target = controller.tcp_pos(data) + np.array([0.03, 0.0, -0.05])

    for _ in range(60):
        err = target - controller.tcp_pos(data)
        action = np.zeros(7)
        action[:3] = np.clip(err / MAX_DELTA_POS, -1.0, 1.0)
        controller.step(data, action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

    assert np.linalg.norm(controller.tcp_pos(data) - target) < 0.015


def test_position_step_is_capped(scene) -> None:
    model, data, _ = scene
    controller = TaskSpaceController(model, data)
    tcp_before = controller.tcp_pos(data).copy()

    action = np.zeros(7)
    action[0] = 1.0  # 2 cm máx en +x
    controller.step(data, action)
    for _ in range(DECIMATION):
        mujoco.mj_step(model, data)

    moved = controller.tcp_pos(data) - tcp_before
    assert 0.003 < moved[0] < 0.025
    assert abs(moved[1]) < 0.005 and abs(moved[2]) < 0.005


def test_zero_action_holds_pose(scene) -> None:
    """Regresión: con acción cero el brazo no acumula el sag del servo."""
    model, data, _ = scene
    controller = TaskSpaceController(model, data)
    tcp_before = controller.tcp_pos(data).copy()
    quat_before = controller.hand_quat(data).copy()

    action = np.zeros(7)
    action[6] = -1.0  # gripper abierto
    for _ in range(60):
        controller.step(data, action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

    drift = float(np.linalg.norm(controller.tcp_pos(data) - tcp_before))
    rot_err = float(np.linalg.norm(rotation_error_world(quat_before, controller.hand_quat(data))))
    assert drift < 0.01, f"drift de TCP con acción cero: {drift * 1000:.1f} mm"
    assert rot_err < np.deg2rad(3.0), f"drift de orientación: {np.rad2deg(rot_err):.2f}°"


def test_determinism(scene) -> None:
    model_a, data_a, params = scene
    model_b, data_b = sb.load_scene(params, config=CONFIG, settle_steps=150)
    ctrl_a = TaskSpaceController(model_a, data_a)
    ctrl_b = TaskSpaceController(model_b, data_b)

    qpos_start = data_a.qpos.copy()
    rng = np.random.default_rng(0)
    for action in rng.uniform(-1.0, 1.0, size=(30, 7)):
        ctrl_a.step(data_a, action)
        ctrl_b.step(data_b, action)
        for _ in range(DECIMATION):
            mujoco.mj_step(model_a, data_a)
            mujoco.mj_step(model_b, data_b)

    assert not np.allclose(data_a.qpos, qpos_start)  # las acciones movieron el brazo
    assert np.array_equal(data_a.qpos, data_b.qpos)
