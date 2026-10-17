"""F1 — tests de escena: determinismo, parseo, límites de spawn y colores."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from vta.envs import scene_builder as sb

pytestmark = pytest.mark.smoke

CONFIG = sb.SceneConfig.from_yaml()


@pytest.fixture(scope="module")
def ref_model(tmp_path_factory: pytest.TempPathFactory) -> mujoco.MjModel:
    """Modelo de referencia (una escena cualquiera) para checks geométricos."""
    params = sb.sample_scene(0, config=CONFIG)
    path = sb.write_scene(params, config=CONFIG, out_dir=tmp_path_factory.mktemp("scenes"))
    return mujoco.MjModel.from_xml_path(str(path))


def test_same_seed_produces_identical_scene(tmp_path) -> None:
    params_a = sb.sample_scene(11, config=CONFIG)
    params_b = sb.sample_scene(11, config=CONFIG)
    assert params_a == params_b
    assert sb.build_scene_xml(params_a, CONFIG) == sb.build_scene_xml(params_b, CONFIG)

    path_a = sb.write_scene(params_a, config=CONFIG, out_dir=tmp_path)
    path_b = sb.write_scene(params_b, config=CONFIG, out_dir=tmp_path)
    assert path_a == path_b
    assert path_a.read_bytes() == path_b.read_bytes()

    other = sb.sample_scene(12, config=CONFIG)
    assert other != params_a


def test_generated_scene_parses_with_expected_names(tmp_path) -> None:
    params = sb.sample_scene(3, color="blue", target_box="b", config=CONFIG)
    path = sb.write_scene(params, config=CONFIG, out_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))

    for body in (sb.HAND_BODY_NAME, sb.CUP_BODY_NAME, sb.CAM_TARGET_NAME, "box_a", "box_b"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, f"falta body {body}"
    for camera in (sb.CAM_SCENE_NAME, sb.CAM_TOP_NAME):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0
    for joint in (*sb.ARM_JOINT_NAMES, *sb.FINGER_JOINT_NAMES, sb.CUP_JOINT_NAME):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0

    # 7 juntas del brazo + 2 dedos + 1 freejoint de la taza
    assert model.njnt == 10

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, sb.INIT_KEY_NAME)
    assert key >= 0, "falta el keyframe init"
    mujoco.mj_resetDataKeyframe(model, data, key)
    cup_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sb.CUP_JOINT_NAME)]
    assert np.allclose(data.qpos[cup_adr:cup_adr + 3], params.cup_xyz, atol=1e-9)
    assert np.allclose(data.qpos[0:7], sb.INIT_ARM_QPOS, atol=1e-9)

    # la geometría de la taza sale de la config, no de números duplicados
    # (un cilindro en MuJoCo tiene size = (radio, media_altura, 0))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_collision")
    assert np.allclose(model.geom_size[gid][:2], (CONFIG.cup.radius, CONFIG.cup.half_height), atol=1e-9)


def test_cup_mass_comes_from_collision_density(ref_model) -> None:
    """Los geoms visuales no aportan masa: m = densidad × volumen del cilindro."""
    cup_id = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_BODY, sb.CUP_BODY_NAME)
    volume = np.pi * CONFIG.cup.radius**2 * 2.0 * CONFIG.cup.half_height
    expected = CONFIG.cup.density * volume
    assert abs(ref_model.body_mass[cup_id] - expected) / expected < 0.01


def test_spawn_bounds_and_clearances(ref_model) -> None:
    """200 seeds: la taza cae dentro del workspace y lejos de cajas/borde."""
    data = mujoco.MjData(ref_model)
    mujoco.mj_forward(ref_model, data)

    def geom_face(geom_name: str, axis: int, sign: float) -> float:
        """Cara de un geom en coordenadas del MUNDO (geom_xpos ya aplica el body)."""
        gid = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        pos = data.geom_xpos[gid]
        size = ref_model.geom_size[gid]
        return float(pos[axis] + sign * size[axis])

    # Cara externa de la pared de la caja A más cercana al centro de la mesa
    # (box_a está en y=-0.32; wall_yp está en y=-0.26 y su cara externa en -0.255)
    box_face = geom_face("box_a_wall_yp", 1, +1.0)
    table_far = geom_face("table_top", 0, +1.0)
    radius = CONFIG.cup.radius

    for seed in range(200):
        params = sb.sample_scene(seed, config=CONFIG)
        x, y, z = params.cup_xyz
        assert CONFIG.workspace_x[0] - 1e-9 <= x <= CONFIG.workspace_x[1] + 1e-9
        assert CONFIG.workspace_y[0] - 1e-9 <= y <= CONFIG.workspace_y[1] + 1e-9
        expected_z = CONFIG.table_top + CONFIG.cup.half_height + CONFIG.cup.spawn_gap
        assert abs(z - expected_z) < 1e-12
        assert abs(y) + radius <= abs(box_face) - 0.019, f"seed {seed}: taza pegada a la caja"
        assert x + radius <= table_far - 0.05, f"seed {seed}: taza fuera del borde de la mesa"
        assert 0.0 <= params.cup_yaw < 2 * np.pi


def test_palette_colors_map_to_materials(tmp_path) -> None:
    for color in CONFIG.palette:
        params = sb.sample_scene(5, color=color, config=CONFIG)
        path = sb.write_scene(params, config=CONFIG, out_dir=tmp_path)
        model = mujoco.MjModel.from_xml_path(str(path))
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_visual")
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[gid]))
        assert mat_name == sb.CUP_MATERIALS[color]


def test_invalid_color_and_box_raise() -> None:
    with pytest.raises(ValueError):
        sb.sample_scene(0, color="pink", config=CONFIG)
    with pytest.raises(ValueError):
        sb.sample_scene(0, target_box="c", config=CONFIG)
    with pytest.raises(ValueError):
        sb.default_scene(color="pink", config=CONFIG)


def test_load_scene_applies_keyframe_and_settles(tmp_path) -> None:
    """El camino compartido env/viewer: keyframe init aplicado + taza apoyada."""
    params = sb.sample_scene(6, config=CONFIG)
    model, data = sb.load_scene(params, config=CONFIG, out_dir=tmp_path, settle_steps=150)

    # tolerancia 1e-2 rad: los actuadores ceden ~6 mrad bajo gravedad
    assert np.allclose(data.qpos[:7], sb.INIT_ARM_QPOS, atol=1e-2)
    assert np.allclose(data.qpos[7:9], sb.INIT_FINGER_QPOS, atol=1e-3)

    cup_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, sb.CUP_JOINT_NAME)]
    assert np.allclose(data.qpos[cup_adr:cup_adr + 2], params.cup_xyz[:2], atol=2e-4)
    assert abs(data.qpos[cup_adr + 2] - (params.cup_xyz[2] - 0.001)) < 0.002

    # el XML quedó escrito en la ruta canónica de out_dir
    assert sb.scene_path_for(params, tmp_path).exists()
