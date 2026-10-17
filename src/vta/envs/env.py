"""Env gymnasium de la escena F1 (mesa + Panda + taza + cajas).

Contrato de observaciones (se congela en F2; ver AGENTS.md §5.3):

    obs = {
        "image":   uint8 [84, 84, 3]  — cam_scene 320x240 → resize INTER_AREA,
        "proprio": float32 [22]       — [ee_pos(3), ee_quat(4), gripper(1),
                                         qpos_arm(7), qvel_arm(7)]
    }
    action = float32 [7] en [-1, 1]   — Δpos(3), Δrot(3), gripper(1)

Notas de contrato:
  - `ee_*` = pose del cuerpo `panda_hand` (marco del hand de menagerie).
  - `gripper` = qpos[finger_joint1] + qpos[finger_joint2] (0=cerrado, 0.08=abierto).

En F1 NO hay controlador todavía: `step()` avanza la física con el brazo
retenido en la pose inicial, y la acción se valida/clipea pero no se aplica
(el mapeo acción→juntas llega en F2 junto al controlador mink).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import gymnasium
import mujoco
import numpy as np

from vta.envs import scene_builder as sb

log = logging.getLogger(__name__)

CONTROL_HZ = 20.0
OBS_IMAGE_SIZE = 84
RENDER_WIDTH = 320
RENDER_HEIGHT = 240
SETTLE_STEPS = 250
PROPRIO_DIM = 22
ACTION_DIM = 7


class PickPlaceEnv(gymnasium.Env):
    """Escena F1: randomiza la taza, carga el modelo y expone obs/render/contactos."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(CONTROL_HZ)}

    def __init__(
        self,
        *,
        config: sb.SceneConfig | None = None,
        scene_dir: Path | str | None = None,
        render_width: int = RENDER_WIDTH,
        render_height: int = RENDER_HEIGHT,
        obs_image_size: int = OBS_IMAGE_SIZE,
        settle_steps: int = SETTLE_STEPS,
    ) -> None:
        super().__init__()
        self._config = config or sb.SceneConfig.from_yaml()
        self._scene_dir = Path(scene_dir) if scene_dir is not None else sb.DEFAULT_SCENE_DIR
        self._render_width = render_width
        self._render_height = render_height
        self._obs_image_size = obs_image_size
        self._settle_steps = settle_steps

        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Dict({
            "image": gymnasium.spaces.Box(
                0, 255, shape=(obs_image_size, obs_image_size, 3), dtype=np.uint8
            ),
            "proprio": gymnasium.spaces.Box(-np.inf, np.inf, shape=(PROPRIO_DIM,), dtype=np.float32),
        })

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.scene_params: sb.SceneParams | None = None
        self.scene_path: Path | None = None

        self._renderer: mujoco.Renderer | None = None
        self._decimation = 1
        self._arm_qpos_adr = np.zeros(0, dtype=np.int64)
        self._arm_qvel_adr = np.zeros(0, dtype=np.int64)
        self._finger_qpos_adr = np.zeros(0, dtype=np.int64)
        self._hand_body_id = -1
        self._cup_body_id = -1

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reinicia la escena.

        `options` acepta:
          - {"params": SceneParams}: escena explícita (tests/eval con splits G*).
          - {"color": str, "target_box": str}: color/caja de la tarea.

        Con seed fija, la escena es determinística (misma seed ⇒ misma XML y
        misma física). Sin seed, se deriva del RNG interno de gymnasium.
        """
        super().reset(seed=seed)
        options = dict(options or {})
        params = options.get("params")
        if params is not None and not isinstance(params, sb.SceneParams):
            raise TypeError("options['params'] debe ser un SceneParams")
        if params is None:
            sample_seed = seed
            if sample_seed is None:
                sample_seed = int(self.np_random.integers(0, 2**31 - 1))
            params = sb.sample_scene(
                sample_seed,
                color=options.get("color"),
                target_box=options.get("target_box"),
                config=self._config,
            )

        self.scene_params = params
        self.scene_path = sb.write_scene(params, config=self._config, out_dir=self._scene_dir)

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self._resolve_indices()
        self._decimation = max(1, round(1.0 / (CONTROL_HZ * float(self.model.opt.timestep))))

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, sb.INIT_KEY_NAME)
        if key_id < 0:
            raise RuntimeError(f"La escena no tiene keyframe {sb.INIT_KEY_NAME!r}: {self.scene_path}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self.model, height=self._render_height, width=self._render_width)

        info = {
            "seed": params.seed,
            "scene": params.to_dict(),
            "scene_xml": self.scene_path.as_posix(),
        }
        return self._build_obs(), info

    def step(self, action: np.ndarray):
        """Avanza 1/20 s de simulación. F1: sin controlador (ver docstring del módulo)."""
        self._require_loaded()
        act = np.array(action, dtype=np.float32, copy=True)
        if act.shape != self.action_space.shape:
            raise ValueError(f"shape de acción {act.shape}; esperado {self.action_space.shape}")
        np.clip(act, self.action_space.low, self.action_space.high, out=act)
        for _ in range(self._decimation):
            mujoco.mj_step(self.model, self.data)
        return self._build_obs(), 0.0, False, False, {}

    def render(self) -> np.ndarray:
        """Frame completo (320x240) de cam_scene."""
        self._require_loaded()
        assert self._renderer is not None
        self._renderer.update_scene(self.data, camera=sb.CAM_SCENE_NAME)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ helpers

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Posición (3,) y quat (4,) en el mundo de un cuerpo con nombre."""
        self._require_loaded()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise KeyError(f"body {body_name!r} no existe en la escena")
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def get_contacts(
        self,
        *,
        bodies: set[str] | None = None,
        min_normal_force: float = 0.0,
    ) -> list[dict]:
        """Contactos activos con nombres resueltos.

        - `bodies`: si se pasa, filtra contactos donde participe alguno de esos
          cuerpos (p. ej. {"cup", "panda_left_finger", "panda_right_finger"}).
        - Nota: los geoms estáticos (mesa, piso) cuelgan del body "world".
        """
        self._require_loaded()
        model, data = self.model, self.data
        buf = np.zeros(6)
        contacts: list[dict] = []
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
            if bodies is not None and not ({body1, body2} & bodies):
                continue
            mujoco.mj_contactForce(model, data, i, buf)
            normal = float(buf[0])
            if normal < min_normal_force:
                continue
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"
            contacts.append({
                "geom1": geom1,
                "geom2": geom2,
                "body1": body1,
                "body2": body2,
                "dist": float(con.dist),
                "normal_force": normal,
            })
        return contacts

    # ------------------------------------------------------------------ internos

    def _require_loaded(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("El env no está inicializado: llamá a reset() primero")

    def _resolve_indices(self) -> None:
        assert self.model is not None
        model = self.model

        def joint_id(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name!r} no existe en la escena {self.scene_path}")
            return jid

        arm = [joint_id(n) for n in sb.ARM_JOINT_NAMES]
        fingers = [joint_id(n) for n in sb.FINGER_JOINT_NAMES]
        self._arm_qpos_adr = np.array([model.jnt_qposadr[j] for j in arm], dtype=np.int64)
        self._arm_qvel_adr = np.array([model.jnt_dofadr[j] for j in arm], dtype=np.int64)
        self._finger_qpos_adr = np.array([model.jnt_qposadr[j] for j in fingers], dtype=np.int64)
        self._hand_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sb.HAND_BODY_NAME))
        self._cup_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sb.CUP_BODY_NAME))

    def _build_obs(self) -> dict[str, np.ndarray]:
        self._require_loaded()
        assert self._renderer is not None

        self._renderer.update_scene(self.data, camera=sb.CAM_SCENE_NAME)
        frame = self._renderer.render()
        image = cv2.resize(
            frame, (self._obs_image_size, self._obs_image_size), interpolation=cv2.INTER_AREA
        )
        image = np.ascontiguousarray(image, dtype=np.uint8)

        hand_pos = self.data.xpos[self._hand_body_id]
        hand_quat = self.data.xquat[self._hand_body_id]
        gripper_width = float(self.data.qpos[self._finger_qpos_adr].sum())
        arm_qpos = self.data.qpos[self._arm_qpos_adr]
        arm_qvel = self.data.qvel[self._arm_qvel_adr]
        proprio = np.concatenate(
            [hand_pos, hand_quat, [gripper_width], arm_qpos, arm_qvel]
        ).astype(np.float32)
        return {"image": image, "proprio": proprio}
