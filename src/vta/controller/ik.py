"""Controlador de espacio de tarea: acción 7D [-1,1] → ctrl de juntas (mink IK).

Contrato (AGENTS §5.2):
    acción = [Δpos(3), Δrot(3), gripper(1)] ∈ [-1,1]^7
      - Δpos: desplazamiento del TCP en marco MUNDO, |Δ| ≤ 2 cm por control step.
      - Δrot: rotación axis-angle en marco MUNDO, |Δ| ≤ 5°/step (pre-multiplicada).
      - gripper: +1 cierra, -1 abre.

El TCP es el punto medio de los pads de los dedos; su offset en el frame de
`panda_hand` se mide del modelo al construir el controlador (no hardcodeado).
Se controla el body `panda_hand` con mink (FrameTask, solver daqp, límites de
configuración) y la salida se escribe en `data.ctrl` (actuadores de posición).

Determinista: mismo (qpos, qvel, acción) → mismo ctrl.
"""

from __future__ import annotations

import logging

import mujoco
import mink
import numpy as np

from vta.envs.scene_builder import ARM_JOINT_NAMES

log = logging.getLogger(__name__)

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ
MAX_DELTA_POS = 0.02           # m por control step
MAX_DELTA_ROT = np.deg2rad(5.0)  # rad por control step

# "Correa" del target virtual respecto de la pose real: evita latigazos cuando
# el brazo está bloqueado (el target se mueve con las acciones, no con la pose).
# Si la correa es larga, un descenso más rápido que lo que el brazo puede seguir
# acumula "recorrido pendiente" y el brazo se pasa de largo al frenar.
TARGET_LEASH = 0.02               # m (≈ un paso de acción completo)
TARGET_MAX_TILT = np.deg2rad(45.0)  # rad

GRIPPER_CTRL_OPEN = 255.0      # ctrl del actuador del gripper: 255 = abierto
GRIPPER_CTRL_CLOSED = 0.0

HAND_BODY_NAME = "panda_hand"
FINGER_BODY_NAMES = ("panda_left_finger", "panda_right_finger")


def gripper_action_to_ctrl(action: float) -> float:
    """+1 → 0 (cerrado), -1 → 255 (abierto), 0 → 127.5."""
    return GRIPPER_CTRL_OPEN * (1.0 - float(np.clip(action, -1.0, 1.0))) / 2.0


# ------------------------------------------------------------------ utilidades

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quat (w,x,y,z) del producto q1 * q2 (rotación de q2 seguida de q1)."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def rotation_error_world(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Vector axis-angle (marco mundo) de la rotación q_current → q_target (camino corto)."""
    q_err = _quat_mul(q_target, _quat_inv(q_current))
    if q_err[0] < 0.0:
        q_err = -q_err
    vec = q_err[1:]
    s = float(np.linalg.norm(vec))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(s, q_err[0]))
    return vec / s * angle


class TaskSpaceController:
    """Convierte acciones normalizadas en `data.ctrl` con IK diferencial (mink)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, dt: float = DT) -> None:
        self.model = model
        self.dt = dt

        self._hand_id = self._body_id(HAND_BODY_NAME)
        self._finger_ids = [self._body_id(name) for name in FINGER_BODY_NAMES]

        # sanity: los 7 primeros actuadores deben ser los joints del brazo, en orden
        self._arm_qpos_adr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        for i, joint_name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if int(model.actuator_trnid[i, 0]) != joint_id:
                raise RuntimeError(f"actuador {i} no mapea a {joint_name} (trnid={int(model.actuator_trnid[i, 0])})")
        self._ctrl_low = model.actuator_ctrlrange[:7, 0].copy()
        self._ctrl_high = model.actuator_ctrlrange[:7, 1].copy()

        self.tcp_local = self._measure_tcp_offset(data)

        self._cfg = mink.Configuration(model)
        self._task = mink.FrameTask(HAND_BODY_NAME, "body", position_cost=1.0, orientation_cost=0.3)
        self._limits = [mink.ConfigurationLimit(model)]

        # target VIRTUAL persistente (mundo): las acciones lo mueven; no se
        # re-ancla a la pose medida (el sag del servo de posición se acumularía).
        self._target_tcp: np.ndarray | None = None
        self._target_quat: np.ndarray | None = None

    # -------------------------------------------------------------- público

    def reset_targets(self, data: mujoco.MjData) -> None:
        """Ancla el target virtual a la pose actual (llamar tras resetear la sim)."""
        self._target_tcp = self.tcp_pos(data).copy()
        self._target_quat = self.hand_quat(data).copy()

    def tcp_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Posición del TCP (punto medio de pads) en el mundo."""
        hand_mat = data.xmat[self._hand_id].reshape(3, 3)
        return data.xpos[self._hand_id] + hand_mat @ self.tcp_local

    def hand_quat(self, data: mujoco.MjData) -> np.ndarray:
        """Quat (w,x,y,z) del body `panda_hand` en el mundo."""
        return data.xquat[self._hand_id].copy()

    def step(self, data: mujoco.MjData, action: np.ndarray) -> None:
        """Aplica una acción normalizada: escribe `data.ctrl` (brazo + gripper).

        Las acciones mueven un target VIRTUAL persistente (no la pose medida): un
        servo de posición sin compensación de gravedad acumula sag si el target
        se re-ancla a la pose real en cada paso. El target queda "atado" a la
        pose real (≤ `TARGET_LEASH` y ≤ `TARGET_MAX_TILT`) para evitar latigazos
        cuando el brazo está bloqueado.
        """
        act = np.clip(np.asarray(action, dtype=np.float64).reshape(7), -1.0, 1.0)
        if self._target_tcp is None:
            self.reset_targets(data)

        self._target_tcp = self._target_tcp + act[:3] * MAX_DELTA_POS

        delta = act[3:6] * MAX_DELTA_ROT
        angle = float(np.linalg.norm(delta))
        if angle > 1e-12:
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, delta / angle, angle)
            self._target_quat = _quat_mul(dq, self._target_quat)  # delta mundo, pre-multiplicado
            self._target_quat /= np.linalg.norm(self._target_quat)

        self._clamp_target(data)

        target_mat = _quat_to_mat(self._target_quat)
        target_hand_pos = self._target_tcp - target_mat @ self.tcp_local

        self._cfg.update(data.qpos)
        self._task.set_target(
            mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(wxyz=self._target_quat), translation=target_hand_pos
            )
        )
        try:
            vel = mink.solve_ik(
                self._cfg, [self._task], self.dt, solver="daqp", damping=1e-4, limits=self._limits
            )
        except mink.NoSolutionFound:
            log.debug("IK sin solución para acción %s; se mantiene la pose", act)
            vel = np.zeros(self.model.nv)
        self._cfg.integrate_inplace(vel, self.dt)

        q_target = np.clip(self._cfg.q[self._arm_qpos_adr], self._ctrl_low, self._ctrl_high)
        data.ctrl[:7] = q_target
        data.ctrl[7] = gripper_action_to_ctrl(act[6])

    # -------------------------------------------------------------- interno

    def _clamp_target(self, data: mujoco.MjData) -> None:
        """Mantiene el target virtual dentro de la correa (posición y tilt)."""
        assert self._target_tcp is not None and self._target_quat is not None
        tcp = self.tcp_pos(data)
        e_pos = self._target_tcp - tcp
        dist = float(np.linalg.norm(e_pos))
        if dist > TARGET_LEASH:
            self._target_tcp = tcp + e_pos * (TARGET_LEASH / dist)

        r_err = rotation_error_world(self._target_quat, self.hand_quat(data))
        r_norm = float(np.linalg.norm(r_err))
        if r_norm > TARGET_MAX_TILT:
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, r_err / r_norm, TARGET_MAX_TILT - r_norm)
            self._target_quat = _quat_mul(dq, self._target_quat)
            self._target_quat /= np.linalg.norm(self._target_quat)

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"body {name!r} no existe en el modelo")
        return int(bid)

    def _measure_tcp_offset(self, data: mujoco.MjData) -> np.ndarray:
        """Offset del TCP en el frame del hand, medido de los pads de los dedos.

        Los pads de contacto son, por dedo, los 2 geoms más cercanos a la línea
        de cierre (menor |y| local). Se promedian los 4 y se fuerza x=y=0.
        """
        hand_pos = data.xpos[self._hand_id]
        hand_mat = data.xmat[self._hand_id].reshape(3, 3)
        pads: list[np.ndarray] = []
        for finger_id in self._finger_ids:
            geoms = [i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) == finger_id]
            local = [hand_mat.T @ (data.geom_xpos[i] - hand_pos) for i in geoms]
            local.sort(key=lambda v: abs(float(v[1])))
            pads.extend(local[:2])
        tcp = np.mean(pads, axis=0)
        tcp[0] = 0.0
        tcp[1] = 0.0
        return tcp
