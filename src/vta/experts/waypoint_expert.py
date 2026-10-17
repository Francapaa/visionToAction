"""Experto scripted de pick-and-place por waypoints (F2).

Fases: PRE_GRASP → DESCEND → CLOSE → LIFT → TRANSFER → DESCEND_INTO_BOX →
OPEN → RETREAT.

Cada fase emite acciones en el MISMO espacio que usará la política (7D
normalizado en [-1,1]) mediante un servo P en espacio de tarea sobre el TCP
del gripper. El ground truth (pose viva de la taza, posición de la caja) se
lee del estado de la simulación en cada paso: SOLO el experto y la evaluación
pueden usarlo, nunca como input de política (AGENTS §5.3).
"""

from __future__ import annotations

import enum
import logging

import mujoco
import numpy as np

from vta.controller.ik import MAX_DELTA_POS, MAX_DELTA_ROT, TaskSpaceController, rotation_error_world
from vta.envs.scene_builder import BOX_BODY_NAMES, CUP_BODY_NAME

log = logging.getLogger(__name__)

# Orientación objetivo: herramienta apuntando hacia abajo (180° sobre +x).
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

HOVER_HEIGHT = 0.12   # pre-grasp: sobre el centro de la taza
LIFT_HEIGHT = 0.60    # altura de transferencia (sobre las cajas, world z)
PLACE_GAP = 0.005     # la taza se suelta esta altura sobre el fondo de la caja

POS_TOL = 0.008       # tolerancia vertical para avanzar de fase
XY_TOL_GRASP = 0.004  # alineación lateral fina antes de descender/cerrar
Z_TOL_GRASP = 0.005   # altura fina del agarre (los pads cubren ±5 mm del TCP)
ROT_TOL = 0.10        # tolerancia de orientación (rad)
STABLE_STEPS = 2      # pasos consecutivos dentro de tolerancia
CLOSE_STEPS = 18      # pasos para cerrar el gripper
OPEN_STEPS = 18       # pasos para abrirlo

# Rampas de velocidad (fracción del cap de 2 cm/step) en fases con carga: el
# tirón inicial con la taza agarrada puede sacarla del gripper por inercia.
LIFT_MAX_ACTION = 0.4       # 8 mm/step al despegar
TRANSFER_MAX_ACTION = 0.5   # 1 cm/step llevando la taza
DESCEND_MAX_ACTION = 0.5    # 1 cm/step al meterla en la caja


class Phase(enum.Enum):
    PRE_GRASP = "pre_grasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    TRANSFER = "transfer"
    DESCEND_INTO_BOX = "descend_into_box"
    OPEN = "open"
    RETREAT = "retreat"
    DONE = "done"


_PHASE_ORDER = (
    Phase.PRE_GRASP, Phase.DESCEND, Phase.CLOSE, Phase.LIFT,
    Phase.TRANSFER, Phase.DESCEND_INTO_BOX, Phase.OPEN, Phase.RETREAT, Phase.DONE,
)

# timeout por fase en control steps: si se agota, se avanza igual (con warning)
_TIMEOUTS = {
    Phase.PRE_GRASP: 100,
    Phase.DESCEND: 100,
    Phase.CLOSE: CLOSE_STEPS,
    Phase.LIFT: 100,
    Phase.TRANSFER: 120,
    Phase.DESCEND_INTO_BOX: 120,
    Phase.OPEN: OPEN_STEPS,
    Phase.RETREAT: 100,
}


class WaypointExpert:
    """Experto determinista que resuelve la tarea de pick-and-place."""

    def __init__(self, model: mujoco.MjModel, controller: TaskSpaceController, *, target_box: str) -> None:
        if target_box not in BOX_BODY_NAMES:
            raise ValueError(f"Caja {target_box!r} desconocida; esperado 'a' o 'b'")
        self.model = model
        self.controller = controller
        self.target_box = target_box

        self._cup_id = self._body_id(CUP_BODY_NAME)
        self._box_id = self._body_id(BOX_BODY_NAMES[target_box])

        # geometría derivada del modelo (nada hardcodeado)
        floor_geom = model.geom(f"box_{target_box}_floor")
        # z del fondo DENTRO del body de la caja; en act() se suma la posición
        # mundo del body (la caja es estática y está soldada al mundo)
        self._floor_local_top = float(floor_geom.pos[2] + floor_geom.size[2])
        self._cup_half = float(model.geom("cup_collision").size[1])

        self.phase = Phase.PRE_GRASP
        self.episode_step = 0
        self.phase_steps = 0
        self._stable = 0
        self._hold_target: np.ndarray | None = None
        self._grasp_xy = np.zeros(2)
        self.history: list[tuple[str, int]] = [(self.phase.value, 0)]

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    # ------------------------------------------------------------------ API

    def act(self, data: mujoco.MjData) -> np.ndarray:
        self.episode_step += 1
        self.phase_steps += 1

        if not self.done and self.phase_steps > _TIMEOUTS[self.phase]:
            log.warning("experto: timeout en fase %s (paso %d)", self.phase.value, self.episode_step)
            self._advance(data)

        if self.done:
            action = np.zeros(7)
            action[6] = -1.0
            return action

        cup_pos = data.xpos[self._cup_id]
        box_pos = data.xpos[self._box_id]

        if self.phase is Phase.PRE_GRASP:
            target = cup_pos + np.array([0.0, 0.0, HOVER_HEIGHT])
            action = self._servo(data, target, gripper=-1.0)
            self._maybe_advance(data, target, xy_tol=XY_TOL_GRASP)
        elif self.phase is Phase.DESCEND:
            target = cup_pos.copy()
            action = self._servo(data, target, gripper=-1.0, max_action=DESCEND_MAX_ACTION)
            self._maybe_advance(data, target, xy_tol=XY_TOL_GRASP, z_tol=Z_TOL_GRASP)
        elif self.phase is Phase.CLOSE:
            # brazo QUIETO durante el cierre: las mordazas capturan y centran la
            # copa entre los pads (perseguirla rompe la captura)
            action = self._servo(data, self._hold_target, gripper=1.0)
            if self.phase_steps >= CLOSE_STEPS:
                self._advance(data)
        elif self.phase is Phase.LIFT:
            # lift vertical puro sobre el xy del agarre: sin tironeo lateral
            # (trackear el xy vivo de la copa bombea su vaivén y la pierde)
            target = np.array([self._grasp_xy[0], self._grasp_xy[1], LIFT_HEIGHT])
            action = self._servo(data, target, gripper=1.0, max_action=LIFT_MAX_ACTION)
            self._maybe_advance(data, target, xy_tol=POS_TOL)
        elif self.phase is Phase.TRANSFER:
            target = np.array([box_pos[0], box_pos[1], LIFT_HEIGHT])
            action = self._servo(data, target, gripper=1.0, max_action=TRANSFER_MAX_ACTION)
            self._maybe_advance(data, target, xy_tol=POS_TOL)
        elif self.phase is Phase.DESCEND_INTO_BOX:
            # La taza cuelga unos mm por debajo del TCP dentro de los pads:
            # compensar ese offset (medido en vivo) para soltarla a PLACE_GAP
            # del fondo, sin aplastarla contra la base de la caja.
            tcp = self.controller.tcp_pos(data)
            offset_z = float(tcp[2] - cup_pos[2])
            floor_top = float(box_pos[2]) + self._floor_local_top
            z_target = floor_top + self._cup_half + PLACE_GAP + offset_z
            target = np.array([box_pos[0], box_pos[1], z_target])
            action = self._servo(data, target, gripper=1.0, max_action=DESCEND_MAX_ACTION)
            self._maybe_advance(data, target, xy_tol=POS_TOL)
        elif self.phase is Phase.OPEN:
            action = self._servo(data, self._hold_target, gripper=-1.0)
            if self.phase_steps >= OPEN_STEPS:
                self._advance(data)
        else:  # RETREAT
            target = np.array([box_pos[0], box_pos[1], LIFT_HEIGHT])
            action = self._servo(data, target, gripper=-1.0)
            self._maybe_advance(data, target, xy_tol=POS_TOL)

        return action

    # -------------------------------------------------------------- interno

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"body {name!r} no existe en el modelo")
        return int(bid)

    def _servo(
        self, data: mujoco.MjData, target_pos: np.ndarray, *, gripper: float, max_action: float = 1.0
    ) -> np.ndarray:
        """Acción 7D que acerca el TCP a `target_pos` manteniendo la orientación DOWN.

        `max_action` limita la fracción del cap de 2 cm/step (rampas suaves).
        """
        tcp = self.controller.tcp_pos(data)
        e_pos = np.asarray(target_pos, dtype=np.float64) - tcp
        a_pos = np.clip(e_pos / MAX_DELTA_POS, -max_action, max_action)

        e_rot = rotation_error_world(DOWN_QUAT, self.controller.hand_quat(data))
        a_rot = np.clip(e_rot / MAX_DELTA_ROT, -1.0, 1.0)

        return np.concatenate([a_pos, a_rot, [gripper]])

    def _at_target(
        self, data: mujoco.MjData, target_pos: np.ndarray, *, xy_tol: float, z_tol: float = POS_TOL
    ) -> bool:
        e = np.asarray(target_pos, dtype=np.float64) - self.controller.tcp_pos(data)
        if float(np.linalg.norm(e[:2])) > xy_tol or abs(float(e[2])) > z_tol:
            return False
        rot_err = float(np.linalg.norm(rotation_error_world(DOWN_QUAT, self.controller.hand_quat(data))))
        return rot_err <= ROT_TOL

    def _maybe_advance(
        self, data: mujoco.MjData, target_pos: np.ndarray, *, xy_tol: float, z_tol: float = POS_TOL
    ) -> None:
        if self._at_target(data, target_pos, xy_tol=xy_tol, z_tol=z_tol):
            self._stable += 1
            if self._stable >= STABLE_STEPS:
                self._advance(data)
        else:
            self._stable = 0

    def _advance(self, data: mujoco.MjData) -> None:
        idx = _PHASE_ORDER.index(self.phase)
        self.phase = _PHASE_ORDER[idx + 1]
        self.phase_steps = 0
        self._stable = 0
        if self.phase in (Phase.CLOSE, Phase.OPEN):
            # congelar el objetivo en la posición actual mientras abre/cierra
            self._hold_target = self.controller.tcp_pos(data).copy()
        if self.phase is Phase.CLOSE:
            # xy del agarre: el lift sube en vertical pura desde acá
            self._grasp_xy = self._hold_target[:2].copy()
        self.history.append((self.phase.value, self.episode_step))
