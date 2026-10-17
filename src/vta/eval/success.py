"""Criterio de éxito de la tarea (AGENTS §5.4).

Definición: el CENTRO de la taza objetivo está dentro del interior de la caja
objetivo (XY dentro del rectángulo interior, z sobre el fondo) y la taza está
parada (`dot(up, world_z) > 0.7`). La geometría se deriva del modelo (geoms de
la caja), no de constantes duplicadas.
"""

from __future__ import annotations

import mujoco
import numpy as np

from vta.envs.scene_builder import BOX_BODY_NAMES, CUP_BODY_NAME

UPRIGHT_THRESHOLD = 0.7


def box_interior(model: mujoco.MjModel, data: mujoco.MjData, target_box: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(centro world, media-extensión interior xy, z de la cara superior del fondo)."""
    if target_box not in BOX_BODY_NAMES:
        raise ValueError(f"Caja {target_box!r} desconocida; esperado 'a' o 'b'")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAMES[target_box])
    center = data.xpos[body_id].copy()

    wall_xm = model.geom(f"box_{target_box}_wall_xm")
    wall_ym = model.geom(f"box_{target_box}_wall_ym")
    hx = float(-(wall_xm.pos[0] + wall_xm.size[0]))
    hy = float(-(wall_ym.pos[1] + wall_ym.size[1]))

    floor = model.geom(f"box_{target_box}_floor")
    floor_top = float(center[2] + floor.pos[2] + floor.size[2])
    return center, np.array([hx, hy]), floor_top


def cup_place_success(model: mujoco.MjModel, data: mujoco.MjData, target_box: str) -> dict:
    """Métricas + veredicto de éxito. `success` es la clave que usan los tests."""
    center, half, floor_top = box_interior(model, data, target_box)

    cup_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUP_BODY_NAME)
    cup_pos = data.xpos[cup_id].copy()
    up_world = data.xmat[cup_id].reshape(3, 3)[:, 2]

    dx = float(cup_pos[0] - center[0])
    dy = float(cup_pos[1] - center[1])
    up_z = float(up_world[2])

    inside_xy = abs(dx) <= half[0] and abs(dy) <= half[1]
    above_floor = float(cup_pos[2]) > floor_top
    upright = up_z > UPRIGHT_THRESHOLD

    return {
        "success": bool(inside_xy and above_floor and upright),
        "dx": dx,
        "dy": dy,
        "cup_z": float(cup_pos[2]),
        "floor_top": floor_top,
        "up_z": up_z,
        "inside_xy": bool(inside_xy),
        "above_floor": bool(above_floor),
        "upright": bool(upright),
    }
