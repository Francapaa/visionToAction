"""F1 — Abre una escena en el viewer interactivo de MuJoCo (Simulate).

Uso:
    .venv\\Scripts\\python.exe scripts\\view_scene.py --random --seed 7
    .venv\\Scripts\\python.exe scripts\\view_scene.py --color green --box b
    .venv\\Scripts\\python.exe scripts\\view_scene.py --expert --seed 4   # experto F2 en vivo
    .venv\\Scripts\\python.exe scripts\\view_scene.py --check        # sin GUI

Controles en la ventana:
    mouse izq: orbitar · mouse der: pan · scroll: zoom
    doble click: seleccionar cuerpo · Ctrl + arrastrar: empujar objetos
    R: nueva escena aleatoria (con --random) o reiniciar al experto (con --expert)
    espacio: pausa/play · cerrar la ventana: salir

La escena se carga por el mismo camino que el env (`load_scene`: keyframe
`init` + asentado), así lo que se ve acá es lo que va a ver la política.
"""

from __future__ import annotations

import argparse
import logging
import time

import mujoco
import mujoco.viewer

from vta.controller.ik import TaskSpaceController
from vta.envs.scene_builder import (
    BOX_BODY_NAMES,
    CUP_JOINT_NAME,
    DEFAULT_SETTLE_STEPS,
    INIT_KEY_NAME,
    SceneConfig,
    SceneParams,
    default_scene,
    load_scene,
    sample_scene,
    scene_id,
)
from vta.eval.success import cup_place_success
from vta.experts.waypoint_expert import WaypointExpert

log = logging.getLogger("view_scene")

FPS = 60.0        # ritmo objetivo del loop del viewer (pacer de tiempo real)
CONTROL_HZ = 20.0  # ritmo del controlador F2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="seed de la escena inicial")
    parser.add_argument("--random", action="store_true", help="samplear posición/luces por seed")
    parser.add_argument("--color", type=str, default=None, help="color de la taza (default: red)")
    parser.add_argument("--box", type=str, default=None, choices=sorted(BOX_BODY_NAMES),
                        help="caja objetivo (default: a)")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--check", action="store_true",
                        help="solo carga y asienta la escena, sin abrir la ventana")
    parser.add_argument("--expert", action="store_true",
                        help="corre el experto scripted F2 en vivo (R reinicia el episodio)")
    return parser.parse_args()


def build_params(args: argparse.Namespace, seed: int, config: SceneConfig) -> SceneParams:
    if args.random:
        return sample_scene(seed, color=args.color, target_box=args.box, config=config)
    return default_scene(color=args.color, target_box=args.box, config=config)


def run_viewer(params: SceneParams, config: SceneConfig, *, settle_steps: int, allow_regenerate: bool) -> bool:
    """Abre el viewer y simula en tiempo real. Devuelve True si hay que regenerar (R)."""
    model, data = load_scene(params, config=config, settle_steps=settle_steps)
    steps_per_frame = max(1, round(1.0 / (FPS * float(model.opt.timestep))))
    regenerate = [False]
    paused = [False]

    def on_key(keycode: int) -> None:
        key = chr(keycode).lower() if 0 < keycode < 0x110000 else ""
        if key == "r" and allow_regenerate:
            regenerate[0] = True
        elif key == " ":
            paused[0] = not paused[0]
            log.info("pausa %s", "activada" if paused[0] else "desactivada")

    log.info(
        "escena %s (%s → caja %s) — R: nueva escena · espacio: pausa · Ctrl+arrastrar: empujar objetos",
        scene_id(params), params.cup_color, params.target_box,
    )
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running() and not regenerate[0]:
            frame_start = time.perf_counter()
            if not paused[0]:
                mujoco.mj_step(model, data, nstep=steps_per_frame)
            viewer.sync()
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - frame_start)))
    return regenerate[0]


def run_expert_viewer(params: SceneParams, config: SceneConfig, *, settle_steps: int) -> None:
    """Corre al experto scripted (F2) en vivo sobre la escena elegida."""
    model, data = load_scene(params, config=config, settle_steps=settle_steps)
    controller = TaskSpaceController(model, data)
    expert = WaypointExpert(model, controller, target_box=params.target_box)
    decimation = max(1, round(1.0 / (CONTROL_HZ * float(model.opt.timestep))))

    restart = [False]
    paused = [False]

    def on_key(keycode: int) -> None:
        key = chr(keycode).lower() if 0 < keycode < 0x110000 else ""
        if key == "r":
            restart[0] = True
        elif key == " ":
            paused[0] = not paused[0]
            log.info("pausa %s", "activada" if paused[0] else "desactivada")

    def reset_episode() -> WaypointExpert:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, INIT_KEY_NAME)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        controller.reset_targets(data)
        return WaypointExpert(model, controller, target_box=params.target_box)

    log.info(
        "experto F2: %s → caja %s (%s) — espacio: pausa · R: reiniciar · cerrar: salir",
        params.cup_color, params.target_box, scene_id(params),
    )
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()
            if restart[0]:
                restart[0] = False
                expert = reset_episode()
                log.info("experto: episodio reiniciado")
            if not paused[0] and not expert.done:
                action = expert.act(data)
                controller.step(data, action)
                mujoco.mj_step(model, data, nstep=decimation)
            viewer.sync()
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - frame_start)))

    metrics = cup_place_success(model, data, params.target_box)
    log.info(
        "experto F2: %s | dx=%.3f dy=%.3f up_z=%.2f",
        "ÉXITO" if metrics["success"] else "fallo", metrics["dx"], metrics["dy"], metrics["up_z"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = SceneConfig.from_yaml()
    seed = args.seed

    while True:
        params = build_params(args, seed, config)

        if args.check:
            model, data = load_scene(params, config=config, settle_steps=args.settle_steps)
            cup_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUP_JOINT_NAME)]
            log.info("check OK — %s | nq=%d | taza z=%.4f",
                     scene_id(params), model.nq, float(data.qpos[cup_adr + 2]))
            return

        if args.expert:
            run_expert_viewer(params, config, settle_steps=args.settle_steps)
            return

        if run_viewer(params, config, settle_steps=args.settle_steps, allow_regenerate=args.random):
            seed += 1
            continue
        return


if __name__ == "__main__":
    main()
