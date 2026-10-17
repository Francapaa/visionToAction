"""F1 — Genera escenas y las renderiza para validación visual.

Uso:
    .venv\\Scripts\\python.exe scripts\\build_scene.py --random --seed 0
    .venv\\Scripts\\python.exe scripts\\build_scene.py --random --seed 0 --n 10 --out reports\\validation\\f1 --video

Salidas (en --out):
    scene_XX_<color>_<caja>.png   render de cam_scene por escena
    f1_scenes.png                 contact sheet con todas las escenas
    scenes.json                   parámetros de randomización de cada escena
    f1_orbit.mp4                  órbita de cámara libre (con --video)

Los XML generados quedan en data/scenes/<scene_id>/scene.xml (fuera de git).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import imageio
import imageio.v3 as iio
import mujoco
import numpy as np

from vta.envs.scene_builder import (
    BOX_BODY_NAMES,
    CAM_SCENE_NAME,
    DEFAULT_SETTLE_STEPS,
    ROOT,
    SceneConfig,
    SceneParams,
    default_scene,
    load_scene,
    sample_scene,
    scene_id,
    scene_path_for,
)

log = logging.getLogger("build_scene")

DEFAULT_TASKS = ("red_a", "red_b", "blue_a", "blue_b", "green_a", "yellow_b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="seed base (escenas usan seed+i)")
    parser.add_argument("--n", type=int, default=1, help="cantidad de escenas a generar")
    parser.add_argument("--random", action="store_true", help="samplear posición/luces por seed")
    parser.add_argument("--tasks", type=str, default=None,
                        help="lista de tareas separadas por coma (default: ciclo rojo/azul/verde/amarillo)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "validation" / "f1")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--video", action="store_true", help="generar órbita f1_orbit.mp4")
    return parser.parse_args()


def parse_task(task: str) -> tuple[str, str]:
    color, sep, box = task.partition("_")
    if not sep or box not in BOX_BODY_NAMES:
        raise ValueError(f"Tarea inválida {task!r}; formato esperado: <color>_<a|b> (ej. red_a)")
    return color, box


def render_scene(params: SceneParams, config: SceneConfig, *, width: int, height: int, settle_steps: int) -> np.ndarray:
    model, data = load_scene(params, config=config, settle_steps=settle_steps)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=CAM_SCENE_NAME)
        return renderer.render().copy()


def contact_sheet(frames: list[np.ndarray], *, cols: int = 5, gap: int = 4) -> np.ndarray:
    rows = math.ceil(len(frames) / cols)
    h, w = frames[0].shape[:2]
    canvas = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 235, dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * (h + gap): r * (h + gap) + h, c * (w + gap): c * (w + gap) + w] = frame
    return canvas


def render_orbit(
    params: SceneParams,
    out_path: Path,
    config: SceneConfig,
    *,
    width: int,
    height: int,
    settle_steps: int,
    n_frames: int = 121,
    fps: int = 30,
) -> None:
    model, data = load_scene(params, config=config, settle_steps=settle_steps)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.25, 0.0, 0.5)
    cam.distance = 1.8
    cam.elevation = -22.0
    with mujoco.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
    ) as writer:
        for k in range(n_frames):
            cam.azimuth = 90.0 + 360.0 * k / (n_frames - 1)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = SceneConfig.from_yaml()
    tasks = args.tasks.split(",") if args.tasks else list(DEFAULT_TASKS)
    args.out.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    first_params: SceneParams | None = None
    for i in range(args.n):
        task = tasks[i % len(tasks)]
        color, box = parse_task(task)
        if args.random:
            params = sample_scene(args.seed + i, color=color, target_box=box, config=config)
        else:
            params = default_scene(color=color, target_box=box, config=config)
        first_params = first_params or params

        frame = render_scene(params, config, width=args.width, height=args.height, settle_steps=args.settle_steps)
        frames.append(frame)
        name = f"scene_{i:02d}_{color}_{box}.png"
        iio.imwrite(args.out / name, frame)

        entries.append({
            "index": i,
            "task": f"{color}_{box}",
            "scene_id": scene_id(params),
            "scene_xml": scene_path_for(params).relative_to(ROOT).as_posix(),
            "params": params.to_dict(),
        })
        log.info("escena %02d (%s) → %s", i, task, name)

    if len(frames) > 1:
        iio.imwrite(args.out / "f1_scenes.png", contact_sheet(frames))

    (args.out / "scenes.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    if args.video and first_params is not None:
        render_orbit(
            first_params, args.out / "f1_orbit.mp4", config,
            width=args.width, height=args.height, settle_steps=args.settle_steps,
        )
        log.info("video: %s", args.out / "f1_orbit.mp4")

    log.info("listo: %d escena(s) en %s", len(frames), args.out)


if __name__ == "__main__":
    main()
