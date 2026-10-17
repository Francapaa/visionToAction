"""F0 — Smoke test de render offscreen.

Valida de punta a punta:
  1. import de mujoco y carga de la escena de menagerie (Panda),
  2. stepping de física,
  3. render offscreen (WGL en Windows),
  4. escritura de video mp4 con imageio-ffmpeg.

Uso:
    .venv\\Scripts\\python.exe scripts\\smoke_render.py

Salida: reports/validation/f0_smoke.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
DEFAULT_OUT = ROOT / "reports" / "validation" / "f0_smoke.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.scene.is_file():
        raise FileNotFoundError(
            f"No existe la escena: {args.scene}\n"
            "Cloná el subset de mujoco_menagerie en assets/ (ver AGENTS.md sección 2)."
        )

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # Cámara libre mirando al brazo (el scene.xml de menagerie no define cámaras fijas).
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 0.5)
    cam.distance = 2.0
    cam.azimuth = 140.0
    cam.elevation = -20.0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    dt = float(model.opt.timestep)
    n_steps = int(args.seconds / dt)
    step_every = max(1, round(1.0 / (args.fps * dt)))

    rng = np.random.default_rng(0)
    ctrl_home = data.ctrl.copy()
    frames: list[np.ndarray] = []
    for i in range(n_steps):
        phase = i * dt
        # Movimiento suave de prueba alrededor de la pose "home" para que el video
        # muestre dinámica real sin desestabilizar la escena.
        data.ctrl[:] = ctrl_home + 0.2 * np.sin(phase + rng.normal(0.0, 0.1, size=model.nu))
        mujoco.mj_step(model, data)
        if i % step_every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=None
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"OK - {len(frames)} frames renderizados ({args.width}x{args.height}), {args.fps} fps.")
    print(f"Video: {args.out}")
    print(f"mujoco {mujoco.__version__} | nq={model.nq} nu={model.nu} nbody={model.nbody}")


if __name__ == "__main__":
    main()
