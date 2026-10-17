"""Colector de demos: corre el experto sobre (tarea, seed) y guarda HDF5 + índice.

Uso típico desde scripts/collect_demos.py. El índice (`index.jsonl`) queda en la
raíz del dataset y los episodios en <out>/train/<task>/demo_<seed>.h5. Los videos
(opcionales) se guardan en <out>/videos/ y NO van dentro del HDF5.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from vta.data.dataset import append_index, write_episode
from vta.envs.env import PickPlaceEnv
from vta.envs.scene_builder import BOX_BODY_NAMES
from vta.eval.success import cup_place_success
from vta.experts.waypoint_expert import WaypointExpert

log = logging.getLogger(__name__)

INDEX_NAME = "index.jsonl"
VIDEO_FPS = 20          # 1 frame por control step (20 Hz) = tiempo real
SETTLE_STEPS = 30       # pasos de reposo tras retirarse el gripper (no grabados)


@dataclass
class EpisodeResult:
    task: str
    color: str
    target_box: str
    seed: int
    success: bool
    steps: int
    path: str
    duration_s: float
    video: str | None = None
    cup_final: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metrics: dict[str, Any] = field(default_factory=dict)
    phases: list[list[Any]] = field(default_factory=list)


def _rel_posix(path: Path, base: Path) -> str:
    """Ruta relativa POSIX si está dentro de `base`; absoluta si no."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def parse_task(task: str) -> tuple[str, str]:
    color, sep, box = task.partition("_")
    if not sep or box not in BOX_BODY_NAMES:
        raise ValueError(f"Tarea inválida {task!r}; formato esperado: <color>_<a|b> (ej. red_a)")
    return color, box


def collect_episode(
    env: PickPlaceEnv,
    task: str,
    seed: int,
    out_dir: Path | str,
    *,
    max_steps: int = 500,
    video_path: Path | str | None = None,
) -> EpisodeResult:
    """Corre un episodio del experto y lo guarda. Devuelve el resumen."""
    start = time.perf_counter()
    out_dir = Path(out_dir)
    color, box = parse_task(task)

    obs, info = env.reset(seed=seed, options={"color": color, "target_box": box})
    assert env.model is not None and env.data is not None and env.controller is not None
    expert = WaypointExpert(env.model, env.controller, target_box=box)

    images: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    cup_pos: list[np.ndarray] = []
    cup_quat: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    contacts: list[str] = []

    writer = None
    if video_path is not None:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264", quality=8, macro_block_size=None)
    try:
        for _ in range(max_steps):
            pos, quat = env.get_body_pose("cup")
            images.append(obs["image"])
            proprio.append(obs["proprio"])
            cup_pos.append(pos)
            cup_quat.append(quat)
            action = expert.act(env.data)
            actions.append(action.astype(np.float32))
            contacts.append(";".join(f"{c['body1']}|{c['body2']}" for c in env.get_contacts()))
            if writer is not None:
                writer.append_data(env.render())
            obs, *_ = env.step(action)
            if expert.done:
                break
        else:
            log.warning("experto no terminó en %d pasos (task=%s seed=%d)", max_steps, task, seed)

        # asentado final: la taza suelta cae los últimos milímetros
        zero = np.zeros(7, dtype=np.float32)
        for _ in range(SETTLE_STEPS):
            env.step(zero)
            if writer is not None:
                writer.append_data(env.render())
    finally:
        if writer is not None:
            writer.close()

    metrics = cup_place_success(env.model, env.data, box)
    steps = len(actions)
    h5_path = out_dir / "train" / task / f"demo_{seed:06d}.h5"
    lighting = {
        "key_pos": info["scene"]["key_pos"],
        "key_diffuse": info["scene"]["key_diffuse"],
        "fill_diffuse": info["scene"]["fill_diffuse"],
    }
    write_episode(
        h5_path,
        images=np.stack(images),
        proprio=np.stack(proprio),
        cup_pos=np.stack(cup_pos),
        cup_quat=np.stack(cup_quat),
        actions=np.stack(actions),
        contacts=contacts,
        attrs={
            "task_id": task,
            "seed": seed,
            "cup_color": color,
            "target_box": box,
            "expert_success": int(metrics["success"]),
            "n_steps": steps,
            "spawn": np.asarray(info["scene"]["cup_xyz"], dtype=np.float64),
            "lighting": json.dumps(lighting, ensure_ascii=False),
            "metrics": json.dumps({k: v for k, v in metrics.items()}, ensure_ascii=False),
        },
    )

    duration = time.perf_counter() - start
    cup_final_pos, _ = env.get_body_pose("cup")
    result = EpisodeResult(
        task=task,
        color=color,
        target_box=box,
        seed=seed,
        success=bool(metrics["success"]),
        steps=steps,
        path=h5_path.as_posix(),
        duration_s=round(duration, 3),
        video=Path(video_path).as_posix() if video_path is not None else None,
        cup_final=tuple(round(float(v), 4) for v in cup_final_pos),
        metrics={k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        phases=[list(h) for h in expert.history],
    )

    append_index(out_dir / INDEX_NAME, {
        "task": task,
        "color": color,
        "target_box": box,
        "seed": seed,
        "success": result.success,
        "steps": steps,
        "duration_s": result.duration_s,
        "file": _rel_posix(h5_path, out_dir),
        "video": _rel_posix(Path(video_path), out_dir) if video_path is not None else None,
        "spawn": [round(float(v), 6) for v in info["scene"]["cup_xyz"]],
        "cup_final": [round(float(v), 4) for v in result.cup_final],
        "phases": result.phases,
    })
    return result


def collect_demos(
    *,
    tasks: list[str] | tuple[str, ...],
    n_per_task: int,
    out_dir: Path | str,
    seed0: int = 0,
    videos: int = 0,
    max_steps: int = 500,
    env: PickPlaceEnv | None = None,
) -> list[EpisodeResult]:
    """Corre la grilla completa y devuelve los resultados (loguea el progreso)."""
    out_dir = Path(out_dir)
    own_env = env is None
    env = env or PickPlaceEnv()
    results: list[EpisodeResult] = []
    total = len(tasks) * n_per_task
    try:
        for task in tasks:
            for i in range(n_per_task):
                seed = seed0 + i
                video_path = None
                if len(results) < videos:
                    video_path = out_dir / "videos" / f"{task}_s{seed:04d}.mp4"
                result = collect_episode(env, task, seed, out_dir, max_steps=max_steps, video_path=video_path)
                results.append(result)
                log.info(
                    "[%d/%d] %s seed=%d → %s | %d pasos | %.1fs",
                    len(results), total, task, seed,
                    "ÉXITO" if result.success else "fallo",
                    result.steps, result.duration_s,
                )
    finally:
        if own_env:
            env.close()

    for task in dict.fromkeys(tasks):
        rows = [r for r in results if r.task == task]
        rate = sum(r.success for r in rows) / len(rows) if rows else 0.0
        log.info("tasa de éxito %s: %d/%d (%.0f%%)", task, sum(r.success for r in rows), len(rows), 100 * rate)
    overall = sum(r.success for r in results) / len(results) if results else 0.0
    log.info("tasa global: %d/%d (%.0f%%)", sum(r.success for r in results), len(results), 100 * overall)
    return results
