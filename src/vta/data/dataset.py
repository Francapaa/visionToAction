"""Escritura/lectura de episodios HDF5 (AGENTS §5.5).

Layout por episodio:
    attrs: task_id, seed, cup_color, target_box, expert_success, n_steps,
           spawn (zona de spawn [x,y,z]), lighting (JSON str), metrics (JSON str)
    obs/image       uint8   (T, 84, 84, 3)
    obs/proprio     float32 (T, 22)
    state/cup_pos   float64 (T, 3)
    state/cup_quat  float64 (T, 4)
    actions         float32 (T, 7)  — espacio normalizado [-1,1]
    contacts        utf-8   (T,)    — "body1|body2;body1|body2" por paso
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

log = logging.getLogger(__name__)

EPISODE_VERSION = 1


def write_episode(
    path: Path | str,
    *,
    images: np.ndarray,
    proprio: np.ndarray,
    cup_pos: np.ndarray,
    cup_quat: np.ndarray,
    actions: np.ndarray,
    contacts: list[str],
    attrs: dict[str, Any],
) -> Path:
    """Escribe un episodio HDF5 (bytes determinísticos salvo compresión)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["version"] = EPISODE_VERSION
        for key, value in attrs.items():
            f.attrs[key] = value
        obs = f.create_group("obs")
        obs.create_dataset("image", data=images, dtype=np.uint8, compression="gzip", compression_opts=4)
        obs.create_dataset("proprio", data=proprio, dtype=np.float32)
        state = f.create_group("state")
        state.create_dataset("cup_pos", data=cup_pos, dtype=np.float64)
        state.create_dataset("cup_quat", data=cup_quat, dtype=np.float64)
        f.create_dataset("actions", data=actions, dtype=np.float32)
        f.create_dataset(
            "contacts", data=np.array(contacts, dtype=object), dtype=h5py.string_dtype("utf-8")
        )
    log.debug("episodio escrito en %s", path)
    return path


def read_episode(path: Path | str) -> dict[str, Any]:
    """Lee un episodio HDF5 completo a memoria (dict de arrays + attrs)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        contacts = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["contacts"][...]
        ]
        return {
            "attrs": dict(f.attrs),
            "image": f["obs/image"][...],
            "proprio": f["obs/proprio"][...],
            "cup_pos": f["state/cup_pos"][...],
            "cup_quat": f["state/cup_quat"][...],
            "actions": f["actions"][...],
            "contacts": contacts,
        }


def append_index(index_path: Path | str, record: dict[str, Any]) -> None:
    """Agrega una línea JSON al índice de episodios (append atómico por línea)."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_index(index_path: Path | str) -> list[dict[str, Any]]:
    index_path = Path(index_path)
    if not index_path.exists():
        return []
    return [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
