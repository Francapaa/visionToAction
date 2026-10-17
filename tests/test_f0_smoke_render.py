"""F0 — el script de smoke render corre de punta a punta y produce un video legible.

Este test ejecuta `scripts/smoke_render.py` con parámetros mínimos (64x48, 0.3 s)
en un subprocess, usando el mismo intérprete de Python del entorno de tests.
Valida: carga de la escena de menagerie, stepping, render offscreen y escritura mp4.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_render.py"
SCENE = ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"


@pytest.mark.smoke
def test_smoke_render_produces_video(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), f"Falta el script: {SCRIPT}"
    assert SCENE.is_file(), f"Falta la escena de menagerie: {SCENE}"

    out = tmp_path / "smoke.mp4"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--width", "64",
            "--height", "48",
            "--seconds", "0.3",
            "--fps", "10",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"smoke_render falló:\n{result.stdout}\n{result.stderr}"
    assert out.is_file(), "No se generó el video de smoke"
    assert out.stat().st_size > 0, "El video de smoke está vacío"
