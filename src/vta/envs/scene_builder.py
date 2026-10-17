"""Construcción paramétrica de la escena F1 (mesa + Panda + taza + cajas).

Uso típico:
    config = SceneConfig.from_yaml()
    params = sample_scene(seed=0, color="red", target_box="a", config=config)
    path   = write_scene(params, config=config)   # data/scenes/<scene_id>/scene.xml
    model  = mujoco.MjModel.from_xml_path(str(path))

Garantías:
  - Determinismo: (seed, color, target_box) ⇒ SceneParams y XML byte a byte.
  - sample_scene consume el RNG en un orden fijo (ver docstring).
  - El XML generado registra toda la randomización; el env además la copia a
    `info` (ver AGENTS.md §5.1).

El layout fijo (mesa, cajas, montaje del robot, cámaras) vive en scenes/base.xml.
Acá viven los rangos de randomización y la geometría parametrizable de la taza.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = ROOT / "scenes" / "base.xml"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "scene_f1.yaml"
PANDA_XML_PATH = ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
DEFAULT_SCENE_DIR = ROOT / "data" / "scenes"

# --- Nombres del modelo (contrato con env.py, tests y el experto de F2) ---
PANDA_PREFIX = "panda_"
ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
HAND_BODY_NAME = "panda_hand"
CUP_BODY_NAME = "cup"
CUP_JOINT_NAME = "cup_joint"
INIT_KEY_NAME = "init"
CAM_SCENE_NAME = "cam_scene"
CAM_TOP_NAME = "cam_top"
CAM_TARGET_NAME = "cam_scene_target"
BOX_BODY_NAMES = {"a": "box_a", "b": "box_b"}
CUP_MATERIALS = {
    "red": "mat_cup_red",
    "blue": "mat_cup_blue",
    "green": "mat_cup_green",
    "yellow": "mat_cup_yellow",
}

# Pose inicial del brazo: "ready" estándar del Panda (codo al frente, mano
# sobre el centro del workspace). Reemplaza al home de menagerie, que deja al
# gripper fuera de cuadro de cam_scene y no es una pose de agarre natural.
# Gripper: ctrl 255 = abierto (tendon en 0.04), 0 = cerrado.
INIT_ARM_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
INIT_FINGER_QPOS = (0.04, 0.04)
INIT_CTRL = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 255.0)

# Detalles visuales de la taza (colisión = cilindro sólido; visual = hueca).
CUP_RIM_THICKNESS = 0.0065
CUP_BOTTOM_THICKNESS = 0.005
CUP_INNER_ZOFF = -0.001


# ---------------------------------------------------------------- helpers

def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _fmt_vec(values: Sequence[float]) -> str:
    return " ".join(_fmt(v) for v in values)


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quat (w, x, y, z) de una rotación de `yaw` alrededor de +z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _as_pair(values: Any, *, name: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in values)
    if lo > hi:
        raise ValueError(f"Rango inválido para {name}: {lo} > {hi}")
    return lo, hi


def _as_vec3(values: Any, *, name: str) -> tuple[float, float, float]:
    vec = tuple(float(v) for v in values)
    if len(vec) != 3:
        raise ValueError(f"{name} debe tener 3 componentes, tiene {len(vec)}: {vec}")
    return vec  # type: ignore[return-value]


def _mid(pair: tuple[float, float]) -> float:
    return 0.5 * (pair[0] + pair[1])


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class CupSpec:
    radius: float
    half_height: float
    density: float
    spawn_gap: float


@dataclass(frozen=True)
class SceneConfig:
    """Rangos de randomización + geometría configurable de la escena F1."""

    seed: int
    table_top: float
    cup: CupSpec
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    yaw_range: tuple[float, float]
    palette: tuple[str, ...]
    default_color: str
    default_box: str
    key_pos_x: tuple[float, float]
    key_pos_y: tuple[float, float]
    key_pos_z: tuple[float, float]
    key_diffuse: tuple[float, float]
    fill_pos: tuple[float, float, float]
    fill_diffuse: tuple[float, float]
    cam_scene_pos: tuple[float, float, float]
    cam_top_pos: tuple[float, float, float]

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "SceneConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cup = CupSpec(**{k: float(v) for k, v in data["cup"].items()})
        config = cls(
            seed=int(data.get("seed", 0)),
            table_top=float(data["table_top"]),
            cup=cup,
            workspace_x=_as_pair(data["workspace"]["x"], name="workspace.x"),
            workspace_y=_as_pair(data["workspace"]["y"], name="workspace.y"),
            yaw_range=_as_pair(data["workspace"]["yaw"], name="workspace.yaw"),
            palette=tuple(str(c) for c in data["palette"]),
            default_color=str(data["default_color"]),
            default_box=str(data["default_box"]),
            key_pos_x=_as_pair(data["lighting"]["key"]["pos_x"], name="lighting.key.pos_x"),
            key_pos_y=_as_pair(data["lighting"]["key"]["pos_y"], name="lighting.key.pos_y"),
            key_pos_z=_as_pair(data["lighting"]["key"]["pos_z"], name="lighting.key.pos_z"),
            key_diffuse=_as_pair(data["lighting"]["key"]["diffuse"], name="lighting.key.diffuse"),
            fill_pos=_as_vec3(data["lighting"]["fill"]["pos"], name="lighting.fill.pos"),
            fill_diffuse=_as_pair(data["lighting"]["fill"]["diffuse"], name="lighting.fill.diffuse"),
            cam_scene_pos=_as_vec3(data["cameras"]["cam_scene"]["pos"], name="cameras.cam_scene.pos"),
            cam_top_pos=_as_vec3(data["cameras"]["cam_top"]["pos"], name="cameras.cam_top.pos"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.palette:
            raise ValueError("La paleta de colores está vacía")
        for color in self.palette:
            if color not in CUP_MATERIALS:
                raise ValueError(f"Color sin material definido: {color!r}")
        if self.default_color not in self.palette:
            raise ValueError(f"default_color {self.default_color!r} no está en la paleta {self.palette}")
        if self.default_box not in BOX_BODY_NAMES:
            raise ValueError(f"default_box inválido: {self.default_box!r} (esperado 'a' o 'b')")
        if self.cup.radius <= CUP_RIM_THICKNESS or self.cup.half_height <= CUP_BOTTOM_THICKNESS:
            raise ValueError("Geometría de taza inválida para el visual hueco de la plantilla")


def _validate_choice(color: str, target_box: str, config: SceneConfig) -> None:
    if color not in config.palette:
        raise ValueError(f"Color {color!r} fuera de la paleta {config.palette}")
    if target_box not in BOX_BODY_NAMES:
        raise ValueError(f"Caja {target_box!r} desconocida; esperado 'a' o 'b'")


# ---------------------------------------------------------------- params

@dataclass(frozen=True)
class SceneParams:
    """Instancia concreta de escena: todo lo que varía entre episodios."""

    seed: int
    cup_color: str
    target_box: str
    cup_xyz: tuple[float, float, float]
    cup_yaw: float
    key_pos: tuple[float, float, float]
    key_diffuse: float
    fill_diffuse: float

    def to_dict(self) -> dict[str, object]:
        """Versión JSON-serializable (se registra en `info` del env)."""
        return {
            "seed": self.seed,
            "cup_color": self.cup_color,
            "target_box": self.target_box,
            "cup_xyz": list(self.cup_xyz),
            "cup_yaw": self.cup_yaw,
            "key_pos": list(self.key_pos),
            "key_diffuse": self.key_diffuse,
            "fill_diffuse": self.fill_diffuse,
        }


def sample_scene(
    seed: int,
    *,
    color: str | None = None,
    target_box: str | None = None,
    config: SceneConfig | None = None,
) -> SceneParams:
    """Samplea una escena determinística.

    El RNG se consume SIEMPRE en este orden (contrato de reproducibilidad):
    1) cup x, 2) cup y, 3) cup yaw, 4) key light x, 5) key light y,
    6) key light z, 7) key light diffuse, 8) fill light diffuse.
    """
    config = config or SceneConfig.from_yaml()
    color = color or config.default_color
    target_box = target_box or config.default_box
    _validate_choice(color, target_box, config)

    rng = np.random.default_rng(seed)
    cup_x = float(rng.uniform(*config.workspace_x))
    cup_y = float(rng.uniform(*config.workspace_y))
    cup_yaw = float(rng.uniform(*config.yaw_range))
    key_x = float(rng.uniform(*config.key_pos_x))
    key_y = float(rng.uniform(*config.key_pos_y))
    key_z = float(rng.uniform(*config.key_pos_z))
    key_diffuse = float(rng.uniform(*config.key_diffuse))
    fill_diffuse = float(rng.uniform(*config.fill_diffuse))

    cup_z = config.table_top + config.cup.half_height + config.cup.spawn_gap
    return SceneParams(
        seed=int(seed),
        cup_color=color,
        target_box=target_box,
        cup_xyz=(cup_x, cup_y, cup_z),
        cup_yaw=cup_yaw,
        key_pos=(key_x, key_y, key_z),
        key_diffuse=key_diffuse,
        fill_diffuse=fill_diffuse,
    )


def default_scene(
    *,
    color: str | None = None,
    target_box: str | None = None,
    config: SceneConfig | None = None,
) -> SceneParams:
    """Escena sin randomización: puntos medios de los rangos (debug/screenshots)."""
    config = config or SceneConfig.from_yaml()
    color = color or config.default_color
    target_box = target_box or config.default_box
    _validate_choice(color, target_box, config)

    return SceneParams(
        seed=config.seed,
        cup_color=color,
        target_box=target_box,
        cup_xyz=(
            _mid(config.workspace_x),
            _mid(config.workspace_y),
            config.table_top + config.cup.half_height + config.cup.spawn_gap,
        ),
        cup_yaw=0.0,
        key_pos=(_mid(config.key_pos_x), _mid(config.key_pos_y), _mid(config.key_pos_z)),
        key_diffuse=_mid(config.key_diffuse),
        fill_diffuse=_mid(config.fill_diffuse),
    )


# ---------------------------------------------------------------- XML

def build_scene_xml(params: SceneParams, config: SceneConfig | None = None) -> str:
    """Renderiza la plantilla scenes/base.xml con los parámetros de la escena."""
    config = config or SceneConfig.from_yaml()
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    cup_inner_size = (
        config.cup.radius - CUP_RIM_THICKNESS,
        config.cup.half_height - CUP_BOTTOM_THICKNESS,
    )
    cup_quat = _yaw_quat(params.cup_yaw)
    mapping = {
        "panda_xml": PANDA_XML_PATH.as_posix(),
        "key_pos": _fmt_vec(params.key_pos),
        "key_diffuse": _fmt_vec((params.key_diffuse,) * 3),
        "fill_pos": _fmt_vec(config.fill_pos),
        "fill_diffuse": _fmt_vec((params.fill_diffuse,) * 3),
        "cup_pos": _fmt_vec(params.cup_xyz),
        "cup_quat": _fmt_vec(cup_quat),
        "cup_material": CUP_MATERIALS[params.cup_color],
        "cup_density": _fmt(config.cup.density),
        "cup_geom_size": _fmt_vec((config.cup.radius, config.cup.half_height)),
        "cup_inner_size": _fmt_vec(cup_inner_size),
        "cup_inner_z": _fmt(CUP_INNER_ZOFF),
        "cam_scene_pos": _fmt_vec(config.cam_scene_pos),
        "cam_top_pos": _fmt_vec(config.cam_top_pos),
        "init_qpos": _fmt_vec((*INIT_ARM_QPOS, *INIT_FINGER_QPOS, *params.cup_xyz, *cup_quat)),
        "init_ctrl": _fmt_vec(INIT_CTRL),
    }
    return template.format_map(mapping)


def scene_id(params: SceneParams) -> str:
    return f"seed{params.seed:04d}_{params.cup_color}_box{params.target_box}"


def write_scene(
    params: SceneParams,
    *,
    config: SceneConfig | None = None,
    out_dir: Path | str | None = None,
) -> Path:
    """Escribe el XML generado (bytes determinísticos) y devuelve su ruta."""
    config = config or SceneConfig.from_yaml()
    base = Path(out_dir) if out_dir is not None else DEFAULT_SCENE_DIR
    scene_dir = base / scene_id(params)
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / "scene.xml"
    path.write_text(build_scene_xml(params, config), encoding="utf-8", newline="\n")
    log.debug("escena escrita en %s", path)
    return path
