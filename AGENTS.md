# AGENTS.md — Visión→Acción (ATLAS)

> Guía operativa del proyecto. Es la fuente de verdad para agentes (OpenCode, Claude, etc.)
> y humanos. Mantenerlo actualizado cuando cambien formatos, comandos o el protocolo
> experimental. Leer completo antes de cualquier cambio.

## 0. Estado actual

- **F0 (setup)** — ✅ cerrada 2026-10-17: venv (Python 3.14) + deps + menagerie + render offscreen validado en Windows (WGL) → `reports/validation/f0_smoke.mp4`.
- **F1 (escena)** — próxima: mesa + taza + caja + cámaras + randomización + registro de contactos.
- Actualizar esta sección al cerrar cada fase.

## 1. Qué es este proyecto

Política neuronal que mapea **(imagen RGB, estado propioceptivo, task_id) → acciones continuas**
en espacio de tarea, entrenada por **imitación** sobre un experto scripted, dentro de una
**escena MuJoCo propia**: brazo Franka Panda + tazas de colores + cajas sobre una mesa.

- Tareas: `"agarrá la taza {color} y ponela en la caja {A|B}"`.
- Colores de entrenamiento: **rojo, azul**. Colores OOD (generalización): **verde, amarillo**.
- Diferencial del proyecto: comparación rigurosa contra **baselines** + estudio cuantitativo de
  **generalización** (posición, color, iluminación, distractores, cámara).
- Condicionamiento: one-hot de task_id (F3–F4) y embeddings CLIP de la instrucción (F5+).

## 2. Stack y versiones

- Python 3.14 (sistema), venv en `.venv/`. Usar `.venv\Scripts\python.exe` directamente.
- Deps en `requirements.txt`: `mujoco>=3.16`, `torch`/`torchvision` (CPU en local), `numpy`,
  `opencv-python`, `mink` (IK diferencial), `gymnasium` (solo interfaz Env),
  `imageio` + `imageio-ffmpeg` (videos), `h5py` (datasets), `PyYAML` (configs), `tqdm`,
  `tensorboard`, `pandas`, `matplotlib`, `pytest`.
- Assets: subset de `mujoco_menagerie/franka_emika_panda` en
  `assets/mujoco_menagerie/` (licencia upstream del modelo, ver su `LICENSE`;
  "Franka" es marca registrada → uso solo educativo/investigación).
  **No editar los assets originales**: cualquier overlay va en `scenes/`.
- Máquina de desarrollo: Windows, i5-9400F, 16 GB RAM, sin CUDA (AMD RX 6500 XT).
  Render offscreen vía `mujoco.Renderer` (WGL). Plan B si falla: WSL2.
- Entrenamientos serios: Kaggle (GPU gratis, 30 h/semana) o Colab. Datasets como
  **Kaggle Private Dataset**.

## 3. Comandos canónicos

```powershell
python -m venv .venv                                  # una sola vez
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\smoke_render.py      # F0: render offscreen + video
.venv\Scripts\python.exe scripts\build_scene.py --random --seed 0   # escena → PNG
.venv\Scripts\python.exe scripts\collect_demos.py --tasks red_a,red_b,blue_a,blue_b --n 80 --out data\demos\v1
.venv\Scripts\python.exe -m pytest -m smoke           # tests rápidos (<60 s)
.venv\Scripts\python.exe -m pytest                    # suite completa
.venv\Scripts\python.exe scripts\train.py --config configs\bc_vision_v1.yaml
.venv\Scripts\python.exe scripts\evaluate.py --ckpt checkpoints\<run>\best.pt --splits g0,g1,g2,g3,g4,g5
```

## 4. Estructura del repo

```
visionToAction/
  AGENTS.md
  requirements.txt
  pyproject.toml     # metadata + config de pytest (markers smoke/slow)
  assets/            # menagerie (subset Panda) — prístino
  scenes/            # plantillas MJCF + overlays propios
  configs/           # YAML por experimento
  scripts/           # smoke_render, build_scene, collect_demos, train, evaluate, export_dataset
  src/vta/
    envs/            # Env gymnasium, scene_builder (XML paramétrico), randomización
    controller/      # IK (mink) + mapeo acción normalizada → juntas
    experts/         # experto scripted (waypoints), plan B SAC+HER
    data/            # collector, dataset HDF5, sampler, augmentations
    policies/        # MLP, CNN encoder, ACT-like, Diffusion Policy, CLIP conditioning
    train/           # loops de BC/DP, logging, checkpoints
    eval/            # evaluator, splits OOD, métricas, tablas
  nbs/               # notebooks de Kaggle
  tests/             # pytest (ver §7)
  data/              # FUERA DE GIT: demos, exports
  checkpoints/       # FUERA DE GIT
  reports/           # runs.jsonl, eval/*.csv, videos chicos, experiments.md
```

## 5. Arquitectura técnica

### 5.1 Escena (MJCF)
- `scenes/base.xml`: mesa + Panda + luces parametrizables + cámaras `cam_scene`
  (tercera persona; render 320×240 → 84×84), `cam_top` (opcional), `cam_wrist` (F5+).
- Taza: cilindro hueco, ~3.5 cm radio, ~8 cm alto; color por material
  (`red/blue/green/yellow` + texturas OOD).
- Cajas: 5 geoms (4 paredes + fondo) soldadas al mundo; caja **A** (izquierda), **B** (derecha).
- Distractores: tazas/objetos extra (F5+).
- `envs/scene_builder.py`: genera el XML paramétrico (spawn, colores, luces, distractores).
  **Toda randomización se registra en `info` del env y en los attrs del HDF5.**
- Determinismo: misma seed → misma escena (hay test).

### 5.2 Controlador (acción → juntas)
- Interfaz compartida política/experto: acción **7D normalizada en [-1,1]**:
  Δpos(3) [máx 2 cm/step], Δrot(3) [tipo axis-angle, máx ~5°/step], gripper(1) [+ cierra].
- Frecuencia de control: **20 Hz** (decimando sim).
- IK diferencial con **mink** (límites articulares; evitación de colisiones como constraint en F5+).
- Determinista y testeado: test de convergencia IK a poses alcanzables.

### 5.3 Observaciones
- `image`: uint8 [84,84,3] de `cam_scene` (F4+ evalúa sumar `cam_wrist`).
- `proprio`: ee_pos(3) + ee_quat(4) + ancho gripper(1) + qpos(7) + qvel(7).
  El contrato exacto se **congela en F2** y se documenta en el código.
- `task`: one-hot de task_id (F3–F4); embedding CLIP del texto (F5+, experimento).
- Ground truth (poses de objetos, contactos): **solo experto y evaluación, nunca input de política**.

### 5.4 Experto scripted
- Máquina de estados de waypoints: pre-grasp → descender → cerrar → levantar → sobre caja →
  descender → abrir → retirar, con pausas de asentamiento.
- Se valida por etapas con video: reach → grasp → lift → place.
- **Criterio de aceptación: ≥90% de éxito en 30 seeds por tarea** (distribución de entrenamiento).
- Definición de éxito: centro de la taza objetivo dentro del interior de la caja objetivo
  (XY dentro del rectángulo, z sobre el fondo) y taza parada (`dot(up, world_z) > 0.7`).
- Plan B si no llega: SAC+HER (stable-baselines3) en la misma escena.

### 5.5 Dataset
- HDF5 por episodio: `data/demos/v1/<split>/<task_id>/demo_000123.h5`
  - grupos: `obs/image`, `obs/proprio`, `state/objects`, `actions`, `contacts`
  - attrs: `task_id`, `seed`, `spawn`, `lighting`, `expert_success`
- Solo episodios con `expert_success=1` entran a entrenamiento.
- **Splits de evaluación (idénticos para TODAS las políticas):**
  - G0 in-distribution · G1 posiciones OOD · G2 colores OOD (verde/amarillo) ·
    G3 iluminación OOD · G4 distractores no vistos · G5 perturbación de cámara.
  - 50 episodios por split × 3 seeds de evaluación.

### 5.6 Políticas y entrenamiento
- F3 baseline: MLP (estado + task) → acción.
- F4 política propia: CNN pequeña (~1–3M params, imagen + proprio + task), augmentations
  de color/ruido, normalización de acciones.
- F5: DAgger (consultas online al experto), action chunking (estilo ACT), mini Diffusion
  Policy (U-Net 1D), comparación CLIP vs one-hot.
- Configs YAML en `configs/`, seeds fijas, checkpoints en `checkpoints/<run_name>/`,
  métricas en `reports/runs.jsonl`.
- Nombre de run: `f{fase}_{multi|single}_{modelo}_{variante}_s{seed}`.
- Local: solo smoke runs (verifican que corre). Runs reales: Kaggle (`nbs/` + scripts).

### 5.7 Evaluación
- `scripts/evaluate.py` corre cualquier política sobre G0–G5 → `reports/eval/<run>.csv`
  (éxito global, éxito por tarea, colisiones totales, pasos promedio).
- Baselines obligatorios en toda tabla: **random, experto (oracle), BC-state**.
- Ninguna afirmación del informe sin tabla reproducible detrás.

## 6. Convenciones

- Conventional commits; commits chicos; tags `f0-done`, `f1-done`, ...
- **Nunca** commitear datos ni checkpoints (`.gitignore`: `data/`, `checkpoints/`, `*.h5`,
  `*.pt`, `*.zip`). Videos chicos para README solo en `reports/`.
- Python con type hints; configs como dataclasses cargadas de YAML; `logging`, no `print`.
- Toda randomización nueva: registrada en `info` y en attrs del HDF5.
- Assets de menagerie prístinos; overlays en `scenes/`.

## 7. Testing (regla dura)

- Cada módulo nuevo llega con sus tests pytest en el mismo commit.
- Mínimos por área:
  - env: determinismo por seed, shapes de obs/acción, límites del action space.
  - controller: convergencia IK, determinismo.
  - expert: ≥90% éxito en 30 seeds (marcar `slow`).
  - dataset: shapes, dtypes, attrs completos, consistencia.
  - eval: determinismo por seed, splits correctos.
- `pytest -m smoke` debe pasar en <60 s antes de cada commit.
- Validación post-implementación por fase = smoke + test de la fase + video corto guardado
  en `reports/validation/`.

## 8. Roadmap y criterios de "done"

| Fase | Entregable | Done cuando... |
|---|---|---|
| F0 Setup | venv + deps + menagerie + render offscreen | video del Panda renderizado offscreen guardado en `reports/validation/` + test `smoke` en verde |
| F1 Escena | mesa + taza + caja + cámaras + randomización + contactos | 10 escenas aleatorias renderizadas (color/luz/posición varían) |
| F2 Experto + dataset | waypoint expert + collector HDF5 | ≥90% éxito experto (30 seeds/tarea) + 320 demos de 4 tareas validadas |
| F3 BC-state | MLP (estado+task) + evaluador + logging | pipeline reproducible end-to-end con seeds fijas |
| F4 BC-vision | CNN + task embedding; smoke local + runs Kaggle | success medible en G0 y tabla vs F3/random |
| F5 Iteraciones | augmentations, DAgger, chunking, mini-DP, CLIP | cada variante = hipótesis + run + resultado en `experiments.md` |
| F6 Generalización + informe | G1–G5 completos, ablations, videos, write-up | tabla maestra + figuras de robustez + README con GIFs |

## 9. Flujo Kaggle/Colab

- Export: `python scripts/export_dataset.py --v v1 --out data/export/v1.zip` (<2 GB).
- Subir como Kaggle Private Dataset `vta-demos-v1`.
- Entrenar: `nbs/kaggle_train.ipynb` (lee `/kaggle/input/vta-demos-v1`).
- Descargar outputs (checkpoints) → `checkpoints/`.
- Registrar cada run en `reports/experiments.md`: hipótesis → comando → resultado → conclusión.

## 10. Decisiones abiertas

- Resolución final de imagen (84×84 vs 64×64): decidir en F4 con smoke runs.
- `mink` como IK por defecto; fallback: DLS propio (~40 líneas, ya referenciado).
- Licencia del repo (MIT vs Apache-2.0): decidir antes de F6.
- Nombre público del proyecto ("ATLAS" es provisorio).
