"""F2 — Colecta demos del experto scripted a HDF5 + índice (dataset v1).

Uso (comando canónico de AGENTS §3):
    .venv\\Scripts\\python.exe scripts\\collect_demos.py --tasks red_a,red_b,blue_a,blue_b --n 80 --out data\\demos\\v1

Ejemplos:
    ... --tasks red_a --n 3 --videos 3 --out data\\demos\\_prueba   # smoke con videos
    ... --n 30 --out data\\demos\\_aceptacion                       # corrida de aceptación (90%)

Salida:
    <out>/train/<task>/demo_<seed>.h5   episodios (solo expert_success=1 entran a entrenamiento)
    <out>/index.jsonl                   índice (una línea JSON por episodio)
    <out>/videos/*.mp4                  primeros N episodios (si --videos > 0)

Exit code 0 si todo corre; 1 si alguna tarea queda por debajo del 90% de éxito (criterio F2).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vta.data.collector import collect_demos

log = logging.getLogger("collect_demos")

TRAIN_TASKS = ("red_a", "red_b", "blue_a", "blue_b")
SUCCESS_TARGET = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=str, default=",".join(TRAIN_TASKS),
                        help="lista separada por comas (default: las 4 tareas de entrenamiento)")
    parser.add_argument("--n", type=int, default=10, help="episodios por tarea (seeds seed0..seed0+n-1)")
    parser.add_argument("--seed0", type=int, default=0, help="primera seed de la grilla")
    parser.add_argument("--out", type=Path, default=Path("data/demos/v1"), help="directorio del dataset")
    parser.add_argument("--videos", type=int, default=0,
                        help="graba los primeros N episodios en <out>/videos")
    parser.add_argument("--max-steps", type=int, default=500, help="tope de pasos de control por episodio")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    if not tasks:
        log.error("--tasks está vacío")
        return 2

    log.info("colectando %d tarea(s) × %d episodio(s) → %s (seeds %d..%d)",
             len(tasks), args.n, args.out, args.seed0, args.seed0 + args.n - 1)
    results = collect_demos(
        tasks=tasks, n_per_task=args.n, out_dir=args.out,
        seed0=args.seed0, videos=args.videos, max_steps=args.max_steps,
    )

    failed = False
    for task in tasks:
        rows = [r for r in results if r.task == task]
        ok = sum(r.success for r in rows)
        rate = ok / max(1, len(rows))
        flag = "" if rate >= SUCCESS_TARGET else "  ← por debajo del criterio F2 (90%)"
        log.info("%s: %d/%d (%.0f%%)%s", task, ok, len(rows), 100 * rate, flag)
        failed = failed or rate < SUCCESS_TARGET
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
