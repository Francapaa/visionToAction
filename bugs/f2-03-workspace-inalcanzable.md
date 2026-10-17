# f2-03 — El workspace de spawn incluía un bolsillo inalcanzable para el Panda

- **Fase:** F2 · **Estado:** resuelto · **Fecha:** 2026-10-17
- **Archivos tocados:** `configs/scene_f1.yaml`; diagnóstico con `data/_grid_script.py` (scratch, no versionado)

## Síntoma

Seeds con la copa cerca de la base del robot (**x ≤ 0.27**) fallaban **siempre** y en
las 4 tareas: el episodio agotaba los 500 pasos atascado en la fase de descenso
("des499"), sin llegar nunca a agarrar. En la aceptación pre-fix: seeds 3 (≈ (0.209, −0.105))
y 11 (≈ (0.224, 0.0)).

## Cómo lo cacé

Grilla de alcance **limpia** (`data/_grid_script.py`: copa teleportada, experto
completo, criterio de éxito oficial), x ∈ [0.18, 0.42] × y ∈ [−0.20, 0.20]:

- Bolsillo inalcanzable: **x ≤ 0.27, y ∈ [−0.14, 0]** → 13/70 celdas con el descenso
  atascado. x ≥ 0.28 y los bordes y = ±0.20: OK.
- Trace del atascado: la **correa del controlador saturada** (~18–20 mm de separación
  target–TCP), TCP en x ≈ 0.300 con la copa en ~0.292, `panda_joint4` pegado a su
  límite inferior (−2.98 vs −3.072 de límite), y contactos del brazo contra su propia
  base/columna (25–56 N en los peores casos).

Señal clave: la "falla" no era de software sino **física**: el brazo simplemente no llega.

## Causa raíz

Con la muñeca hacia abajo (pose de agarre) el Panda no alcanza ese sector: el
antebrazo/codo choca con su columna y las juntas topan antes de que el TCP llegue al
objetivo. El workspace de spawn de F1 (`x ∈ [0.18, 0.52]`) se había definido a ojo,
sin medir la alcanzabilidad real.

## Fix

`configs/scene_f1.yaml`: `workspace.x: [0.18, 0.52] → [0.28, 0.52]`, con un comentario
en el YAML explicando el motivo (la correa se satura, el descenso no converge).
Nada más: la escena sigue randomizando dentro de la región demostrablemente alcanzable.

## Evidencia

- Post-recorte: aceptación **120/120** con spawn x ∈ [0.292, 0.509] (30 seeds × 4 tareas).
- Tests de spawn (200 seeds) y clearances en verde con el nuevo rango.
- Residual conocido: celdas "don" esporádicas fuera del bolsillo
  ((0.20,−0.20), (0.22,−0.20), (0.22,+0.20), (0.30,+0.07) en la grilla) — la corrida
  de aceptación de 120 episodios no las reprodujo; quedan en observación.

## Lección

- Medir el espacio alcanzable con el **criterio real** (episodio completo con el
  experto y el predicado de éxito), no con la cinemática teórica ni a ojo.
- Toda randomización de spawn debe restringirse a la región demostrablemente
  alcanzable, y dejar el motivo documentado junto al rango.
