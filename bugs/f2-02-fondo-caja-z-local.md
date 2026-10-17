# f2-02 — El experto aplastaba la copa contra el fondo de la caja (z local ≠ z mundo)

- **Fase:** F2 · **Estado:** resuelto · **Fecha:** 2026-10-17
- **Archivos tocados:** `src/vta/experts/waypoint_expert.py`

## Síntoma

Casi todos los episodios terminaban con `timeout en fase descend_into_box
(paso ~300–370)` (120 pasos de fase agotados) aunque el resultado final fuera exitoso:
el brazo bajaba hasta **clavar la copa contra el fondo de la caja** y seguía empujando
(3–5 cm de sobre-recorrido) hasta que el timeout forzaba el OPEN. Episodios de
~310–420 pasos, con la fase final patológica.

## Cómo lo cacé

Trace paso a paso de la fase (tcp, copa y contactos):

- `tcp_z ≈ 0.451` estable y `cup_z ≈ 0.449` (fondo de la caja en 0.41): la copa ya
  estaba apoyada y apretada.
- Sin embargo, el **target** de la fase seguía en `zt ≈ +0.057`. El servo jamás podía
  alcanzarlo → empujaba para siempre.

Los dos números (0.057 vs 0.457) delataron un error de marco de referencia + un offset
sistemático.

## Causa raíz

`_box_floor_top = floor_geom.pos[2] + floor_geom.size[2]`: en MJCF, `geom.pos` es
**local al body** (acá, dentro de `box_a`/`box_b`): daba 0.005 + 0.005 = **0.01** en vez
de la z **mundo** del fondo (0.41). El target Z apuntaba "al piso de la mesa".

Segundo error latente: la copa cuelga ~4 mm por debajo del TCP dentro de los pads; el
target del TCP no lo compensaba (liberar el TCP a "5 mm del fondo" dejaba la copa en el fondo).

## Fix

`waypoint_expert.py`:

- `_floor_local_top` (z del fondo dentro del body) + `floor_top = box_pos[2] + _floor_local_top`
  calculado del **estado vivo** (z mundo), no del XML.
- `offset_z = tcp_z − cup_z` medido en vivo: el target del TCP deja el **fondo de la
  copa** a `PLACE_GAP` (~5 mm) del fondo de la caja.

## Evidencia

- Post-fix: **0 timeouts** de `descend_into_box` (antes: casi todos los episodios).
- Liberación limpia a 5 mm del fondo (copa final en `z = 0.450` = 0.41 + 0.04, exacto).
- Episodios 198–305 pasos (mediana 229) vs ~310–420 antes del fix.

## Lección

- En MJCF, `geom.pos`/`body.pos` de bodies anidados son **locales**: todo target de
  tarea se calcula en mundo desde el estado (o sumando la pose del body padre).
- El "colgado" del objeto dentro del gripper es un offset sistemático que hay que
  **medir**, no asumir: liberar a la altura correcta requiere compensarlo.
