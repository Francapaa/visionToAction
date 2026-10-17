# f2-05 — Sobrepaso del descenso por "recorrido pendiente" del target virtual

- **Fase:** F2 · **Estado:** resuelto · **Fecha:** 2026-10-17
- **Archivos tocados:** `src/vta/experts/waypoint_expert.py` (rampas), `src/vta/controller/ik.py` (correa)

## Síntoma

Con descensos comandados a fondo (acción ±1 → 2 cm/step), el brazo llegaba a la copa
**pasándose de largo** (sobrepaso de varios mm) y frenaba tarde: golpecitos a la taza
y captura inestable (el gate de la fase tardaba en estabilizar).

## Cómo lo cacé

Trace de tcp vs target paso a paso durante el descenso: el target virtual iba ~1 cm
adelantado (llegaba antes a la meta), y al pedir el freno el brazo **atravesaba** la
posición objetivo antes de asentarse — la dinámica real no sigue el target 1:1 (servo P).

## Causa raíz

El target virtual avanza 2 cm/step pero el brazo sigue ~0.7–1 cm/step: en descensos
largos el target **se adelanta** y acumula "recorrido pendiente". Al frenar, ese
pendiente se descarga de golpe: sobrepaso y oscilación alrededor del objetivo.

## Fix

Dos capas:

- Controlador: **correa de 2 cm** (el target nunca se separa más que eso del TCP real,
  ver [`f2-04`](f2-04-controlador-drift-sag.md)) — acota el pendiente máximo.
- Experto: rampas de velocidad por fase en tractos con contacto:
  `DESCEND_MAX_ACTION=0.5` (1 cm/step al descender/entrar a la caja),
  `LIFT_MAX_ACTION=0.4` (8 mm/step al despegar con la copa cargada),
  `TRANSFER_MAX_ACTION=0.5` (1 cm/step llevando la taza).

## Evidencia

- Trace post-fix: convergencia monótona al centro de la copa, sin sobrepaso.
- Los descensos de la aceptación final (**120/120**) y de los videos de validación
  (`reports/validation/f2/`) son limpios, sin golpes a la taza.

## Lección

Con acciones relativas + target persistente, hay que **limitar la velocidad en fases
de contacto**: el "recorrido pendiente" del target es un oscilador en potencia.
