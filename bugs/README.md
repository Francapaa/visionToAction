# Registro de bugs — Visión→Acción (ATLAS)

Post-mortems de los bugs que fuimos encontrando: qué se vio, cómo lo cazamos, la causa
raíz, el fix y la lección. **Un archivo por bug**, índice abajo. El detalle de runs y
métricas asociadas va en [`reports/experiments.md`](../reports/experiments.md); acá va
el análisis técnico.

Convención de este registro: cada bug con fix no trivial se documenta en la misma fase
en que se resuelve (índice acá + `<fase>-NN-<slug>.md`). Las hipótesis descartadas
también se escriben: son las que evitan re-caminar el mismo terreno.

## Índice

| # | Bug | Fase | Estado | Archivo |
|---|---|---|---|---|
| f2-01 | Copa eyectada del agarre en el lift (micro-rodadura sobre las esquinitas de los pads) | F2 | resuelto | [`f2-01`](f2-01-agarre-micro-rodadura.md) |
| f2-02 | El experto aplastaba la copa contra el fondo de la caja (z local ≠ z mundo) | F2 | resuelto | [`f2-02`](f2-02-fondo-caja-z-local.md) |
| f2-03 | El workspace de spawn incluía un bolsillo inalcanzable para el Panda | F2 | resuelto | [`f2-03`](f2-03-workspace-inalcanzable.md) |
| f2-04 | Drift del controlador con acción cero (target re-anclado a la pose medida) | F2 | resuelto | [`f2-04`](f2-04-controlador-drift-sag.md) |
| f2-05 | Sobrepaso del descenso por "recorrido pendiente" del target virtual | F2 | resuelto | [`f2-05`](f2-05-descenso-sobrepaso.md) |
| f2-06 | Diagnóstico inválido por editar un `env.model` obsoleto tras `reset()` | F2 | resuelto (proceso) | [`f2-06`](f2-06-test-modelo-obsoleto.md) |
| f2-07 | Anti-patrón: perseguir el xy vivo de la copa durante el cierre (revertido) | F2 | revertido | [`f2-07`](f2-07-close-tracking-copa.md) |

## Plantilla

````md
# <id> — <título corto>

- **Fase:** <F0|F1|…> · **Estado:** <resuelto|revertido|mitigado|abierto> · **Fecha:** <AAAA-MM-DD>
- **Archivos tocados:** `<paths>`

## Síntoma
Los hechos observados, sin interpretación.

## Cómo lo cacé
El proceso de diagnóstico: mediciones, benches, traces e hipótesis descartadas.

## Causa raíz
El mecanismo real (la parte que, si no queda escrita, se repite).

## Fix
Qué se cambió exactamente y dónde.

## Evidencia
Cómo se comprobó el fix (tests, métricas, clips).

## Lección
La regla general para no volver a pisarlo.
````
