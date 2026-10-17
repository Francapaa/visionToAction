# f2-07 — Anti-patrón: perseguir el xy vivo de la copa durante el cierre (revertido)

- **Fase:** F2 · **Estado:** revertido · **Fecha:** 2026-10-17
- **Archivos tocados:** `src/vta/experts/waypoint_expert.py` (reversión; quedan comentarios en el código)

## Síntoma

Para "ayudar" al centrado de la copa, cambié la fase CLOSE (y el arranque del LIFT)
para que el brazo **persiguiera el xy vivo de la copa** mientras las mordazas cerraban.
Regresión fuerte: seeds que pasaban (s0, s5, s24) empezaron a fallar — la captura se
rompía, la copa salía disparada o quedaba torcida entre los pads.

## Causa raíz

El servo perseguidor **bombea** el vaivén de la copa: los pads empujan en fase con la
oscilación (que ellos mismos alimentan) y la terminan eyectando. El cierre necesita un
brazo **quieto**: las mordazas (pads + tendón) capturan y centran la copa solas, como
un gripper real. Perseguir el objeto en la fase de contacto íntimo es un lazo de
realimentación positiva con la oscilación.

## Fix

Revertido: durante el cierre el experto mantiene `_hold_target` (congelado al entrar a
la fase) con acción de brazo nula; el tramo inicial del lift sale vertical puro sobre el
xy del agarre. Quedaron **comentarios** en el código explicando por qué no se re-intenta.

## Evidencia

- A/B: con tracking → s0/s5/s24 fallan; revertido → vuelven los verdes.
- La aceptación final (**120/120**) y el dataset v1 corren con el comportamiento
  revertido (brazo quieto en el cierre).

## Lección

En fases de contacto íntimo (cierre de mordazas, extracción), no perseguir variables
vivas del objeto: el centrado lo hace la **geometría/fricción de las mordazas**, no el
control. Un cambio que "ayuda" pero rompe regresiones se revierte y se documenta el
porqué (acá, con comentario en el código).
