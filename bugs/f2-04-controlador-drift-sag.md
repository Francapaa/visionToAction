# f2-04 — Drift del controlador con acción cero (target re-anclado a la pose medida)

- **Fase:** F2 · **Estado:** resuelto · **Fecha:** 2026-10-17
- **Archivos tocados:** `src/vta/controller/ik.py`, `tests/test_controller.py`, `scripts/view_scene.py`

## Síntoma

Con **acción cero** (el brazo debería quedarse quieto), el TCP derivaba lentamente: el
servo de posición de MuJoCo no compensa gravedad perfectamente y el error se acumulaba
paso a paso. En una corrida de 100 pasos de acción nula, ~3.5 mm de drift y creciendo.

## Cómo lo cacé

- Medición directa: episodio con acciones de brazo nulas, pose del TCP registrada por
  paso → deriva monotónica.
- Comparación de patrones de referencia:
  (a) re-anclar el target a la pose **medida** en cada paso → drift;
  (b) target virtual que solo se mueve con las **acciones** → quieto (salvo el error
  numérico del servo, que ya no se acumula porque nunca se "acepta" como referencia).

## Causa raíz

Re-anclar el target a la pose medida hace que cualquier error del servo (sag por
gravedad, fricción articular) quede **adoptado** como nueva referencia: el error se
integra paso a paso. Es el equivalente en control a caminar mirándose los pies.

## Fix

`TaskSpaceController` con **target virtual persistente**:

- Las acciones **mueven el target** (Δ ≤ 2 cm/step; Δrot ≤ 5°/step); acción cero lo deja donde está.
- **Correa (leash)**: el target nunca se separa más de `TARGET_LEASH = 2 cm` de la pose
  real, y el tilt máximo respecto de la orientación real es `TARGET_MAX_TILT = 45°`
  → si el brazo queda bloqueado (contacto, límite articular) el target no acumula
  recorrido pendiente (ver f2-05).
- `reset_targets(data)`: ancla el target a la pose al resetear; el viewer lo llama al
  regenerar escena.

## Evidencia

- `test_zero_action_holds_pose` (regresión, `tests/test_controller.py`): 100 pasos de
  acción cero → drift ~**3.5 mm** total del TCP.
- El contrato quedó documentado en `AGENTS.md` §5.2 (base para las políticas F3+).

## Lección

Servo de posición puro + re-anclaje por paso ⇒ drift garantizado. El patrón estable
para control por acciones **relativas** es target virtual persistente con **correa**
respecto de la pose real.
