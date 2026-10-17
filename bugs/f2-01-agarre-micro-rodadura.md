# f2-01 — Copa eyectada del agarre en el lift (micro-rodadura sobre las esquinitas de los pads)

- **Fase:** F2 · **Estado:** resuelto · **Fecha:** 2026-10-17
- **Archivos tocados:** `scenes/base.xml` (geom `cup_collision` y visuales de la taza)

## Síntoma

Con la copa agarrada y el TCP prácticamente quieto, la copa **derrapaba** ~0.5 mm por
paso de control (~10 mm/s) a través de los pads durante el lift. Al llegar al borde de
los pads la normal se inclinaba y la **eyectaba**: los dedos cerraban en falso
(gripper 0.035 → 0) y la copa caía. Se veía con el seed 24 (spawn ≈ (0.292, −0.038)) en
las 4 tareas.

Datos que no cerraban a primera vista:

- Los contactos reportaban `friction=1.5` y había ~8 contactos normales (±y) de ~0.44 N.
- La fricción **sí** sostenía el peso en z (≈0.054 N por contacto × 8 ≈ 0.43 N = peso).
- Pero en la dirección del derrape (x) la fuerza tangencial era ≈0. ¡Y el solver
  convergía! (Fricción "de equilibrio" en z, ausente en x.)

## Cómo lo cacé

1. **Fuerzas de contacto** (`mj_contactForce` + frames): por contacto, normal ±y,
   tangencial principal alineada con la gravedad (z) — no con el deslizamiento (x).
2. **Bench aislado** (MuJoCo puro: cilindro + 2 cajas estáticas + patada de 5 cm/s):
   con cajas grandes 8.5×20×30 mm la fricción **frena en ~0.03 s** → el motor de física
   está bien.
3. **Bench con la geometría exacta** de los pads del menagerie (las 4 cajitas de
   6×4×6 mm por dedo, en sus posiciones reales): la copa frena… y después **re-acelera
   sola hasta −0.10 m/s**. ¡Creep reproducido en aislamiento! (No era el brazo ni el tendón.)
4. Hipótesis descartadas por el camino: subir la fricción de deslizamiento (3× no cambia
   nada: sigue re-acelerando), `solref`/`solimp` suaves, opciones del solver (todas
   default y `niter=1`).

## Causa raíz

Los pads del Panda (menagerie) no son una superficie plana: son **cajas chicas** y la
copa apoyaba sobre sus **esquinitas**. Con `condim=3` (solo fricción de deslizamiento)
nada resiste la **micro-rodadura**: la copa "rueda/camina" microscópicamente sobre las
aristas — como caminar un mueble — y avanza sin deslizar de verdad. Por eso la fricción
de deslizamiento no la frena: casi no hay deslizamiento que frenar.

## Fix

`scenes/base.xml`, geom `cup_collision` (+ ajustes en el mismo geom):

- `condim="6"` + `friction="1.5 0.3 0.001"`: **fricción torsional 0.3** — la que ancla la copa.
- `priority="2"` + `solref="0.008 1"` + `solimp="0.95 0.99 0.001"`: contactos más
  rígidos; sin esto el gripper sobre-cierra sobre la taza "blanda" y la cuña de los
  dedos la clava en la mesa.
- `mass="0"` en los geoms **visuales** de la taza: MuJoCo les asignaba su densidad
  default (1000) y la taza pesaba ~10× lo esperado (la masa sale solo del geom de
  colisión, con la densidad de la config).

Queda un comentario largo en el XML explicando el porqué, para que no se revierta "por limpieza".

## Evidencia

- **Kick-test** en el env (patada de 5 cm/s al soltar y acción cero después):
  - Antes: la velocidad se absorbía ~90% y **re-aceleraba sola hasta ~+0.03 m/s**.
  - Con el fix: |vx| ≲ 3 mm/s (≈0) durante los pasos siguientes.
- Seed 24 pasa en ambas cajas.
- Variante descartada medida: fricción de deslizamiento 3.0 → el creep persiste
  (confirma que la causa era la torsional, no la fricción "raspante").
- Aceptación posterior: **120/120 (100%)** en 30 seeds × 4 tareas.

## Lección

- Para objetos cilíndricos agarrados de costado entre pads pequeños, la fricción de
  deslizamiento sola no alcanza: hace falta **resistencia torsional** (`condim≥4`;
  acá `condim=6`, que agrega también la rodante).
- Aislar el fenómeno en un **bench con la geometría exacta** es la forma más rápida de
  separar "bug de física/MuJoCo" de "dinámica del sistema completo".
