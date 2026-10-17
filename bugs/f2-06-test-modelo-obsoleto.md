# f2-06 — Diagnóstico inválido por editar un `env.model` obsoleto tras `reset()`

- **Fase:** F2 · **Estado:** resuelto (lección de proceso) · **Fecha:** 2026-10-17
- **Archivos tocados:** ninguno (bug de metodología de diagnóstico; se usó para el bench de f2-01)

## Síntoma

Investigando el derrape de la copa ([`f2-01`](f2-01-agarre-micro-rodadura.md)), probé
5 variantes de parámetros del contacto "en vivo" (fricción 1.0 / 1.5 / 3.0, `solref`
blando, `solimp`) y **las 5 dieron resultados bit-idénticos**. Conclusión errónea
tentadora: "los parámetros no importan / MuJoCo ignora mis cambios". Estuve a punto de
descartar la vía que después resultó ser la correcta.

## Causa raíz

`env.reset()` **reconstruye** `env.model`/`env.data` (vuelve a compilar la escena). Mi
patrón de test era:

```python
env.reset(seed=24)
m = env.model             # capturado UNA vez...
...
env.reset(seed=24)        # ...¡y reset() reemplaza el modelo!
m.geom_friction[...] = …  # esto edita el modelo VIEJO, que el env ya no usa
```

`env.step()` usa el `env.model` nuevo: las ediciones iban a un objeto muerto. Las
"variantes" eran idénticas porque **el experimento nunca cambió nada**.

## Fix (proceso)

- Re-capturar `env.model` **después de cada reset** (o editar el XML y recién ahí
  resetear — que es lo que hace el pipeline real de escena).
- Verificar que el cambio "pegó" leyendo el parámetro **desde el estado vivo**
  (p. ej. `c.friction` de un contacto activo en `env.data.contact`), no desde la
  variable que edité.

Con eso corregido, el test discriminó perfecto: `condim=6` + torsional mató el creep
de la copa; subir la fricción de deslizamiento 3× no. Y el bench aislado (geometría
exacta de los pads) confirmó el mecanismo.

## Evidencia

- Antes (modelo obsoleto): 5 variantes → timings de vx idénticos hasta el último decimal.
- Después (modelo vivo): `condim=6` + torsional → vx ≈ 0; `friction=3.0` → sigue
  re-acelerando (+0.02…+0.04 m/s).

## Lección

En emuladores que reconstruyen el modelo por episodio, **jamás** asumir que una
referencia vieja al modelo sigue viva. Y todo resultado negativo "demasiado limpio"
merece un chequeo de "¿de verdad mi cambio llegó al sistema?" antes de descartar la
hipótesis.
