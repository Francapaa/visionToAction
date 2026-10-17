# Registro de experimentos

Toda corrida (local, Kaggle o Colab) se registra acá con el formato
**hipótesis → comando → resultado → conclusión**. Incluir también los runs fallidos.

| Fecha | Run | Fase | Hipótesis | Comando | Resultado | Conclusión |
|---|---|---|---|---|---|---|
| 2026-10-17 | f0_smoke | F0 | MuJoCo renderiza offscreen (WGL) en Windows y escribe mp4 con imageio-ffmpeg | `.venv\Scripts\python.exe scripts\smoke_render.py` | 118 frames 320×240 (249 KB); `pytest -m smoke` → 1 passed (1.8 s); torch 2.14.1+cpu | ✅ WGL funciona, no hace falta WSL2. F0 cerrada |
| 2026-10-17 | f1_scenes | F1 | La escena (Panda vía `attach` + taza + cajas) es determinística por (seed, color, caja) y la randomización de posición/luz/color se ve en los renders | `.venv\Scripts\python.exe scripts\build_scene.py --random --seed 0 --n 12 --out reports\validation\f1 --video` | 12 renders 320×240 + contact sheet + órbita 121 frames; XML byte-idéntico por seed (test); spawn dentro del workspace con ≥1.9 cm a las cajas (200 seeds, test); `pytest -m smoke` → 14 passed (6.4 s) | ✅ F1 cerrada: escena aleatorizada, determinista y testeada; lista para el controlador + experto (F2) |
