# Registro de experimentos

Toda corrida (local, Kaggle o Colab) se registra acá con el formato
**hipótesis → comando → resultado → conclusión**. Incluir también los runs fallidos.

| Fecha | Run | Fase | Hipótesis | Comando | Resultado | Conclusión |
|---|---|---|---|---|---|---|
| 2026-10-17 | f0_smoke | F0 | MuJoCo renderiza offscreen (WGL) en Windows y escribe mp4 con imageio-ffmpeg | `.venv\Scripts\python.exe scripts\smoke_render.py` | 118 frames 320×240 (249 KB); `pytest -m smoke` → 1 passed (1.8 s); torch 2.14.1+cpu | ✅ WGL funciona, no hace falta WSL2. F0 cerrada |
