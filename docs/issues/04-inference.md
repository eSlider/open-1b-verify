Epic: #EPIC

## Цель

Поднять open-1b локально и доказать работоспособность на нашем железе
(Intel Arc B580, Vulkan; без NVIDIA/CUDA).

## Критерии приёмки

- [ ] llama.cpp собран с Vulkan; бэкенд активен на Arc B580
- [ ] веса HF (`Gensyn/open-1b-sft`, fallback `-base`) скачаны, provenance = точная ревизия
- [ ] конвертация в GGUF; собран/выбран квант
- [ ] `llama-cli`/`llama-server` даёт связный ответ; замер tok/s и VRAM
- [ ] отчёт с точными командами и caveat'ами

## Прочее

- Сервисы — только docker compose (не systemd).
- Обвязка — Go; Python только через `uv` (канонический `convert_hf_to_gguf.py`).

## Статус

🔄 выполняется фоновым SE-субагентом (2026-09-28).
