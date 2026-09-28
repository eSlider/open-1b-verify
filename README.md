# 1b — Gensyn open-1b

Независимая верификация *auditable training* Gensyn **open-1b** и локальный запуск модели.

- **Epic:** vado/1b#1
- **Отчёты:** [`reports/`](reports/)
- **Тела issue:** [`docs/issues/`](docs/issues/)

## Итоги (кратко)

| Задача | Статус |
|--------|--------|
| Аудит init-unit open-1b (CPU) | ✅ **MATCH** — `state_hash 16554a11…`; см. [`reports/audit-open1b-init.md`](reports/audit-open1b-init.md) |
| Аудит тренировочного интервала | ⛔ заблокирован (нет опубликованного interval-unit + RAM/диск) |
| Локальный inference на Arc B580 | 🔄 в работе |

## Ссылки

- Анонс: https://www.gensyn.ai/news/introducing-open-1b-auditable-training
- Harness: https://github.com/gensyn-ai/open-transformers
- Audit app: https://open1b.gensyn.ai/

## Ограничения (по состоянию на 2026-09)

- `repop` (воспроизводимые ядра) — **не open source**, только wheel под «Gensyn Reproducibility License v1.0».
- v3-хэш **не покрывает** RNG, курсор data-stream, spike state и ключи `meta.json`.
- В self-serve audit-ките опубликованы **только init-units**; interval-units — «added per run once … published».
