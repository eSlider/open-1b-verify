Epic: #EPIC

## Цель

Независимо воспроизвести опубликованный **init-хэш** open-1b официальным harness'ом
на CPU (Linux x86-64, без NVIDIA).

## Критерии приёмки

- [x] kit `pt-f91110a0d0c8_rp-c9ca6e71e673` скачан; sha256 **всех** файлов сверены с `kit.json`
- [x] `repop.build_info().commit == kit.json.repop_commit` (`c9ca6e71e673`)
- [x] `pretrain-audit-replay --from-init --until-step 0 --config-name 1b_repop_v2 --device cpu` → `match=true`
- [x] отчёт с raw-выводом и таблицей sha256

## Результат (2026-09-28)

✅ **MATCH**. `state_hash = 16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed`,
wall 58.3 s, peak RSS 16.8 GB, exit 0.

Отчёт: `audit/REPORT.md`.

## Ограничение

Init-unit по README **не** проверяет тренировочный интервал — это отдельная задача
`#AUDIT-INTERVAL`.
