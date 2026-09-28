Epic: #EPIC

## Цель

Воспроизвести **тренировочный интервал** (replay шага против опубликованного hash) —
именно это доказывает заявление «bitwise auditable training», в отличие от init-unit.

## Критерии приёмки

- [ ] получен interval-unit с canonical target hash (published commitment на шаг)
- [ ] скачан чекпоинт + шарды, потреблённые интервалом
- [ ] `pretrain-audit-replay --checkpoint … --until-step N --gcs-root … --expect-hash …` → `MATCH`
- [ ] отчёт с raw-выводом

## Блокер (2026-09-28)

- ⛔ В kit `pt-f91110a0d0c8_rp-c9ca6e71e673` `trajectory.json` содержит **только init-units**
  (`100m_smoke_repop`, `1b_repop_run3`, `1b_repop_v2`). Interval-units «added per run once its
  checkpoints and state_hash files are published» — **не опубликованы**. Self-serve кит
  не определяет commit на шаг.
- ⛔ Ресурсы хоста: нужно ~24 ГБ RAM (есть ~11 ГБ) и ≥60 ГБ диска (есть 39 ГБ);
  один чекпоинт `ckpt/step_000000100/` = 19.31 ГБ.
- Решение по разблокировке (владелец — PO/пользователь): запросить у audit-app interval-unit,
  либо выполнять на хосте с ≥32 ГБ RAM / ≥60 ГБ диска (предпочтительно NVIDIA `--device cuda`).

## Примечание

Per-step log доступен на `gs://gensyn-open-1b/logs/state_hashes.jsonl`; harness не умеет
автоподстановку, а audit-app гейтит сабмиты — это тоже часть блокера.
