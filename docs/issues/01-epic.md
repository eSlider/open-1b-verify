# Epic: Gensyn open-1b — верификация auditable training и локальный inference

**Цель.** Независимо проверить заявление Gensyn о *bitwise auditable training* (open-1b) на нашем железе и запустить модель локально; зафиксировать факты, ограничения и решения.

## Контекст

- Анонс 2026-09-15: https://www.gensyn.ai/news/introducing-open-1b-auditable-training (+ PRNewswire 302878850)
- Harness: https://github.com/gensyn-ai/open-transformers (Apache-2.0)
- Audit CLI: https://github.com/gensyn-ai/gensyn-audit
- Audit app: https://open1b.gensyn.ai/ (manifest: https://open1b.gensyn.ai/manifest.json)
- Модель: HF `Gensyn/open-1b-base`, `-midtrained-93B`, `-sft`
- Модель: 1.61B total / 1.08B non-emb, 24 слоя, 400B токенов, 80 957 шагов, 48×H100
- 2dph info-leaf (PO, 2026-09-28): `6703767319c6e2f9fa6d63465934d72e`

## Дети эпика

- `#AUDIT-INIT` — Аудит init-unit open-1b (CPU, Linux x86-64)
- `#AUDIT-INTERVAL` — Аудит тренировочного интервала open-1b (заблокировано)
- `#INFERENCE` — Локальный запуск open-1b на Arc B580 (llama.cpp Vulkan)
- `#REPORT` — Сводный отчёт и запись фактов в 2dph

## Критерий закрытия эпика

Все дочерние issues закрыты с доказательствами (тест/лог/хэш). Заблокированный
интервальный аудит может остаться открытым, если блокер явно задокументирован
(нет опубликованного interval-unit + ресурсы хоста).
