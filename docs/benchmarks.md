# Benchmarks

Speed of **open-1b** on our hardware, with a reference comparison against the
model we run in production (`gemma-4-E2B`). All numbers are single-stream
(one request at a time) on one GPU.

## Hardware / setup

- **GPU:** 1× Intel Arc B580 (BMG G21, 12 GB), Vulkan (Mesa), llama.cpp `build-vulkan` (b10893).
- **CPU:** AMD Ryzen 7 7700X (16 vCPU), 29 GB RAM.
- **Serving flags:** `-ngl 99 -fa off --jinja --spec-type ngram-map-k,ngram-cache -t 16 --parallel 1`.
- `-fa off` is mandatory on these Arc/Vulkan builds → KV cache is f16 (no `-ctk q8_0`).

## open-1b (our Go-built GGUF, f16)

| Metric | Value |
|---|---|
| Decode | **~112–117 tok/s** (e.g. 256 tok @ 111.7; 96 tok @ 117.2) |
| Prefill | **~892 tok/s** (961 prompt tok in 1077 ms) |
| VRAM | **~3.6 GiB** (10971 → 7258 MiB free) |
| Context | 4096 (model limit; the requested 32k is capped) |
| Weights | f16, ~3.2 GB |

## Reference comparison (1× Arc B580)

| Model | Weights | Context | Decode | Prefill |
|---|---|---|---|---|
| **open-1b-sft** (this repo) | f16 ~3.2 GB | 4096 | ~114 tok/s | ~892 tok/s |
| **gemma-4-E2B** (prod) | Q4 ~1.5 GiB | 64k | **~214 tok/s** | **~1659 tok/s** |

*gemma figures are from our 2026-09-20 controlled benchmark (decode 256 tok /
prefill 1698 tok) on the same card and build.*

**Reading:** gemma is roughly **2× faster**, ~**2.4× smaller**, and has **16×
the context** of open-1b. For any practical workload (classification, extraction,
translation, tool-calling) gemma wins outright. open-1b's reason to exist is not
throughput or quality but **auditable training**.

## Caveats

- This is **not** a controlled A/B: the open-1b figures are a short first-run
  measurement with different prompts; the gemma figures come from a separate
  controlled benchmark. Use them as an order-of-magnitude orientation.
- The `ngram` speculative decoder helps task-dependent amounts; open-1b's
  baseline (no-spec) number was not measured here.
- Quality: open-1b scores **25.4** on the OLMo2 suite vs **31.9** for OLMo2-1B,
  i.e. below an ordinary 1B model. It is a proof of *verifiability*, not a strong
  assistant.
