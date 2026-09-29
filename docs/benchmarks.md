# Benchmarks

open-1b on our hardware, with the production model, gemma-4-E2B, as reference.
Single stream, one GPU.

## Setup

- GPU: one Intel Arc B580 (BMG G21, 12 GB), Vulkan, llama.cpp build-vulkan (b10893).
- CPU: AMD Ryzen 7 7700X, 29 GB RAM.
- Flags: `-ngl 99 -fa off --jinja --spec-type ngram-map-k,ngram-cache -t 16 --parallel 1`.
- `-fa off` is required on these Arc/Vulkan builds, so the KV cache stays in f16.

## open-1b, Go-built GGUF, f16

| Metric | Value |
| --- | --- |
| Decode | 112 to 117 tok/s |
| Prefill | about 892 tok/s |
| VRAM | about 3.6 GiB |
| Context | 4096, the model limit |
| Weights | f16, about 3.2 GB |

## Reference, one Arc B580

| Model | Weights | Context | Decode | Prefill |
| --- | --- | --- | --- | --- |
| open-1b-sft | f16, about 3.2 GB | 4096 | about 114 tok/s | about 892 tok/s |
| gemma-4-E2B | Q4, about 1.5 GiB | 64k | about 214 tok/s | about 1659 tok/s |

The gemma figures come from our 2026-09-20 benchmark on the same card and build.

gemma is about twice as fast, about 2.4 times smaller, and has sixteen times the
context. For classification, extraction, translation, or tool calls, gemma is the
better choice. open-1b is interesting for its training audit, not for its speed.

## Notes

- Not a controlled comparison. The open-1b numbers are a short first run with
  different prompts; the gemma numbers come from a separate controlled benchmark.
- No speculative-decoder-off measurement for open-1b.
- Quality: open-1b scores 25.4 on the OLMo2 suite, against 31.9 for OLMo2-1B.
