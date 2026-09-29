# hf2gguf

A Go 1.27 tool that converts a Hugging Face open-1b checkpoint to GGUF and
verifies the result byte for byte. No Python, no torch, no pip. Library:
[go-gguf](https://github.com/cymertek/go-gguf).

For open-1b it reproduces llama.cpp's `convert_hf_to_gguf.py` output exactly,
down to the same SHA-256. The output is deterministic and the check is built in,
so the conversion can be trusted without trusting a Python environment.

## Why Go, and why this shape

| | hf2gguf | llama.cpp `convert_hf_to_gguf.py` |
| --- | --- | --- |
| Runtime | one Go binary | Python 3 with torch, numpy, transformers, gguf-py |
| Setup | `go build` | pip or uv, wheels, pinned versions |
| Startup | immediate | imports torch and its chain |
| Memory | streams one tensor at a time, O(1) per tensor | loads tensors through torch |
| Output check | built-in byte-level `verify` | none |
| Output | byte-identical to a reference file | the reference |
| Scope | open-1b only | many architectures |
| Quantization | none, F16 only | many quant types |
| Dependencies | go-gguf | a large Python tree |

The advantages that matter here:

- No Python runtime. Nothing to install or pin, and it runs where Python is
  unavailable or unwanted.
- Deterministic output, checked byte for byte against a reference.
- Small and readable. One binary, one small module, no large dependency tree.
- Streaming. Tensors are converted one at a time; memory does not grow with the
  model size.
- Easy to build and ship. `go build` and a single file in CI images.

## Limits

- Narrow by design. It maps one Llama-derived architecture and does not claim to
  convert arbitrary models.
- Metadata is copied from a reference GGUF, not derived from the config alone.
  The config and tokenizer are parsed and validated, but the reference stays
  authoritative.
- F16 only. There is no quantization to Q4, Q8, or other ggml types.
- It depends on a third-party GGUF writer for serialization.

## Build

```bash
go build ./...
go vet ./...
go test -race ./...        # the full test includes the 3.2 GB byte comparison
```

## Commands

### dump

Reads the reference GGUF and writes a JSON manifest of all metadata pairs (key,
type, value, or array element type, count, and `raw_sha256`) and every tensor
(name, shape, ggml type, byte count).

```bash
go run . dump -gguf models/open-1b-sft-f16.gguf -o models/open-1b-sft-f16.manifest.json
```

`go-gguf` returns an empty value for `BTypeArray`, so `rawkv.go` walks the KV
section directly to keep the tokenizer arrays as raw bytes. Writing still goes
through the `go-gguf` writer.

### convert

```bash
go run . convert \
  -ref              models/open-1b-sft-f16.gguf \
  -safetensors      models/open-1b-sft/model.safetensors \
  -config           models/open-1b-sft/config.json \
  -tokenizer-config models/open-1b-sft/tokenizer_config.json \
  -tokenizer        models/open-1b-sft/tokenizer.json \
  -o                models/open-1b-sft-f16-go.gguf
```

- Metadata is copied verbatim from the reference.
- Tensors are mapped by name, with `blocks.i.ffn.w_gate_up.weight` split into
  `ffn_gate` and `ffn_up`, and `*.weight_scale` dropped.
- Payloads stream from safetensors and convert F32 to F16 on the fly, using
  round-to-nearest, ties to even.
- The config and tokenizer flags are optional. When given, they are parsed and
  validated against the reference; a mismatch is fatal.

### verify

```bash
go run . verify -ref models/open-1b-sft-f16.gguf -got models/open-1b-sft-f16-go.gguf
```

Compares metadata on raw wire bytes and tensors by shape, type, and full payload.
Reports the first differing offset on failure.

```
metadata: 39 checked, 0 mismatches
tensors:  220 checked, 220 byte-identical, 0 mismatches
verification PASSED
```

## Layout

| File | Purpose |
| --- | --- |
| `main.go` | dispatch, manifest types, dump |
| `rawkv.go` | read-only GGUF KV walker with array support |
| `safetensors.go` | safetensors header and byte access |
| `hfconfig.go` | HF config and tokenizer parsing, validation |
| `convert.go` | metadata reproduction, tensor mapping, F32 to F16 |
| `verify.go` | verify command and comparison engine |
| `convert_test.go` | F16 rounding, KV round-trip, full-file verification |
