# hf2gguf

Pure-Go (Go 1.27) tool to convert a HuggingFace `open-1b` safetensors checkpoint
into GGUF **byte-identically** to a supplied reference GGUF, plus a byte-level
verifier. Built for epic `vado/1b#1` (Gensyn open-1b local bring-up).

Library: [`github.com/cymertek/go-gguf`](https://github.com/cymertek/go-gguf)
(reader + writer). No Python anywhere in the tool.

## Build

```bash
go build ./...
go vet ./...
go test -race ./...        # full test includes the 3.2 GB byte comparison
```

## Subcommands

### `dump` — the reproduction contract

Reads the reference GGUF and emits a JSON manifest of **all** metadata KV pairs
(key, type, value / array elem-type + count + `raw_sha256`) and every tensor
(name, shape, ggml type, nbytes).

```bash
go run . dump -gguf models/open-1b-sft-f16.gguf -o models/open-1b-sft-f16.manifest.json
```

`go-gguf` exposes scalar/string metadata and tensors; its `MetadataEntry.Value()`
returns an empty value for `BTypeArray`, so `rawkv.go` walks the KV section
directly to preserve tokenizer arrays (`tokens`, `merges`, `token_type`) as raw
bytes. Writing still goes through `go-gguf`'s writer — nothing hand-rolls GGUF
serialization.

### `convert` — emit our GGUF

```bash
go run . convert \
  -ref          models/open-1b-sft-f16.gguf \
  -safetensors  models/open-1b-sft/model.safetensors \
  -config       models/open-1b-sft/config.json \
  -tokenizer-config models/open-1b-sft/tokenizer_config.json \
  -tokenizer    models/open-1b-sft/tokenizer.json \
  -o            models/open-1b-sft-f16-go.gguf
```

- Metadata is copied verbatim from the reference KV section.
- Tensors are mapped HF FQN → GGUF name (see the module docs in `convert.go`),
  with `blocks.{i}.ffn.w_gate_up.weight` split into `ffn_gate`/`ffn_up`, and
  `*.weight_scale` dropped (the reference has none).
- Payloads stream from safetensors and are converted F32→F16 with IEEE
  round-to-nearest-even on the fly (O(1) memory per tensor).
- `-config` / `-tokenizer-config` / `-tokenizer` are optional; when given they
  are parsed and **validated** against the reference (mismatch is fatal).

### `verify` — byte-level equality

```bash
go run . verify -ref models/open-1b-sft-f16.gguf -got models/open-1b-sft-f16-go.gguf
```

Compares metadata on raw wire bytes and tensors by shape, type and full payload
equality (streaming, first-diff offset reported on failure).

```
metadata: 39 checked, 0 mismatches
tensors:  220 checked, 220 byte-identical, 0 mismatches
verification PASSED: 39 metadata keys and 220 tensors byte-identical
```

## Layout

| File | Purpose |
|---|---|
| `main.go` | dispatch, manifest types, `dump` |
| `rawkv.go` | read-only GGUF KV walker (array support) |
| `safetensors.go` | safetensors header/byte access |
| `hfconfig.go` | HF config/tokenizer parsing + validation |
| `convert.go` | metadata reproduction, tensor mapping, F32→F16 |
| `verify.go` | `verify` command + comparison engine |
| `convert_test.go` | F16 rounding, KV round-trip, full-file verification |
