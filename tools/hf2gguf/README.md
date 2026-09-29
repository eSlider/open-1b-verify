# hf2gguf

A Go 1.27 tool that converts a Hugging Face `open-1b` safetensors checkpoint to
GGUF, byte-identical to a supplied reference file, with a byte-level verifier.

Library: [go-gguf](https://github.com/cymertek/go-gguf), reader and writer. No
Python anywhere.

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
