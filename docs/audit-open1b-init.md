# Audit: open-1b init unit on CPU

- Date: 2026-09-28
- Host: brainstation, Linux x86-64, kernel 7.3.0, no NVIDIA GPU
- Harness: `gensyn-ai/open-transformers` at `d7b7b674`
- Workdir: `audit/`
- Verdict: match on the `1b_repop_v2` init unit. No training interval was replayed.

## Kit

The audit app publishes the mapping at `https://open1b.gensyn.ai/manifest.json`:

```
run:   1b_repop_v2
       initStateHash 16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed
audit: kit pt-f91110a0d0c8_rp-c9ca6e71e673
```

All four published kits declare the same init commitment, so the choice of kit
does not change the hash under test.

- pretrain commit: `f91110a0d0c84fa3bad808f365567d9c655d10b2`
- repop commit: `c9ca6e71e673660afc6c87de2592fa6f62c87b15`
- torch: `2.10.0`

## Toolchain

| Component | Version |
| --- | --- |
| Python | 3.11.16, via uv. System Python untouched. |
| uv | 0.12.17 |
| PyTorch | 2.10.0+cpu |
| repop | 0.1.5, build commit `c9ca6e71e673`, matches kit.json |
| pretrain | 0.1.0, kit wheel |
| Go | 1.27, for all glue and checks |

The runbook targets Linux with an NVIDIA GPU and points at the CUDA 12.9 index.
This host has no GPU. The README names CPU as the reference device, so the pinned
torch version was installed from the CPU index. Init is device-independent by
construction, so the result is unaffected.

## Commands

```bash
git clone --depth 1 https://github.com/gensyn-ai/open-transformers
go run ./tools/kitinspect kits/*/
go run ./tools/kitfetch kits/pt-f91110a0d0c8_rp-c9ca6e71e673 kit-pt-f91110a0d0c8_rp-c9ca6e71e673

uv venv --python 3.11 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0"
uv pip install --python .venv \
    ./kit-pt-f91110a0d0c8_rp-c9ca6e71e673/repop-0.1.5-cp311-cp311-linux_x86_64.whl \
    ./kit-pt-f91110a0d0c8_rp-c9ca6e71e673/pretrain-0.1.0-cp311-cp311-linux_x86_64.whl

.venv/bin/pretrain-audit-replay --from-init --until-step 0 \
    --config-name 1b_repop_v2 --device cpu \
    --expect-hash 16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed
```

## Checksums

All five kit files were hashed before install. The Go fetcher accepts a file only
on a full match and deletes rejected bytes. Expected values come from kit.json.

| File | SHA-256 | Result |
| --- | --- | --- |
| `pretrain-0.1.0-cp311-cp311-linux_x86_64.whl` | `ecc49d1c05c0e2876a59d98d877795769d6e9620eb66e4b4c866b16f695fb174` | match |
| `pretrain-0.1.0-cp311-cp311-macosx_14_0_arm64.whl` | `fbd23a54aa0bf87775e86493d67f759d1c1f4cc1dcd63ad991fba59dcb92491c` | match |
| `repop-0.1.5-cp311-cp311-linux_x86_64.whl` | `15f171643871211b2a772ee83c94833c0be2eef0495790cedc33813c95147ff0` | match |
| `repop-0.1.5-cp311-cp311-macosx_14_0_arm64.whl` | `d8658e0327c7278812ecb6dd455cd4cf8723a974a78788adba41d5f747289c69` | match |
| `trajectory.json` | `9435515c37d8956d0c81d5bc78c0a383bb6bf27c57b5dbaf1f8be564df0b324a` | match |

Only the two linux_x86_64 wheels were installed.

## Result

```json
{
  "step": 0,
  "state_hash": "16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed",
  "mode": "init",
  "device": "cpu",
  "expected":  "16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed",
  "match": true
}
```

Log line: `from-init: regenerated init state_hash=16554a1119745f1b expected=16554a1119745f1b MATCH=True`.

Resources: 58.3 s wall, peak RSS 16.8 GB, no swap, exit 0.

## Limits

1. No training interval. An init-only unit says nothing about the forward pass,
   the backward pass, the optimizer, or the data stream. The per-step claim is
   not exercised.
2. repop is closed source. Only compiled wheels ship, under the Gensyn
   Reproducibility Licence. The kernels that make training reproducible cannot be
   read or checked. The init unit does not even run the kernels changed by the
   recent repop repin.
3. Hash coverage. The v3 hash covers weights, optimizer state, gradients, and the
   batch digest. It omits the RNG, the data-stream cursor, spike state, and the
   meta.json descriptor keys.
4. Init is the easy case. init.py samples Philox on CPU and byte-copies, so it is
   device-independent by design. The hard cross-device claim lives in the
   training kernels, which this unit does not touch.
5. Provenance of the published run is not established. We only show that the
   pinned artifacts reproduce the published commitment on this machine.

## Interval replay was not attempted

This is a decision, not a stall.

- Not defined by the kit. This kit's trajectory.json contains only three init
  units (`100m_smoke_repop`, `1b_repop_run3`, `1b_repop_v2`). No interval unit
  with a canonical target hash is published, so there is no commitment to check.
- Memory. A step replay needs about 24 GB. This host had about 11 GB free, and
  the init unit already peaked at 16.8 GB.
- Disk. One checkpoint `ckpt/step_000000100/` is 19.31 GB; the full `ckpt/` tree
  is 15.64 TB. With 39 GB free, a single checkpoint is tight.

Re-attempt on a host with 32 GB or more RAM and 60 GB or more disk, ideally with
a CUDA GPU.

## Files

- `open-transformers/`: harness clone
- `kits/`: manifests
- `kit-pt-f91110a0d0c8_rp-c9ca6e71e673/`: verified wheels
- `.venv/`: Python 3.11 audit environment
- `tools/`: Go glue (`kitinspect`, `trajinspect`, `kitfetch`, `gslist`)
- `init-replay.log`: full replay output
