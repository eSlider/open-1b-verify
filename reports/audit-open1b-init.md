# Gensyn OPEN-1B audit — independent replay (CPU, Linux x86-64)

- **Date:** 2026-09-28
- **Host:** `brainstation`, Linux x86-64, kernel `7.3.0-070300rc1-generic` (Ubuntu), **no NVIDIA GPU**
- **Auditor:** SE sub-agent (independent)
- **Workdir:** `/root/projects/gensyn/audit`
- **Harness:** `gensyn-ai/open-transformers` @ `d7b7b674cd851f7a16815229f060658180cb772b`
  ("Initial commit: the OPEN-1B pretraining and audit-replay harness", 2026-09-15), cloned via git.
- **Verdict:** ✅ **MATCH** on the `1b_repop_v2` init unit. ❌ No training-interval
  replay run (not defined by this kit and not feasible on this host — see §7).

---

## 1. Kit selection

Four kits were listed. All four `trajectory.json` files declare the **same**
`1b_repop_v2` init commitment
`16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed`, so the choice
of kit does not change the init hash under test. The authoritative mapping is
published by the audit app itself at `https://open1b.gensyn.ai/manifest.json`:

```json
"run":   { "id": "20260722-213626-ad3276b", "model": "1b_repop_v2",
           "initStateHash": "16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed" },
"audit": { "kit": { "id": "pt-f91110a0d0c8_rp-c9ca6e71e673",
                    "urlBase": "https://storage.googleapis.com/gensyn-audit-public/audit-kit/pt-f91110a0d0c8_rp-c9ca6e71e673" } }
```

**Kit under test: `pt-f91110a0d0c8_rp-c9ca6e71e673`.**

- `pretrain_commit`: `f91110a0d0c84fa3bad808f365567d9c655d10b2`
- `repop_commit`:    `c9ca6e71e673660afc6c87de2592fa6f62c87b15`
- `torch_requirement`: `torch==2.10.0`

(The other three kits — `pt-cb4ac29438bc_rp-…`, `pt-f6bc086dfebd_rp-…`,
`pt-f5eece09abe6_rp-…` — carry the same init commitment but are older
pretrain/repop pins. Two share the `c9ca6e71e673` repop but a different pretrain
commit; the app manifest picks `f91110a0d0c8`.)

## 2. Toolchain

| Component | Version / identity |
|---|---|
| Python | 3.11.16 (managed by `uv`; system Python 3.14 untouched) |
| `uv` | 0.12.17 |
| PyTorch | `2.10.0+cpu` (`torch.version.cuda == None`) |
| `repop` | 0.1.5, `build_info().commit == c9ca6e71e673660afc6c87de2592fa6f62c87b15` ✅ (== kit.json `repop_commit`); backends `["cpu","cuda"]` |
| `pretrain` | 0.1.0 (kit wheel) |
| Go (glue tools) | go1.27.0 (fabric rule: all glue/parsing in Go) |

**Deviation, stated plainly:** the RUNBOOK targets Linux hosts with an NVIDIA GPU
and points them at the CUDA 12.9 PyTorch index. This host has no GPU, and the
README names CPU as the *reference* device, so the exact pin `torch==2.10.0` was
installed from PyTorch's official **CPU** index (`.../whl/cpu`, wheel
`2.10.0+cpu`). The pinned version is unchanged; only the CPU build variant was
chosen deliberately, not by an accidental PyPI fallback. Init is device-independent
by construction (see §6), so this does not weaken the init result.

Python (`uv`-managed) is used **only** because the vendor harness is Python and
ships as wheels; all download/verification/listing glue is Go under `tools/`
(`kitinspect`, `trajinspect`, `kitfetch`, `gslist`).

## 3. Commands run

```bash
# workdir + docs
mkdir -p /root/projects/gensyn/audit && cd /root/projects/gensyn/audit
git clone --depth 1 https://github.com/gensyn-ai/open-transformers

# kit manifests + trajectories (Go inspector)
go run ./tools/kitinspect kits/*/
go run ./tools/trajinspect kits/*/trajectory.json

# download + per-file sha256 verification against kit.json (Go tool,
# downloads to .part, verifies, removes on mismatch)
go run ./tools/kitfetch kits/pt-f91110a0d0c8_rp-c9ca6e71e673 \
                        kit-pt-f91110a0d0c8_rp-c9ca6e71e673

# Python 3.11 env
uv venv --python 3.11 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0"
uv pip install --python .venv \
    ./kit-.../repop-0.1.5-cp311-cp311-linux_x86_64.whl \
    ./kit-.../pretrain-0.1.0-cp311-cp311-linux_x86_64.whl

# provenance assertions
.venv/bin/python -c "import repop,json; print(repop.build_info())"

# THE AUDIT (init unit)
PRETRAIN_AUDIT_MEMLOG=1 /usr/bin/time -v .venv/bin/pretrain-audit-replay \
    --from-init --until-step 0 \
    --config-name 1b_repop_v2 \
    --device cpu \
    --expect-hash 16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed
```

## 4. SHA-256 verification table (all 5 kit files)

Every file was hashed **before** install; the Go fetcher adopts a file only on a
full 64-hex match and deletes rejected bytes. Expected values are read from the
kit's own `kit.json`.

| File | expected (`kit.json`) | computed | result |
|---|---|---|---|
| `pretrain-0.1.0-cp311-cp311-linux_x86_64.whl` | `ecc49d1c05c0e2876a59d98d877795769d6e9620eb66e4b4c866b16f695fb174` | same | ✅ OK |
| `pretrain-0.1.0-cp311-cp311-macosx_14_0_arm64.whl` | `fbd23a54aa0bf87775e86493d67f759d1c1f4cc1dcd63ad991fba59dcb92491c` | same | ✅ OK |
| `repop-0.1.5-cp311-cp311-linux_x86_64.whl` | `15f171643871211b2a772ee83c94833c0be2eef0495790cedc33813c95147ff0` | same | ✅ OK |
| `repop-0.1.5-cp311-cp311-macosx_14_0_arm64.whl` | `d8658e0327c7278812ecb6dd455cd4cf8723a974a78788adba41d5f747289c69` | same | ✅ OK |
| `trajectory.json` | `9435515c37d8956d0c81d5bc78c0a383bb6bf27c57b5dbaf1f8be564df0b324a` | same | ✅ OK |

Byte sizes also matched `kit.json`. Only the two **linux_x86_64** wheels were
installed; the macOS arm64 wheels were downloaded and verified but not installed.

## 5. Init-unit result — MATCH

Raw result object printed by the harness (stdout, exit status 0):

```json
{
  "step": 0,
  "consumed_tokens": 0,
  "state_hash": "16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed",
  "mode": "init",
  "repop": { "commit": "c9ca6e71e673660afc6c87de2592fa6f62c87b15",
             "backends": ["cpu", "cuda"], "cuda_arch_list": "7.5 8.0 9.0" },
  "device": "cpu",
  "expected":  "16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed",
  "match": true,
  "cpu_threads": { "requested": "auto", "source": "runtime-default",
                   "torch_num_threads": 8, "omp_num_threads": null, "mkl_num_threads": null }
}
```

Log line: `from-init: regenerated init state_hash=16554a1119745f1b expected=16554a1119745f1b MATCH=True`.
Config `1b_repop_v2`, seed `42`, loss path `fused CE+z-loss`. The only warning is
the expected `not distributed + no CUDA — parallelize applies AC only (FSDP
skipped; dev mode)`, i.e. the documented CPU reference path.

**Resource notes:** wall **58.3 s**, user 78.5 s / sys 61.1 s, **peak RSS
16.80 GB** (16,804,956 KB), **0 swaps**, exit 0. (README/RUNBOOK size this unit at
~21.9 GB peak on CPU; we measured 16.8 GB. Free RAM was ~11 GB and the unit still
completed without swapping, so the doc figure is conservative.)

## 6. What this proves — and what it does not

**Proves.** The bytes installed are the bytes `kit.json` published (all five
SHA-256s match). The `repop` kernel build is the one the kit claims
(`build_info().commit == repop_commit`). Running the vendor harness on this
CPU, from seed, the `1b_repop_v2` **initial state** — seeded device-independent
trunc-normal init plus zeroed optimizer moments — reproduces the published
canonical hash `16554a11…` **bit-for-bit**. Independent third-party
reproduction of the run's init commitment: ✅.

**Does not prove.**

1. **No training interval.** Per the README: "An initialization-only unit does
   not verify a training interval." A match here says nothing about any forward,
   backward, AdamW or data-stream step. The claimed bitwise reproducibility of
   *optimizer steps* is **not** tested by this run.
2. **`repop` is closed source.** Only compiled wheels are shipped; the kernel
   arithmetic that makes training bitwise reproducible cannot be read or
   independently re-derived from this kit. The init unit does not even exercise
   the kernels that the recent `c9ca6e71e673` repin changed (the trajectory notes
   say exactly this), so a match is *not* evidence for the kernel-level BFR
   contract at that commit.
3. **v3-hash coverage.** The README states the v3 hash covers weights, optimizer
   state, gradients and the batch digest, but **not** RNG, the data-stream
   cursor, spike state, or the `meta.json` descriptor keys. Even a matching
   interval hash would leave those unverified.
4. **Init is the easy case by design.** `init.py` samples Philox on CPU and
   byte-copies, so it is device-independent; it is expected to match across
   devices. The hard cross-device claim lives in the training kernels, which
   this unit does not touch.
5. Provenance of the *published run* is not established; only that the pinned
   artifacts reproduce the published commitment on this machine.

## 7. Step-interval replay — deliberately not attempted

Stopping after the init unit is a considered decision, not a stall:

- **Not defined by the kit.** This kit's `trajectory.json` (`init-units-v1`)
  contains **only 3 init units** — `100m_smoke_repop`, `1b_repop_run3`,
  `1b_repop_v2`. Its own note says: "Step-interval units are added per run once
  its checkpoints and state_hash files are published." **No interval unit with a
  canonical target hash is published here**, so the kit defines no
  step-interval commitment to check. (The run's per-step log exists at
  `gs://gensyn-open-1b/logs/state_hashes.jsonl`, but the harness has no automatic
  lookup and the audit app gates submissions.)
- **Memory.** A step-interval replay needs ~24 GB (README), and the audit's
  auto-offload only reaches "every published unit" at that figure. This host had
  ~11 GB RAM available (29 GB total, 18 GB in use); an init unit already peaked at
  16.8 GB. The heavy replay would swap-thrash and would not fit honestly.
- **Disk/size.** One checkpoint `ckpt/step_000000100/` is **19.31 GB**
  (150 objects); the full `ckpt/` prefix is **15.64 TB** (810 checkpoint dirs).
  With 39 GB free, a single checkpoint is tight against free space and the
  documented `--from-init + --gcs-root` combination is broken (wrong shard
  anchor), so a from-init interval would first require staging shards via
  `--data-root`.

Re-attempting an interval audit should be done on a host with ≥32 GB free RAM
(NVIDIA GPU preferred: `--device cuda`) and ≥60 GB free disk.

## 8. Files produced

- `open-transformers/` — harness clone (docs read: `README.md`,
  `docs/audit-replay-usage.md`, `scripts/audit_volunteer/RUNBOOK.md`)
- `kits/<kit>/kit.json`, `kits/<kit>/trajectory.json` — manifests
- `kit-pt-f91110a0d0c8_rp-c9ca6e71e673/` — verified wheels + trajectory
- `.venv/` — Python 3.11 audit env
- `init-replay.log` — full captured replay output + `/usr/bin/time -v`
- `tools/{kitinspect,trajinspect,kitfetch,gslist}` — Go glue
- `REPORT.md` — this file

No secrets were written. Nothing outside the workdir was modified or deleted.
