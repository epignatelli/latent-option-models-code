---
name: oom-diagnosis
description: Use this skill when a process crashes with MemoryError, OOM kill, or unexpectedly large RSS. Covers diagnosing the cause, estimating actual allocation size, and choosing between pre-allocation strategies (np.empty, memmap, chunked write).
---

## Skill: OOM Diagnosis and Fix in NumPy/PyTorch Pipelines

**Trigger:** `MemoryError`, process killed by OOM killer, `np.concatenate` or `np.empty` hangs or crashes, training loop killed after loading a batch.

---

## Diagnosis Checklist

### 1. Estimate the allocation before running
```python
bytes = shape[0] * shape[1] * shape[2] * dtype.itemsize
gb = bytes / 1e9
print(f"{gb:.1f} GB")
```
If `gb > 10`, do not use `np.empty`, `np.zeros`, or `np.concatenate` without first checking the available memory budget.

### 2. Check the actual memory limit
```bash
cat /proc/sys/vm/overcommit_memory   # 0=heuristic, 1=always, 2=never
cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # per-process cgroup limit
free -h
```
**`overcommit_memory=1` does not protect you from a cgroup limit.** Per-process allocations fail above the cgroup ceiling even if the OS would allow the overcommit.

### 3. Identify the eager allocation
| Pattern | Behaviour | Memory cost |
|---|---|---|
| `np.concatenate(list_of_arrays)` | Allocates full output at once | sum of all input shapes × dtype |
| `np.empty(shape)` | Allocates full array immediately (no copy-on-write) | shape × dtype |
| `np.zeros(shape)` | Same as empty | shape × dtype |
| `np.memmap(..., mode="w+")` | File-backed; OS pages in on access | ~0 RSS until written |
| `np.load(path)` without `mmap_mode` | Decompresses entire file into RAM | compressed file × ~5-10× |

---

## Fixes

### Fix A — Replace `np.concatenate` with pre-allocated fill
```python
# Before (OOM if total > limit):
result = np.concatenate([f["tty_chars"] for f in files])

# After (two-pass):
total = sum(f["tty_chars"].shape[0] for f in files)   # Pass 1: shapes only
buf = np.empty((total, H, W), dtype=np.uint8)          # still eager — use Fix B if > limit
offset = 0
for f in files:
    n = f["tty_chars"].shape[0]
    buf[offset:offset+n] = f["tty_chars"]
    offset += n
```

### Fix B — Use memmap for large allocations
```python
import tempfile

tmp_root = os.path.dirname(output_path)   # same FS to avoid cross-device error
with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
    mm = np.memmap(os.path.join(tmpdir, "buf.bin"), dtype=np.uint8, mode="w+", shape=(total, H, W))
    offset = 0
    for path, n in valid:
        with np.load(path) as f:
            mm[offset:offset+n] = f["tty_chars"].astype(np.uint8)
        offset += n
    mm.flush()
    np.savez_compressed(output_path, data=mm)
    del mm   # MUST delete before TemporaryDirectory.__exit__
```

### Fix C — Tune buffer_size for training
Each loaded npz decompresses fully. For a `GameBuffer` holding `buffer_size` files:
```
peak_ram = buffer_size × chunk_bytes + 1 × chunk_bytes   (one loading slot)
chunk_bytes = T × H × W × 2 bytes   (chars + colors, uint8)
```
Solve for `buffer_size` given your RAM budget:
```python
budget_gb = 200
chunk_gb = T * H * W * 2 / 1e9
buffer_size = int((budget_gb - chunk_gb) / chunk_gb)
```

---

## Pitfalls

| Situation | Pitfall |
|---|---|
| `overcommit_memory=1` on compute cluster | Gives false confidence; cgroup limit still kills the process |
| Memmap `TemporaryDirectory` cleanup | `del` the memmap objects before the `with` block exits or get `OSError: Directory not empty` |
| `np.load(compressed_npz)` | Decompresses entire file — even `mmap_mode` doesn't help for compressed npz |
| Cross-device memmap | Putting `TemporaryDirectory` in `/tmp` when output is on `/scratch` causes rename errors at savez time |

---

## Evidence
Learned during nao-top10 conversion (session 2026-05-23). OOM #1: `np.concatenate` needing 20.8 GiB. OOM #2: `np.empty` needing 50.8 GiB on a larger player. Root cause: per-process cgroup limit of ~16-20 GiB, unaffected by `overcommit_memory=1`. Resolution: two-pass + memmap on the same filesystem as the output.
