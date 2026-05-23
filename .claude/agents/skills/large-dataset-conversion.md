---
name: large-dataset-conversion
description: Use this skill when converting large raw datasets (>1 GB per item) into npz/memmap format. Covers two-pass conversion, memmap write, game-boundary chunking, and TemporaryDirectory cleanup pitfalls.
---

## Skill: Large Dataset Conversion (Two-Pass + Memmap)

**Trigger:** Converting raw trajectory files into consolidated npz files where a single player's data exceeds available RAM, or where `np.concatenate` / `np.empty` would allocate more than ~10 GB in a single call.

---

## Pattern

### Pass 1 — shapes only, no data in memory
Read every source file to collect `(path, n_frames)` tuples. Record `H, W` from the first valid file. Do not load arrays.

```python
valid: list[tuple[str, int]] = []
H = W = None
for path in sorted(source_files):
    with np.load(path) as f:
        shape = f["tty_chars"].shape
    if H is None: H, W = shape[1], shape[2]
    valid.append((path, shape[0]))
```

### Chunk at natural boundaries (never mid-sequence)
Group games into chunks of ≤ `MAX_FRAMES` by accumulating at game boundaries. Never split a game across chunks.

```python
MAX_FRAMES = 2_000_000
chunks: list[list[int]] = []
current, total = [], 0
for idx, (_, n) in enumerate(valid):
    if current and total + n > MAX_FRAMES:
        chunks.append(current); current, total = [], 0
    current.append(idx); total += n
if current: chunks.append(current)
```

### Pass 2 — memmap write per chunk
Use a `TemporaryDirectory` on the **same filesystem** as the output (avoids cross-device moves). Delete memmap objects **before** the `with` block exits — otherwise `shutil.rmtree` raises `OSError: Directory not empty`.

```python
import tempfile

tmp_root = os.path.dirname(output_path)
with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
    mm_chars  = np.memmap(os.path.join(tmpdir, "c.bin"),   dtype=np.uint8, mode="w+", shape=(total_frames, H, W))
    mm_colors = np.memmap(os.path.join(tmpdir, "col.bin"), dtype=np.uint8, mode="w+", shape=(total_frames, H, W))
    offset = 0
    for path, n in chunk_valid:
        with np.load(path) as f:
            mm_chars[offset:offset+n]  = f["tty_chars"].astype(np.uint8)
            mm_colors[offset:offset+n] = np.clip(f["tty_colors"].astype(np.int16), 0, 31).astype(np.uint8)
        offset += n
    mm_chars.flush(); mm_colors.flush()
    np.savez_compressed(output_path, tty_chars=mm_chars, tty_colors=mm_colors,
                        offsets=np.array(offsets_list, dtype=np.int64))
    del mm_chars, mm_colors  # MUST come before TemporaryDirectory.__exit__
```

### Skip if output already exists
Before Pass 2, check if all chunk outputs exist and load their metadata from disk. This makes the function idempotent and safe to re-run after partial failure.

---

## Tools
- `Bash` — check disk space (`df -h`), inspect shapes before writing
- `Read` — audit conversion script before running
- The conversion itself runs on the compute node (network filesystem); use `Bash` only for local verification

---

## Failure Modes

| Symptom | Root cause | Fix |
|---|---|---|
| `MemoryError` on `np.empty(shape)` | Per-process cgroup limit even when `overcommit_memory=1` | Switch to memmap |
| `OSError: Directory not empty` | Memmap objects still alive when `TemporaryDirectory.__exit__` calls `shutil.rmtree` | `del mm_chars, mm_colors` before `with` block exits |
| Cross-device rename error | `TemporaryDirectory` on a different filesystem than output | Pass `dir=os.path.dirname(output_path)` to `TemporaryDirectory` |
| Training OOM with large buffer | Each chunk decompresses fully into RAM; `buffer_size=N` means N × chunk_size in RAM | Calculate peak = N × chunk_bytes + 1 chunk loading; tune `buffer_size` to fit budget |

---

## Evidence
Learned during nao-top10 dataset conversion (session 2026-05-23). Players up to 28M frames; 50.8 GiB allocation failure drove switch from `np.empty` to memmap. Chunking at 2M frames per file resolved the training RAM budget problem.
