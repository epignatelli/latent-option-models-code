---
name: dead-code-removal
description: Use this skill to systematically find and remove dead code across a Python project. Covers tracing symbol references, cleaning imports, removing dead config fields, deleting dead test files, and updating public API exports.
---

## Skill: Dead Code Removal

**Trigger:** A refactor has made one load path, one class, or one config field obsolete. Or you've confirmed that a set of functions are no longer called anywhere.

---

## Pattern

### 1. Identify the dead root
Start from the highest-level entry point that is no longer used. Examples: a loader that was replaced by a new one, a config field whose consumers were deleted, a test file for a class that no longer exists.

### 2. Grep for all references
Before deleting anything, grep the entire repo for every symbol you plan to remove. Do this in one pass so you catch cross-file dependencies you didn't know about.

```bash
grep -r "SymbolA\|SymbolB\|SymbolC" --include="*.py" --include="*.yaml" .
```

Categorise each hit as: **caller** (keep until it's also dead), **import** (update), or **string/comment** (update or delete).

### 3. Work leaf-to-root
Delete in dependency order: remove callers before the things they call. If A calls B and B calls C, delete A first, then B, then C.

### 4. Clean up in this order
1. Delete dead functions/classes from source files
2. Remove their imports from the files that imported them
3. Remove them from `__init__.py` exports and `__all__`
4. Remove dead config dataclass fields
5. Remove dead config fields from all YAML configs
6. Delete dead test files entirely (don't gut them — an empty test file is worse than none)
7. Update any `description:` strings in agent files that reference the old names

### 5. Verify
```bash
python -m py_compile <edited_files>           # syntax
grep -r "DeadSymbol" --include="*.py" .       # no strays
```

---

## Rules

- **Delete, don't comment out.** Commented-out code is noise; git history is the record.
- **Delete test files wholesale** when the class they test is gone. Gutted test files with dead imports are harder to maintain than no file.
- **Trust grep over intuition.** You will miss cross-file uses if you rely on memory.
- **Private → public rename counts as dead code cleanup** if the underscore prefix is the only thing preventing clean use. Do it in the same pass.

---

## Tools
- `Bash` with `grep -r` — find all symbol references
- `Read` — understand what each file exports before editing
- `Edit` with `replace_all: true` — bulk rename (e.g. `_GameBuffer` → `GameBuffer`)
- `Bash` with `python -m py_compile` — fast syntax check without needing the full runtime

---

## Failure Modes

| Symptom | Root cause | Fix |
|---|---|---|
| Import error after deletion | Missed a reference in a file you didn't check | Grep more broadly; check `__init__.py`, test files, agent `.md` files |
| Config field removal breaks experiment runs | YAML configs still had the old field; dataclass ignores unknown keys silently | Always grep YAMLs, not just Python files |
| Test file still imports deleted symbol | Deleted the module but not the test | Delete the whole test file, not just the class |
| Agent description references old class name | String match won't catch it; agent loads wrong context | Search `.md` files too in the final grep pass |

---

## Evidence
Learned during LOM codebase cleanup (session 2026-05-23). Removed ~430 lines of dead NLE loader chain (`TrajectoryDataset`, `load_nld_nao`, `load_nao_top10`, etc.) after `NpzTrajectoryDataset` became the sole load path. Dead fields in `DataCfg` survived across three YAML configs until explicitly grepped.
