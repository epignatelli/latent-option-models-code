---
name: code-simplification
description: Use this skill to reduce unnecessary complexity in a codebase — fewer files, classes, and functions without losing correctness. Covers identifying thin wrappers, near-duplicate logic, premature abstractions, and redundant config fields.
---

## Skill: Code Simplification

**Trigger:** The codebase has grown; something feels harder to navigate than it should. Or a refactor has left scaffolding that no longer carries weight. Or a review surfaces classes/functions that exist "just in case."

---

## Principle

An abstraction must earn its place. It earns its place when:
- It is used in **three or more** genuinely independent call sites, OR
- It encodes a **non-obvious invariant** that would be re-derived (and probably gotten wrong) each time if inlined

Everything else is indirection without payoff. Delete it.

---

## Pattern

### 1. Audit for thin layers
Look for these shapes — each is a candidate for deletion:

| Shape | Question to ask |
|---|---|
| Class with one method | Can that method just be a function? |
| Base class with one subclass | Is the split doing anything, or just adding a file to navigate? |
| Helper module with one function | Does this function live better inside its only caller? |
| Wrapper function that calls one other function | What does the wrapper add? |
| Config field that is always the same value | Is this actually variable, or a frozen constant pretending to be configurable? |
| Function whose body is a single expression | Inline it. |

### 2. Audit for near-duplicates
Grep for structural similarity: same argument names, same return shape, same logic with one parameter flipped. Near-duplicates are almost always a sign that one generalisation was missed.

```bash
# Find functions with similar names — often signals duplication
grep -n "^def \|^class " lom/*.py | sort
```

Merge the near-duplicates into one function with a parameter controlling the difference. Only do this if the merged version is genuinely clearer than either original — three nearly-identical lines beat a clever abstraction that requires mental substitution to follow.

### 3. Flatten unnecessary hierarchy
- One-level inheritance where the child just sets defaults → use a factory function or a single class with default arguments
- Nested config dataclasses where one level is always accessed together → flatten into one

### 4. Remove config fields that are not varied
A config field that is never changed across any experiment is a constant. Move it out of the config and into the code as a literal. If it might vary in future, leave a comment — but don't make it configurable until it actually varies.

### 5. Collapse single-use modules
A file that exports exactly one thing and is imported in exactly one place should have its content inlined into the importer. The file boundary adds navigation cost with no payoff.

---

## Rules

- **Simplify toward the caller, not the implementation.** The caller's perspective is what matters: does the API surface feel minimal and coherent?
- **Don't add an abstraction to remove duplication — remove both duplicates instead.** If two functions do nearly the same thing and neither is right, delete both and write one correct version.
- **Flat is better than nested.** A long function with clear local variable names is easier to follow than a call chain through three helper functions.
- **Never simplify and refactor in the same pass.** Simplification removes code. Refactoring moves code. Doing both at once makes the diff unreadable and the blame history useless.
- **Don't simplify across a trust boundary.** Public API surfaces, serialised configs, and test fixtures must stay stable even when the internals are collapsed.

---

## Tools
- `Bash` with `grep -n "^def \|^class "` — inventory all functions and classes in a file
- `Bash` with `grep -r "ClassName\|fn_name"` — check how many callers exist before deleting
- `Read` — understand the full body before judging whether it earns its abstraction
- `Edit` with `replace_all: true` — inline a function by replacing all calls with the body

---

## Failure Modes

| Symptom | Root cause | Fix |
|---|---|---|
| Broke a public API caller | Simplified something that was exported | Always grep for external callers before collapsing |
| Merged two functions that look identical but differ in a subtle invariant | Structural similarity masked a semantic difference | Read both bodies fully before merging; add a test that exercises the difference |
| Inlined a function that was tested directly | The test now has no target | Delete the test or redirect it to the inlined location |
| Removed a config field that one experiment actually used | Grep missed a YAML | Always grep YAMLs and scripts, not just Python files |

---

## Evidence
Proposed by Eduardo Pignatelli as a standing principle for the LOM codebase (2026-05-23). The project tendency is to accumulate scaffolding during rapid prototyping; this skill exists to periodically pay down that debt.
