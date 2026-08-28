---
name: handoff
description: Use when the user asks for a handoff, says to write or update the handoff file, is wrapping up a working session, or wants a record of what this conversation covered and what the next session should do.
---

# Write the handoff entry

`docs/handoff.md` is the only file that answers "what was I doing?" after a break. It is a
session log, newest entry on top — not a findings archive.

**Scope:** this skill writes exactly one new entry in `docs/handoff.md`, then commits and
pushes. It does not edit any other doc.

**First, check the findings are logged.** Measurements, verdicts, card statuses, new traps
and verified facts belong in `experiments.md` / `proposals.md` / `traps.md` / `CLAUDE.md`,
and that is the `/update` skill's job. If anything from this session is still unlogged, run
`/update` first, then come back here. The handoff entry then *points at* those entries
rather than restating them.

## Gather before writing

Do not write the entry from memory alone. Reconcile the conversation against live state:

```powershell
$env:PYTHONUTF8 = "1"
git log --oneline -20
git status --short
kaggle kernels status tiankljucanin/rsna-knee-train
kaggle competitions submissions rsna-knee-abnormality-detection --csv
```

Also check any background tasks still running — an unfinished poll or kernel is the single
most important thing the next session needs to know about.

## Entry structure — every slot is REQUIRED

Insert the new entry directly after the intro block's `---`, above the previous entry. Match
the existing entries' formatting.

```markdown
## YYYY-MM-DD — one-line summary of what this session was

### ⏳ Still in flight as this was written (HH:MM)   ← omit ONLY if nothing is running

| In flight | What it is | Started | How to check | How to read it |
|---|---|---|---|---|

### Where things stand

| | Status |
|---|---|
(✅ / ⏳ / ❌ per component: pipeline, cache, submissions, repo, …)

### What we talked about and decided

- The main threads of the conversation, including choices made and options rejected —
  especially decisions whose reasoning is not visible in the code or the git log.

### What we figured out

Numbered findings that changed what we do, each with the number that supports it and a
pointer to its experiments.md / traps.md entry. Not a restatement of those entries — the
one sentence explaining why it mattered.

### ⏭ Next action, in order

1. Concrete enough to execute without re-deriving anything: the exact command, the file,
   and the rule for reading the result ("within ±0.01 = faithful; a real drop means …").

### Open decisions for Tian

- Judgement calls and anything that needs a browser or a human.

### Things that will bite if forgotten

- Operational traps that this session actually hit.
```

## Rules for the content

- **The in-flight table is the most valuable thing in the file.** For each running item give
  how to check it *and* how to interpret what comes back, including what a suspicious result
  would look like. The next session may start hours later with no context.
- **Next actions must be executable.** A step that says "analyse the results" is not a step.
  Give the command, the file it writes, and the threshold that decides the verdict.
- **Record decisions, not just outcomes.** Why we chose depth over breadth, why a card was
  deferred — the git log already has the what.
- **Keep it short and concrete.** This file is read under time pressure.
- **Never rewrite an older entry.** New entry on top; the history stays.
- Preserve the `## Template for the next entry` block at the bottom of the file.

## Then commit and push

```powershell
git add -A
git commit -m "Handoff: <one-line summary of the session>"
git push origin main
```

## Common mistakes

| Mistake | Fix |
|---|---|
| Writing a handoff while results are unlogged elsewhere | Run `/update` first; handoff points, it does not archive. |
| Omitting the in-flight table because "it will be done soon" | It will not be, and nobody will know how to read it. |
| "Next action: continue the work" | Give the command and the decision rule. |
| Appending at the bottom of the file | Newest entry goes on top. |
| Restating all of experiments.md | Link to it. The handoff is state, not evidence. |
| Editing a previous session's entry to correct it | Append the correction in the new entry instead. |
