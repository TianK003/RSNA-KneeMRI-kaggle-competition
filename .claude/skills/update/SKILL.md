---
name: update
description: Use when a measurement, kernel result, submission score, bug, or new idea from this session is not yet written into docs/ — when the user says "update the docs", "log this", "record what we found", when a Kaggle run or submission returns a number, or when a silent failure mode is discovered.
---

# Update the lab notebook

Route every new finding to exactly one doc, with a verdict, then commit and push.

**This skill owns:** `docs/experiments.md`, `docs/proposals.md`, `docs/traps.md`,
`docs/brainstorm.md`, and the verified-facts / state lines in `CLAUDE.md`.

**This skill never touches `docs/handoff.md`.** That is the `/handoff` skill's file. If the
user wants a session log too, run this first, then `/handoff`.

## Routing table — every finding goes to exactly one file

| What you have | Goes to | Required form |
|---|---|---|
| A number that came back from a run, a script, or a submission | `experiments.md` | New `###` entry under the right `##` section, **with a verdict** |
| A public LB score | `experiments.md` **Submissions table** *and* the Scoreboard | One row: `#`, date, kernel version, config change, OOF, LB, notes |
| An idea we have not run yet | `proposals.md` | A full P-nn card (template in that file) + a row in the ranked index |
| A card whose status moved | `proposals.md` | Edit the card's `Status:` **and** its ranked-index row — both, or the index lies |
| A card whose result is now measured | `experiments.md` (result) + `proposals.md` (card reduced to a pointer) | Untried ideas never go to experiments.md; measured ones never stay in proposals.md |
| A bug, or a failure that looked fine while being wrong | `traps.md` | Numbered entry in the correct tier (1 = corrupts results silently, 2 = wastes a session, 3 = friction) |
| A fact about the data, competition, or repo state we verified this session | `CLAUDE.md` | Edit in place; say how it was verified and on what date |
| A question that needs a browser or a human decision | `brainstorm.md` | Open-questions table |

If a finding seems to belong in two places, it is usually one measurement (experiments.md)
plus one operational rule learned from it (traps.md). Write both, and cross-link them.

## The two conventions that keep these files worth reading

**1. `experiments.md` is append-only.** Never delete or rewrite an entry. A result we later
overturned is more useful than a tidy file, because it records why we changed course. Mark
corrections inline instead:

```markdown
**CORRECTED 2026-09-01:** this was measured against gold-copied targets; the real figure is …
```

Superseded scoreboard rows get `❌ superseded (reason, see P-nn)` — they stay.

**2. Every experiments.md entry carries a verdict, and the verdict is gated by the noise
floor.** Pick from ✅ KEEP / ❌ DEAD END / 🔁 INCONCLUSIVE / ⏳ PENDING.

| Metric | Set | A delta smaller than this is **🔁 INCONCLUSIVE, not a win** |
|---|---|---|
| OOF macro-AUC vs teacher | all 4,407 studies | ~0.01 (asserted until P-02 measures it) |
| Gold macro-AUC | the 58 labelled studies | 0.05 macro (per-label Hanley–McNeil SE ≈ 0.09) |
| Public LB | hidden test | 0.005 (the top ten span 0.006 in total) |

Before writing ✅ KEEP, state the delta and the floor it beat in the entry itself. If you
cannot, the verdict is 🔁 INCONCLUSIVE. Label changes are judged on **coverage per language
+ OOF over all 4,407**, never on the 58 gold alone.

## Steps

1. **Collect what is actually new.** Read the session for measurements, scores, bugs, and
   ideas. Then pull live state rather than trusting memory:

   ```powershell
   $env:PYTHONUTF8 = "1"
   kaggle kernels status tiankljucanin/rsna-knee-train
   kaggle competitions submissions rsna-knee-abnormality-detection --csv
   git log --oneline -15
   ```

   A pending run that has since finished is a finding. Check background tasks too.

2. **Check it is not already settled.** Grep `experiments.md` before adding anything — an
   entry that repeats a settled question, or resurrects a ❌ DEAD END without a new reason,
   is worse than no entry.

   ```powershell
   Select-String -Path docs/experiments.md -Pattern "<keyword>"
   ```

3. **Write each finding into its one file** per the routing table, matching the surrounding
   entries' format exactly (heading style, date prefix, table columns).

4. **Reconcile `proposals.md`'s ranked index** with the cards you touched. The index is the
   file's entry point; a stale status there sends the next session down a dead path.

5. **Update `CLAUDE.md` only for verified facts** — the "State as of" line, a corrected
   number, a new hard constraint. Do not restate an experiment there; link to it.

6. **Commit and push.**

   ```powershell
   git add -A
   git commit -m "Docs: <what was learned, not 'update docs'>"
   git push origin main
   ```

## Commit message

Name the finding, not the action. `Docs: submission #2 scored 0.841, cache run v8 timings`
tells the next session something; `Update docs` does not.

## Common mistakes

| Mistake | Why it hurts |
|---|---|
| Recording a 0.02 gold-AUC gain as ✅ KEEP | Below the 0.05 floor. That is how the top public notebook overfit its LB. |
| Rewriting an experiments.md entry to fix it | Destroys the record of why we changed course. Append a `**CORRECTED**` line instead. |
| Adding an untested idea to experiments.md | That file is only for things that were run. Untried ideas are cards in proposals.md. |
| Editing a card but not the ranked index | The index is what gets read first. |
| Writing a session narrative here | Session state belongs in handoff.md via `/handoff`. |
| Logging a number without the config that produced it | An unattributable number cannot be reproduced or trusted later. |
