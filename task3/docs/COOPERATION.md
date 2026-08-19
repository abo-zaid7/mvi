# Evidence of cooperation

The brief requires evidence that the group gave and received clear instructions during
development — "GitHub issues, commit messages, peer-review comments or meeting records" —
and the marking criteria reserve the top band for "strong evidence of group or team work
via records of weekly or higher frequency meetings".

**This file is a scaffold, not a record.** The tables below are empty on purpose. They must
be filled in with what actually happened, by the people it happened to. Minutes written
after the fact by one person, or invented, are worth nothing if the group is asked about
them in the viva — and the viva is where this gets tested.

---

## 1. What counts as evidence, and where it lives

| Kind of evidence | Where it goes | Who maintains it |
|---|---|---|
| Commit messages | `git log` in this repository | everyone, continuously |
| Task assignment and status | GitHub Issues, or §3 below if not using GitHub | the member who owns the task |
| Code review comments | GitHub pull request reviews, or §4 below | the reviewer |
| Meeting records | §2 below | rotating minute-taker |
| Contribution statement | `task3/credits.json`, shown in the GUI's Credits window | agreed by all four |

### Getting the repository history started

The project was not under version control while the individual tasks were being built, so
the history begins with the Task 3 work. That is worth stating plainly in the report rather
than disguising — an honest short history reads better than a fabricated long one.

```bash
cd my-first-project
git init
git add .
git commit -m "Add Task 1, Task 2 and Task 3 GUI"
```

`.gitignore` already excludes the two virtual environments, the dataset, the model
binaries and `task3/uploads/`, so the repository stays small enough to push.

From this point on, commit in small pieces with messages that say *why*, not *what* — the
diff already says what. Prefer:

```
Fix box readout losing the "simulated" label after a prediction

The server clamps the box to the image and returns it, and the client was
re-rendering the readout with simulated=false, so a simulated box was
relabelled as a hand-drawn one.
```

over `fix bug`.

---

## 2. Meeting records

Copy the block below for each meeting. Weekly is the minimum the criteria reward; the
minute-taker should rotate so the record is not one person's account.

### Meeting template

```
### Meeting N — YYYY-MM-DD, HH:MM–HH:MM, <location or platform>

Present:
Absent (and why):
Minutes taken by:

Progress since the last meeting
- <member>: <what was finished, what slipped and why>

Decisions taken
- <decision, and the reason it was taken>

Instructions given and accepted
- <who> asked <who> to <what>, by <when>   → accepted / renegotiated to <when>

Blockers raised
- <blocker> — owner, and what unblocking it needs

Actions before the next meeting
- [ ] <member> — <action> — due YYYY-MM-DD

Next meeting: YYYY-MM-DD
```

### Records

<!-- Add real meetings here, newest last. Do not pre-fill. -->

_None recorded yet._

---

## 3. Task assignment

Use GitHub Issues if the group has a remote; one issue per task, assigned, with the
acceptance criteria in the body. If not, keep the table here up to date and reference it
from the report.

| # | Task | Owner | Requested by | Agreed | Due | Status |
|---|---|---|---|---|---|---|
| | | | | | | |

Suggested split for the Task 3 work, matching the roles in `credits.json` — adjust to what
the group actually agreed:

| Area | Files |
|---|---|
| Server, model registry, dual-runtime bridge | `app.py`, `backends.py`, `task1_worker.py` |
| Visualisation layer and dataset browser | `render.py`, viewer part of `static/js/app.js` |
| Multilingual layer and accessibility settings | `static/js/i18n.js`, `static/css/style.css` |
| Audio, spoken readout, walkthrough, session log | `static/js/audio.js`, `make_sounds.py`, help content |
| Documentation | `README.md`, `docs/CHAPTER5_GUI.md`, this file |

---

## 4. Peer review

Every non-trivial change should be read by someone who did not write it. On GitHub that is
a pull request review; without a remote, record it here.

### Review template

```
### Review R — YYYY-MM-DD
Change: <branch, commit range, or files>
Author: <member>          Reviewer: <member>

Comments
1. <file>:<line> — <observation> → <what the author did about it>
2. ...

Outcome: approved / changes requested / approved with follow-up issue #N
```

### Reviews

<!-- Add real reviews here. -->

_None recorded yet._

---

## 5. Defects found and fixed during Task 3 development

This is a factual record of the defects found while testing the GUI, and it is useful
evidence in its own right — it shows the interface was tested rather than assumed to work.
The entries below are real; add the reviewer's name against each one if it was found by
someone other than the author.

| # | Defect | Cause | Fix |
|---|---|---|---|
| 1 | Task 1 model would not load in the server process | NumPy 2.x pickle read under NumPy 1.24 | Dual-runtime bridge: forest served by a child process on `mvi-env` |
| 2 | An uploaded PNG was scored against itself | The mask path was derived by string replacement, which returns the scan's own path for uploads | Upload masks resolved from `uploads/masks/` explicitly |
| 3 | `predict` passed the model path where the image path was expected | Variable shadowing in `backends.predict` | Renamed to `model_path`; image path passed through |
| 4 | FLOPs shown were always the first model's | `complexity.json` is keyed by evaluation name, not file name | Explicit filename → key mapping |
| 5 | The "working" spinner never went away | A class selector's `display:flex` outranks the browser's `[hidden]` rule | `[hidden] { display: none !important }` |
| 6 | The empty-state overlay sat below centre | Percentage `max-height` inside a grid row of indefinite height | Overlays positioned absolutely with `inset: 0` |
| 7 | The viewer controls drew on top of the action bar | `height: 100%` on the viewer panel fought the stage's minimum height | Panel sizes by flex; the column scrolls instead |
| 8 | The uploads list included the `masks` directory as a scan | Directory not filtered out of the listing | Files with image extensions only |
| 9 | A stray click on the scan deleted the drawn box | A zero-size drag was treated as a new box | The previous box is restored, with a warning |
| 10 | Switching language discarded the current result | The language handler called `refreshImages()`, which reselects the scan | Option labels re-rendered without touching the selection |
| 11 | The specification sheet stayed in the previous language | It was only rendered at model load | The last spec is kept and re-rendered on language change |
| 12 | Arabic rendered "41 features per pixel" as "features per pixel 41" | The bidirectional algorithm moves a leading digit in an RTL paragraph | Latin identifiers and numbers pinned to `direction: ltr` |
| 13 | The first Task 2 prediction reported ~590 ms instead of ~50 ms | Keras compiles the graph on the first call | A throwaway prediction at load time |
| 14 | "Address already in use" on macOS with nothing running | Port 5000 is held by the AirPlay Receiver | Default port moved to 5050, overridable |
| 15 | The Segment button slid sideways as the hint text changed | The hint had a maximum width, not a fixed one | Fixed width reserved for the hint |

---

## 6. Checklist before submission

- [ ] `git init` done and the history has real, individually attributable commits
- [ ] At least one meeting record per week of the assignment period, in §2
- [ ] Task assignments recorded in §3 or as GitHub Issues, with owners
- [ ] At least one peer review per member in §4
- [ ] `task3/credits.json` has the real names, TP numbers and contributions
- [ ] The CReDiT roles in `credits.json` match what each member actually did
- [ ] Every member can explain any part of the GUI they are asked about
