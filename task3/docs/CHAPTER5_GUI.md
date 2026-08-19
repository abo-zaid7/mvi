# Chapter 5 — Group Assignment GUI

Material for Chapter 5 of the group report, in the structure the brief specifies:
5.1 framework, 5.2 technical documentation, 5.3 cultural features supported by literature.

> **Before submission.** References marked `[ADD CITATION]` are places where a specific
> paper from the past 5–10 years is needed and has deliberately been left blank rather
> than guessed at. Each one says what to search for. The references that *are* given —
> the W3C standards, the CRediT taxonomy, Okabe–Ito, Kirschner — are real and can be
> cited as written, but check the exact edition you are quoting.

---

## 5.1 Framework for developing the GUI

### 5.1.1 What the interface had to do

The brief sets six functional requirements: load the models from Tasks 1 and 2, visualise
the ground truth, visualise the segmented boundaries, display metrics numerically, switch
between English and one other language, and provide audio for important functions, plus a
CReDiT credits window. Three constraints came from the group rather than the brief:

1. **Two incompatible Python environments.** Task 1 is forbidden from using TensorFlow, so
   it was built in a separate virtual environment. The two environments cannot be merged
   (§5.1.3), so whatever framework was chosen had to tolerate the models living in
   different processes.
2. **Four developers on three operating systems.** macOS, Windows and Linux were all in
   use in the group.
3. **A live demonstration under time pressure.** The marking criteria require the GUI to
   be defended "with no bugs or errors", which favours a small, inspectable codebase over
   a large dependency tree.

### 5.1.2 Choice of framework

| Option | Why it was rejected or chosen |
|---|---|
| **Tkinter** | In the standard library and needs no install, but image compositing, a translucent overlay and right-to-left text all have to be built by hand, and it has no usable route to text-to-speech. Rejected. |
| **PyQt / PySide** | Capable, and it has real RTL support. Rejected on licensing complexity (GPL/commercial for Qt bindings) and because a Qt build on three operating systems was a poor use of the time available. |
| **Streamlit / Gradio** | Fastest to a demo, but the interaction model is a script re-run on every widget change. Dragging a box on an image and keeping a loaded model warm both fight that model, and the appearance is fixed by the framework, which conflicts directly with the "highly polished" and "additional inclusive design features" criteria. Rejected. |
| **Flask + HTML/CSS/JavaScript** | **Chosen.** |

Flask was chosen for five concrete reasons:

- **It runs on the same interpreter as the models.** The server is `tf-env`'s Python, so
  the three Keras networks are loaded in-process and stay warm between predictions.
- **The browser is the one runtime all four of us already had**, identical on all three
  operating systems, and the marker needs nothing but a browser.
- **Right-to-left support is free and correct.** Browsers implement the Unicode
  bidirectional algorithm; with CSS logical properties, one `dir="rtl"` attribute mirrors
  the entire layout (§5.3.2). No other option offered that for the cost.
- **Speech synthesis is built in.** The Web Speech API gives a spoken readout of the
  results in the selected language with no extra dependency — where `playsound`, which the
  brief suggests, plays fixed clips only.
- **The dependency footprint is one package.** Flask. Everything else the GUI needs was
  already installed for Task 2, and the front end has no build step, no bundler and no
  third-party JavaScript at all — the whole client is three plain `.js` files.

The cost of the choice is honest and worth stating: a web page cannot read arbitrary local
files, so the user picks scans from the dataset or uploads them, rather than browsing the
filesystem. For a demonstration tool that is a fair trade.

### 5.1.3 Architecture — the dual-runtime model bridge

This is the part of the design that is not standard and needs explaining.

Task 1's Random Forest was trained and pickled in `mvi-env` under NumPy 2.x. Task 2
requires TensorFlow 2.13, which pins NumPy to 1.24. A NumPy 2.x pickle cannot be read
under NumPy 1.24 — it fails with `ModuleNotFoundError: No module named 'numpy._core'`. The
options were:

- **Retrain the forest under `tf-env`.** Rejected. It would produce a different model from
  the one Chapter 1 reports, so the GUI would no longer be demonstrating the graded work.
- **Upgrade NumPy in `tf-env`.** Rejected. It breaks TensorFlow 2.13, and with it all three
  Task 2 models.
- **Bridge the two environments.** Chosen.

```
                    browser (one page, no build step)
                              │  JSON over HTTP
                              ▼
              ┌───────────────────────────────────┐
              │  app.py  —  Flask, on tf-env      │
              │                                   │
              │  backends.py                      │
              │    ├── Task2Runner ── 3 × Keras   │   in-process, kept warm
              │    │                  models      │
              │    └── Task1Bridge                │
              │           │  JSON lines / stdio   │
              └───────────┼───────────────────────┘
                          ▼
              ┌───────────────────────────────────┐
              │  task1_worker.py — on mvi-env     │
              │    Random Forest + 41 features    │   loaded once, stays resident
              └───────────────────────────────────┘
```

`app.py` starts the worker with `mvi-env`'s interpreter the first time a Task 1 model is
loaded. They exchange one JSON object per line over a pipe; masks cross as base64 PNGs.
The forest is loaded once and stays resident, so a Task 1 prediction through the bridge
costs the same as calling the segmenter directly — measured at 60–75 ms per scan, against
the 63 ms reported in Chapter 1.

Three details make it reliable rather than merely working:

- **A lock around the pipe.** Flask's development server is threaded; two requests writing
  into the same stdin would interleave and desync the protocol permanently.
- **A lock around Keras prediction.** Graph execution is not thread-safe. At one scan per
  request, serialising costs nothing and removes a whole class of intermittent failure
  from a live demonstration.
- **A warm-up prediction at load time.** The first call to a Keras model compiles the
  graph, which takes about half a second. Without a throwaway prediction at load, the
  first real scan would report roughly ten times the true latency and the figure on screen
  would contradict the figure in the report.

The same design would let a fifth model in a fourth environment be added by writing one
more worker, which is the general argument for it: the bridge isolates a dependency
conflict instead of resolving it by compromising one of the models.

### 5.1.4 Consistency of measurement

Everything the interface reports is computed by `task2/metrics.py` at 512 × 512 — the same
code and the same resolution both individual tasks were scored at. A Task 1 Dice score and
a Task 2 Dice score shown in this interface are therefore directly comparable, and both are
comparable with the tables in Chapters 1–4. FLOPs, parameter counts and reference latency
are read from `task2/outputs/complexity.json`, the measurements Task 2 already published,
rather than being re-derived — so the specification sheet in the GUI and the specification
table in the report cannot drift apart.

---

## 5.2 Technical documentation

The full installation, operation and troubleshooting documentation is `task3/README.md`,
which covers, as the brief requires: model loading (§4.1), image input (§4.2), metric
interpretation (§4.5), GUI operation (§4.3–4.8) and troubleshooting (§5). It is written
for someone who has not seen the project before.

### 5.2.1 Summary for the report

**Install.** The GUI adds exactly one dependency, Flask, to the existing Task 2
environment: `../tf-env/bin/python -m pip install flask`. The six audio cues are generated
locally by `make_sounds.py` rather than downloaded, so nothing has to be fetched and there
is no licence to clear.

**Run.** `cd task3 && ./run_gui.sh`, then open `http://127.0.0.1:5050`. The launcher checks
both environments exist, installs Flask if it is missing, generates the sounds if they are
absent, and starts the server. Port 5050 rather than Flask's usual 5000 because on macOS
port 5000 is held by the AirPlay Receiver, which otherwise produces a confusing
"address already in use" on a machine where nothing of the group's is running;
`MVI_GUI_PORT` overrides it.

**Use.** Four steps, listed in the interface and ticked off as they are completed: load a
model, choose a scan, mark the lesion (Task 1 only), run. Every step also has a keyboard
shortcut.

**Error handling.** Every route returns JSON with an `ok` field and, on failure, a readable
`error` message; the client turns every failure into an on-screen message, a screen-reader
announcement and an audio cue. The paths that were explicitly tested to fail cleanly:
no model loaded, Task 1 with no box, a box smaller than 8 px, a model path outside the
project folder, a directory-traversal attempt in a model or image path, a non-existent
model or scan, an unsupported upload type, a file that is not decodable as an image, a
malformed box, an unknown view name, and non-numeric render parameters. None of them
produces a stack trace or a silent no-op.

### 5.2.2 Verification carried out

| Checked | Result |
|---|---|
| All four models load and predict | Pass. RF 17.85 MB; MHA-ResUNet 8.84 M params / 14.93 GFLOPs; pruned 3.28 M / 5.08 GFLOPs; U-Net (PSO) 4.37 M / 7.62 GFLOPs |
| Bridge fidelity — GUI vs direct call | Identical to 4 d.p. (e.g. scan `00257_me`: Dice 0.9763, HD95 3.16 both ways) |
| Task 1 latency through the bridge | 60–75 ms/scan, against 63 ms reported in Chapter 1 |
| Task 2 latency after warm-up | 51–53 ms/scan, stable from the first scan |
| All six views plus the export strip render | Pass, all seven |
| Metrics against `task2/outputs/results_per_image.csv` | Consistent |
| The 13 error paths in §5.2.1 | All return a readable message; no stack traces |
| Three languages, full interface | Pass, including the specification sheet and all dynamically built text |
| Language change preserves the current result | Pass (was a defect, fixed) |
| Arabic right-to-left mirroring | Pass — layout, sliders, progress ticks and meter all mirror; Latin identifiers and numbers stay left-to-right |
| Audio assets | All six WAVs valid mono PCM, served as `audio/x-wav`, decode correctly in-browser |
| Accessibility settings | Verified applied: 17 px base at "large", high-contrast tokens, Okabe–Ito palette |
| Browser console and server log after a full session | No errors, no 404s |

**Not verified, and stated as such:** audible playback and the spoken readout could not be
confirmed in the automated browser used for testing, which has no audio output device —
the assets, the wiring and the fallback behaviour were verified instead. Both should be
checked by ear on the demonstration machine before the presentation.

### 5.2.3 Evidence of instruction given and received during development

See `docs/COOPERATION.md`. The repository history is the primary record; that file collects
the meeting minutes, the peer-review comments and the task assignments alongside it.

---

## 5.3 Features that consider cultural differences

### 5.3.1 Language, and why Bahasa Melayu and Arabic

The brief asks for English plus one other language. The interface has three, and the third
is the one that does the real work.

**Bahasa Melayu** is the national language of the country the group studies in and the
language a Malaysian clinical user is most likely to prefer. It is the obvious second
language on the merits, not merely the nearest one to hand.

**Arabic was added specifically because it is written right to left.** A second
left-to-right language only exercises the string table. A right-to-left language exercises
the *layout*, and that distinction matters: an interface that swaps its words but keeps its
structure mirrored the wrong way is measurably harder to use for a right-to-left reader,
because reading order, scanning order and the expected position of a "next" control all
reverse together. Adding Arabic forced the interface to be built so that it genuinely
mirrors, which is a structural property that cannot be retrofitted by translation alone.

> `[ADD CITATION]` — one recent (2016–2024) empirical study of right-to-left interface
> localisation or bidirectional UI usability, to support the claim that mirroring the
> layout, not only the text, affects task performance for RTL readers. Search terms:
> "right-to-left interface localisation usability", "bidirectional UI mirroring Arabic
> usability", "RTL layout adaptation user performance".

Every visible string is in `static/js/i18n.js`, keyed and looked up — including tooltips,
screen-reader labels, image alternative text and the metric glossary. A translated
interface that leaves its tooltips and accessibility labels in English is only half
translated, and a screen-reader user in the second language would get the worse half.

### 5.3.2 Layout mirroring, implemented once

The stylesheet uses CSS **logical properties** throughout — `margin-inline-start` rather
than `margin-left`, `inset-inline-start` rather than `left`, `border-inline-start` rather
than `border-left`. Selecting Arabic sets `dir="rtl"` on the document and the whole
interface mirrors from that single attribute: the three columns swap order, the sliders
fill from the right, the "next scan" arrow reverses, the progress ticks and the coloured
edge markers on the log entries move to the other side, and the 0.85 threshold mark on the
Dice meter moves with the bar.

Two deliberate exceptions, both correct:

- **Numbers, file names and model identifiers stay left-to-right.** Arabic writes numerals
  left to right even inside right-to-left text. Without this, the bidirectional algorithm
  moves a leading digit to the far end of the line and "41 features per pixel" renders as
  "features per pixel 41" — which was an actual defect found in testing and fixed.
- **The medical images are never mirrored.** Anatomical left and right are clinically
  meaningful; flipping a scan to match reading direction would be a patient-safety defect,
  not a localisation feature. Only the interface around the image mirrors.

That second point is the sharpest example of the general principle: localisation has to
stop at the boundary of the clinical data.

> `[ADD CITATION]` — for the general principle of adapting interface structure rather than
> only content to the user's locale. Search terms: "culturally adaptive user interface
> design", "cultural dimensions interface localisation", "localisation beyond translation
> health information system". Hofstede's cultural-dimensions framework is the classical
> reference but predates the 5–10 year window, so a recent application of it to interface
> design is what is wanted here.

### 5.3.3 Colour: not assuming the reader sees what we see

The default overlay palette is the green/red pairing radiology tools conventionally use.
Green and red is also precisely the combination that around 1 in 12 men with northern
European ancestry cannot reliably separate, so the interface offers the **Okabe–Ito
colour-universal palette** as an alternative, and the legend swatches change with it so the
key never contradicts the image.

More importantly, **colour is never the only carrier of meaning**. Every masked region also
has a distinct outline; the difference view labels its regions in the legend in words; the
Dice verdict is a sentence as well as a coloured bar; and the progress ticks are drawn
shapes, not coloured dots. That is WCAG 2.1 success criterion 1.4.1 (Use of Colour), and it
is what makes the colour-blind-safe palette an improvement rather than a substitute for
one.

- Okabe, M. and Ito, K. (2008) *Color Universal Design (CUD) — How to make figures and
  presentations that are friendly to colourblind people.* (The palette itself predates the
  citation window; it is cited as the source of the specific colour values.)
- W3C (2018) *Web Content Accessibility Guidelines (WCAG) 2.1*, W3C Recommendation,
  5 June 2018 — SC 1.4.1 Use of Colour, SC 1.4.3 Contrast (Minimum), SC 1.4.11
  Non-text Contrast.
- W3C (2023) *Web Content Accessibility Guidelines (WCAG) 2.2*, W3C Recommendation,
  5 October 2023.

### 5.3.4 Learner styles, and an honest note about them

The module's CLO3 names visual, auditory, kinesthetic, social and solo learners, and the
interface addresses all five:

| Style | How the interface addresses it |
|---|---|
| **Visual** | Six views of the same result — both boundaries, ground truth alone, prediction alone, the colour-coded difference map, the probability heat map, and the scan alone. Adjustable overlay fill and line weight. The Dice bar with the 0.85 threshold marked on it. |
| **Auditory** | Six distinct cues on the events that matter (model loaded, prediction finished, error, prompt), plus a spoken readout of the result that follows the interface language. Rising pitch for success and falling for error, so the meaning survives poor speakers. |
| **Kinesthetic** | The lesion box is *dragged* on the image — the user's contribution to Task 1 is a physical gesture, not a number typed into a field. Sliders for the overlay, drag-and-drop upload, and a keyboard shortcut for every action. |
| **Social** | A timestamped session log, exportable as CSV, so a pair or group at one screen has a shared record of what was tried; a "same scan, each model" comparison table for discussion; and a one-click PNG export of the current case to paste into a report or a message. |
| **Solo** | A four-step checklist that ticks itself off, so someone working alone can see what remains without being walked through it; a per-view caption explaining what is being looked at; a metric glossary in plain language; and a self-paced walkthrough in the Help window. |

**The honest caveat, which belongs in the report.** The "learning styles" model — that
learners have a fixed modality and are better served by teaching matched to it — has not
survived experimental testing, and continuing to design on the assumption that it has is
criticised directly in the education literature (Kirschner, 2017). The defence of the
features above is therefore *not* that they match each user to a channel. It is that
**redundancy across channels helps everyone**: a result available as an image, as numbers,
as a sentence and as speech is robust to a noisy demonstration room, to a projector with
poor colour, to a user who is looking away, and to a screen-reader user — and it satisfies
WCAG's multiple-means-of-representation requirements as a side effect. Presented that way
the feature set is defensible on evidence; presented as learning-style matching it is not.

- Kirschner, P. A. (2017) 'Stop propagating the learning styles myth', *Computers &
  Education*, 106, pp. 166–171.

> `[ADD CITATION]` — one recent source on multimodal or redundancy-based interface design,
> or on universal design for learning, to carry the positive argument. Search terms:
> "multimodal redundancy interface design", "universal design for learning digital
> interface", "multiple means of representation software".

### 5.3.5 Other inclusive design decisions

- **Three text sizes and a high-contrast mode.** High contrast raises the border tokens as
  well as the text, because raising text contrast alone makes panels dissolve into the
  background.
- **Reduced motion**, both as a setting and by honouring the operating system's
  `prefers-reduced-motion`.
- **A light and a dark theme.** Dark suits reading MRI in a dimmed reporting room; light
  suits a projector in a bright room. Both are full themes, not an inverted filter.
- **Keyboard operable throughout**, with a visible focus ring on every control
  (`:focus-visible`, so it does not appear on ordinary mouse clicks).
- **A screen-reader live region.** Every message, view change and result is announced.
- **Icons are line-art SVG, not emoji.** Emoji render differently on every platform and
  several of the obvious candidates carry a skin tone or an implied gender that an
  interface used across cultures has no business asserting.
- **No text baked into images.** All labels are live text, so they translate, scale with
  the text-size setting and are readable by a screen reader. The only exception is the
  exported PNG strip, which is a deliberate artefact for the report.
- **British English spelling** consistently ("tumour", "visualise"), matching the module
  and the report.
- **Uncertainty is stated, not hidden.** A scan with no ground truth mask shows a dash and
  says why, rather than a zero that looks like a real score; and when a model returns an
  empty or confidently wrong mask, the interface names that failure mode and says such a
  case would be referred back for a manual box. Presenting an unreliable number as a
  reliable one is an accessibility failure as much as an honesty failure.

> `[ADD CITATION]` — one recent source on clinical-user-interface design or on
> uncertainty communication in medical AI, to support §5.3.5's last point. Search terms:
> "uncertainty communication clinical decision support interface", "silent failure medical
> AI user interface", "clinician trust segmentation visualisation".

### 5.3.6 What the interface does *not* claim

Three limitations, stated so the report does not overreach:

1. **The three translations were produced by the group, not by a certified medical
   translator.** They are adequate for a demonstration interface. Clinical deployment in
   Bahasa Melayu or Arabic would require professional translation and review of the
   clinical terms specifically.
2. **Only the interface is localised, not the model.** BRISC 2025 is a single dataset from
   a single acquisition context; changing the interface language does nothing about
   population or scanner differences in the underlying data, and the report should not
   suggest otherwise.
3. **Accessibility has been designed for, not audited.** WCAG criteria were designed
   against and checked manually; no assistive-technology user testing and no formal
   conformance audit were carried out.

---

## References cited above

- Brand, A., Allen, L., Altman, M., Hlava, M. and Scott, J. (2015) 'Beyond authorship:
  attribution, contribution, collaboration, and credit', *Learned Publishing*, 28(2),
  pp. 151–155. — the CRediT taxonomy used in the credits window.
- Kirschner, P. A. (2017) 'Stop propagating the learning styles myth', *Computers &
  Education*, 106, pp. 166–171.
- NISO (2022) *CRediT, Contributor Roles Taxonomy*, ANSI/NISO Z39.104-2022.
  https://credit.niso.org/
- Okabe, M. and Ito, K. (2008) *Color Universal Design (CUD)*.
- W3C (2018) *Web Content Accessibility Guidelines (WCAG) 2.1*, W3C Recommendation.
  https://www.w3.org/TR/WCAG21/
- W3C (2023) *Web Content Accessibility Guidelines (WCAG) 2.2*, W3C Recommendation.
  https://www.w3.org/TR/WCAG22/

Plus the five `[ADD CITATION]` slots above, which need papers from 2016–2024.
