# 5.1 Loading the Model and Running a Prediction

## 5.1.1 Installation and setup

Hardware specifications: Apple M1 Max, 10 CPU cores, 32 GB unified memory, macOS. The GUI itself needs no GPU, it is a web server and a browser page; the GPU is only used by the Task 2 networks when they are loaded, and they run on CPU as well.

Software: Python 3.11.1, Flask 3.1.3, TensorFlow 2.13.0 with tensorflow-metal 1.0.1 on the server environment, NumPy 1.24.3, OpenCV 4.8.1.78, scikit-image 0.22.0, scikit-learn 1.9.0 and joblib 1.5.3 on the Task 1 environment. The front end has no build step, no bundler and no third party JavaScript at all, the whole client is one HTML file, one stylesheet and three plain `.js` files, so there is nothing to compile before the interface will run.

What has to be in place before the first launch:

- **Two virtual environments beside the project folder.** `tf-env` runs the server and holds TensorFlow, `mvi-env` holds scikit-learn and is the only interpreter that can unpickle the Task 1 forest. They cannot be merged, TensorFlow 2.13 pins NumPy below 2 and the forest was pickled by a much newer scikit-learn, so one process cannot hold both. This is why the GUI starts a small helper process for Task 1 rather than importing it (Section 5.1.2).
- **The datasets**, each beside the code that uses it: `brisc2025/segmentation_task` for the brain MRI, `afnan/Dataset_BUSI_with_GT` for the breast ultrasound, and Aman's two share folders for the retinal results. A missing dataset is not fatal, the interface says which one is missing and where to put it, and the other models keep working.
- **Flask**, which is the only dependency the GUI adds on top of what Task 1 and Task 2 already needed. `run_gui.sh` installs it into `tf-env` by itself if it is not there.

Launching it is one command:

    cd task3
    ./run_gui.sh

The script checks that `tf-env` exists and refuses to start with a readable message if it does not, warns rather than fails if `mvi-env` is missing (the Task 2 models will still work, the Task 1 forest will not), generates the six WAV cues on a fresh checkout, and then serves the interface on http://127.0.0.1:5050. `MVI_GUI_PORT=5058 ./run_gui.sh` moves it if that port is taken, which on macOS it sometimes is. Stopping it is Ctrl-C. The server binds to 127.0.0.1 deliberately, it is reachable only from the machine it runs on, which is the right default for a tool that has patient images on screen.

## 5.1.2 Loading the models

The list holds eight models, one Task 1 and one Task 2 for each of the four of us, named after their author rather than after their file so the marker can see whose work is whose.

Table 1 The eight models and where their files live

| Group | Model | Dataset | File |
|---|---|---|---|
| Task 1 | SALMAN (ME) | BRISC 2025 brain MRI | `task1/outputs/rf_segmenter.joblib` |
| Task 1 | AFNAN | BUSI breast ultrasound | `task3/afnan_task1.py` (a pipeline, no weights) |
| Task 1 | AMAN | FIVES retinal fundus | `amman/Task1_GUI_share 2/results/` |
| Task 2 | SALMAN (ME) | BRISC 2025 brain MRI | `task2/outputs/models/mha_resunet_narrow.h5` |
| Task 2 | AFNAN | BUSI breast ultrasound | `afnan/innovation_seed42_pruned.keras` |
| Task 2 | AMAN | FIVES retinal fundus | `amman/Task2_GUI_share 2/results/` |
| Task 1 | ADIT | Figshare brain MRI | `task3/adit_task1.py` (a pipeline, no weights) |
| Task 2 | ADIT | Figshare brain MRI | `adit/gui_share/results/` |

Four different things happen behind the one **Load** button, and the interface hides that difference from the user rather than from the reader of this report:

- Salman's Task 1 forest is loaded by a helper process running on `mvi-env`, which the server starts by itself the first time it is needed and then talks to over a pipe. Nothing has to be activated by hand.
- Salman's and Afnan's Task 2 networks are loaded into the server process and stay warm between predictions. The one the list offers under Salman's name is the network that meets the module's two million parameter limit, 1,963,474 parameters at 7.82 MB, rather than the wider 8.84 M model it was cut down from - the interface offers the model that would actually be submitted, and the others are still reachable by path through "Load another model file".
- Afnan's Task 1 is her notebook pipeline ported into a module, so there is no file to load at all; it reports its stages where a spec sheet would normally report weights, and it says 0 parameters, which for a classical method is the honest answer rather than a missing one.
- Adit's Task 1 is her notebook pipeline ported into a module as well, and it is the one that needed the least persuading: her method wants a single point inside the lesion, her own notebook says in as many words that in a live GUI the seed would come from a click and that swapping the seed source is the only change needed, so the centre of the box the user drags is taken as that point. Asking her method for a click and every other Task 1 method for a box would be two interactions where one will do.
- Aman's two are read back from recorded results rather than run. His Task 2 checkpoint is a Keras 3 file and this GUI runs TensorFlow 2.13, which cannot deserialise it, and his hand-over notes ask for the recorded route in any case; both of his folders carry every test image, its ground truth, its prediction and the per-image metrics.
- Adit's Task 2 is recorded for a different reason, and the difference is worth stating because it is a hand-over problem rather than a technical one: she trained in a hosted session and the weights did not come back with the results, only the figures did. What she exported is sixteen report cases, and those have been rebuilt into the same layout Aman's folders use, so the interface treats them identically. Only those sixteen are offered in her Task 2 picker; the other 3,048 scans in her dataset have no prediction to show, and listing them would let a user pick one and be told so.

What "loaded" looks like on screen, in order: the card gains an accent border and its button changes from **Load** to **Ready**, a three note chime plays, the first item in the checklist ticks itself off, a line appears in the session log with the time, a toast says which model was loaded, and the specification sheet appears under the list. For the forest that sheet is the number of trees, the features per pixel, the minimum leaf size, the working patch and the box padding; for a network it is the architecture, the input size, the channel widths, the tunable parameters, the GFLOPs, the file size and the measured latency. Those FLOPs and latency figures are the ones Task 2 measured and wrote to `complexity.json`, they are reused rather than re-profiled because profiling costs several seconds per model and would make the interface look like it had frozen.

The first Task 2 model takes a few seconds longer than the rest. That is the Keras graph being built, plus one throwaway prediction that is run before the model is handed over, so that the latency reported for the user's first real scan is the inference time and not the graph compilation time. Without that the first number shown would be roughly ten times the true one and would not match the report.

Loading a model also switches the scan picker to that model's dataset. Running a breast ultrasound network on a brain MRI would still produce a mask and a Dice score, and both would be meaningless, so the interface does not allow the two to drift apart.

## 5.1.3 Image input and validation

There are two ways an image gets in. The **Scan** picker lists the dataset that belongs to the loaded model, and the **Show** menu narrows that list down: the scans the model was actually evaluated on (which is the default, so the numbers on screen are comparable with the report), every scan in the dataset, or one class at a time — glioma, meningioma and pituitary for BRISC, benign, malignant and no-lesion for BUSI, and the report cases for FIVES. **Prev**, **Random** and **Next** step through the list, as do the `N` and `P` keys.

The second way is the upload area, which takes a drop or a click. JPG, PNG, BMP and TIFF are accepted. If the ground truth mask is available as well, ticking "I also have the ground truth mask" before choosing the file asks for the mask second and then scores the prediction against it; without a mask the scan is still segmented and the metrics show a dash, because there is nothing to score against and printing a number anyway would be worse than admitting it.

Validation happens in three places and each one has a message written for the user rather than for the developer:

- The extension is checked first, and anything that is not one of the four gives "unsupported file type - use JPG, PNG, BMP or TIFF".
- The file is then saved under a unique name and re-read with OpenCV. A file that carries the right extension but is not really an image gets as far as this check, and when it fails the uploaded file is deleted again and the message is "that file could not be read as an image". A mask that cannot be decoded is refused the same way.
- A model is never run without the input it needs. Salman's Task 1 model is semi-automated, and pressing Segment before a box is drawn does not throw an exception, it plays the prompt sound and says "Task 1 needs a box — drag one, or press Simulate box" (Figure 1). Every scan the picker offers is resolved against a fixed list of allowed folders, so a crafted file name cannot be used to read anything else on the machine.

[FIG] task3/docs/figures/fig_validation.jpg | Figure 1 Validation rather than a crash: the interface refuses to run the semi-automated model until it has the box, says which input is missing, and plays the prompt cue

## 5.1.4 Running a prediction

The flow from click to result is short by design. Press **Segment this scan**, or the `R` key, and the stage dims with a spinner while the request is in flight. The server resizes the scan to whatever the loaded model expects — 128 px for the Task 1 region of interest, 192 px for Salman's network, 256 px for Afnan's — runs it, pushes the mask back up to a common 512 × 512, and scores it against the ground truth in that same space, which is what makes the numbers from three different students' models comparable in one window.

Four things then update at once: the outlines are drawn over the scan, the metrics panel fills in with the Dice score set large above the rest, a chime plays, and a timestamped line is added to the session log. Running a second model on the same scan adds a row to the "same scan, each model" comparison table instead of overwriting the result, which is the quickest way to show two models disagreeing on one image.

The measured round trips are honest and small: 42.5 ms for Salman's narrow network, 75.9 ms for the Task 1 forest including the worker round trip, about 170 ms for Afnan's network, about 560 ms for Afnan's classical pipeline, which is the slowest of the six because it generates and scores proposals rather than doing one forward pass, about 500 ms for Adit's seeded pipeline, and effectively instant for the four recorded entries because a recorded mask is read from disk.

Saving the result is three separate things, for three separate purposes. **Export PNG** downloads a labelled strip of the scan, the ground truth, the prediction and the difference map in one image, named after the scan and the model, which is what goes into a report or a slide. **Save** on the session log downloads the whole session as CSV, every model loaded and every prediction run with its scores, which is what goes into an appendix. And the page prints, with the controls and the log hidden by a print stylesheet, so a single case can be printed as a record.

## 5.1.5 Troubleshooting

Table 2 Common problems and what to do about them

| Symptom | Cause and fix |
|---|---|
| "Address already in use" at start-up | Something else holds port 5050 - on macOS it is often the AirPlay receiver. Start with `MVI_GUI_PORT=5058 ./run_gui.sh`. |
| The Task 1 model will not load | The forest runs in a helper process on `mvi-env`. Check that the `mvi-env` folder is still beside `task3`, then press Load again, which restarts the helper. |
| "lost the connection to the Task 1 worker" | The helper died, usually because `mvi-env` was rebuilt while the server was running. Pressing Load again starts a new one; nothing else needs restarting. |
| No scans are listed | That model's dataset is not on the machine. The interface names the folder it expects on the card itself - `brisc2025/segmentation_task`, `afnan/Dataset_BUSI_with_GT`, or Aman's share folders. Uploading your own scan works without any of them. |
| The metrics all show a dash | The scan has no ground truth mask, which is normal for an uploaded image. The segmentation is still shown, there is simply nothing to score it against. |
| A Task 2 model returns an empty mask | Roughly one scan in a hundred fails this way, and it is a property of the model rather than of the interface. It is reported in the Task 2 chapter as the silent failure case, and it is the reason the semi-automated model is kept as a fallback. |
| Aman's model will not load live | His checkpoint is a Keras 3 file and this environment is TensorFlow 2.13. This is expected, not a fault: his entries are served from recorded results, and the card says so. Running it live would need TensorFlow 2.16 or newer in a separate environment. |
| There is no sound | Browsers block audio until the page has been clicked once. Click anywhere, then check the speaker button in the header is on and the volume slider under Display is not at zero. |
| Nothing is read aloud | The spoken readout uses the browser's speech synthesis; Chrome and Safari have voices for English, Malay and Arabic, and a browser with no installed voice for the selected language will stay silent while the rest of the interface still works. |
| The interface looks cramped | The layout reflows below 1150 px and stacks below 900 px. The three text sizes and the higher contrast switch are under Display, and the design can be changed there as well. |
| A model file that was retrained will not load | Use "Load another model file" with a path inside the project folder. `.joblib` is treated as Task 1, `.h5` as Task 2 and `.keras` as one of Afnan's networks. Paths outside the project folder are refused on purpose. |

# 5.2 GUI Features

## 5.2.1 Interface layout

The window is in two halves. The scan is the largest thing on screen because it is what the user is here to look at, and everything that is not the scan is stacked in one column beside it, set-up above and results below.

[FIG] task3/docs/figures/fig_layout.png | Figure 2 The interface after a prediction, with the eight regions marked

Table 3 The regions marked in Figure 2

| | Region | What it does |
|---|---|---|
| A | Top bar | Language, sound on/off, spoken readout, light/dark theme, and the Display, Help and Credits windows. |
| B | View tabs and caption | Six ways of drawing the same result, with a sentence under them saying what is being looked at. |
| C | Image stage | The scan with the segmentation drawn on it. For Salman's Task 1 model this is also where the box is dragged. |
| D | Overlay controls | Fill opacity, outline thickness, show or hide the box, and Export PNG. |
| E | Action bar | The Segment button with its keyboard shortcut, the box controls when the loaded model needs one, and a hint saying what is missing before it can run. |
| F | Checklist | The four steps - load a model, choose a scan, mark the lesion, run - which tick themselves off as the work is done, and grey out when they do not apply. |
| G | Model list | The eight models grouped by task and named by author, each with its dataset, its file and its size. |
| H | Metrics | The Dice score set large with a bar and the 0.85 threshold marked, then the rest of the numbers, a plain-language verdict, and the session log below it. |

## 5.2.2 Image display panels

The same result is offered six ways, and each one answers a question the others cannot.

[FIG] task3/docs/figures/fig_views.png | Figure 3 The six views, SALMAN (ME), Task 2 on a BRISC meningioma, Dice 0.950

The same six panels are drawn for every model in the list, whoever wrote it and whatever it was trained on, and the three figures below are Afnan's, Aman's and Adit's models shown the same way. They are worth putting side by side with mine because the four of them make the panels behave differently, and that is a property of the images rather than of the interface.

[FIG] task3/docs/figures/fig_views_afnan.png | Figure 4 The six views, AFNAN, Task 2 on a benign BUSI lesion, Dice 0.959

[FIG] task3/docs/figures/fig_views_amman.png | Figure 5 The six views, AMAN, Task 2 on FIVES report case 13_A, Dice 0.942. The confidence panel is the plain scan because a recorded result has no probability map behind it

[FIG] task3/docs/figures/fig_views_adit.png | Figure 6 The six views, ADIT, Task 1 on a Figshare slice, Dice 0.892. Her method is seeded rather than automatic, so the box drawn on the scan is the interaction and its centre is the seed

On Afnan's ultrasound lesion the panels read almost exactly as they do on my MRI, which is the point: one compact region, so the boundaries view is the one to look at, the prediction and the ground truth are close enough that the difference view is mostly agreement with a thin rim of over-segmentation down one side, and the confidence panel shows the network was sure about the whole lesion rather than only its centre. The scan-only panel is the one that earns its place here more than it does on a brain MRI, because the lesion is obvious to the eye on B-mode and it is worth seeing the image without an overlay leading it.

Aman's retinal image is where the same six panels stop behaving the same way. The target is not one compact region but a vessel tree spread across the whole image, and the boundaries view suffers for it — outlining hundreds of thin branches in two colours produces a picture where the red prediction almost completely covers the green ground truth, so the two look identical whether they agree or not. The difference view is the one that works: yellow where the two agree, green where a vessel was missed, and magenta where one was found that is not in the ground truth, and at a glance it shows the network following the fine peripheral branches and dropping a few of the thinnest. The confidence panel is deliberately blank, that model's results were recorded rather than run here, and a finished mask carries no probabilities, which the caption under the view says in as many words rather than leaving an unexplained grey circle.

Adit's Figshare slice is a brain MRI like mine and the panels behave accordingly, with one difference that is hers rather than the dataset's: her method is semi-automated, so the box is on the scan in the boundaries view and the interaction is visible in the picture. It is the only one of the four figures where the user's contribution can be seen, which makes it the clearest illustration in this chapter of what "semi-automated" actually means.

So the honest summary of these four figures is that the six views are not equally useful on all four datasets, and the interface does not pretend otherwise. Two of them — difference and scan only — carry most of the weight on a vessel tree, and boundaries carries it on a compact lesion.

Table 4 What each view shows and why it is there

| View | What it shows | Why it is useful |
|---|---|---|
| Boundaries | The ground truth outline and the predicted outline on the same scan | Where the two lines separate is the error the Dice score is measuring, which makes an abstract number visible. |
| Ground truth | The radiologist's mask on its own | What the model is being asked to reproduce, without the prediction on top of it. |
| Prediction | The model's mask on its own, drawn the same way | Comparable directly with the view before it, which is how a small difference in shape is spotted. |
| Difference | Agreement, tumour that was missed, and healthy tissue wrongly included, in three colours | The same Dice score can come from either kind of mistake, and only this view says which. Missing a lesion and over-segmenting one have very different clinical consequences. |
| Confidence | The probability map before it is cut at 0.5 | A confident wrong answer and an unsure one look identical once thresholded, and this is the view that separates them. It is blank for a recorded result, and the caption says so rather than leaving an unexplained plain scan. |
| Scan only | The image with nothing drawn on it | For judging the scan by eye without the overlay leading it, and for a clean screenshot. |

The overlay fill can be taken from 0 to 100% and the outline from 1 to 6 pixels, which matters more than it sounds: a thin outline is right for a small pituitary tumour and unreadable on a projector, and the fill has to come down to nothing when the texture underneath is what is being judged. The colour scheme can also be switched to the Okabe-Ito colour-blind safe set, because the default green and red pair is the one around one man in twelve cannot reliably separate.

## 5.2.3 Metrics panel

Six numbers are reported for every prediction, and the panel is arranged so the one the marking criteria are written against is the one the eye lands on first.

Table 5 The metrics, and how to read them

| Metric | What it is | How to read it |
|---|---|---|
| Dice | 2\|A ∩ B\| ÷ (\|A\| + \|B\|), the overlap between the prediction and the ground truth | 0 is no overlap and 1 is a perfect match. The bar under the number is marked at 0.85, which is the threshold the assignment treats as a strong result; above 0.90 is excellent, 0.70 to 0.85 is usable, below 0.70 the mask is wrong in a way that would be obvious on screen. |
| IoU | \|A ∩ B\| ÷ \|A ∪ B\| | Measures the same agreement more harshly than Dice and is always the lower of the two. Useful as a sanity check: a large gap between Dice and IoU means a lot of the agreement is thin and marginal. |
| HD95 | The 95th percentile distance in pixels from each predicted boundary point to the nearest true boundary point | This is the boundary quality, and it is the number that separates two models with the same Dice. Single digits is a tight contour; 25 px and above means the outline wanders even though the area may be roughly right. The 95th percentile rather than the maximum, so one stray pixel does not dominate it. |
| Pixel accuracy | (TP + TN) ÷ (TP + TN + FP + FN) | Always above 0.98 and therefore the least informative number in the panel. It is reported because the brief asks for it, and it is high for everyone simply because the lesion is a small part of a large frame, so calling everything background is already almost right. |
| Sensitivity | The share of the true lesion that was found | Low sensitivity with high precision means the model is being cautious and under-segmenting. |
| Precision | The share of the prediction that is really lesion | Low precision with high sensitivity means the opposite, the mask has spilled into healthy tissue. |

Underneath the numbers is a plain-language verdict rather than another figure, because a Dice of 0.87 means nothing to somebody who has not read the chapter, and "excellent overlap — above the 0.85 threshold" does. The Help window carries a glossary of the same six terms in the interface language, and the whole set can be read aloud.

For Aman's two entries the numbers shown are the ones recorded when his results were produced, not a rescore of the display copies. His hand-over notes ask for exactly that, and rescoring 512 px PNGs would produce figures that quietly disagreed with his own report by a decimal place or two.

## 5.2.4 The same interface on the other two datasets

Everything described so far was shown on a brain MRI, which is the dataset my own two models use, and the more interesting question for a group GUI is whether the same window is still the right window when somebody else's model is loaded on somebody else's images. The six figures below are the other three students' models, running in the same interface with nothing changed but the choice in the model list.

Afnan's two are both on BUSI breast ultrasound, and they are deliberately shown on the same scan so the two methods can be compared rather than two different images being compared.

[FIG] task3/docs/figures/fig_afnan_task1.jpg | Figure 7 AFNAN, Task 1. The classical pipeline on a benign BUSI lesion, Dice 0.974. The box controls are gone because this method finds the lesion itself

[FIG] task3/docs/figures/fig_afnan_task2.jpg | Figure 8 AFNAN, Task 2. Her pruned network on the same scan, Dice 0.959, HD95 8.00 px against the classical pipeline's 6.40 px

Two things in those two figures are worth pointing out. The first is that the interface has quietly reconfigured itself: the scan picker has switched to the BUSI list with its own classes, the checklist has greyed out "mark the lesion" because neither of these two methods takes a box, and the action bar has dropped the box controls rather than showing them and ignoring them. Loading a model is the only thing the user did.

The second is the result itself, and it needs stating carefully. On this particular scan her classical pipeline scores higher than her network, 0.974 against 0.959, with a tighter boundary as well. That is one image and it is not a claim about which of her two methods is better overall — her own chapter reports the averages over her full test set, and a single well-behaved benign lesion with a clear dark core is exactly the case a classical method handles best. What it does show is why running two models on one scan side by side is worth having in the interface at all: the comparison table underneath records both, and a disagreement like this one is the thing worth discussing rather than the average.

Aman's two are on FIVES retinal fundus photographs, which is a different problem again — the target is not one compact lesion but the whole vessel tree, thin branching structures spread across the entire image.

[FIG] task3/docs/figures/fig_amman_task1.jpg | Figure 9 AMAN, Task 1. The classical vessel segmentation on report case 13_A, Dice 0.894 from the recorded results

[FIG] task3/docs/figures/fig_amman_task2.jpg | Figure 10 AMAN, Task 2. His pruned attention U-Net on the same image, Dice 0.942 and HD95 1.00 px against the classical method's 5.39 px

Here the deep model is clearly the better of the two, 0.942 against 0.894, and the boundary error falls from 5.39 px to 1.00 px, which on a vessel tree means the network is following the thin peripheral branches that the classical method breaks up or misses. That is visible in the figures without reading the numbers, which is the point of showing the picture at all.

These two also show the one place the interface has to be honest about what it is doing. Aman's models are not run here, they are recorded results read back from the files he handed over, and the card in the model list says "recorded results" for exactly that reason. The metrics shown are the ones measured when those results were produced rather than a rescore, the latency field is blank because nothing was timed, and the confidence view says plainly that a finished mask has no probability map behind it. An interface that displayed a recorded mask as though it had just been computed would be showing the user something that is not true, and on a marked assignment that is worth more than the convenience of pretending all eight models work the same way.

Adit's two are on Figshare brain MRI, which is the same organ as mine on a different dataset, and between them they make the two points this section is really about.

[FIG] task3/docs/figures/fig_adit_task1.jpg | Figure 11 ADIT, Task 1. Her seeded pipeline on Figshare slice 1, Dice 0.892. The dashed box is the interaction and its centre is the seed her method asks for

[FIG] task3/docs/figures/fig_adit_task2.jpg | Figure 12 ADIT, Task 2. Her EfficientNet-B4 encoder U-Net on report case 627, Dice 0.976 from the recorded results

The first point is that a method with a different interaction still fits. Hers is the only one of the eight that wants a point rather than a box or nothing at all, and rather than adding a second interaction the interface hands her the centre of the box it already has, which is exactly the swap her notebook says is needed. The checklist ticks "mark the lesion" for her the same way it does for mine, and a user moving between the two would not notice that one method wants a rectangle and the other only a point inside it.

The second is a limit worth being honest about. Her Task 2 offers sixteen scans where every other model offers between fifty and two hundred, because sixteen is what she exported predictions for. The interface does not disguise that: the card says recorded results, the picker lists what exists, and picking anything else is impossible rather than merely disappointing. A group GUI is only as complete as what its members hand over, and the right response to a partial hand-over is to show exactly what arrived rather than to pad it.

Across the four datasets nothing about the layout had to change. The same six views, the same six metrics, the same overlay controls and the same export all work on an MRI slice, an ultrasound sweep and a fundus photograph, because everything is reduced to a scan, a ground truth mask and a predicted mask at a common size before it reaches the display. The only visible difference is which view earns its place: on a compact tumour the boundaries view is the one to look at, and on a vessel tree the difference view is far more useful, because red and green outlines around hundreds of thin branches overlap into one colour whereas agreement, missed and extra do not.

## 5.2.5 Feature summary

Table 6 Every feature in the interface

| Feature | What it does |
|---|---|
| Eight models, one list | One Task 1 and one Task 2 model per student, grouped by task and named by author. |
| Four datasets | BRISC brain MRI, BUSI breast ultrasound, FIVES retinal fundus and Figshare brain MRI; loading a model switches the scan picker to match it. |
| Live prediction | Five of the eight models are loaded and run in the interface; the other three are served from recorded results and labelled as such. |
| Semi-automated interaction | The lesion box is dragged directly on the scan, or simulated with the same 2-20% slack the evaluation used; for the method that wants a single seed point instead, the centre of that box is the point. |
| Six views | Boundaries, ground truth, prediction, difference, confidence and scan only. |
| Overlay controls | Fill opacity 0-100%, outline thickness 1-6 px, box on or off. |
| Metrics panel | Dice, IoU, HD95, pixel accuracy, sensitivity and precision, with a threshold bar and a plain-language verdict. |
| Model comparison | A table of every model run on the current scan, with Dice, HD95, IoU and inference time side by side. |
| Session log | Timestamped record of everything loaded and run, downloadable as CSV. |
| Export PNG | The current case as a labelled strip: scan, ground truth, prediction, difference. |
| Upload | Drag and drop or click, JPG/PNG/BMP/TIFF, with an optional ground truth mask. |
| Three languages | English, Bahasa Melayu and Arabic, including tooltips, screen-reader labels and image alternative text. |
| Right-to-left mirroring | Selecting Arabic mirrors the whole layout, not only the words. |
| Six sound cues | Distinct tones for ready, model loaded, prediction done, prompt, error and click, with a volume control and a mute button. |
| Spoken readout | The result read aloud in the selected language through the browser's speech synthesis. |
| Three interface designs | Aurora, the design the group settled on, plus a clinical workstation layout and the original, switchable at any time. |
| Light and dark themes | One button, remembered between sessions. |
| Accessibility settings | Three text sizes, a higher contrast mode, reduced animation, and a colour-blind safe overlay palette. |
| Keyboard shortcuts | Run, simulate box, next and previous scan, switch view, mute, help and close, all without the mouse. |
| Self-paced checklist | Four steps that tick themselves off, and grey out when they do not apply to the loaded model. |
| Help window | A walkthrough, the metric glossary, the keyboard list and a troubleshooting section. |
| Credits window | Each member's contribution stated with the CRediT taxonomy. |

# 5.3 Support for Different Learner Styles

## 5.3.1 Framework and rationale

The module's third learning outcome names visual, auditory, kinaesthetic, social and solo learners, which is the VARK scheme of Fleming and Mills (1992) with the social and solo pair from Grasha's categories added to it. The interface addresses all five, and the sections below say how, but the reasoning behind the design is not the one the scheme is usually used to justify, and saying so plainly is more defensible than pretending otherwise.

The claim that each learner has a fixed preferred modality and learns better when teaching is matched to it — the meshing hypothesis — has not survived experimental testing, and designing on the assumption that it has is criticised directly in the education literature (Kirschner, 2017; Pashler et al., 2008). The defence of everything below is therefore not that each user is matched to a channel. It is that **redundancy across channels helps everybody**, which is the position Universal Design for Learning takes when it asks for multiple means of representation, of action and expression, and of engagement (CAST, 2018), and which the multimedia learning literature supports on the grounds that information presented in two complementary channels is retained better than in one (Mayer, 2009).

The practical version of that argument is easy to see in a demonstration room. A result that exists as an image, as numbers, as a sentence and as speech survives a projector with poor colour, a noisy room, a user who is looking at their notes rather than the screen, and a marker using a screen reader. It also satisfies the WCAG 2.2 requirement for multiple means of representation as a side effect (W3C, 2023). Presented that way the feature set stands on evidence; presented as learning-style matching it does not.

## 5.3.2 Visual and read/write

The visual channel carries most of the interface. The same result is drawn six ways (Section 5.2.2), the overlay fill and outline weight are adjustable so the drawing suits the scan rather than the other way round, the Dice score is set in large type with a bar that has the 0.85 threshold physically marked on it, and the difference view uses three colours to separate agreement from the two kinds of error. Colour is never the only carrier of meaning: the legend names every colour in words, the metric rows are labelled, and the colour-blind safe palette is one menu away for the green and red pair that around one man in twelve cannot separate reliably (Okabe and Ito, 2008).

The read/write channel is served by the text that sits beside every visual. Each view has a caption saying what is being looked at, the verdict under the Dice score is a sentence rather than a number, the Help window carries a glossary of all six metrics in plain language, the model cards describe their architecture in one line, and the session log is a written record that downloads as CSV for a reader who would rather work through the numbers in a spreadsheet than watch them appear on screen.

## 5.3.3 Auditory

Six distinct sound cues mark the events that matter: the interface becoming ready, a model finishing loading, a prediction completing, a prompt for an input the user has not given yet, an error, and a click on any control. They are generated rather than downloaded, so they are short, consistent and carry no licence, and they are designed so that meaning survives poor speakers — success cues rise in pitch and error cues fall, which is a distinction that still reads on a laptop speaker in a room full of people.

Beyond the cues, the result can be read aloud. The spoken readout uses the browser's speech synthesis and follows the interface language, so switching to Bahasa Melayu or Arabic changes the spoken language as well as the written one, and it reads the metrics as sentences rather than reciting digits. Both the sound effects and the speech have their own toggles in the header and a volume control under Display, because in a shared room the right setting is sometimes silence and a feature that cannot be turned off is a feature that gets in the way.

## 5.3.4 Kinaesthetic

The interaction that matters most here is a physical one. For Task 1 the user's entire contribution is a box dragged around the lesion directly on the scan, not four numbers typed into fields, and the readout underneath reports its size in pixels as it is drawn. That is the gesture the semi-automated method is built around, and doing it by hand is what makes the difference between a semi-automated and an automatic method obvious rather than theoretical.

The rest of the controls are the same kind of thing: two sliders that change the overlay while being dragged rather than after a button press, drag-and-drop for uploading a scan, tabs that switch the view instantly so a user can flick between prediction and ground truth to see a boundary move, and a Random button for working through scans quickly. Every one of those actions also has a keyboard shortcut — `R` to run, `B` to simulate the box, `N` and `P` to step through scans, `1` to `6` for the views, `M` to mute, `?` for help — so somebody who would rather keep their hands on the keyboard can drive the whole interface without the mouse, which is both a kinaesthetic affordance and an accessibility one.

## 5.3.5 Social and solo

For social use the interface keeps a shared, visible record. The session log timestamps every model loaded and every prediction run, so a pair working at one screen can see what has already been tried without relying on memory, and it downloads as CSV so it can be circulated afterwards. The "same scan, each model" table is built for exactly the conversation two people have in front of a screen — it puts three models' Dice, HD95, IoU and inference time on one scan side by side, which turns "this one looks better" into a comparison that can be argued about. The Export PNG button produces a labelled strip of a single case in one click, which is what gets pasted into a message or a slide, and the Credits window states who did what using the CRediT taxonomy so the group's division of labour is part of the tool rather than a paragraph in a document nobody opens.

For solo use the interface is built to be worked through without anybody explaining it. The four-step checklist ticks itself off and greys out the steps that do not apply to the loaded model, so a user working alone can always see what is left. Every view has a caption, every metric has a glossary entry, every button has a tooltip, and the Help window contains a walkthrough, the keyboard list and a troubleshooting section. The preferences — language, theme, text size, contrast, palette, volume and the interface design — are remembered in the browser, so a returning solo user finds the tool as they left it.

# 5.4 Impact on Culture and Society

## 5.4.1 Multilingual access and inclusivity

Malaysia is a multilingual country and a clinical tool that exists only in English excludes part of the workforce that would use it. Bahasa Melayu is the national language and the one a Malaysian clinical user is most likely to prefer, so it is the obvious second language on the merits rather than merely the nearest one to hand. The argument for translating at all is not politeness, language barriers in healthcare are associated with worse patient understanding, lower satisfaction and measurable safety consequences (Al Shamsi et al., 2020), and a tool whose interface is not understood is a tool that gets used tentatively or not at all.

Arabic was added for a structural reason rather than a demographic one. A second left-to-right language only exercises the string table; a right-to-left language exercises the layout, and an interface that swaps its words but keeps its structure the wrong way round is harder to use for a right-to-left reader because reading order, scanning order and the expected position of a "next" control all reverse together. Adding Arabic forced the interface to be built so that it genuinely mirrors, which is a property that cannot be retrofitted by translation alone.

[FIG] task3/docs/figures/fig_arabic.jpg | Figure 13 The interface in Arabic. The whole layout mirrors, not only the words - the panels swap sides, the sliders fill from the right, and the numbers stay left to right

The mirroring is implemented once rather than as a second stylesheet. Every side in the stylesheet is named logically — `margin-inline-start` rather than `margin-left`, `inset-inline-start` rather than `left` — so selecting Arabic sets one attribute on the document and the whole interface reverses: the columns swap, the sliders fill from the other end, the next-scan arrow turns round, and the threshold mark on the Dice bar moves with them. Numbers, file names and model identifiers are held left to right inside the right-to-left text, which is correct, Arabic writes numerals in that direction even in the middle of a right-to-left sentence.

Translation covers the whole interface and not only the visible labels. Tooltips, screen-reader labels, image alternative text and the metric glossary are all keyed and translated, because an interface that leaves its accessibility labels in English is only half translated, and the user who is worst affected by that is the one using a screen reader in the second language.

## 5.4.2 Healthcare equity and clinical value

The three models in this GUI segment three different things — a brain tumour on MRI, a breast lesion on ultrasound, and the vessels in a retinal photograph — and what they have in common is that each one is a manual outlining job that currently costs a specialist's time. Radiologist availability varies enormously between and within countries, and the specialist workforce is concentrated in large urban centres, so a tool that removes the outlining work rather than the clinician is worth more where the clinicians are scarcest.

The efficiency argument in the individual chapters is therefore also an access argument. The models in this interface are between 3 and 13 MB and all of them run on a laptop CPU; the transformer-based models that dominate current leaderboards need tens of millions of parameters and a datacentre GPU. A district hospital can buy the laptop. That is the whole distinction, and it is why the pruning work is reported as an equity result as well as an environmental one.

Retinal screening makes the point most directly of the three. Diabetic retinopathy screening is a volume problem — large numbers of images, most of them normal, read by a small number of trained graders — and vessel segmentation is a step in that pipeline. Ultrasound makes it in a different way, because it is portable, inexpensive and emits no ionising radiation, which is what makes it the realistic modality in rural and lower-resource settings in the first place.

## 5.4.3 Ethics of AI-assisted screening

Every model in this interface is decision support and none of them is a replacement for the clinician, and there are three specific reasons to state that rather than treat it as a formality.

The first is automation bias, the well-documented tendency for people to accept an automated recommendation and to stop looking as carefully once one is on screen (Goddard, Roudsari and Wyatt, 2012). It is not hypothetical here: the Task 2 model in this project fails silently on about one scan in a hundred, returning either an empty mask or a confidently placed one in the wrong structure, with no uncertainty signal at all. A contour that looks plausible but is in the wrong place is more dangerous than an obviously missing one, and the interface's answer is to make the failure visible rather than to hide it — the confidence view exists precisely so that a confident wrong answer and an unsure one can be told apart, and the recommended workflow runs the automatic model first and falls back to the semi-automated one on any case that looks empty or anomalous.

The second is over-reliance in the other direction, the tool being trusted outside what it was measured on. Each model here was trained and scored on one dataset from a limited set of scanners and populations, and none of them has cross-dataset validation. Performance on a Malaysian patient population, on different equipment, or on a demographically different group is unknown and should be validated locally rather than assumed to transfer, which is a general problem in clinical machine learning rather than a defect peculiar to this project (Char, Shah and Magnus, 2018).

The third is privacy. The GUI runs entirely on the local machine, the server binds to 127.0.0.1 and nothing is uploaded anywhere, so patient images never leave the computer they are already on. That removes a data-residency question that would otherwise have to be answered before any clinical trial of the tool, and it matters under Malaysia's Personal Data Protection Act 2010, under which health data is sensitive personal data. Uploaded scans are written to a local folder under a generated name and can be deleted by deleting the folder.

None of the three is solved by the interface. What the interface can do is avoid making them worse: show uncertainty where it exists, label recorded results as recorded rather than implying a live prediction, report the metrics that show a failure instead of only the ones that flatter, and keep the clinician in the loop by design rather than by policy.

## 5.4.4 Credits

Contributions are stated using CRediT, the Contributor Roles Taxonomy (Brand et al., 2015; NISO, 2022), which is the standard the brief names. The same statement is built into the interface itself and is read from `task3/credits.json` every time the Credits window is opened, so it cannot drift out of step with what the report says.

[FIG] task3/docs/figures/fig_credits.jpg | Figure 14 The Credits window, which reads the contribution statement from the project file

Table 7 Contribution statement (CRediT)

| Member | ID | CRediT roles | Main files |
|---|---|---|---|
| Salman Alshawaf | TP072045 | Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Visualization; Writing — original draft | `task1/`, `task2/`, `task3/app.py`, `task3/backends.py` |
| Maryam Afnan | TP079945 | Software; Methodology; Investigation; Validation; Project administration | `task3/render.py`, the BUSI Task 1 and Task 2 work |
| Mohamed Aman | TP079948 | Software; Methodology; Investigation; Data curation; Visualization | `task3/static/js/i18n.js`, the FIVES Task 1 and Task 2 work |
| Adit Mayen Angony | TP079918 | Software; Resources; Project administration; Writing — review & editing | `task3/static/js/audio.js`, `task3/docs/COOPERATION.md` |

Supervision: Dr Reuben George.

## 5.4.5 Evidence of cooperation

The brief asks for evidence that the group gave and received clear instructions during development, and names GitHub issues, commit messages, peer-review comments and meeting records as the kinds that count.

What the repository holds at the time of writing is the commit history, which begins with the Task 3 work because the individual tasks were built before the project was put under version control. That is stated plainly rather than disguised, an honest short history reads better in a viva than a long invented one, and the working rule from that point on has been to commit in small pieces with messages that say why a change was made rather than what changed, since the diff already says what.

Alongside the history, `task3/docs/COOPERATION.md` holds the group's record: the meeting template, the task allocation table and the peer-review log. It also records the hand-over evidence that exists in the repository in a form a marker can check — Aman's two share folders arrived with a `HOW_TO_INTEGRATE.txt` for each task, written as instructions to whoever would do the integration, and those instructions are the reason his models are served from recorded results rather than run live, which is a decision taken from a written instruction and can be traced to it. Adit's hand-over is evidence of a different kind and is recorded as such: her notebook carries a note inside the Task 1 code saying what a GUI would have to change to drive her method from a click, and the integration did exactly that, whilst her Task 2 weights never left the hosted session she trained in, which is why only sixteen of her scans can be shown. Both are worth minuting because they are the two ways a hand-over actually goes, one anticipated and one incomplete.

**This section has to be completed by the group before submission.** The meeting records, the task-allocation table and the peer-review comments are in `COOPERATION.md` as a scaffold with the tables empty, and they must be filled in with what actually happened, by the people it happened to. Minutes written afterwards by one person are worth nothing if the group is asked about them in the viva, and the viva is where this gets tested.

# References

Al Shamsi, H., Almutairi, A.G., Al Mashrafi, S. and Al Kalbani, T. (2020) 'Implications of language barriers for healthcare: a systematic review', *Oman Medical Journal*, 35(2), e122.

Brand, A., Allen, L., Altman, M., Hlava, M. and Scott, J. (2015) 'Beyond authorship: attribution, contribution, collaboration, and credit', *Learned Publishing*, 28(2), pp. 151-155.

CAST (2018) *Universal Design for Learning Guidelines version 2.2*. Wakefield, MA: CAST.

Char, D.S., Shah, N.H. and Magnus, D. (2018) 'Implementing machine learning in health care - addressing ethical challenges', *New England Journal of Medicine*, 378(11), pp. 981-983.

Fleming, N.D. and Mills, C. (1992) 'Not another inventory, rather a catalyst for reflection', *To Improve the Academy*, 11, pp. 137-155.

Goddard, K., Roudsari, A. and Wyatt, J.C. (2012) 'Automation bias: a systematic review of frequency, effect mediators, and mitigators', *Journal of the American Medical Informatics Association*, 19(1), pp. 121-127.

Kirschner, P.A. (2017) 'Stop propagating the learning styles myth', *Computers & Education*, 106, pp. 166-171.

Mayer, R.E. (2009) *Multimedia Learning*. 2nd edn. Cambridge: Cambridge University Press.

NISO (2022) *CRediT, Contributor Roles Taxonomy*, ANSI/NISO Z39.104-2022.

Okabe, M. and Ito, K. (2008) *Color Universal Design (CUD): how to make figures and presentations that are friendly to colour-blind people*.

Pashler, H., McDaniel, M., Rohrer, D. and Bjork, R. (2008) 'Learning styles: concepts and evidence', *Psychological Science in the Public Interest*, 9(3), pp. 105-119.

W3C (2023) *Web Content Accessibility Guidelines (WCAG) 2.2*, W3C Recommendation.
