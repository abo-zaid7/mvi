/*
 * Task 3 GUI - front end.
 *
 * Layout of this file:
 *
 *   state / helpers        the one object holding what is selected
 *   settings               preferences saved in localStorage
 *   models                 listing, loading, the spec sheet
 *   images                 listing, picking, uploading
 *   the box                dragging the Task 1 rectangle on the canvas
 *   running                calling the model and showing the result
 *   views                  the tab strip and the rendered PNG
 *   metrics                the numbers, the verdict, the spoken readout
 *   log / compare          session history and model-vs-model table
 *   credits                the CReDiT window
 *   keyboard / start-up
 */

const state = {
  models: [],
  loadedModel: null,       // path of the model currently in memory
  loadedTask: null,        // 1 or 2
  loadedDataset: null,     // "brisc", "busi" or "fives" - the scans it expects
  needsBox: false,         // whether the loaded model waits for a user box
  loadedInfo: null,        // its spec sheet, kept so it can be re-translated
  datasets: {},            // what /api/state says is actually on disk
  dataset: "brisc",        // the dataset the scan picker is currently showing
  images: [],
  imageIndex: -1,
  box: null,               // [top, left, bottom, right] in 512-space
  boxSimulated: false,     // whether that box came from the evaluation simulator
  view: "boundaries",
  result: null,
  comparisons: [],
  logEntries: [],
};

const settings = {
  language: "en",
  skin: "aurora",
  theme: "dark",
  sound: true,
  speech: false,
  palette: "clinical",
  textSize: "normal",
  contrast: false,
  reduceMotion: false,
  volume: 60,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));


/* ------------------------------------------------------------------ helpers */

function announce(message) {
  /* One live region for screen readers.  Clearing it first forces the reader to
   * speak again even when the new message is identical to the old one. */
  const region = $("#live-region");
  region.textContent = "";
  window.setTimeout(() => { region.textContent = message; }, 60);
}

function toast(message, kind = "info") {
  const element = document.createElement("div");
  element.className = `toast ${kind}`;
  element.textContent = message;
  $("#toasts").append(element);

  if (kind === "error") {
    audio.error();
    element.setAttribute("role", "alert");
  }
  announce(message);

  window.setTimeout(() => {
    element.classList.add("leaving");
    window.setTimeout(() => element.remove(), 400);
  }, kind === "error" ? 6000 : 3200);
}

async function api(path, options = {}) {
  /* Every call comes back through here so a failure always produces a readable
   * message and an error sound, rather than a silent dead button. */
  let response;
  try {
    response = await fetch(path, options);
  } catch (networkError) {
    throw new Error("The server is not responding. Is app.py still running?");
  }

  let body = null;
  try {
    body = await response.json();
  } catch (parseError) {
    body = null;
  }

  if (!response.ok || (body && body.ok === false)) {
    throw new Error((body && body.error) || `Request failed (${response.status})`);
  }
  return body;
}

function busy(on) {
  $("#stage-busy").hidden = !on;
  $("#run-model").disabled = on;
}

function fixed(value, places = 3) {
  return Number(value).toFixed(places);
}

function percent(value) {
  return Math.round(Number(value) * 100);
}


/* ----------------------------------------------------------------- settings */

function loadSettings() {
  try {
    const saved = JSON.parse(window.localStorage.getItem("mvi-gui-settings") || "{}");
    Object.assign(settings, saved);
  } catch (error) {
    /* A corrupt entry is not worth complaining about - the defaults are fine. */
  }

  /* The head script has already resolved the skin, taking ?skin= into account,
   * so that is the authority here rather than the stored value. */
  settings.skin = document.documentElement.dataset.skin || settings.skin;
}

function applySkin(name) {
  /* The skin stylesheet is layered on top of style.css rather than replacing it,
   * so "original" means removing the extra <link> and nothing else.  The head of
   * index.html does the same thing before the first paint; this is the version
   * that runs when the choice changes while the page is open. */
  const root = document.documentElement;
  root.dataset.skin = name;

  const existing = document.getElementById("skin-css");
  if (name === "original") {
    if (existing) existing.remove();
    return;
  }

  const href = `${root.dataset.skinBase || "/static/css/"}skin-${name}.css`;
  if (existing) {
    if (existing.getAttribute("href") !== href) existing.href = href;
    return;
  }

  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.id = "skin-css";
  link.href = href;
  document.head.appendChild(link);
}


function saveSettings() {
  try {
    window.localStorage.setItem("mvi-gui-settings", JSON.stringify(settings));
  } catch (error) {
    /* Private browsing can refuse to store anything.  Not fatal. */
  }
}

function applySettings() {
  const root = document.documentElement;
  root.dataset.theme = settings.theme;
  root.dataset.textSize = settings.textSize;
  root.dataset.contrast = settings.contrast ? "high" : "normal";
  root.dataset.motion = settings.reduceMotion ? "reduced" : "full";

  audio.setEnabled(settings.sound);
  audio.setSpeaking(settings.speech);
  audio.setVolume(settings.volume / 100);

  applySkin(settings.skin);

  $("#language").value = settings.language;
  $("#skin").value = settings.skin;
  $("#palette").value = settings.palette;
  $("#text-size").value = settings.textSize;
  $("#high-contrast").checked = settings.contrast;
  $("#reduce-motion").checked = settings.reduceMotion;
  $("#volume").value = settings.volume;

  const soundButton = $("#toggle-sound");
  soundButton.setAttribute("aria-pressed", String(settings.sound));
  soundButton.querySelector('[data-icon="sound-on"]').hidden = !settings.sound;
  soundButton.querySelector('[data-icon="sound-off"]').hidden = settings.sound;

  $("#toggle-speech").setAttribute("aria-pressed", String(settings.speech));
  $("#toggle-theme").setAttribute("aria-pressed", String(settings.theme === "light"));

  i18n.apply(settings.language);
}


/* ------------------------------------------------------------------- models */

/* The list is split by task, and inside each task there is one model per
 * student, named after its author.  The three of them are on three different
 * datasets, so each card says which one it wants rather than the heading saying
 * it for a whole group. */
const MODEL_GROUPS = [
  { task: 1, label: "models.task1" },
  { task: 2, label: "models.task2" },
];

function renderModels() {
  const container = $("#model-list");
  container.innerHTML = "";

  MODEL_GROUPS.forEach(({ task, label }) => {
    const group = state.models.filter((model) => model.task === task);
    if (!group.length) return;

    const heading = document.createElement("p");
    heading.className = "group-label";
    heading.textContent = i18n.t(label);
    container.append(heading);

    group.forEach((model) => {
      const isLoaded = state.loadedModel === model.id;
      const dataset = state.datasets[model.dataset] || {};

      /* Two different ways a card can be unusable: the model file itself is
       * missing, or the scans it was trained on are not on this machine.  Both
       * are said on the card, because finding out by loading it and landing in
       * an empty picker is worse. */
      const missingScans = dataset.present === false;
      const usable = model.available !== false;

      const card = document.createElement("div");
      card.className = `model-card${isLoaded ? " loaded" : ""}`;

      const info = document.createElement("div");
      info.className = "model-info";
      info.innerHTML = `
        <span class="model-name"></span>
        <span class="model-note"></span>
        <span class="model-file"></span>`;
      info.querySelector(".model-name").textContent = model.name;

      /* The note is one line with an ellipsis to keep the cards the same height,
       * so the full text goes in the tooltip. */
      const note = info.querySelector(".model-note");
      note.textContent = model.note;
      note.title = model.note;

      const details = [model.file];
      if (model.size_mb) details.push(`${model.size_mb} MB`);
      if (model.recorded) details.push(i18n.t("models.recorded"));
      info.querySelector(".model-file").textContent = details.join(" · ");

      if (!usable || missingScans) {
        const warning = document.createElement("span");
        warning.className = "hint warn";
        warning.textContent = usable
          ? i18n.t("models.missingData", dataset.expected_at || "")
          : i18n.t("models.unavailable");
        info.append(warning);
      }

      const button = document.createElement("button");
      button.className = `button small${isLoaded ? " ghost" : ""}`;
      button.textContent = isLoaded ? i18n.t("models.ready") : i18n.t("models.load");
      button.disabled = isLoaded || !usable;
      button.addEventListener("click", () => loadModel(model.id, button));

      card.append(info, button);
      container.append(card);
    });
  });

  $("#model-count").textContent = state.models.length;
}

async function refreshModels() {
  const data = await api("/api/state");
  state.models = data.models;
  state.datasets = data.datasets || {};
  renderModels();
}

async function loadModel(modelId, button) {
  const original = button ? button.textContent : null;
  if (button) {
    button.disabled = true;
    button.textContent = i18n.t("models.loading");
  }

  try {
    const data = await api("/api/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelId }),
    });

    state.loadedModel = modelId;
    state.loadedTask = data.task;
    /* Only Salman's forest is box-driven; Afnan's and Amman's Task 1 methods
     * find the lesion themselves, so the box controls stay out of their way. */
    const entry = state.models.find((item) => item.id === modelId);
    state.needsBox = entry ? entry.needs_box !== false : data.task === 1;
    state.loadedDataset = data.dataset || "brisc";
    state.loadedInfo = data.info;

    renderModels();
    renderSpecSheet(data.info);
    updateTaskUi();
    updateSteps();

    /* A model only understands the pictures it was trained on, so loading one
     * from the other dataset swaps the scan picker over to match.  Running a
     * breast ultrasound network on a brain MRI would produce a mask and a Dice
     * score that mean nothing at all. */
    if (state.loadedDataset !== state.dataset) {
      await useDataset(state.loadedDataset);
    } else {
      /* Same dataset, different model, so the picker still has to be refetched:
       * two models can share a dataset and still be scored on different scans -
       * Adit's Task 1 is reported on 50 Figshare slices and her Task 2 on the
       * sixteen cases she exported predictions for. */
      await refreshImages();
    }

    const model = state.models.find((entry) => entry.id === modelId);
    const name = model ? model.name : modelId.split("/").pop();

    audio.modelLoaded();
    toast(i18n.t("toast.modelLoaded", name), "success");
    addLog("model", name);
  } catch (error) {
    toast(error.message, "error");
    if (button) {
      button.disabled = false;
      button.textContent = original;
    }
  }
}

/* Which spec fields to show, in this order.  Anything the back end did not
 * report is skipped rather than shown empty. */
const SPEC_ORDER = [
  "kind", "dataset", "input_size", "trees", "features", "min_samples_leaf",
  "patch_size", "roi_padding", "filters", "base_filters", "layers", "params_M",
  "sparsity_pct", "gflops", "size_mb", "measured_latency_ms", "tta",
  "load_seconds",
];

const SPEC_UNITS = {
  sparsity_pct: " %",
  size_mb: " MB",
  measured_latency_ms: " ms",
  load_seconds: " s",
  params_M: " M",
  input_size: " px",
  patch_size: " px",
};

function renderSpecSheet(info) {
  const list = $("#spec-list");
  list.innerHTML = "";

  SPEC_ORDER.forEach((key) => {
    const value = info[key];
    if (value === undefined || value === null) return;

    const term = document.createElement("dt");
    term.textContent = i18n.t(`spec.${key}`);

    const definition = document.createElement("dd");
    if (Array.isArray(value)) {
      definition.textContent = value.join(" · ");
    } else if (typeof value === "boolean") {
      definition.textContent = i18n.t(value ? "spec.yes" : "spec.no");
    } else if (key === "dataset") {
      definition.textContent = i18n.t(`dataset.${value}`);
    } else {
      definition.textContent = `${value}${SPEC_UNITS[key] || ""}`;
    }

    list.append(term, definition);
  });

  $("#spec-sheet").hidden = false;
}

/* Task 1 needs a box and Task 2 does not, so the box controls only appear when
 * they are actually usable.  Showing a dead control is worse than hiding it. */
function updateTaskUi() {
  const wantsBox = state.needsBox === true;
  $("#box-controls").hidden = !wantsBox;
  $("#box-canvas").classList.toggle("drawable", wantsBox);
  updateRunHint();
}

function updateRunHint() {
  const hint = $("#run-hint");
  if (!state.loadedModel) hint.textContent = i18n.t("run.hint");
  else if (state.imageIndex < 0) hint.textContent = i18n.t("run.hint.image");
  else if (state.needsBox && !state.box) hint.textContent = i18n.t("run.hint.box");
  else hint.textContent = i18n.t("run.hint.ready");
}


/* ------------------------------------------------------------------- images */

/* Rebuild the dropdown from state.images.  Kept separate from fetching so that
 * switching language can relabel the list - the tumour type in each label is
 * translated - without touching the selection or the current result. */
function renderImageOptions() {
  const select = $("#image-select");
  select.innerHTML = "";

  state.images.forEach((image, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    /* BRISC and BUSI classes have translations; the FIVES conditions arrive as
     * plain text from Amman's CSV, so an untranslated key falls back to it. */
    const key = `tumour.${image.tumour_type}`;
    const translated = i18n.t(key);
    const label = translated === key ? image.tumour_type : translated;
    option.textContent = `${image.name}  ·  ${label}`;
    select.append(option);
  });

  $("#image-count").textContent = state.images.length;
  if (state.imageIndex >= 0) select.value = String(state.imageIndex);
}

/* Switch the scan picker to a dataset: hide the class filters that belong to
 * the other one, relabel the panel, and fetch the new listing. */
async function useDataset(dataset) {
  state.dataset = dataset;

  const filterSelect = $("#image-filter");
  $$("#image-filter option[data-dataset]").forEach((option) => {
    const mine = option.dataset.dataset === dataset;
    option.hidden = !mine;
    option.disabled = !mine;
  });

  /* If the filter that was selected belonged to the other dataset it is now
   * hidden, and a select stuck on a hidden option looks broken. */
  if (filterSelect.selectedOptions[0] && filterSelect.selectedOptions[0].disabled) {
    filterSelect.value = "evaluated";
  }

  renderDatasetNote();
  state.imageIndex = -1;
  await refreshImages();
}

function renderDatasetNote() {
  const info = state.datasets[state.dataset] || {};
  $("#dataset-note").textContent = info.present === false
    ? i18n.t("images.missing", info.expected_at || "")
    : i18n.t(`dataset.${state.dataset}`);
}

async function refreshImages() {
  const filter = $("#image-filter").value;

  try {
    /* The loaded model goes with the request: Amman's two recorded sets cover
     * the same 200 scans under the same names, and only the model says which
     * set's input pictures the picker should be offering. */
    const data = await api(
      `/api/images?dataset=${encodeURIComponent(state.dataset)}` +
      `&filter=${encodeURIComponent(filter)}` +
      (state.loadedModel ? `&model=${encodeURIComponent(state.loadedModel)}` : ""));
    state.images = data.images;
  } catch (error) {
    state.images = [];
    state.imageIndex = -1;
    $("#image-count").textContent = "0";
    $("#image-select").innerHTML = "";
    toast(error.message, "error");
    return;
  }

  renderImageOptions();

  if (state.images.length) {
    const keep = state.imageIndex >= 0 && state.imageIndex < state.images.length;
    selectImage(keep ? state.imageIndex : 0);
  } else {
    state.imageIndex = -1;
  }
}

function selectImage(index) {
  if (!state.images.length) return;

  state.imageIndex = Math.min(Math.max(index, 0), state.images.length - 1);
  $("#image-select").value = String(state.imageIndex);

  /* A new scan invalidates the old box, the old prediction and the old
   * comparison table - none of them belong to this image. */
  clearBox();
  state.result = null;
  clearComparisons();
  showResults(null);

  $("#viewer").removeAttribute("src");
  $("#stage-empty").hidden = false;

  updateSteps();
  updateRunHint();
}

function currentImageName() {
  const image = state.images[state.imageIndex];
  return image ? image.name : null;
}

async function uploadScan(file, maskFile) {
  const form = new FormData();
  form.append("image", file);
  if (maskFile) form.append("mask", maskFile);

  try {
    const data = await api("/api/upload", { method: "POST", body: form });

    $("#image-filter").value = "all";
    state.imageIndex = -1;
    await refreshImages();

    const index = state.images.findIndex((image) => image.name === data.name);
    if (index >= 0) selectImage(index);

    audio.predictionDone();
    toast(i18n.t("toast.uploaded", file.name), "success");
    addLog("upload", file.name);
  } catch (error) {
    toast(error.message, "error");
  }
}


/* --------------------------------------------------------------------- box */

/* The canvas is 512x512 internally but displayed at whatever size fits, so
 * pointer coordinates have to be scaled.  Doing it from the bounding box rather
 * than a stored ratio means it keeps working when the window is resized. */
function toImageCoords(event) {
  const canvas = $("#box-canvas");
  const rect = canvas.getBoundingClientRect();

  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;

  return {
    x: Math.min(Math.max(Math.round(x), 0), canvas.width - 1),
    y: Math.min(Math.max(Math.round(y), 0), canvas.height - 1),
  };
}

function drawBoxOverlay() {
  const canvas = $("#box-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.box) return;

  const [top, left, bottom, right] = state.box;

  context.setLineDash([7, 5]);
  context.lineWidth = 2;
  context.strokeStyle = "#f0b429";
  context.strokeRect(left, top, right - left, bottom - top);

  /* Corner ticks: they make the handles obvious without adding drag targets,
   * and they survive being printed in greyscale for the report. */
  context.setLineDash([]);
  context.lineWidth = 3;
  const size = 14;
  [[left, top, 1, 1], [right, top, -1, 1],
   [left, bottom, 1, -1], [right, bottom, -1, -1]].forEach(([x, y, dx, dy]) => {
    context.beginPath();
    context.moveTo(x + dx * size, y);
    context.lineTo(x, y);
    context.lineTo(x, y + dy * size);
    context.stroke();
  });
}

function setBox(box, simulated = false) {
  state.box = box;
  state.boxSimulated = Boolean(box) && simulated;
  drawBoxOverlay();

  const readout = $("#box-readout");
  if (!box) {
    readout.textContent = i18n.t("run.box.none");
  } else {
    const height = box[2] - box[0];
    const width = box[3] - box[1];
    readout.textContent = i18n.t(state.boxSimulated ? "run.box.simulated"
                                                   : "run.box.drawn",
                                 width, height);
  }

  updateSteps();
  updateRunHint();
}

function clearBox() {
  setBox(null);
}

function setUpBoxDrawing() {
  const canvas = $("#box-canvas");
  let start = null;
  let previous = null;          /* the box being replaced, in case the drag fails */

  canvas.addEventListener("pointerdown", (event) => {
    if (state.loadedTask !== 1 || state.imageIndex < 0) return;

    canvas.setPointerCapture(event.pointerId);
    start = toImageCoords(event);
    previous = { box: state.box, simulated: state.boxSimulated };
    event.preventDefault();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!start) return;

    const now = toImageCoords(event);
    state.box = [Math.min(start.y, now.y), Math.min(start.x, now.x),
                 Math.max(start.y, now.y), Math.max(start.x, now.x)];
    drawBoxOverlay();
  });

  const finish = (event) => {
    if (!start) return;

    const now = toImageCoords(event);
    const box = [Math.min(start.y, now.y), Math.min(start.x, now.x),
                 Math.max(start.y, now.y), Math.max(start.x, now.x)];
    start = null;

    /* A stray click on the scan is a box of nearly zero size.  The previous box
     * is put back rather than thrown away - losing a carefully drawn box to a
     * misplaced click is far more annoying than the warning is helpful. */
    if (box[2] - box[0] < 8 || box[3] - box[1] < 8) {
      setBox(previous.box, previous.simulated);
      toast(i18n.t("toast.boxTooSmall"), "warn");
      audio.prompt();
      return;
    }

    setBox(box, false);
    audio.click();
  };

  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", () => {
    start = null;
    if (previous) setBox(previous.box, previous.simulated);
  });
}

async function simulateBox() {
  const name = currentImageName();
  if (!name) {
    toast(i18n.t("toast.noImage"), "warn");
    audio.prompt();
    return;
  }

  try {
    const data = await api("/api/box", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: name }),
    });
    setBox(data.box, true);
    audio.click();
  } catch (error) {
    toast(error.message, "error");
  }
}


/* ------------------------------------------------------------------ running */

async function runModel() {
  if (!state.loadedModel) {
    toast(i18n.t("toast.noModel"), "warn");
    audio.prompt();
    return;
  }

  const name = currentImageName();
  if (!name) {
    toast(i18n.t("toast.noImage"), "warn");
    audio.prompt();
    return;
  }

  if (state.needsBox && !state.box) {
    toast(i18n.t("toast.noBox"), "warn");
    audio.prompt();
    return;
  }

  const button = $("#run-model");
  const label = button.querySelector("span");
  const originalLabel = label.textContent;
  label.textContent = i18n.t("run.running");
  busy(true);

  try {
    const data = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.loadedModel,
        image: name,
        box: state.needsBox ? state.box : null,
      }),
    });

    state.result = data;

    /* The server clamps the box to the image, so it can come back slightly
     * different from what was sent - show what was actually used. */
    if (data.box) setBox(data.box, state.boxSimulated);

    await refreshView();
    showResults(data);
    addComparison(data);
    addLog("predict", name, data);
    updateSteps();

    audio.predictionDone();
    if (data.metrics) {
      toast(i18n.t("toast.predicted", fixed(data.metrics.dice)), "success");
      speakResults(data);
    } else {
      toast(i18n.t("toast.predictedNoTruth"), "info");
    }
  } catch (error) {
    toast(error.message, "error");
  } finally {
    label.textContent = originalLabel;
    busy(false);
  }
}


/* -------------------------------------------------------------------- views */

function viewUrl() {
  const parameters = new URLSearchParams({
    view: state.view,
    palette: settings.palette,
    opacity: String(Number($("#opacity").value) / 100),
    thickness: $("#thickness").value,
    show_box: $("#show-box").checked ? "1" : "0",
    /* The browser caches by URL and every render is a fresh PNG at the same
     * address, so a changing parameter is needed to defeat it. */
    t: String(Date.now()),
  });
  return `/api/view?${parameters.toString()}`;
}

async function refreshView() {
  if (!state.result) return;

  const image = $("#viewer");
  await new Promise((resolve) => {
    image.onload = resolve;
    image.onerror = resolve;
    image.src = viewUrl();
  });

  $("#stage-empty").hidden = true;
  drawBoxOverlay();
}

function setView(view) {
  state.view = view;

  $$('.view-tabs [role="tab"]').forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });

  /* A recorded result is a finished mask, not a probability map, so the
   * confidence view has nothing to colour - say that rather than leaving a
   * plain scan on screen with no explanation. */
  let caption = i18n.t(`views.${view}.caption`);
  if (view === "heatmap" && state.result && state.result.recorded) {
    caption = i18n.t("views.heatmap.recorded");
  }
  $("#view-caption").textContent = caption;
  $("#legend").dataset.view = view;

  refreshView();
  audio.click();
  announce(i18n.t(`views.${view}`));
}


/* ------------------------------------------------------------------ metrics */

const METRIC_ROWS = ["iou", "hd95", "accuracy", "sensitivity", "precision"];

function showResults(data) {
  const hasResult = Boolean(data);
  $("#results").hidden = !hasResult;
  $("#no-results").hidden = hasResult;

  if (!hasResult) {
    $("#result-model").textContent = "";
    return;
  }

  const model = state.models.find((entry) => entry.id === data.model);
  $("#result-model").textContent = model ? model.name : "";

  const metrics = data.metrics;

  if (!metrics) {
    /* No ground truth: show the honest reason rather than zeros that look real. */
    $("#dice-value").textContent = "—";
    $("#dice-bar").style.width = "0%";
    $("#dice-verdict").textContent = i18n.t("metrics.noTruth");
    $("#metric-grid").innerHTML = "";
    $("#interpretation").textContent = "";
    return;
  }

  $("#dice-value").textContent = fixed(metrics.dice);
  $("#dice-bar").style.width = `${percent(metrics.dice)}%`;
  $("#dice-bar").dataset.band = diceBand(metrics.dice, data.empty_prediction);
  $("#dice-verdict").textContent = verdictText(metrics.dice, data.empty_prediction);

  const grid = $("#metric-grid");
  grid.innerHTML = "";

  METRIC_ROWS.forEach((key) => {
    /* Not every model reports every metric - the recorded results carry
     * whatever their author measured, and Adit's per-image file has no
     * sensitivity or precision column.  A row that would read NaN is left out
     * rather than printed, because a missing measurement and a measurement of
     * zero are not the same thing. */
    const value = metrics[key];
    if (value === undefined || value === null || Number.isNaN(Number(value))) return;

    const term = document.createElement("dt");
    term.textContent = i18n.t(`metrics.${key}`);

    const definition = document.createElement("dd");
    definition.textContent = key === "hd95"
      ? `${fixed(metrics.hd95, 2)} px`
      : fixed(value);

    grid.append(term, definition);
  });

  /* Recorded results were not timed here, so there is no latency to report. */
  if (data.latency_ms !== null && data.latency_ms !== undefined) {
    const latency = document.createElement("dt");
    latency.textContent = i18n.t("metrics.latency");
    const latencyValue = document.createElement("dd");
    latencyValue.textContent = `${data.latency_ms} ms`;
    grid.append(latency, latencyValue);
  }

  renderInterpretation(data);
}

function diceBand(dice, empty) {
  if (empty || dice === 0) return "failed";
  if (dice >= 0.85) return "excellent";
  if (dice >= 0.70) return "good";
  if (dice >= 0.50) return "fair";
  return "poor";
}

function verdictText(dice, empty) {
  return i18n.t(`verdict.${diceBand(dice, empty)}`);
}

/* Plain-language reading of the numbers.  The point is that the metrics are for
 * people who have not met them before: the marker, and any clinician in the
 * demo audience. */
function renderInterpretation(data) {
  const container = $("#interpretation");
  container.innerHTML = "";

  const metrics = data.metrics;
  const has = (key) => metrics[key] !== undefined && metrics[key] !== null
    && !Number.isNaN(Number(metrics[key]));

  const lines = [];
  if (has("hd95")) lines.push(i18n.t("interp.hd95", fixed(metrics.hd95, 1)));
  /* Same reason as the metric rows: a recorded result only carries what its
   * author measured, and a sentence about NaN% of the tumour is worse than no
   * sentence at all. */
  if (has("sensitivity") && has("precision")) {
    lines.push(i18n.t("interp.sens", percent(metrics.sensitivity),
                      percent(metrics.precision)));
  }

  if (data.truth_pixels !== null && data.truth_pixels !== undefined) {
    lines.push(i18n.t("metrics.pixels", data.predicted_pixels, data.truth_pixels));
  }

  lines.forEach((line) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    container.append(paragraph);
  });

  /* The silent failure is the honest limitation of the Task 2 model, so the
   * interface names it when it happens instead of just showing a zero. */
  if (data.empty_prediction || metrics.dice === 0) {
    const warning = document.createElement("p");
    warning.className = "warning-note";
    warning.textContent = i18n.t("interp.silent");
    container.append(warning);
  }
}

function speakResults(data) {
  if (!data.metrics) return;

  const model = state.models.find((entry) => entry.id === data.model);
  const spoken = [
    model ? model.name : "",
    `${i18n.t("metrics.dice")} ${fixed(data.metrics.dice, 2)}.`,
    `${i18n.t("metrics.hd95")} ${fixed(data.metrics.hd95, 1)}.`,
    verdictText(data.metrics.dice, data.empty_prediction),
  ].join(". ");

  audio.speak(spoken, i18n.speechTag());
}


/* ----------------------------------------------------------- compare / log */

function addComparison(data) {
  if (!data.metrics) return;

  const model = state.models.find((entry) => entry.id === data.model);
  const name = model ? model.name : data.model.split("/").pop();

  /* One row per model per scan: re-running the same model replaces its row
   * rather than filling the table with duplicates. */
  state.comparisons = state.comparisons.filter((row) => row.name !== name);
  state.comparisons.push({
    name,
    dice: data.metrics.dice,
    hd95: data.metrics.hd95,
    iou: data.metrics.iou,
    ms: data.latency_ms,
  });

  renderComparisons();
}

function renderComparisons() {
  const body = $("#compare-body");
  body.innerHTML = "";

  $("#compare-panel").hidden = state.comparisons.length < 2;
  if (state.comparisons.length < 2) return;

  const best = Math.max(...state.comparisons.map((row) => row.dice));

  state.comparisons.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.dice === best) tr.className = "best";

    [row.name, fixed(row.dice), fixed(row.hd95, 1), fixed(row.iou), row.ms]
      .forEach((value, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        cell.textContent = value;
        tr.append(cell);
      });

    body.append(tr);
  });
}

function clearComparisons() {
  state.comparisons = [];
  renderComparisons();
}

/* The session log is there so a pair or a group working at one screen has a
 * record of what was tried, and can save it and attach it to the report. */
function addLog(kind, subject, data = null) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    kind,
    subject,
    dice: data && data.metrics ? fixed(data.metrics.dice) : null,
    hd95: data && data.metrics ? fixed(data.metrics.hd95, 1) : null,
    model: data ? data.model : null,
  };
  state.logEntries.push(entry);
  renderLog();
}

function renderLog() {
  const list = $("#log");
  list.innerHTML = "";

  if (!state.logEntries.length) {
    const empty = document.createElement("li");
    empty.className = "log-empty";
    empty.textContent = i18n.t("log.empty");
    list.append(empty);
    return;
  }

  /* Newest first: with a long session the interesting entry is the last one. */
  state.logEntries.slice().reverse().forEach((entry) => {
    const item = document.createElement("li");
    item.className = `log-item ${entry.kind}`;

    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = entry.time;

    const text = document.createElement("span");
    text.className = "log-text";
    text.textContent = entry.dice
      ? `${entry.subject} — Dice ${entry.dice}, HD95 ${entry.hd95}`
      : entry.subject;

    item.append(time, text);
    list.append(item);
  });
}

function downloadLog() {
  if (!state.logEntries.length) {
    toast(i18n.t("log.empty"), "warn");
    return;
  }

  const header = "time,action,subject,model,dice,hd95\n";
  const rows = state.logEntries.map((entry) => [
    entry.time, entry.kind, entry.subject,
    entry.model ? entry.model.split("/").pop() : "",
    entry.dice || "", entry.hd95 || "",
  ].map((field) => `"${String(field).replace(/"/g, '""')}"`).join(","));

  const blob = new Blob([header + rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "gui_session_log.csv";
  link.click();
  URL.revokeObjectURL(url);

  toast(i18n.t("toast.logSaved"), "success");
}


/* ------------------------------------------------------------------ credits */

async function showCredits() {
  const container = $("#credits-content");

  try {
    const data = await api("/api/credits");
    const credits = data.credits;
    container.innerHTML = "";

    const project = document.createElement("div");
    project.className = "credits-project";
    project.innerHTML = "<h3></h3><p></p>";
    project.querySelector("h3").textContent = credits.project.title;
    project.querySelector("p").textContent =
      `${credits.project.module} · ${credits.project.dataset}`;
    container.append(project);

    credits.members.forEach((member) => {
      const card = document.createElement("article");
      card.className = "credit-card";

      const heading = document.createElement("header");
      const name = document.createElement("h4");
      name.textContent = member.name;
      const meta = document.createElement("span");
      meta.className = "credit-meta";
      meta.textContent = [member.student_id, member.chapter]
        .filter(Boolean).join(" · ");
      heading.append(name, meta);

      const roles = document.createElement("ul");
      roles.className = "role-chips";
      member.roles.forEach((role) => {
        const chip = document.createElement("li");
        chip.textContent = role;
        roles.append(chip);
      });

      const contribution = document.createElement("p");
      contribution.className = "credit-text";
      contribution.textContent = member.contribution;

      card.append(heading, roles, contribution);

      if (member.artefacts && member.artefacts.length) {
        const files = document.createElement("p");
        files.className = "credit-files";
        files.textContent = `${i18n.t("credits.artefacts")}: ${member.artefacts.join(", ")}`;
        card.append(files);
      }

      container.append(card);
    });

    if (credits.supervision) {
      const supervisor = document.createElement("p");
      supervisor.className = "credit-supervisor";
      supervisor.textContent =
        `${i18n.t("credits.supervision")}: ${credits.supervision.name}`;
      container.append(supervisor);
    }

    if (credits.project.statement_url) {
      $("#credits-link").href = credits.project.statement_url;
    }
  } catch (error) {
    container.textContent = error.message;
  }

  $("#credits-modal").showModal();
  audio.click();
}


/* -------------------------------------------------------------- glossary */

function renderGlossary() {
  const list = $("#metric-glossary");
  list.innerHTML = "";

  ["dice", "iou", "accuracy", "hd95", "sensitivity", "precision"].forEach((key) => {
    const term = document.createElement("dt");
    term.textContent = i18n.t(`metrics.${key}`);
    const definition = document.createElement("dd");
    definition.textContent = i18n.t(`glossary.${key}`);
    list.append(term, definition);
  });
}


/* ---------------------------------------------------------------- progress */

/* The four-step checklist on the left.  Someone working through this on their
 * own can see what is done and what is next without being walked through it. */
function updateSteps() {
  const done = {
    model: Boolean(state.loadedModel),
    image: state.imageIndex >= 0,
    box: state.loadedTask === 2 ? Boolean(state.loadedModel) : Boolean(state.box),
    run: Boolean(state.result),
  };

  let nextFound = false;
  $$("#steps li").forEach((item) => {
    const step = item.dataset.step;
    const complete = done[step];

    item.classList.toggle("done", complete);
    item.classList.toggle("next", !complete && !nextFound);
    if (!complete) nextFound = true;
  });

  /* For a Task 2 model the box step does not apply, so it is dimmed rather than
   * left looking like an outstanding task. */
  const boxStep = $('#steps li[data-step="box"]');
  boxStep.classList.toggle("not-applicable", state.loadedTask === 2);
}


/* ----------------------------------------------------------------- language */

function changeLanguage(code) {
  settings.language = code;
  saveSettings();
  i18n.apply(code);

  /* Anything rendered from JavaScript has to be rebuilt, because i18n.apply only
   * rewrites the elements that carry a data-i18n attribute in the markup.
   *
   * Note what is *not* called here: refreshImages, which reselects the scan and
   * would throw away the loaded model's result.  Changing language must not cost
   * the user the prediction they are looking at. */
  renderModels();
  renderGlossary();
  renderLog();
  renderComparisons();
  renderImageOptions();
  renderDatasetNote();
  if (state.loadedInfo) renderSpecSheet(state.loadedInfo);
  setView(state.view);
  updateRunHint();
  setBox(state.box, state.boxSimulated);
  if (state.result) showResults(state.result);

  toast(i18n.t("toast.language", i18n.t("meta.name")), "info");
}


/* ----------------------------------------------------------------- keyboard */

function setUpKeyboard() {
  document.addEventListener("keydown", (event) => {
    /* Never hijack a key the user is typing into a field, and leave dialogs to
     * handle their own keys. */
    const tag = event.target.tagName;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
    if (document.querySelector("dialog[open]")) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;

    const key = event.key.toLowerCase();

    if (key >= "1" && key <= "6") {
      const tab = $$('.view-tabs [role="tab"]')[Number(key) - 1];
      if (tab) setView(tab.dataset.view);
      return;
    }

    switch (key) {
      case "r": runModel(); break;
      case "b": if (state.needsBox) simulateBox(); break;
      case "n": selectImage(state.imageIndex + 1); break;
      case "p": selectImage(state.imageIndex - 1); break;
      case "m":
        settings.sound = !settings.sound;
        applySettings();
        saveSettings();
        break;
      case "?": $("#help-modal").showModal(); break;
      default: break;
    }
  });
}


/* ---------------------------------------------------------------- start-up */

function setUpEvents() {
  $("#language").addEventListener("change", (event) => changeLanguage(event.target.value));

  $("#toggle-sound").addEventListener("click", () => {
    settings.sound = !settings.sound;
    applySettings();
    saveSettings();
  });

  $("#toggle-speech").addEventListener("click", () => {
    settings.speech = !settings.speech;
    applySettings();
    saveSettings();
    if (settings.speech && state.result) speakResults(state.result);
  });

  $("#toggle-theme").addEventListener("click", () => {
    settings.theme = settings.theme === "dark" ? "light" : "dark";
    applySettings();
    saveSettings();
    audio.click();
  });

  $("#open-settings").addEventListener("click", () => {
    $("#settings-modal").showModal();
    audio.click();
  });
  $("#open-help").addEventListener("click", () => {
    renderGlossary();
    $("#help-modal").showModal();
    audio.click();
  });
  $("#open-credits").addEventListener("click", showCredits);

  /* Switching design only swaps the stylesheet - no state is touched, so a
   * loaded model and a finished result survive the change. */
  $("#skin").addEventListener("change", (event) => {
    settings.skin = event.target.value;
    applySkin(settings.skin);
    saveSettings();
    audio.click();
  });

  /* Settings that change how things are drawn need a re-render, not just a save. */
  $("#palette").addEventListener("change", (event) => {
    settings.palette = event.target.value;
    saveSettings();
    document.documentElement.dataset.palette = settings.palette;
    refreshView();
  });

  $("#text-size").addEventListener("change", (event) => {
    settings.textSize = event.target.value;
    applySettings();
    saveSettings();
  });

  $("#high-contrast").addEventListener("change", (event) => {
    settings.contrast = event.target.checked;
    applySettings();
    saveSettings();
  });

  $("#reduce-motion").addEventListener("change", (event) => {
    settings.reduceMotion = event.target.checked;
    applySettings();
    saveSettings();
  });

  $("#volume").addEventListener("input", (event) => {
    settings.volume = Number(event.target.value);
    audio.setVolume(settings.volume / 100);
  });
  $("#volume").addEventListener("change", () => {
    saveSettings();
    audio.click();
  });

  $("#load-custom").addEventListener("click", () => {
    const path = $("#custom-path").value.trim();
    if (!path) {
      toast(i18n.t("models.custom.hint"), "warn");
      audio.prompt();
      return;
    }
    loadModel(path, null);
  });

  $("#image-filter").addEventListener("change", () => {
    state.imageIndex = -1;
    refreshImages();
  });
  $("#image-select").addEventListener("change", (event) => {
    selectImage(Number(event.target.value));
  });
  $("#prev-image").addEventListener("click", () => selectImage(state.imageIndex - 1));
  $("#next-image").addEventListener("click", () => selectImage(state.imageIndex + 1));
  $("#random-image").addEventListener("click", () => {
    if (state.images.length) {
      selectImage(Math.floor(Math.random() * state.images.length));
      audio.click();
    }
  });

  /* Upload: click, keyboard, and drag-and-drop all reach the same place. */
  const zone = $("#upload-zone");
  zone.addEventListener("click", () => $("#upload-image").click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#upload-image").click();
    }
  });
  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, () => zone.classList.remove("dragging"));
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) handleChosenScan(file);
  });

  $("#upload-image").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) handleChosenScan(file);
    event.target.value = "";
  });

  $("#upload-mask").addEventListener("change", (event) => {
    const mask = event.target.files[0];
    if (mask && pendingScan) {
      uploadScan(pendingScan, mask);
      pendingScan = null;
    }
    event.target.value = "";
  });

  $("#simulate-box").addEventListener("click", simulateBox);
  $("#clear-box").addEventListener("click", () => {
    clearBox();
    audio.click();
  });
  $("#run-model").addEventListener("click", runModel);

  $$('.view-tabs [role="tab"]').forEach((tab) => {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  });

  $("#opacity").addEventListener("input", (event) => {
    $("#opacity-value").textContent = `${event.target.value}%`;
  });
  $("#opacity").addEventListener("change", refreshView);

  $("#thickness").addEventListener("input", (event) => {
    $("#thickness-value").textContent = event.target.value;
  });
  $("#thickness").addEventListener("change", refreshView);

  $("#show-box").addEventListener("change", refreshView);

  $("#export-view").addEventListener("click", () => {
    if (!state.result) {
      toast(i18n.t("toast.noResult"), "warn");
      audio.prompt();
      return;
    }
    window.location.href = `/api/export?palette=${settings.palette}`;
    toast(i18n.t("toast.exported"), "success");
  });

  $("#speak-metrics").addEventListener("click", () => {
    if (!state.result) return;
    if (!settings.speech) {
      settings.speech = true;
      applySettings();
      saveSettings();
    }
    speakResults(state.result);
  });

  $("#clear-compare").addEventListener("click", () => {
    clearComparisons();
    toast(i18n.t("toast.cleared"), "info");
  });
  $("#clear-log").addEventListener("click", () => {
    state.logEntries = [];
    renderLog();
    toast(i18n.t("toast.cleared"), "info");
  });
  $("#download-log").addEventListener("click", downloadLog);
}

/* Held between choosing a scan and choosing its mask, when the user has said
 * they have one. */
let pendingScan = null;

function handleChosenScan(file) {
  if ($("#upload-with-mask").checked) {
    pendingScan = file;
    toast(i18n.t("images.upload.mask"), "info");
    audio.prompt();
    $("#upload-mask").click();
  } else {
    uploadScan(file, null);
  }
}

async function start() {
  loadSettings();
  applySettings();
  audio.load();

  document.documentElement.dataset.palette = settings.palette;

  setUpEvents();
  setUpBoxDrawing();
  setUpKeyboard();
  renderGlossary();
  renderLog();
  setView(state.view);
  updateTaskUi();          /* hides the box controls until a Task 1 model is in */
  updateSteps();

  try {
    await refreshModels();
    await useDataset(state.dataset);
  } catch (error) {
    toast(error.message, "error");
  }
}

document.addEventListener("DOMContentLoaded", start);
