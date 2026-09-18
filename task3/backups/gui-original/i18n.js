/*
 * Translations for the interface.
 *
 * The brief asks for English plus one other language.  There are three here:
 *
 *   en  English         - the language of the module
 *   ms  Bahasa Melayu   - the national language of the country the group studies
 *                         in, and the one a Malaysian clinical user is most
 *                         likely to want
 *   ar  Arabic          - included on purpose because it is written
 *                         right-to-left, which forces the layout itself to
 *                         mirror rather than just swapping the words out
 *
 * Arabic is the interesting case.  Switching to it sets dir="rtl" on the
 * document, and because the stylesheet is written with logical properties
 * (margin-inline-start rather than margin-left) the whole interface mirrors:
 * the panels swap sides, the sliders fill from the right, the "next scan" arrow
 * turns round.  Numbers stay left-to-right, which is correct - Arabic writes
 * numerals in that direction even inside right-to-left text.
 */

const TRANSLATIONS = {

  /* ------------------------------------------------------------- English */
  en: {
    "meta.name": "English",
    "meta.dir": "ltr",
    "meta.speech": "en-GB",

    "app.title": "Brain Tumour Segmentation",
    "app.subtitle": "Task 1 and Task 2 models, one interface",

    "header.language": "Language",
    "header.sound": "Sound effects on or off",
    "header.speech": "Read the results aloud",
    "header.theme": "Light or dark theme",
    "header.settings": "Display settings",
    "header.settings.short": "Display",
    "header.help": "How to use this",
    "header.help.short": "Help",
    "header.credits": "Group contributions",
    "header.credits.short": "Credits",

    "a11y.setup": "Set-up",
    "a11y.viewer": "Image viewer",
    "a11y.results": "Results",
    "a11y.viewerImage": "The MRI scan with the segmentation drawn on it",
    "a11y.boxCanvas": "Drag across the scan to mark the lesion",

    "steps.model": "Load a model",
    "steps.image": "Choose a scan",
    "steps.box": "Mark the lesion",
    "steps.run": "Run the model",

    "models.title": "Models",
    "models.load": "Load",
    "models.loading": "Loading…",
    "models.loaded": "Loaded model",
    "models.ready": "Ready",
    "models.custom": "Load another model file",
    "models.custom.hint": "A path inside the project folder. Use .joblib for a Task 1 forest or .h5 for a Task 2 network.",
    "models.custom.path": "Model file path",
    "models.task1": "Task 1 · semi-automated",
    "models.task2": "Task 2 · deep learning",
    "models.busi": "Task 2 · breast ultrasound (BUSI)",
    "models.missingData": "The BUSI scans are not here yet. Put Dataset_BUSI_with_GT at {0}.",

    "spec.trees": "Trees",
    "spec.features": "Features per pixel",
    "spec.min_samples_leaf": "Minimum leaf size",
    "spec.patch_size": "Working patch",
    "spec.roi_padding": "Box padding",
    "spec.size_mb": "Model size",
    "spec.kind": "Architecture",
    "spec.input_size": "Input size",
    "spec.params_M": "Tunable parameters",
    "spec.trainable_params": "Trainable weights",
    "spec.layers": "Layers",
    "spec.filters": "Channel widths",
    "spec.base_filters": "Base filters",
    "spec.gflops": "GFLOPs",
    "spec.measured_latency_ms": "Measured latency",
    "spec.load_seconds": "Loaded in",
    "spec.dataset": "Trained on",
    "spec.tta": "Test time augmentation",
    "spec.sparsity_pct": "Weights pruned to zero",
    "spec.yes": "Yes",
    "spec.no": "No",

    "dataset.brisc": "BRISC 2025 · brain MRI",
    "dataset.busi": "BUSI · breast ultrasound",

    "images.title": "Scan",
    "images.filter": "Show",
    "images.filter.all": "All test scans",
    "images.filter.evaluated": "The evaluated scans only",
    "images.missing": "These scans are missing. Put the dataset at {0}.",
    "images.pick": "Scan",
    "images.prev": "Previous scan",
    "images.prev.short": "Prev",
    "images.next": "Next scan",
    "images.next.short": "Next",
    "images.random": "Random",
    "images.upload": "Drop your own scan here, or click to choose",
    "images.upload.hint": "JPG, PNG, BMP or TIFF. Without a mask file the scan can still be segmented, but there is nothing to score it against.",
    "images.upload.mask": "I also have the ground truth mask",
    "images.uploaded": "Uploaded",

    "tumour.glioma": "Glioma",
    "tumour.meningioma": "Meningioma",
    "tumour.pituitary": "Pituitary",
    "tumour.no_tumour": "No tumour",
    "tumour.uploaded": "Uploaded",
    "tumour.unknown": "Unknown",
    "tumour.benign": "Benign",
    "tumour.malignant": "Malignant",
    "tumour.normal": "No lesion",

    "run.title": "Run",
    "run.button": "Segment this scan",
    "run.running": "Segmenting…",
    "run.hint": "Load a model to begin.",
    "run.hint.image": "Choose a scan.",
    "run.hint.box": "Drag a box round the lesion, or press Simulate box.",
    "run.hint.ready": "Ready to run.",
    "run.box.hint": "Task 1 is semi-automated: it needs a rough box round the lesion. Drag one on the image, or let the app simulate the clinician's box.",
    "run.box.simulate": "Simulate box",
    "run.box.clear": "Clear",
    "run.box.none": "No box yet",
    "run.box.drawn": "Box: {0} × {1} px",
    "run.box.simulated": "Simulated box: {0} × {1} px",

    "views.title": "Views",
    "views.boundaries": "Boundaries",
    "views.boundaries.caption": "Ground truth and predicted outlines on the same scan. Where the two lines separate is the error the Dice score measures.",
    "views.truth": "Ground truth",
    "views.truth.caption": "The radiologist's mask on its own — what the model is being asked to reproduce.",
    "views.prediction": "Prediction",
    "views.prediction.caption": "The model's mask on its own, drawn the same way as the ground truth so the two views can be compared directly.",
    "views.difference": "Difference",
    "views.difference.caption": "Agreement, tumour that was missed, and healthy tissue wrongly included. The same Dice score can come from either kind of mistake.",
    "views.heatmap": "Confidence",
    "views.heatmap.caption": "The probability map before it is cut at 0.5. A confident wrong answer and an unsure one look identical once thresholded.",
    "views.scan": "Scan only",
    "views.scan.caption": "The MRI slice with nothing drawn on it.",

    "stage.empty": "Load a model and pick a scan to begin",
    "stage.busy": "Working…",

    "legend.truth": "Ground truth",
    "legend.prediction": "Model prediction",
    "legend.missed": "Missed",
    "legend.extra": "Over-segmented",
    "legend.overlap": "Agreement",

    "controls.opacity": "Fill",
    "controls.thickness": "Line",
    "controls.showBox": "Show box",
    "controls.export": "Export PNG",

    "metrics.title": "Metrics",
    "metrics.none": "No result yet. Run a model to see the scores.",
    "metrics.dice": "Dice score",
    "metrics.dice.short": "Dice",
    "metrics.iou": "Intersection over Union",
    "metrics.iou.short": "IoU",
    "metrics.accuracy": "Pixel accuracy",
    "metrics.accuracy.short": "Accuracy",
    "metrics.hd95": "HD95 boundary error",
    "metrics.hd95.short": "HD95",
    "metrics.sensitivity": "Sensitivity",
    "metrics.sensitivity.short": "Sens.",
    "metrics.precision": "Precision",
    "metrics.precision.short": "Prec.",
    "metrics.latency": "Inference time",
    "metrics.latency.short": "ms",
    "metrics.threshold": "0.85 — the distinction threshold",
    "metrics.speak": "Read the results aloud",
    "metrics.noTruth": "This scan has no ground truth mask, so it cannot be scored. The segmentation is still shown.",
    "metrics.pixels": "{0} px predicted, {1} px in the ground truth",

    "verdict.excellent": "Excellent overlap — above the 0.85 threshold.",
    "verdict.good": "Good overlap, just under 0.85.",
    "verdict.fair": "Usable, but the boundary needs correcting.",
    "verdict.poor": "Poor overlap — this case would need to be redone by hand.",
    "verdict.failed": "The model returned nothing at all for this scan.",

    "interp.hd95": "The predicted boundary sits about {0} pixels from the true one.",
    "interp.sens": "It found {0}% of the tumour and {1}% of what it marked was really tumour.",
    "interp.silent": "Note: an empty or wildly wrong mask with no warning is the failure mode this system cannot yet detect on its own. In use, a case like this would be sent back for a manual box.",

    "compare.title": "Same scan, each model",
    "compare.model": "Model",
    "compare.ms": "ms",
    "compare.clear": "Clear",

    "log.title": "Session log",
    "log.download": "Save",
    "log.clear": "Clear",
    "log.empty": "Nothing yet",

    "help.title": "How to use this",
    "help.walkthrough": "Walkthrough",
    "help.step1": "Pick a model on the left and press Load. A three-note chime means it is in memory, and its size, parameter count and FLOPs appear underneath.",
    "help.step2": "Choose a scan. The default list is the 200 scans Task 2 was scored on, so the numbers you see match the report.",
    "help.step3": "For a Task 1 model, drag a rough box round the lesion, or press Simulate box to use the same loose box the evaluation used. Task 2 models need no box.",
    "help.step4": "Press Segment. The outlines appear on the scan and the scores appear on the right.",
    "help.metrics": "Reading the metrics",
    "help.keys": "Keyboard",
    "help.key.run": "Run the model",
    "help.key.box": "Simulate the box",
    "help.key.nav": "Next / previous scan",
    "help.key.views": "Switch view",
    "help.key.mute": "Mute or unmute",
    "help.key.help": "This window",
    "help.key.close": "Close a window",
    "help.trouble": "If something goes wrong",
    "help.trouble.q1": "The Task 1 model will not load",
    "help.trouble.a1": "The forest runs in a separate helper process using mvi-env. Check that the mvi-env folder is still next to task3, then press Load again — that restarts the helper.",
    "help.trouble.q2": "No scans are listed",
    "help.trouble.a2": "The BRISC 2025 folder has to sit beside task3 as brisc2025/segmentation_task. Uploading your own scan works without it.",
    "help.trouble.q3": "Metrics show a dash",
    "help.trouble.a3": "That scan has no ground truth mask, so there is nothing to score against. The segmentation is still shown.",
    "help.trouble.q4": "There is no sound",
    "help.trouble.a4": "Browsers block audio until the page has been clicked once. Click anywhere, and check the speaker button in the header is on.",

    "glossary.dice": "How much the two masks overlap, from 0 to 1. The marking criteria are written against this one; 0.85 and above is the distinction band.",
    "glossary.iou": "The same idea as Dice but stricter, so it always reads a little lower.",
    "glossary.accuracy": "The share of pixels labelled correctly. It looks impressive on brain MRI whatever the model does, because the tumour is a tiny part of the image — which is exactly why Dice is quoted instead.",
    "glossary.hd95": "How far the predicted outline sits from the true one, in pixels, ignoring the worst 5% of points. Lower is better. Overlap can look fine while this is bad.",
    "glossary.sensitivity": "The share of the tumour the model found. Missing tumour is the dangerous error.",
    "glossary.precision": "The share of what the model marked that really was tumour.",

    "settings.title": "Display settings",
    "settings.intro": "These preferences are remembered in this browser.",
    "settings.palette": "Colour scheme for the overlays",
    "settings.palette.clinical": "Clinical (green / red)",
    "settings.palette.accessible": "Colour-blind safe (Okabe-Ito)",
    "settings.palette.hint": "Green and red is the pair around 1 in 12 men cannot separate reliably.",
    "settings.textSize": "Text size",
    "settings.textSize.normal": "Normal",
    "settings.textSize.large": "Large",
    "settings.textSize.xlarge": "Very large",
    "settings.contrast": "Higher contrast",
    "settings.motion": "Reduce animation",
    "settings.volume": "Sound volume",

    "credits.title": "Contributions",
    "credits.intro": "Contributions are stated using CRediT, the Contributor Roles Taxonomy.",
    "credits.standard": "About CRediT",
    "credits.roles": "Roles",
    "credits.contribution": "What they did",
    "credits.artefacts": "Files",
    "credits.supervision": "Supervision",
    "credits.chapter": "Chapter",

    "modal.close": "Close",
    "modal.done": "Done",

    "toast.modelLoaded": "{0} is loaded",
    "toast.predicted": "Done — Dice {0}",
    "toast.predictedNoTruth": "Segmented. No ground truth, so no score.",
    "toast.uploaded": "{0} uploaded",
    "toast.exported": "Image saved",
    "toast.logSaved": "Log saved",
    "toast.language": "Interface language: {0}",
    "toast.noModel": "Load a model first",
    "toast.noImage": "Choose a scan first",
    "toast.noBox": "Task 1 needs a box — drag one, or press Simulate box",
    "toast.boxTooSmall": "That box is too small — drag a larger area",
    "toast.noResult": "Run a model first",
    "toast.cleared": "Cleared"
  },

  /* -------------------------------------------------------- Bahasa Melayu */
  ms: {
    "meta.name": "Bahasa Melayu",
    "meta.dir": "ltr",
    "meta.speech": "ms-MY",

    "app.title": "Segmentasi Tumor Otak",
    "app.subtitle": "Model Tugasan 1 dan Tugasan 2, satu antara muka",

    "header.language": "Bahasa",
    "header.sound": "Hidupkan atau matikan bunyi",
    "header.speech": "Bacakan keputusan",
    "header.theme": "Tema cerah atau gelap",
    "header.settings": "Tetapan paparan",
    "header.settings.short": "Paparan",
    "header.help": "Cara menggunakan",
    "header.help.short": "Bantuan",
    "header.credits": "Sumbangan kumpulan",
    "header.credits.short": "Kredit",

    "a11y.setup": "Persediaan",
    "a11y.viewer": "Pemapar imej",
    "a11y.results": "Keputusan",
    "a11y.viewerImage": "Imej MRI dengan segmentasi dilukis di atasnya",
    "a11y.boxCanvas": "Seret pada imej untuk menanda lesi",

    "steps.model": "Muatkan model",
    "steps.image": "Pilih imej",
    "steps.box": "Tanda lesi",
    "steps.run": "Jalankan model",

    "models.title": "Model",
    "models.load": "Muat",
    "models.loading": "Memuatkan…",
    "models.loaded": "Model dimuatkan",
    "models.ready": "Sedia",
    "models.custom": "Muatkan fail model lain",
    "models.custom.hint": "Laluan di dalam folder projek. Guna .joblib untuk hutan Tugasan 1 atau .h5 untuk rangkaian Tugasan 2.",
    "models.custom.path": "Laluan fail model",
    "models.task1": "Tugasan 1 · separa automatik",
    "models.task2": "Tugasan 2 · pembelajaran mendalam",
    "models.busi": "Tugasan 2 · ultrabunyi payudara (BUSI)",
    "models.missingData": "Imej BUSI belum ada di sini. Letakkan Dataset_BUSI_with_GT di {0}.",

    "spec.trees": "Bilangan pokok",
    "spec.features": "Ciri per piksel",
    "spec.min_samples_leaf": "Saiz daun minimum",
    "spec.patch_size": "Saiz tampalan kerja",
    "spec.roi_padding": "Pelapik kotak",
    "spec.size_mb": "Saiz model",
    "spec.kind": "Seni bina",
    "spec.input_size": "Saiz input",
    "spec.params_M": "Parameter boleh laras",
    "spec.trainable_params": "Pemberat boleh latih",
    "spec.layers": "Lapisan",
    "spec.filters": "Lebar saluran",
    "spec.base_filters": "Penapis asas",
    "spec.gflops": "GFLOP",
    "spec.measured_latency_ms": "Kependaman diukur",
    "spec.load_seconds": "Dimuatkan dalam",
    "spec.dataset": "Dilatih dengan",
    "spec.tta": "Penambahan data masa ujian",
    "spec.sparsity_pct": "Pemberat dipangkas kepada sifar",
    "spec.yes": "Ya",
    "spec.no": "Tidak",

    "dataset.brisc": "BRISC 2025 · MRI otak",
    "dataset.busi": "BUSI · ultrabunyi payudara",

    "images.title": "Imej",
    "images.filter": "Tunjukkan",
    "images.filter.all": "Semua imej ujian",
    "images.filter.evaluated": "Imej yang dinilai sahaja",
    "images.missing": "Imej ini tiada. Letakkan set data di {0}.",
    "images.pick": "Imej",
    "images.prev": "Imej sebelumnya",
    "images.prev.short": "Sebelum",
    "images.next": "Imej seterusnya",
    "images.next.short": "Seterus",
    "images.random": "Rawak",
    "images.upload": "Lepaskan imej anda di sini, atau klik untuk memilih",
    "images.upload.hint": "JPG, PNG, BMP atau TIFF. Tanpa fail topeng, imej masih boleh disegmen tetapi tiada rujukan untuk memberi skor.",
    "images.upload.mask": "Saya juga mempunyai topeng rujukan",
    "images.uploaded": "Dimuat naik",

    "tumour.glioma": "Glioma",
    "tumour.meningioma": "Meningioma",
    "tumour.pituitary": "Pituitari",
    "tumour.no_tumour": "Tiada tumor",
    "tumour.uploaded": "Dimuat naik",
    "tumour.unknown": "Tidak diketahui",
    "tumour.benign": "Jinak",
    "tumour.malignant": "Ganas",
    "tumour.normal": "Tiada lesi",

    "run.title": "Jalankan",
    "run.button": "Segmenkan imej ini",
    "run.running": "Menyegmen…",
    "run.hint": "Muatkan model untuk bermula.",
    "run.hint.image": "Pilih satu imej.",
    "run.hint.box": "Seret kotak di sekeliling lesi, atau tekan Simulasi kotak.",
    "run.hint.ready": "Sedia untuk dijalankan.",
    "run.box.hint": "Tugasan 1 adalah separa automatik: ia memerlukan kotak kasar di sekeliling lesi. Seret satu pada imej, atau biarkan aplikasi mensimulasikan kotak klinisian.",
    "run.box.simulate": "Simulasi kotak",
    "run.box.clear": "Kosongkan",
    "run.box.none": "Belum ada kotak",
    "run.box.drawn": "Kotak: {0} × {1} px",
    "run.box.simulated": "Kotak simulasi: {0} × {1} px",

    "views.title": "Paparan",
    "views.boundaries": "Sempadan",
    "views.boundaries.caption": "Garis rujukan dan garis ramalan pada imej yang sama. Di mana kedua-dua garis terpisah, itulah ralat yang diukur oleh skor Dice.",
    "views.truth": "Topeng rujukan",
    "views.truth.caption": "Topeng pakar radiologi sahaja — inilah yang model diminta menghasilkan semula.",
    "views.prediction": "Ramalan",
    "views.prediction.caption": "Topeng model sahaja, dilukis sama seperti topeng rujukan supaya kedua-dua paparan boleh dibandingkan secara langsung.",
    "views.difference": "Perbezaan",
    "views.difference.caption": "Persetujuan, tumor yang tertinggal, dan tisu sihat yang disalah masukkan. Skor Dice yang sama boleh datang daripada mana-mana satu kesilapan ini.",
    "views.heatmap": "Keyakinan",
    "views.heatmap.caption": "Peta kebarangkalian sebelum dipotong pada 0.5. Jawapan salah yang yakin dan jawapan tidak pasti kelihatan serupa selepas pengambangan.",
    "views.scan": "Imej sahaja",
    "views.scan.caption": "Kepingan MRI tanpa apa-apa lukisan.",

    "stage.empty": "Muatkan model dan pilih imej untuk bermula",
    "stage.busy": "Sedang bekerja…",

    "legend.truth": "Topeng rujukan",
    "legend.prediction": "Ramalan model",
    "legend.missed": "Tertinggal",
    "legend.extra": "Lebih segmen",
    "legend.overlap": "Persetujuan",

    "controls.opacity": "Isian",
    "controls.thickness": "Garisan",
    "controls.showBox": "Tunjuk kotak",
    "controls.export": "Eksport PNG",

    "metrics.title": "Metrik",
    "metrics.none": "Belum ada keputusan. Jalankan model untuk melihat skor.",
    "metrics.dice": "Skor Dice",
    "metrics.dice.short": "Dice",
    "metrics.iou": "Persilangan atas Kesatuan",
    "metrics.iou.short": "IoU",
    "metrics.accuracy": "Ketepatan piksel",
    "metrics.accuracy.short": "Ketepatan",
    "metrics.hd95": "Ralat sempadan HD95",
    "metrics.hd95.short": "HD95",
    "metrics.sensitivity": "Kepekaan",
    "metrics.sensitivity.short": "Peka",
    "metrics.precision": "Kepersisan",
    "metrics.precision.short": "Persis",
    "metrics.latency": "Masa inferens",
    "metrics.latency.short": "ms",
    "metrics.threshold": "0.85 — ambang kepujian tertinggi",
    "metrics.speak": "Bacakan keputusan",
    "metrics.noTruth": "Imej ini tiada topeng rujukan, jadi ia tidak boleh diberi skor. Segmentasi masih dipaparkan.",
    "metrics.pixels": "{0} px diramal, {1} px dalam topeng rujukan",

    "verdict.excellent": "Pertindihan cemerlang — melebihi ambang 0.85.",
    "verdict.good": "Pertindihan baik, sedikit di bawah 0.85.",
    "verdict.fair": "Boleh digunakan, tetapi sempadan perlu dibetulkan.",
    "verdict.poor": "Pertindihan lemah — kes ini perlu diulang secara manual.",
    "verdict.failed": "Model tidak mengembalikan apa-apa untuk imej ini.",

    "interp.hd95": "Sempadan ramalan berada kira-kira {0} piksel daripada sempadan sebenar.",
    "interp.sens": "Ia menemui {0}% daripada tumor dan {1}% daripada yang ditandanya benar-benar tumor.",
    "interp.silent": "Nota: topeng yang kosong atau amat salah tanpa apa-apa amaran adalah mod kegagalan yang sistem ini belum dapat kesan sendiri. Dalam penggunaan sebenar, kes seperti ini akan dihantar semula untuk kotak manual.",

    "compare.title": "Imej sama, setiap model",
    "compare.model": "Model",
    "compare.ms": "ms",
    "compare.clear": "Kosongkan",

    "log.title": "Log sesi",
    "log.download": "Simpan",
    "log.clear": "Kosongkan",
    "log.empty": "Belum ada apa-apa",

    "help.title": "Cara menggunakan",
    "help.walkthrough": "Panduan langkah demi langkah",
    "help.step1": "Pilih satu model di sebelah kiri dan tekan Muat. Bunyi tiga nada bermakna ia sudah berada dalam ingatan, dan saiz, bilangan parameter serta FLOP akan muncul di bawahnya.",
    "help.step2": "Pilih satu imej. Senarai lalai ialah 200 imej yang digunakan untuk menilai Tugasan 2, jadi nombor yang anda lihat sepadan dengan laporan.",
    "help.step3": "Untuk model Tugasan 1, seret kotak kasar di sekeliling lesi, atau tekan Simulasi kotak untuk menggunakan kotak longgar yang sama seperti dalam penilaian. Model Tugasan 2 tidak memerlukan kotak.",
    "help.step4": "Tekan Segmenkan. Garis sempadan muncul pada imej dan skor muncul di sebelah kanan.",
    "help.metrics": "Membaca metrik",
    "help.keys": "Papan kekunci",
    "help.key.run": "Jalankan model",
    "help.key.box": "Simulasi kotak",
    "help.key.nav": "Imej seterusnya / sebelumnya",
    "help.key.views": "Tukar paparan",
    "help.key.mute": "Bisukan atau nyalakan bunyi",
    "help.key.help": "Tetingkap ini",
    "help.key.close": "Tutup tetingkap",
    "help.trouble": "Jika ada masalah",
    "help.trouble.q1": "Model Tugasan 1 tidak mahu dimuatkan",
    "help.trouble.a1": "Hutan rawak berjalan dalam proses pembantu berasingan menggunakan mvi-env. Pastikan folder mvi-env masih bersebelahan task3, kemudian tekan Muat sekali lagi — itu akan memulakan semula pembantu.",
    "help.trouble.q2": "Tiada imej disenaraikan",
    "help.trouble.a2": "Folder BRISC 2025 perlu berada bersebelahan task3 sebagai brisc2025/segmentation_task. Memuat naik imej sendiri masih boleh tanpanya.",
    "help.trouble.q3": "Metrik menunjukkan sengkang",
    "help.trouble.a3": "Imej itu tiada topeng rujukan, jadi tiada apa untuk diberi skor. Segmentasi masih dipaparkan.",
    "help.trouble.q4": "Tiada bunyi",
    "help.trouble.a4": "Pelayar menghalang audio sehingga halaman diklik sekali. Klik di mana-mana, dan pastikan butang pembesar suara di bar atas dihidupkan.",

    "glossary.dice": "Sejauh mana kedua-dua topeng bertindih, dari 0 hingga 1. Kriteria pemarkahan ditulis berdasarkan metrik ini; 0.85 dan ke atas ialah jalur kepujian tertinggi.",
    "glossary.iou": "Idea yang sama seperti Dice tetapi lebih ketat, jadi nilainya selalu sedikit lebih rendah.",
    "glossary.accuracy": "Bahagian piksel yang dilabel dengan betul. Nilainya kelihatan tinggi pada MRI otak walau apa pun yang model lakukan, kerana tumor hanya sebahagian kecil daripada imej — itulah sebabnya Dice yang dilaporkan.",
    "glossary.hd95": "Sejauh mana garis ramalan daripada garis sebenar, dalam piksel, dengan mengabaikan 5% titik terburuk. Lebih rendah lebih baik. Pertindihan boleh kelihatan baik walaupun nilai ini buruk.",
    "glossary.sensitivity": "Bahagian tumor yang ditemui oleh model. Tumor yang tertinggal adalah kesilapan yang berbahaya.",
    "glossary.precision": "Bahagian daripada apa yang ditanda model yang benar-benar tumor.",

    "settings.title": "Tetapan paparan",
    "settings.intro": "Pilihan ini diingati dalam pelayar ini.",
    "settings.palette": "Skema warna untuk lapisan tindanan",
    "settings.palette.clinical": "Klinikal (hijau / merah)",
    "settings.palette.accessible": "Selamat buta warna (Okabe-Ito)",
    "settings.palette.hint": "Hijau dan merah ialah pasangan yang kira-kira 1 daripada 12 lelaki tidak dapat bezakan dengan yakin.",
    "settings.textSize": "Saiz teks",
    "settings.textSize.normal": "Biasa",
    "settings.textSize.large": "Besar",
    "settings.textSize.xlarge": "Sangat besar",
    "settings.contrast": "Kontras lebih tinggi",
    "settings.motion": "Kurangkan animasi",
    "settings.volume": "Kekuatan bunyi",

    "credits.title": "Sumbangan",
    "credits.intro": "Sumbangan dinyatakan menggunakan CRediT, Taksonomi Peranan Penyumbang.",
    "credits.standard": "Mengenai CRediT",
    "credits.roles": "Peranan",
    "credits.contribution": "Apa yang dilakukan",
    "credits.artefacts": "Fail",
    "credits.supervision": "Penyeliaan",
    "credits.chapter": "Bab",

    "modal.close": "Tutup",
    "modal.done": "Selesai",

    "toast.modelLoaded": "{0} telah dimuatkan",
    "toast.predicted": "Selesai — Dice {0}",
    "toast.predictedNoTruth": "Telah disegmen. Tiada topeng rujukan, jadi tiada skor.",
    "toast.uploaded": "{0} dimuat naik",
    "toast.exported": "Imej disimpan",
    "toast.logSaved": "Log disimpan",
    "toast.language": "Bahasa antara muka: {0}",
    "toast.noModel": "Muatkan model dahulu",
    "toast.noImage": "Pilih imej dahulu",
    "toast.noBox": "Tugasan 1 memerlukan kotak — seret satu, atau tekan Simulasi kotak",
    "toast.boxTooSmall": "Kotak itu terlalu kecil — seret kawasan yang lebih besar",
    "toast.noResult": "Jalankan model dahulu",
    "toast.cleared": "Telah dikosongkan"
  },

  /* --------------------------------------------------------------- Arabic */
  ar: {
    "meta.name": "العربية",
    "meta.dir": "rtl",
    "meta.speech": "ar-SA",

    "app.title": "تجزئة أورام المخ",
    "app.subtitle": "نماذج المهمة الأولى والثانية في واجهة واحدة",

    "header.language": "اللغة",
    "header.sound": "تشغيل أو إيقاف المؤثرات الصوتية",
    "header.speech": "قراءة النتائج بصوت مسموع",
    "header.theme": "مظهر فاتح أو غامق",
    "header.settings": "إعدادات العرض",
    "header.settings.short": "العرض",
    "header.help": "طريقة الاستخدام",
    "header.help.short": "مساعدة",
    "header.credits": "إسهامات الفريق",
    "header.credits.short": "الإسهامات",

    "a11y.setup": "الإعداد",
    "a11y.viewer": "عارض الصور",
    "a11y.results": "النتائج",
    "a11y.viewerImage": "صورة الرنين المغناطيسي مع التجزئة مرسومة عليها",
    "a11y.boxCanvas": "اسحب على الصورة لتحديد الورم",

    "steps.model": "حمّل نموذجًا",
    "steps.image": "اختر صورة",
    "steps.box": "حدّد الورم",
    "steps.run": "شغّل النموذج",

    "models.title": "النماذج",
    "models.load": "تحميل",
    "models.loading": "جارٍ التحميل…",
    "models.loaded": "النموذج المحمّل",
    "models.ready": "جاهز",
    "models.custom": "تحميل ملف نموذج آخر",
    "models.custom.hint": "مسار داخل مجلد المشروع. استخدم ‎.joblib‎ لغابة المهمة الأولى أو ‎.h5‎ لشبكة المهمة الثانية.",
    "models.custom.path": "مسار ملف النموذج",
    "models.task1": "المهمة الأولى · نصف آلية",
    "models.task2": "المهمة الثانية · تعلّم عميق",
    "models.busi": "المهمة الثانية · الموجات فوق الصوتية للثدي (BUSI)",
    "models.missingData": "صور BUSI غير موجودة بعد. ضع Dataset_BUSI_with_GT في {0}.",

    "spec.trees": "عدد الأشجار",
    "spec.features": "السمات لكل بكسل",
    "spec.min_samples_leaf": "أصغر حجم للورقة",
    "spec.patch_size": "حجم الرقعة",
    "spec.roi_padding": "هامش الإطار",
    "spec.size_mb": "حجم النموذج",
    "spec.kind": "المعمارية",
    "spec.input_size": "حجم الدخل",
    "spec.params_M": "المعاملات القابلة للضبط",
    "spec.trainable_params": "الأوزان القابلة للتدريب",
    "spec.layers": "الطبقات",
    "spec.filters": "عروض القنوات",
    "spec.base_filters": "المرشحات الأساسية",
    "spec.gflops": "جيجا فلوب",
    "spec.measured_latency_ms": "زمن الاستجابة المقيس",
    "spec.load_seconds": "زمن التحميل",
    "spec.dataset": "دُرِّب على",
    "spec.tta": "زيادة البيانات وقت الاختبار",
    "spec.sparsity_pct": "أوزان مُقلَّمة إلى صفر",
    "spec.yes": "نعم",
    "spec.no": "لا",

    "dataset.brisc": "BRISC 2025 · تصوير الدماغ بالرنين",
    "dataset.busi": "BUSI · موجات فوق صوتية للثدي",

    "images.title": "الصورة",
    "images.filter": "اعرض",
    "images.filter.all": "كل صور الاختبار",
    "images.filter.evaluated": "الصور المُقيَّمة فقط",
    "images.missing": "هذه الصور غير موجودة. ضع مجموعة البيانات في {0}.",
    "images.pick": "الصورة",
    "images.prev": "الصورة السابقة",
    "images.prev.short": "السابق",
    "images.next": "الصورة التالية",
    "images.next.short": "التالي",
    "images.random": "عشوائي",
    "images.upload": "أفلت صورتك هنا، أو انقر للاختيار",
    "images.upload.hint": "‏JPG أو PNG أو BMP أو TIFF. بدون ملف القناع يمكن تجزئة الصورة، لكن لا يوجد مرجع لحساب الدرجات.",
    "images.upload.mask": "لديّ أيضًا القناع المرجعي",
    "images.uploaded": "تم الرفع",

    "tumour.glioma": "ورم غراوي",
    "tumour.meningioma": "ورم سحائي",
    "tumour.pituitary": "ورم نخامي",
    "tumour.no_tumour": "لا يوجد ورم",
    "tumour.uploaded": "مرفوعة",
    "tumour.unknown": "غير معروف",
    "tumour.benign": "حميد",
    "tumour.malignant": "خبيث",
    "tumour.normal": "لا توجد آفة",

    "run.title": "التشغيل",
    "run.button": "جزّئ هذه الصورة",
    "run.running": "جارٍ التجزئة…",
    "run.hint": "حمّل نموذجًا للبدء.",
    "run.hint.image": "اختر صورة.",
    "run.hint.box": "اسحب إطارًا حول الورم، أو اضغط محاكاة الإطار.",
    "run.hint.ready": "جاهز للتشغيل.",
    "run.box.hint": "المهمة الأولى نصف آلية: تحتاج إلى إطار تقريبي حول الورم. اسحب إطارًا على الصورة، أو اترك التطبيق يحاكي إطار الطبيب.",
    "run.box.simulate": "محاكاة الإطار",
    "run.box.clear": "مسح",
    "run.box.none": "لا يوجد إطار بعد",
    "run.box.drawn": "الإطار: {0} × {1} بكسل",
    "run.box.simulated": "إطار محاكى: {0} × {1} بكسل",

    "views.title": "طرق العرض",
    "views.boundaries": "الحدود",
    "views.boundaries.caption": "حدود القناع المرجعي وحدود التنبؤ على الصورة نفسها. حيث يتباعد الخطان يكون الخطأ الذي يقيسه معامل دايس.",
    "views.truth": "القناع المرجعي",
    "views.truth.caption": "قناع أخصائي الأشعة وحده — وهو ما يُطلب من النموذج إعادة إنتاجه.",
    "views.prediction": "التنبؤ",
    "views.prediction.caption": "قناع النموذج وحده، مرسوم بالطريقة نفسها كالقناع المرجعي حتى تُقارن الصورتان مباشرة.",
    "views.difference": "الفرق",
    "views.difference.caption": "التوافق، والورم الذي فُقد، والنسيج السليم الذي أُدرج خطأً. المعامل نفسه قد ينتج عن أيٍّ من هذين الخطأين.",
    "views.heatmap": "الثقة",
    "views.heatmap.caption": "خريطة الاحتمالات قبل قطعها عند ٠٫٥. الجواب الخاطئ الواثق والجواب المتردد يتشابهان تمامًا بعد العَتَبة.",
    "views.scan": "الصورة فقط",
    "views.scan.caption": "مقطع الرنين المغناطيسي دون أي رسم عليه.",

    "stage.empty": "حمّل نموذجًا واختر صورة للبدء",
    "stage.busy": "جارٍ العمل…",

    "legend.truth": "القناع المرجعي",
    "legend.prediction": "تنبؤ النموذج",
    "legend.missed": "مفقود",
    "legend.extra": "تجزئة زائدة",
    "legend.overlap": "توافق",

    "controls.opacity": "التعبئة",
    "controls.thickness": "الخط",
    "controls.showBox": "إظهار الإطار",
    "controls.export": "تصدير PNG",

    "metrics.title": "المقاييس",
    "metrics.none": "لا توجد نتيجة بعد. شغّل نموذجًا لعرض الدرجات.",
    "metrics.dice": "معامل دايس",
    "metrics.dice.short": "دايس",
    "metrics.iou": "التقاطع على الاتحاد",
    "metrics.iou.short": "IoU",
    "metrics.accuracy": "دقة البكسل",
    "metrics.accuracy.short": "الدقة",
    "metrics.hd95": "خطأ الحدود HD95",
    "metrics.hd95.short": "HD95",
    "metrics.sensitivity": "الحساسية",
    "metrics.sensitivity.short": "الحساسية",
    "metrics.precision": "الضبط",
    "metrics.precision.short": "الضبط",
    "metrics.latency": "زمن الاستنتاج",
    "metrics.latency.short": "مللي ثانية",
    "metrics.threshold": "٠٫٨٥ — عتبة التقدير الممتاز",
    "metrics.speak": "اقرأ النتائج بصوت مسموع",
    "metrics.noTruth": "هذه الصورة بلا قناع مرجعي، لذلك لا يمكن حساب الدرجات. التجزئة معروضة على أي حال.",
    "metrics.pixels": "‏{0} بكسل متنبأ بها، و{1} بكسل في القناع المرجعي",

    "verdict.excellent": "تطابق ممتاز — أعلى من عتبة ٠٫٨٥.",
    "verdict.good": "تطابق جيد، أقل بقليل من ٠٫٨٥.",
    "verdict.fair": "قابل للاستخدام، لكن الحدود تحتاج إلى تصحيح.",
    "verdict.poor": "تطابق ضعيف — هذه الحالة تحتاج إلى إعادة يدوية.",
    "verdict.failed": "لم يُرجع النموذج أي شيء لهذه الصورة.",

    "interp.hd95": "تبعد الحدود المتنبأ بها نحو {0} بكسل عن الحدود الحقيقية.",
    "interp.sens": "عثر على {0}% من الورم، و{1}% من الذي حدّده كان ورمًا فعلًا.",
    "interp.silent": "ملاحظة: القناع الفارغ أو الخاطئ تمامًا دون أي تحذير هو نمط الفشل الذي لا يستطيع هذا النظام كشفه بنفسه بعد. عمليًا، تُعاد حالة كهذه لتحديد الإطار يدويًا.",

    "compare.title": "الصورة نفسها، كل نموذج",
    "compare.model": "النموذج",
    "compare.ms": "مللي ثانية",
    "compare.clear": "مسح",

    "log.title": "سجل الجلسة",
    "log.download": "حفظ",
    "log.clear": "مسح",
    "log.empty": "لا شيء بعد",

    "help.title": "طريقة الاستخدام",
    "help.walkthrough": "خطوة بخطوة",
    "help.step1": "اختر نموذجًا على الجانب واضغط تحميل. سماع ثلاث نغمات صاعدة يعني أنه في الذاكرة، وسيظهر حجمه وعدد معاملاته وعدد عملياته الحسابية تحته.",
    "help.step2": "اختر صورة. القائمة الافتراضية هي الصور المئتان التي قُيّمت عليها المهمة الثانية، فتطابق الأرقام ما في التقرير.",
    "help.step3": "لنموذج المهمة الأولى، اسحب إطارًا تقريبيًا حول الورم، أو اضغط محاكاة الإطار لاستخدام الإطار الفضفاض نفسه المستخدم في التقييم. نماذج المهمة الثانية لا تحتاج إطارًا.",
    "help.step4": "اضغط جزّئ. تظهر الحدود على الصورة وتظهر الدرجات على الجانب.",
    "help.metrics": "قراءة المقاييس",
    "help.keys": "لوحة المفاتيح",
    "help.key.run": "تشغيل النموذج",
    "help.key.box": "محاكاة الإطار",
    "help.key.nav": "الصورة التالية / السابقة",
    "help.key.views": "تبديل العرض",
    "help.key.mute": "كتم الصوت أو تشغيله",
    "help.key.help": "هذه النافذة",
    "help.key.close": "إغلاق النافذة",
    "help.trouble": "إذا حدث خطأ",
    "help.trouble.q1": "نموذج المهمة الأولى لا يُحمّل",
    "help.trouble.a1": "تعمل الغابة العشوائية في عملية مساعدة منفصلة تستخدم mvi-env. تأكد من أن مجلد mvi-env لا يزال بجانب task3، ثم اضغط تحميل مرة أخرى — هذا يعيد تشغيل المساعد.",
    "help.trouble.q2": "لا توجد صور في القائمة",
    "help.trouble.a2": "يجب أن يكون مجلد BRISC 2025 بجانب task3 باسم brisc2025/segmentation_task. رفع صورتك الخاصة يعمل بدونه.",
    "help.trouble.q3": "المقاييس تظهر شرطة",
    "help.trouble.a3": "تلك الصورة بلا قناع مرجعي، فلا يوجد ما تُقاس عليه. التجزئة معروضة على أي حال.",
    "help.trouble.q4": "لا يوجد صوت",
    "help.trouble.a4": "تمنع المتصفحات الصوت حتى يُنقر على الصفحة مرة واحدة. انقر في أي مكان، وتأكد أن زر مكبر الصوت في الأعلى مُشغَّل.",

    "glossary.dice": "مقدار تطابق القناعين، من ٠ إلى ١. معايير التقييم مكتوبة على أساس هذا المقياس؛ ٠٫٨٥ وما فوق هو نطاق التقدير الممتاز.",
    "glossary.iou": "الفكرة نفسها كمعامل دايس لكنها أكثر تشددًا، فتظهر قيمتها أقل دائمًا بقليل.",
    "glossary.accuracy": "نسبة البكسلات المصنفة تصنيفًا صحيحًا. تبدو مرتفعة على صور المخ مهما فعل النموذج، لأن الورم جزء ضئيل من الصورة — ولهذا يُذكر معامل دايس بدلًا منها.",
    "glossary.hd95": "بُعد الحدود المتنبأ بها عن الحدود الحقيقية بالبكسل، مع تجاهل أسوأ ٥٪ من النقاط. الأقل أفضل. قد يبدو التطابق جيدًا وهذا المقياس سيئًا.",
    "glossary.sensitivity": "نسبة الورم التي عثر عليها النموذج. تفويت الورم هو الخطأ الخطير.",
    "glossary.precision": "نسبة ما حدّده النموذج والذي كان ورمًا بالفعل.",

    "settings.title": "إعدادات العرض",
    "settings.intro": "تُحفظ هذه التفضيلات في هذا المتصفح.",
    "settings.palette": "نظام ألوان الطبقات",
    "settings.palette.clinical": "إكليني (أخضر / أحمر)",
    "settings.palette.accessible": "آمن لعمى الألوان (أوكابي-إيتو)",
    "settings.palette.hint": "الأخضر والأحمر هما الزوج الذي لا يستطيع نحو رجل واحد من كل اثني عشر التمييز بينهما بثقة.",
    "settings.textSize": "حجم النص",
    "settings.textSize.normal": "عادي",
    "settings.textSize.large": "كبير",
    "settings.textSize.xlarge": "كبير جدًا",
    "settings.contrast": "تباين أعلى",
    "settings.motion": "تقليل الحركة",
    "settings.volume": "مستوى الصوت",

    "credits.title": "الإسهامات",
    "credits.intro": "تُذكر الإسهامات وفق تصنيف CRediT لأدوار المساهمين.",
    "credits.standard": "عن CRediT",
    "credits.roles": "الأدوار",
    "credits.contribution": "ما قام به",
    "credits.artefacts": "الملفات",
    "credits.supervision": "الإشراف",
    "credits.chapter": "الفصل",

    "modal.close": "إغلاق",
    "modal.done": "تم",

    "toast.modelLoaded": "تم تحميل {0}",
    "toast.predicted": "تم — دايس {0}",
    "toast.predictedNoTruth": "تمت التجزئة. لا يوجد قناع مرجعي، فلا توجد درجة.",
    "toast.uploaded": "تم رفع {0}",
    "toast.exported": "تم حفظ الصورة",
    "toast.logSaved": "تم حفظ السجل",
    "toast.language": "لغة الواجهة: {0}",
    "toast.noModel": "حمّل نموذجًا أولًا",
    "toast.noImage": "اختر صورة أولًا",
    "toast.noBox": "المهمة الأولى تحتاج إطارًا — اسحب واحدًا، أو اضغط محاكاة الإطار",
    "toast.boxTooSmall": "هذا الإطار صغير جدًا — اسحب منطقة أوسع",
    "toast.noResult": "شغّل نموذجًا أولًا",
    "toast.cleared": "تم المسح"
  }
};


/*
 * The i18n helper.
 *
 * Kept tiny on purpose: a lookup, a placeholder substitution, and one function
 * that walks the document and rewrites every tagged element.
 */
const i18n = {
  language: "en",

  /* Look a key up, falling back to English and then to the key itself, so a
   * missing translation shows readable text rather than an empty gap. */
  t(key, ...values) {
    const table = TRANSLATIONS[this.language] || TRANSLATIONS.en;
    let text = table[key];
    if (text === undefined) text = TRANSLATIONS.en[key];
    if (text === undefined) return key;

    return text.replace(/\{(\d+)\}/g, (match, index) => {
      const value = values[Number(index)];
      return value === undefined ? match : value;
    });
  },

  languages() {
    return Object.keys(TRANSLATIONS).map((code) => ({
      code,
      name: TRANSLATIONS[code]["meta.name"],
    }));
  },

  direction() {
    return this.t("meta.dir");
  },

  speechTag() {
    return this.t("meta.speech");
  },

  /* Switch language and rewrite the interface. */
  apply(language) {
    if (TRANSLATIONS[language]) this.language = language;

    const root = document.documentElement;
    root.lang = this.language;
    root.dir = this.direction();

    document.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = this.t(element.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-title]").forEach((element) => {
      element.title = this.t(element.dataset.i18nTitle);
    });
    document.querySelectorAll("[data-i18n-aria]").forEach((element) => {
      element.setAttribute("aria-label", this.t(element.dataset.i18nAria));
    });
    document.querySelectorAll("[data-i18n-alt]").forEach((element) => {
      element.alt = this.t(element.dataset.i18nAlt);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
      element.placeholder = this.t(element.dataset.i18nPlaceholder);
    });

    document.title = this.t("app.title") + " - Machine Vision Group Assignment";
  },
};
