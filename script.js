let model;
const classes = ["Anthracnose", "Bacterial Leaf Spot", "Healthy", "Leaf Spot"];

const els = {
  status: document.getElementById("status"),
  img: document.getElementById("uploadedImage"),
  video: document.getElementById("videoElement"),
  file: document.getElementById("imageUpload"),
  startCam: document.getElementById("startWebcam"),
  stopCam: document.getElementById("stopWebcam"),
  predictCam: document.getElementById("predictWebcam"),
  canvas: document.getElementById("canvas"),
  predSummary: document.getElementById("predSummary"),
  predTable: document.getElementById("predTable"),
  predBody: document.querySelector("#predTable tbody"),
  shapeBadge: document.getElementById("shapeBadge"),
};

const showStatus = (msg, type = "") => {
  els.status.textContent = msg;
  els.status.className = `small ${type || "muted"}`;
};

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function loadModel() {
  try {
    await tf.ready();
    // Cache-busting query param to avoid stale GitHub Pages model.json
    model = await tf.loadLayersModel("tfjs_model/model.json?v=" + Date.now());
    const shape = model.inputs[0].shape; // [null, H, W, C]
    els.shapeBadge.textContent = JSON.stringify(shape);
    showStatus("Model loaded ✓", "ok");
    console.log("✅ Model loaded. Input shape:", shape);
  } catch (err) {
    console.error(err);
    showStatus("Failed to load model: " + (err?.message || err), "err");
  }
}

/** Get H,W from the model itself instead of hard-coding */
function getTargetHW() {
  if (!model) return { H: 224, W: 224 };
  const shape = model.inputs[0].shape;
  return { H: shape[1] || 224, W: shape[2] || 224 };
}

/** Draw media -> canvas -> tensor [1,H,W,3] normalized 0..1 */
function preprocessToTensor(mediaEl) {
  const { H, W } = getTargetHW();
  els.canvas.width = W;
  els.canvas.height = H;
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(mediaEl, 0, 0, W, H);
  return tf.tidy(() =>
    tf.browser.fromPixels(els.canvas).toFloat().div(255).expandDims(0)
  );
}

function renderPredictions(probArray) {
  els.predBody.innerHTML = "";
  els.predTable.style.display = "table";
  const rows = classes
    .map((label, i) => ({ label, p: probArray[i] || 0 }))
    .sort((a, b) => b.p - a.p);

  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.label}</td><td>${(r.p * 100).toFixed(2)}%</td>`;
    els.predBody.appendChild(tr);
  });

  const top = rows[0];
  els.predSummary.textContent = `Top: ${top.label} (${(top.p * 100).toFixed(2)}%)`;
  els.predSummary.className = "";
}

async function predictOn(mediaEl) {
  if (!model) {
    showStatus("Model not loaded yet.", "err");
    return;
  }
  try {
    const x = preprocessToTensor(mediaEl);
    const y = model.predict(x);
    const out = await y.data();
    x.dispose();
    if (y.dispose) y.dispose();

    const probs = Array.from(out).some(v => v < 0 || v > 1)
      ? softmax(Array.from(out))
      : Array.from(out);

    renderPredictions(probs);
  } catch (err) {
    console.error(err);
    showStatus("Prediction error: " + (err?.message || err), "err");
  }
}

/* ---------- Image upload ---------- */
els.file.addEventListener("change", e => {
  const file = e.target.files?.[0];
  if (!file) return;
  els.video.style.display = "none";
  els.img.style.display = "block";
  els.img.onload = () => predictOn(els.img);
  els.img.src = URL.createObjectURL(file);
});

/* ---------- Webcam ---------- */
let stream;
els.startCam.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    els.video.srcObject = stream;
    els.video.style.display = "block";
    els.img.style.display = "none";
    showStatus("Webcam started ✓", "ok");
  } catch (err) {
    console.error(err);
    showStatus("Webcam error: " + (err?.message || err), "err");
  }
});

els.stopCam.addEventListener("click", () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    showStatus("Webcam stopped", "muted");
  }
  els.video.style.display = "none";
});

els.predictCam.addEventListener("click", () => {
  if (!els.video.srcObject) {
    showStatus("Start the webcam first.", "err");
    return;
  }
  predictOn(els.video);
});

/* ---------- Boot ---------- */
loadModel();


