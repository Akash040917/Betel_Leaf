// === Configuration ===
// Path to TFJS model directory (contains model.json + shard bins)
const MODEL_URL = 'tfjs_model/model.json';

// Replace with your exact class names in the order used during training
const CLASS_NAMES = [
  'Healthy_Red',
  'Healthy_Green',
  'BacterialLeafSpot_Green',
  'AnthracnoseAffected_Green'
];

// Expected input size of your model (typical MobileNetV2 TFJS: 224x224)
const INPUT_SIZE = 224;

// Optional: MobileNetV2 expects [0,1] or [-1,1] depending on preprocessing.
// If you used tf.keras.applications.mobilenet_v2.preprocess_input, use scaleToMinusOneToOne=true.
const scaleToMinusOneToOne = false;

// === State ===
let model;
let stream;
let autoTimer;

// === DOM ===
const statusEl = document.getElementById('status');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview');
const predictImageBtn = document.getElementById('predict-image');
const imageResult = document.getElementById('image-result');

const video = document.getElementById('video');
const startCamBtn = document.getElementById('start-cam');
const stopCamBtn = document.getElementById('stop-cam');
const predictFrameBtn = document.getElementById('predict-frame');
const autoToggle = document.getElementById('auto-toggle');
const camResult = document.getElementById('cam-result');

// === Utils ===
function setStatus(msg) { statusEl.textContent = msg; }

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function renderPredictions(container, probs) {
  container.innerHTML = '';
  const top = probs
    .map((p, i) => ({ label: CLASS_NAMES[i] ?? `Class ${i}`, p }))
    .sort((a, b) => b.p - a.p);

  const best = top[0];
  const title = document.createElement('div');
  title.className = 'pred';
  title.innerHTML = `<strong>Prediction:</strong> ${best.label} (${(best.p*100).toFixed(2)}%)`;
  container.appendChild(title);

  top.forEach(({ label, p }) => {
    const row = document.createElement('div');
    row.className = 'pred';
    row.style.marginTop = '6px';
    row.textContent = `${label}: ${(p*100).toFixed(2)}%`;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('span');
    fill.style.width = `${(p*100).toFixed(2)}%`;
    bar.appendChild(fill);
    container.appendChild(row);
    container.appendChild(bar);
  });
}

function preprocessImg(sourceEl) {
  return tf.tidy(() => {
    let img = tf.browser.fromPixels(sourceEl).toFloat();
    // Resize to model input
    img = tf.image.resizeBilinear(img, [INPUT_SIZE, INPUT_SIZE], true);

    // Normalize
    if (scaleToMinusOneToOne) {
      // [-1,1] scaling
      img = img.div(127.5).sub(1.0);
    } else {
      // [0,1]
      img = img.div(255.0);
    }

    // Add batch dimension
    return img.expandDims(0);
  });
}

async function predictFromElement(el) {
  const input = preprocessImg(el);
  let output = model.predict(input);

  // If model outputs logits, apply softmax; if it already outputs probs, this still works if values sum~1
  const data = Array.from(await output.data());
  tf.dispose([input, output]);

  const probs = softmax(data);
  return probs;
}

// === Event handlers ===
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);
  predictImageBtn.disabled = false;
});

predictImageBtn.addEventListener('click', async () => {
  if (!model || !previewImg.complete) return;
  setStatus('Predicting image…');
  const probs = await predictFromElement(previewImg);
  renderPredictions(imageResult, probs);
  setStatus('Ready');
});

startCamBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
    await video.play();
    stopCamBtn.disabled = false;
    predictFrameBtn.disabled = false;
    setStatus('Camera started');
  } catch (err) {
    console.error(err);
    setStatus('Camera access denied or unavailable');
  }
});

stopCamBtn.addEventListener('click', () => {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
    autoToggle.checked = false;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  stopCamBtn.disabled = true;
  predictFrameBtn.disabled = true;
  setStatus('Camera stopped');
});

predictFrameBtn.addEventListener('click', async () => {
  if (!model || !video.srcObject) return;
  const probs = await predictFromElement(video);
  renderPredictions(camResult, probs);
});

autoToggle.addEventListener('change', () => {
  if (autoToggle.checked) {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(async () => {
      if (model && video.srcObject) {
        const probs = await predictFromElement(video);
        renderPredictions(camResult, probs);
      }
    }, 500); // update every 500ms
  } else {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
    }
  }
});

// === Load model on start ===
(async function init() {
  try {
    setStatus('Loading model…');
    model = await tf.loadLayersModel(MODEL_URL);
    // Warm-up
    tf.tidy(() => model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])));
    setStatus('Model loaded. Ready.');
    predictImageBtn.disabled = !previewImg.src;
  } catch (err) {
    console.error(err);
    setStatus('Failed to load model. Check MODEL_URL and file paths.');
  }
})();

