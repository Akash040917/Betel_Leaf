let model = null;
const classLabels = [
  'Anthracnose Green',
  'BacterialLeafSpot Green',
  'Healthy Green',
  'Healthy Red'
];

async function loadModel() {
  try {
    document.getElementById('status').innerText = '⏳ Loading model...';
    model = await tf.loadLayersModel('model/model.json');
    document.getElementById('status').innerText = '✅ Model loaded';
    document.getElementById('predictBtn').disabled = false;
  } catch (err) {
    console.error('Failed to load model:', err);
    document.getElementById('status').innerText = '❌ Failed to load model.';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  loadModel();

  document.getElementById('imageUpload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = document.getElementById('preview');
    img.src = URL.createObjectURL(file);
    img.style.display = 'block';
  });

  document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!model) {
      alert('Model not loaded yet. Please wait.');
      return;
    }
    const img = document.getElementById('preview');
    if (!img.src) { 
      alert('Please upload an image first'); 
      return; 
    }

    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();

    const predictions = await model.predict(tensor).data();
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const prob = predictions[maxIndex];
    document.getElementById('result').innerText =
      `${classLabels[maxIndex]} — ${(prob * 100).toFixed(2)}%`;
  });
});




