let model;
const classes = ["Class 1", "Class 2", "Class 3", "Class 4"]; // Replace with actual class names

// Load the TFJS model
async function loadModel() {
  try {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("✅ Model loaded successfully");
    console.log("Expected input shape:", model.inputs[0].shape);
  } catch (error) {
    console.error("❌ Error loading model:", error);
  }
}

// Preprocess image to match model input
function preprocessImage(image) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 224, 224);
  
  let tensor = tf.browser.fromPixels(canvas)
    .toFloat()
    .div(255.0) // Normalize
    .expandDims(0); // Add batch dimension
  
  return tensor;
}

// Predict function
async function predictImage(image) {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }
  
  const inputTensor = preprocessImage(image);
  const prediction = model.predict(inputTensor);
  const output = await prediction.data();

  // Find highest probability class
  const maxIndex = output.indexOf(Math.max(...output));
  document.getElementById("prediction").innerText =
    `Prediction: ${classes[maxIndex]} (${(output[maxIndex] * 100).toFixed(2)}%)`;
  
  console.log("Prediction probabilities:", output);
}

// Handle image upload
document.getElementById("imageUpload").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const img = document.getElementById("uploadedImage");
  img.onload = () => predictImage(img);
  img.src = URL.createObjectURL(file);
  img.style.display = "block";
});

// Webcam handling
let stream;
document.getElementById("startWebcam").addEventListener("click", async () => {
  const video = document.getElementById("videoElement");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error("Error starting webcam:", err);
  }
});

document.getElementById("stopWebcam").addEventListener("click", () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
});

document.getElementById("predictWebcam").addEventListener("click", () => {
  const video = document.getElementById("videoElement");
  predictImage(video);
});

// Load model on page start
loadModel();
