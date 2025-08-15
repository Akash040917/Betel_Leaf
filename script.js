let model;

// Load the TFJS model
async function loadModel() {
    try {
        model = await tf.loadLayersModel('tfjs_model/model.json');
        console.log("✅ Model loaded successfully");
        console.log("Model input shape:", model.inputs[0].shape);
    } catch (error) {
        console.error("❌ Error loading model:", error);
    }
}

// Preprocess the uploaded image to [1, 224, 224, 3]
function preprocessImage(image) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    // Draw and resize the image to 224x224
    ctx.drawImage(image, 0, 0, 224, 224);

    // Convert to tensor
    let tensor = tf.browser.fromPixels(canvas)
        .toFloat()
        .expandDims(0); // Add batch dimension: [1, 224, 224, 3]

    // Normalize if your model expects values between 0 and 1
    tensor = tensor.div(255.0);

    return tensor;
}

// Handle file upload
document.getElementById('imageUpload').addEventListener('change', async (event) => {
    if (!model) {
        alert("Model is not loaded yet.");
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = async () => {
        const inputTensor = preprocessImage(img);
        const prediction = model.predict(inputTensor);

        // Get first value if it's a single prediction
        const output = await prediction.data();
        document.getElementById('predictionOutput').textContent = output.join(', ');

        console.log("Prediction array:", output);
    };
    img.src = URL.createObjectURL(file);
});

// Load model when page opens
loadModel();
