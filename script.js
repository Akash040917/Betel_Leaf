let model;

// Load your TFJS model
async function loadModel() {
    // Make sure the path points to your model.json in tfjs_model folder
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Model loaded successfully!");
}

// Run prediction when user clicks
async function runPrediction() {
    const inputValue = parseFloat(document.getElementById('inputValue').value);
    if (isNaN(inputValue)) {
        alert("Please enter a valid number");
        return;
    }

    // Convert input to tensor
    const inputTensor = tf.tensor2d([inputValue], [1, 1]); // Change shape if your model expects different input
    const outputTensor = model.predict(inputTensor);
    
    // Extract value and show
    const outputValue = (await outputTensor.data())[0];
    document.getElementById('outputValue').textContent = outputValue.toFixed(4);
}

// Setup button listener
document.getElementById('predictButton').addEventListener('click', runPrediction);

// Load the model on page load
loadModel();
