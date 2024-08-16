# Monkeypox Prediction Model

## Overview

This project provides a web-based interface for predicting whether an image contains Monkeypox or not using a pre-trained TensorFlow model. The application allows users to upload an image, and the model will classify the image into one of two categories: `MonkeyPox` or `Other`.

## Features

- **Image Upload**: Users can upload images directly from their local filesystem.
- **Model Prediction**: The model predicts the class of the uploaded image.
- **Image Preview**: Displays the uploaded image before prediction.
- **Results Display**: Shows the predicted class name after processing.

## Requirements

- TensorFlow.js
- Modern web browser (Chrome, Firefox, etc.)

## Setup

### 1. Load TensorFlow.js

Include the TensorFlow.js library in your HTML file:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
```

### 2. HTML Structure

Create an HTML file with the following structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monkeypox Prediction</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <style>
        /* Add your CSS styles here */
    </style>
</head>
<body>
    <div id="header">
        <span class="das">DAS</span><span class="medhub">medhub</span>
    </div>
    <h1>Monkeypox Prediction Model</h1>
    <input type="file" id="imageUpload" accept="image/*" />
    <button id="predictButton" onclick="predict()">Predict</button>
    <img id="imagePreview" />
    <p id="result"></p>

    <script>
        /* Add your JavaScript code here */
    </script>
</body>
</html>
```

### 3. JavaScript Code

Add the following JavaScript code to handle model loading, image processing, and prediction:

```javascript
let model;
const classNames = ['MonkeyPox', 'Other'];

async function loadModel() {
    try {
        model = await tf.loadGraphModel('monkeypox_model/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

async function predict() {
    const imageUpload = document.getElementById('imageUpload');
    const file = imageUpload.files[0];
    if (!file) {
        alert('Please upload an image!');
        return;
    }

    // Clear previous prediction result and image preview
    document.getElementById('result').innerText = '';
    document.getElementById('imagePreview').src = '';

    // Display the uploaded image
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.src = URL.createObjectURL(file);

    // Read the image and create a tensor
    try {
        const img = await createImageBitmap(file);
        const tensor = tf.browser.fromPixels(img)
            .resizeBilinear([300, 300])
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255));

        console.log('Tensor shape:', tensor.shape);

        // Disable button while predicting
        const predictButton = document.getElementById('predictButton');
        predictButton.disabled = true;

        // Predict the class
        const prediction = await model.predict(tensor).data();
        console.log('Prediction:', prediction);

        const classIdx = prediction.indexOf(Math.max(...prediction));
        console.log('Predicted class index:', classIdx);

        const className = classNames[classIdx];
        document.getElementById('result').innerText = 'Predicted Class name: ' + className;

        // Re-enable button after prediction
        predictButton.disabled = false;
    } catch (error) {
        console.error('Error during prediction:', error);
        document.getElementById('result').innerText = 'Error making prediction.';
        document.getElementById('imagePreview').src = '';
    }
}

window.onload = loadModel;

document.getElementById('imageUpload').addEventListener('change', () => {
    document.getElementById('result').innerText = '';
    document.getElementById('imagePreview').src = '';
});
```

## Usage

1. **Open the HTML file** in a web browser.
2. **Upload an image** by clicking the file input.
3. **Click the "Predict" button** to get the prediction result.
4. **View the result** and the uploaded image.

## Troubleshooting

- **Model Not Loading**: Ensure that the `model.json` file and model weights are correctly placed in the `monkeypox_model` directory.
- **Prediction Errors**: Verify that the image preprocessing matches the model's training process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact us at [dasmedhub@outlook.com).

---

