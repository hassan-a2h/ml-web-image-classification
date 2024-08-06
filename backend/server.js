const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const path = require("path");
const fs = require("fs");
const sharp = require("sharp");
const cors = require("cors");

const app = express();
app.use(cors());
const upload = multer({ dest: "uploads/" });

let model;
let labels;

// Load the model and labels at startup
(async () => {
  try {
    model = await tf.loadGraphModel("file://./assets/model.json");
    labels = JSON.parse(fs.readFileSync("./assets/labels.json", "utf-8"));
    console.log("Model and labels loaded");
  } catch (error) {
    console.error("Error loading model or labels:", error);
  }
})();

// Image classification endpoint
app.post("/classify", upload.single("image"), async (req, res) => {
  const imagePath = req.file.path;

  console.log("imagePath", imagePath);

  try {
    // Use sharp to resize the image to 128x128 (MobileNet's expected input size)
    const imageBuffer = await sharp(imagePath)
      .resize(128, 128, { fit: "cover" })
      .toBuffer();

    // Decode image and convert it to a tensor
    const imageTensor = tf.node
      .decodeImage(imageBuffer, 3)
      .expandDims()
      .toFloat()
      .div(tf.scalar(255)); // Normalize to [0, 1]

    // Get predictions from the model
    const predictions = await model.predict(imageTensor);
    const probabilities = await predictions.data();

    // Get the indices of the top 5 predictions
    const topIndices = Array.from(probabilities)
      .map((p, i) => [p, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, 5)
      .map(([, index]) => index);

    // Map predictions to output
    const top5 = topIndices.map((index) => ({
      label: labels[index],
      probability: (probabilities[index] * 100).toFixed(2),
    }));

    console.log("Predictions:", top5);

    res.json({ predictions: top5 });
  } catch (error) {
    console.error("Error processing the request:", error);
    res.status(500).send("Error processing the request");
  } finally {
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
