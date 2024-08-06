const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const mobilenet = require("@tensorflow-models/mobilenet");
const path = require("path");
const fs = require("fs");
const sharp = require("sharp");
const cors = require("cors");

const app = express();
app.use(cors());
const upload = multer({ dest: "uploads/" });

let model;

// Load the MobileNet model at startup
(async () => {
  try {
    model = await mobilenet.load();
    console.log("MobileNet model loaded");
  } catch (error) {
    console.error("Error loading MobileNet model:", error);
  }
})();

// Image classification endpoint
app.post("/classify", upload.single("image"), async (req, res) => {
  const imagePath = req.file.path;

  console.log("imagePath", imagePath);

  try {
    // Get image metadata
    const { width, height } = await sharp(imagePath).metadata();

    // Calculate crop dimensions
    const cropSize = Math.min(width, height);
    const left = Math.round((width - cropSize) / 2);
    const top = Math.round((height - cropSize) / 2);

    // Use sharp to crop and resize the image
    const imageBuffer = await sharp(imagePath)
      .extract({ left, top, width: cropSize, height: cropSize }) // Crop to a central square
      .resize(224, 224) // MobileNet expects 224x224 input size
      .toBuffer();

    // Decode image and convert it to a tensor
    const imageTensor = tf.node
      .decodeImage(imageBuffer)
      .toFloat()
      .expandDims()
      .div(tf.scalar(255)); // Normalize to [0, 1]

    // Get predictions from the model
    const predictions = await model.classify(imageTensor);

    // Map predictions to output
    const top5 = predictions.slice(0, 5).map((prediction) => ({
      label: prediction.className,
      probability: (prediction.probability * 100).toFixed(2), // Convert to percentage
    }));

    console.log("Predictions:", predictions); // Debugging output

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
