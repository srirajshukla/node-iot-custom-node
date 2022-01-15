const tf = require('@tensorflow/tfjs-node');
const labels = require('./labels.js');
const fs = require('fs');

// load COCO-SSD graph model from TensorFlow Hub
const loadModel = async function (modelUrl, fromTFHub) {
  // console.log(`loading model from ${modelUrl}`);
  var handler = tf.io.fileSystem('./model.json')
  console.log("loading model from", handler.path);
  // const model = await tf.loadLayersModel("file://D://workspace//nodered//nodered-dev//model.json");
  // const model = await tf.loadGraphModel("https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1", {fromTFHub: true});
  // console.log("model = ", model);
  
  if (fromTFHub) {
    console.log("loading model from", modelUrl);
    model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
  } else {
    model = await tf.loadGraphModel(modelUrl);
  }
  console.log(`model loaded from ${modelUrl}`);
  return model;
}

// convert image to Tensor
const processInput = function (imagePath) {
  console.log(`preprocessing image`);
  // const image = fs.readFileSync(imagePath);
  // const buf = Buffer.from(image);

  // const uint8array = new Uint8Array(buf);
  // return tf.node.decodeImage(uint8array, 3).expandDims();
  // return uint8array;  
  const uint8array = new Uint8Array(imagePath);
  console.log("got the image");

  return tf.node.decodeImage(uint8array, 3).expandDims();
}

const maxNumBoxes = 5;

 // process the model output into a friendly JSON format
 const processOutput = function (prediction, height, width) {
   console.log('processOutput');

   const [maxScores, classes] = extractClassesAndMaxScores(prediction[0]);
   const indexes = calculateNMS(prediction[1], maxScores);

   return createJSONresponse(prediction[1].dataSync(), maxScores, indexes, classes, height, width);
 }

 // determine the classes and max scores from the prediction
 const extractClassesAndMaxScores = function (predictionScores) {
   console.log('calculating classes & max scores');

   const scores = predictionScores.dataSync();
   const numBoxesFound = predictionScores.shape[1];
   const numClassesFound = predictionScores.shape[2];

   const maxScores = [];
   const classes = [];

   // for each bounding box returned
   for (let i = 0; i < numBoxesFound; i++) {
     let maxScore = -1;
     let classIndex = -1;

     // find the class with the highest score
     for (let j = 0; j < numClassesFound; j++) {
       if (scores[i * numClassesFound + j] > maxScore) {
         maxScore = scores[i * numClassesFound + j];
         classIndex = j;
       }
     }

     maxScores[i] = maxScore;
     classes[i] = classIndex;
   }

   return [maxScores, classes];
 }

 // perform non maximum suppression of bounding boxes
 const calculateNMS = function (outputBoxes, maxScores) {
   console.log('calculating box indexes');

   const boxes = tf.tensor2d(outputBoxes.dataSync(), [outputBoxes.shape[1], outputBoxes.shape[3]]);
   const indexTensor = tf.image.nonMaxSuppression(boxes, maxScores, maxNumBoxes, 0.5, 0.5);

   return indexTensor.dataSync();
 }

 // create JSON object with bounding boxes and label
 const createJSONresponse = function (boxes, scores, indexes, classes, height, width) {
   console.log('create JSON output');

   const count = indexes.length;
   const objects = [];

   for (let i = 0; i < count; i++) {
     const bbox = [];

     for (let j = 0; j < 4; j++) {
       bbox[j] = boxes[indexes[i] * 4 + j];
     }

     const minY = bbox[0] * height;
     const minX = bbox[1] * width;
     const maxY = bbox[2] * height;
     const maxX = bbox[3] * width;

     objects.push({
       bbox: [minX, minY, maxX, maxY],
       label: labels[classes[indexes[i]]],
       score: scores[indexes[i]]
     });
   }

   return objects;
 }

 module.exports = {
   loadModel: loadModel,
   processInput: processInput,
   processOutput: processOutput
 }

