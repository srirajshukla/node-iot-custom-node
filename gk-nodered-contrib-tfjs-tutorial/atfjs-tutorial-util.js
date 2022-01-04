const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const modelUrl = 'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';

let model;

// load the cocosd model 
const loadModel = async function () {
    console.log(`Loading model from ${modelUrl}`);

    model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
    return model;
}

const processImage = function (imagePath) {
    console.log(`Processing image ${imagePath}`);

    const image = fs.readFileSync(imagePath);
    const buf = Buffer.from(image);
    const uint8array = new Uint8Array(buf);

    return tf.node.decodeImage(uint8array, 3).expandDims();
}

const runModel = function (inputTensor) {
    return model.executeAsync(inputTensor);
}

const extractClassesAndMaxScores = function (predictionScores) {
    console.log(`Extracting classes and max scores`);

    const scores = predictionScores.dataSync();
    const numBoxesFound = predictionScores.shape[1];
    const numClassesFound = predictionScores.shape[2];

    const maxScores = [];
    const classes = [];

    // for each bounding box
    for (let i = 0; i < numBoxesFound; i++) {
        let maxScore = -1;
        let classIndex = -1;

        // find the class with highest score
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

const maxNumBoxes = 5;

const calculateNMS = function(outputBoxes, maxScores) {
    console.log("calculating bounding boxes");

    const boxes = tf.tensor2d(
        outputBoxes.dataSync(), [outputBoxes.shape[1], outputBoxes.shape[3]]
        );
    
    const indexTensor = tf.image.nonMaxSuppression(
        boxes,maxScores,maxNumBoxes, 0.5, 0.5
    );
    
    return indexTensor.dataSync();
}

const labels = require('./labels.js');

const createJSONResponse = function (boxes, scores, indexes, classes, width, height) {
    console.log("creating JSON response");
    const count = indexes.length;
    const objects = [];

    for(let i=0; i<count; i++){
        let bbox = [];

        for(let j=0; j<4; j++){
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

const processOutput = function (prediction, width, height) {
    console.log("processing output");

    const [maxScores, classes] = extractClassesAndMaxScores(prediction[0]);
    const indexes = calculateNMS(prediction[1], maxScores);

    return createJSONResponse(prediction[1].dataSync(), maxScores, indexes, classes, width, height);
}

if (process.argv.length < 3) {
    console.log('Usage: node run-tfjs-model.js <image-path>');
} else{
    const imagePath = process.argv[2];
    let width = inputTensor.shape[2];
    let height = inputTensor.shape[1];
    
    loadModel().then(model => {
            const inputTensor = processImage(imagePath);
            return runModel(inputTensor);
    }).then(predictions => {
        const output = processOutput(predictions, width, height);
        console.log(output);
    })
}

module.exports = {
    loadModel:loadModel,
    processInput: processImage,
    processOutput: processOutput
};