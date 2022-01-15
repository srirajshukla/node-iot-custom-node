let model;

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');


const loadModel = async function () {
    const model_path = "file://D:/workspace/nodered/testingTF/fashionp/model.json";
    model = await tf.loadLayersModel(model_path);
    console.log(model);
    return model;
}


const processInput = function(msg) {
    console.log("got input data to be processed")
    console.log(msg);
    
    const uint8array = new Uint8Array(msg);
    console.log(uint8array);

    let imageTensor = tf.node.decodeImage(uint8array, 1);

    console.log(imageTensor);
    imageTensor = imageTensor.reshape([1,28,28]);

    console.log(imageTensor);

    return imageTensor;
}

const processOutput = function (prediction) {
    return {
        label: "Prediction of the Image",
        predictionTensor: prediction,
    };
}


const xormodel = async function(a, b) {

    console.log(__dirname);
    const model_path = "file:///D:/workspace/nodered/testingTF/fashionp/model.json";
    try{
    const model = await tf.loadLayersModel(model_path);
    } catch (error){
        console.log(error);
    }
    console.log(model);

    // const imageBuffer = fs.readFileSync('D:/workspace/nodered/nodered-dev/fashion-predictor/fashionp/name.png')
    // const uint8array = new Uint8Array(imageBuffer);
    // let imageTensor = tf.node.decodeImage(uint8array, 1);
    // imageTensor = imageTensor.reshape([1, 28, 28])


    // console.log(imageTensor);
    // "D:\workspace\nodered\nodered-dev\fashion-predictor\fashionp\name.png"


    // const output = model.predict(imageTensor);

    // console.log(output)

    
    // const outputData = output.dataSync();
    // console.log(outputData)
    
    // const input2d = tf.tensor2d([[a,b]]);
    // const output = model.predict(input2d)
    // const outputData = output.dataSync();

    // console.log(`Probability = ${outputData}\tPredicted Value = ${Number(outputData[0] > 0.5)}`)
}
console.log(__dirname);
module.exports = {
    loadModel: loadModel,
    processInput: processInput,
    processOutput: processOutput,
    xormodel: xormodel,
}