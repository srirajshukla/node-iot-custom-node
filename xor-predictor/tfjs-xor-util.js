let model;

const tf = require('@tensorflow/tfjs');

const loadModel = async function () {
    const model_path = "file:///workspace/nodered/testingTF/xormod/model.json";
    model = await tf.loadLayersModel(model_path);

    return model;
}


const processInput = function(_a,_b) {
    const a = Number(_a);
    const b = Number(_b);

    const input2d = tf.tensor2d([[a,b]]);

    return input2d;
}

const processOutput = function (prediction) {
    const outData = prediction.dataSync();
    console.log(outData);
    return Number(outData[0] > 0.5);
}


const xormodel = async function(a, b) {

    loadModel()
    const model = await tf.loadLayersModel(model_path);
    // console.log(model);

    const input2d = tf.tensor2d([[a,b]]);
    const output = model.predict(input2d)
    const outputData = output.dataSync();

    console.log(`Probability = ${outputData}\tPredicted Value = ${Number(outputData[0] > 0.5)}`)
}

module.exports = {
    loadModel: loadModel,
    processInput: processInput,
    processOutput: processOutput,
}