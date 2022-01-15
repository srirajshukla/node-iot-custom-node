let model;

const tf = require('@tensorflow/tfjs');

const loadModel = async function () {
    // const model_path = "file:///D:/workspace/nodered/nodered-dev/xor-predictor/fashionp/model.json";
    // console.log(model_path)
    // model = await tf.loadLayersModel(model_path).catch(err => {
    //     console.log(err)
    // });
    // console.log(model);
    // return model;

    const model_path = "file:///workspace/nodered/testingTF/fashionp/model.json";
    model = await tf.loadLayersModel(model_path);

    return model;

}


const processInput = async function(img) {
    const imageBuffer = await fs.readFile(img)
    const uint8array = new Uint8Array(imageBuffer);
    return tf.node.decodeImage(uint8array, 3);
}

const processOutput = function (input2d) {
    const output = model.predict()

    const outData = output.dataSync();
    console.log(outData);
    return outData;
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

model = loadModel()

// const input2d = processInput("./name.png");
// const output = processOutput(input2d);
// console.log(output)

// module.exports = {
//     loadModel: loadModel,
//     processInput: processInput,
//     processOutput: processOutput,
// }