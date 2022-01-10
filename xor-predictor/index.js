// export the node module
module.exports = function (RED) {
    // import helper module
    const tfmodel = require("./tfjs-xor-util");

    // load the model
    async function loadModel(config, node) {
        node.model = await tfmodel.loadModel(); // tensorflow model
        console.log("model loaded");
    }

    // define the node's behavior
    function XORPredictor(config) {
        // initialize the features
        RED.nodes.createNode(this, config);
        const node = this;

        node.status({
            fill: "red",
            shape: "ring",
            text: "loading the model"
        });
        loadModel(config, node)
            .then(() => {
                node.status({
                    fill: "green",
                    shape: "dot",
                    text: "model loaded"
                });
            })
            .catch((err) => {
                console.error("error in loading model: ", err);
                node.status({
                    fill: "red",
                    shape: "ring",
                    text: "model not loaded"
                });
            });

        // register a listener to get called whenever a message arrives at the node
        node.on("input", function (msg) {
            // preprocess the incoming image
            const inputTensor = tfmodel.processInput(config.a, config.b);

            inputTensor.print();
            // get the prediction
            
            const prediction = node.model.predict(inputTensor)
            console.log(prediction)
            msg.payload = tfmodel.processOutput(prediction);
            // send the prediction out
            node.send(msg);
        
        });
    }

    // register the node with the runtime
    RED.nodes.registerType("xor-predictor-node", XORPredictor);
};