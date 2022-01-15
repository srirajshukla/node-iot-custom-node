// export the node module
module.exports = function (RED) {
    // import helper module
    const tfmodel = require("./fashion-pred");

    // load the model
    async function loadModel(config, node) {
        node.model = await tfmodel.loadModel(); // tensorflow model
        console.log("fashion model loaded");
    }

    // define the node's behavior
    function FashionPredictor(config) {
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
            console.log("sending input data to be processed");
            const inputTensor = tfmodel.processInput(msg.payload);

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
    RED.nodes.registerType("fashion-predictor-node", FashionPredictor);
};