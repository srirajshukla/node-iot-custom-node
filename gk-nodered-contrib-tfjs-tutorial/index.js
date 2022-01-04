// export the node module
module.exports = function(RED) {
  // import helper module
  const tfmodel = require('./tfjs-tutorial-util');

  // load the model
  async function loadModel (config, node) {
    node.model = await tfmodel.loadModel(config.modelUrl, config.fromHub);
  }

  // define the node's behavior
  function TfjsTutorialNode(config) {
    // initialize the features
    RED.nodes.createNode(this, config);
    const node = this

    node.status({fill:"red", shape:"ring", text:"loading the model"});
    loadModel(config, node).then(() => {
      node.status({fill:"green", shape:"dot", text:"model loaded"});
    }).catch(err => {
      node.status({fill:"red", shape:"ring", text:"model not loaded"});
    });

    // register a listener to get called whenever a message arrives at the node
    node.on('input', function (msg) {
      // preprocess the incoming image
      const inputTensor = tfmodel.processInput(msg.payload);
      
      // get image/input shape
      const height = inputTensor.shape[1];
      const width = inputTensor.shape[2];

      // get the prediction
      node.model
        .executeAsync(inputTensor)
        .then(prediction => {
          msg.payload = tfmodel.processOutput(prediction, height, width);
          // send the prediction out
          node.send(msg);
        });
    });
  }

  // register the node with the runtime
  RED.nodes.registerType('tfjs-tutorial-node', TfjsTutorialNode);
}