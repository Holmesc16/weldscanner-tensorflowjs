const tf = require('@tensorflow/tfjs-node');
const path = require('path');

let model;

const loadModel = async () => {
if (!model) {
    const modelPath = path.join(__dirname, '..', '_trained_models', 'your_model_name', 'model.json');
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded successfully.');

    console.log('Model inputs:', model.inputs.map(input => ({ name: input.name, shape: input.shape })));
    console.log('Model Layers:');
    
    model.layers.forEach((layer, index) => {
        console.log(`Layer ${index}: ${layer.name} Output Shape: ${layer.outputShape}, Inbound Nodes: ${layer.inboundNodes.length}`);
        });
    }
    return model;
};

module.exports = {
    loadModel
}