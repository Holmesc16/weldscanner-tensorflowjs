const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { findLayersByName } = require('../utils/gradCam');

let model;

exports.loadModel = async () => {
    if (!model) {
        const modelPath = path.join(__dirname, '..', '_trained_models', 'weldscanner_quality_model_v2', 'model.json');
        model = await tf.loadLayersModel(`file://${modelPath}`);
        console.log('Model loaded successfully.');

        // Access the last convolutional layer
        const lastConvLayerName = 'conv_pw_13_relu'; // Adjust if necessary
        const lastConvLayer = findLayerByName(model, lastConvLayerName);
        if (!lastConvLayer) {
            throw new Error(`Could not find layer ${lastConvLayerName} in the model.`);
        }
        model.lastConvLayerOutput = lastConvLayer.output;
    }
    return model;
};

// Helper function to find a layer by name
function findLayerByName(model, layerName) {
    for (const layer of model.layers) {
        if (layer.name === layerName) {
            return layer;
        } else if (layer instanceof tf.Sequential || layer instanceof tf.LayersModel) {
            const nestedLayer = findLayerByName(layer, layerName);
            if (nestedLayer) {
                return nestedLayer;
            }
        }
    }
    return null;
}
