const tf = require('@tensorflow/tfjs-node');
const path = require('path');

const modelPath = path.join(__dirname, '..', 'trained_models', 'weldscanner_quality_model', 'model.json');
let model;

const loadModel = async () => {
    if (!model) {
        model = await tf.loadLayersModel(`file://${modelPath}`);
    }
    return model;
}

exports.predict = async (tensor) => {
    const loadedModel = await loadModel();
    return loadedModel.predict(tensor);
};
