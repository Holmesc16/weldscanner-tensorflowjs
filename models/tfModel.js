const tf = require('@tensorflow/tfjs-node');
const path = require('path');

let model;

const loadModel = async () => {
  if (!model) {
    const modelPath = path.join(__dirname, '..', '_trained_models', 'weldscanner_quality_model', 'model.json');
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded successfully.');
    console.log('Model inputs: ', model.inputs.map(input => input.name))
  }
  return model;
};

module.exports = {
  loadModel
};
