const tf = require('@tensorflow/tfjs-node');

let model;

async function loadModel() {
    if (!model) {
        model = await tf.loadLayersModel('file://path/to/your/saved/model/model.json');
    }
    return model;
}

async function predict(inputTensor) {
    const model = await loadModel();
    return model.predict(inputTensor);
}

module.exports = {
    predict
};