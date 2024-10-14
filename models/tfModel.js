const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { findLayersByName } = require('../utils/gradCam');

let model;

const loadModel = async () => {
if (!model) {
    const baseModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    
    // truncate the model to exclude the last layers (e.g., classification layers)
    const truncatedBaseModel = tf.model({
        inputs: baseModel.inputs,
        outputs: baseModel.layers[baseModel.layers.length - 4].output // exclude the last 4 layers
    });
    truncatedBaseModel.trainable = false;

    // create new inputs for the image and category
    const imageInput = tf.input({ shape: [224, 224, 3], name: 'imageInput' });
    const categoryInput = tf.input({ shape: [3], name: 'categoryInput' });

    const baseModelOutput = truncatedBaseModel.apply(imageInput);
    console.log(`Base model output shape: ${baseModelOutput.shape}`);

    // get output of last conv layer
    const lastConvLayerName = 'conv_pw_13_relu'
    const lastConvLayer = findLayersByName(truncatedBaseModel, lastConvLayerName);
    const lastConvLayerOutput = lastConvLayer.output;

    // build model
    const baseModelFlattened = tf.layers.flatten().apply(baseModelOutput);
    const categoryDense = tf.layers.dense({ units: 32, activation: 'relu', name: 'categoryDense' }).apply(categoryInput);

    const concatenated = tf.layers.concatenate().apply([baseModelFlattened, categoryDense]);

    let x = concatenated;
    x = tf.layers.dense({ units: 128, activation: 'relu', name: 'dense_1'}).apply(x);
    x = tf.layers.dropout({ rate: 0.5, name: 'dropout_1'}).apply(x);
    x = tf.layers.dense({ units: 64, activation: 'relu', name: 'dense_2'}).apply(x);
    x = tf.layers.dropout({ rate: 0.5, name: 'dropout_2'}).apply(x);
    
    const output = tf.layers.dense({ units: 1, activation: 'sigmoid', name: 'output' }).apply(x);
    const outputReshaped = tf.layers.reshape({ targetShape: [1] }).apply(output);

    model = tf.model({ inputs: [imageInput, categoryInput], outputs: outputReshaped });

   const modelPath = path.join(__dirname, '..', '_trained_models', 'weldscanner_quality_model_v2', 'model.json');
   await model.loadWeights(`file://${modelPath}`);
   console.log('Model loaded successfully.');
   console.log('Model inputs:', model.inputs.map(input => ({ name: input.name, shape: input.shape })));
   console.log('Model Layers:');
   model.layers.forEach((layer, index) => {
        console.log(`Layer ${index}: ${layer.name} Output Shape: ${layer.outputShape}, Inbound Nodes: ${layer.inboundNodes.length}`);
    });
    model.lastConvLayerOutput = lastConvLayerOutput;
}
    return model;
};

module.exports = {
    loadModel
}