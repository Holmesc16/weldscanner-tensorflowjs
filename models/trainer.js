const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const path = require('path');
const { createDataset } = require('../controllers/imageController.js');

async function loadPretrainedModel() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const truncatedModel = tf.model({
        inputs: mobilenet.inputs,
        outputs: mobilenet.layers[mobilenet.layers.length - 4].output
    });
    return truncatedModel;
}

async function createModel() {
    const baseModel = await loadPretrainedModel();
    baseModel.trainable = false;

    const imageInput = tf.input({ shape: [224, 224, 3], name: 'imageInput' });
    const baseModelOutput = baseModel.apply(imageInput);

    const baseModelFlattened = tf.layers.flatten().apply(baseModelOutput);

    const categoryInput = tf.input({ shape: [3], name: 'categoryInput' });
    
    const categoryDense = tf.layers.dense({ units: 32, activation: 'relu', name: 'categoryDense' }).apply(categoryInput);

    const concatenated = tf.layers.concatenate().apply([baseModelFlattened, categoryDense]);

    let x = concatenated;
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
    x = tf.layers.dropout({ rate: 0.5 }).apply(x);

    const output = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x);
    const outputReshaped = tf.layers.reshape({ targetShape: [1] }).apply(output);

    const model = tf.model({ inputs: [imageInput, categoryInput], outputs: outputReshaped });
    console.log(model.summary());

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

const printShapes = {
    onBatchEnd: (batch, logs) => {
        const xsBatch = batch.xs;
        const ysBatch = xsBatch.map(({ xs, ys }) => ys);

        ysBatch.forEach(async (ysTensor) => {
            console.log(`Batch label shape: ${ysTensor.shape || 'unknown'}`);
        });
    }
 }

async function trainModel() {
    const model = await createModel();
    const { trainDataset, valDataset, totalSize } = await createDataset(16);

    // Train the model using the datasets
    await model.fitDataset(trainDataset, {
        epochs: 10,
        validationData: valDataset,
        callbacks: [tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 }), printShapes]
    });

    const modelPath = path.join(__dirname, '..', '_trained_models', 'weldscanner_quality_model_v2');
    await model.save(`file://${modelPath}`);
    console.log('Model trained and saved.');
}

module.exports = {
    trainModel
};
