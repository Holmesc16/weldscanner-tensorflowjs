const tf = require('@tensorflow/tfjs-node');
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

    const model = tf.sequential();
    model.add(baseModel);
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

async function trainModel() {
    const model = await createModel();
    const { dataset, totalSize } = await createDataset(16);

    const totalBatches = await dataset.cardinality().then(num => num)
    const valBatches = Math.floor(totalBatches * 0.2)
    const trainBatches = totalBatches - valBatches;

    const valDataset = dataset.take(valBatches);
    const trainDataset = dataset.skip(valBatches);

    const shuffledTrainDataset = trainDataset.shuffle(1000).batch(16);
    const batchedValDataset = valDataset.batch(16);

    await shuffledTrainDataset.forEachAsync(sample => {
        if (!sample.xs || !sample.ys) {
            console.error('Invalid sample detected, skipping...');
        }
        console.log('shuffled training dataset: ', sample.xs.shape, sample.ys.shape);
    });

    await batchedValDataset.forEachAsync(sample => {
        if (!sample.xs || !sample.ys) {
            console.error('Invalid sample detected, skipping...');
        }
        console.log('batched validation dataset: ', sample.xs.shape, sample.ys.shape);
    });
    
    await model.fitDataset(shuffledTrainDataset, {
        epochs: 10,
        validationData: batchedValDataset,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 })
    });

    const modelPath = path.join(__dirname, '..', 'trained_models', 'weldscanner_quality_model');
    await model.save(`file://${modelPath}`);
    console.log('Model trained and saved.');
}

module.exports = {
    trainModel
};