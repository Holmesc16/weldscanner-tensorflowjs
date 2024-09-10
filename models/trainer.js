const tf = require('@tensorflow/tfjs-node');
const tfvis = require('@tensorflow/tfjs-vis');
const { JSDOM } = require('jsdom');
const { processDataInBatches } = require('../controllers/imageController');
const inspectWeights = require('../visualizations/inspectWeights');
const evaluateModel = require('../evaluations/evaluateModel');
const runHyperparameterOptimization = require('../utils/hyperparameterOptimizer');

// Create a mock browser environment
const { window } = new JSDOM(`<!DOCTYPE html><head></head><body></body>`);
global.document = window.document;
global.HTMLElement = window.HTMLElement;

const trainModel = async (dataGenerator, params) => {
    console.log('Starting model training with parameters:', params);

    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [150, 150, 3],
        filters: params.filters,
        kernelSize: params.kernelSize,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: params.l2 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.conv2d({
        filters: params.filters,
        kernelSize: params.kernelSize,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: params.l2 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' })); // Further reduced units
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = { name: 'Model Training', styles: { height: '1000px' }, tab: 'Weldscanner Model' };
    const callbacks = [];

    if (tfvis) {
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
        callbacks.push(fitCallbacks);
    }

    const earlyStoppingCallback = tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: 5 // Reduced patience
    });
    callbacks.push(earlyStoppingCallback, tf.node.tensorBoard('./logs'));

    // Add custom callbacks for logging
    callbacks.push(tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: 5,
        verbose: 1
    }));

    // Custom logging for epochs and batches
    callbacks.push({
        onEpochEnd: async (epoch, logs) => {
            console.log(`Epoch ${epoch + 1} / 10: loss = ${logs.loss}, val_loss = ${logs.val_loss}, accuracy = ${logs.acc}, val_accuracy = ${logs.val_acc}`);
        },
        onBatchEnd: async (batch, logs) => {
            console.log(`Batch ${batch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
        }
    });

    await model.fitDataset(dataGenerator, {
        epochs: 10,
        validationData: dataGenerator,
        callbacks
    });

    console.log('Model training completed.');
    return model;
};

// Export the main function so it can be called from app.js
const main = async () => {
    console.log('Starting data processing...');
    const dataGenerator = await processDataInBatches(); // Ensure this returns a tf.data.Dataset
    console.log('Data processing completed.');

    console.log('Starting hyperparameter search...');
    let bestParams = await runHyperparameterOptimization();
    if (!bestParams) {
        console.log('No best parameters found, using default parameters.');
        bestParams = {
            filters: 32, // Reduced filters
            kernelSize: 3,
            l2: 0.001,
            batchSize: 16 // Reduced batch size
        };
    }
    console.log('Hyperparameter search completed.');
    console.log('Best hyperparameters:', bestParams);

    const model = await trainModel(dataGenerator, bestParams);

    const modelPath = 'file://trained_models/weldscanner_quality_model';
    await model.save(modelPath);
    console.log('Model trained and saved.');

    console.log('Evaluating model...');
    await evaluateModel(model, dataGenerator);
    console.log('Model evaluation completed.');

    console.log('Inspecting weights...');
    await inspectWeights(`${modelPath}/model.json`);
    console.log('Weights inspection completed.');
};

module.exports = { main };
