// models/trainer.js

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { createDataset } = require('../controllers/imageController.js');
const runHyperparameterOptimization = require('../utils/hyperparameterOptimizer.js');

const trainModel = async () => {
    await tf.ready();
    
    console.log('Starting hyperparameter optimization...');
    const bestParams = await runHyperparameterOptimization();
    console.log('Best hyperparameters:', bestParams);

    console.log('Starting data processing...');
    const dataGenerator = await createDataset(bestParams.batchSize);
    console.log('Data processing completed.');

    // Create model with best hyperparameters
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [150, 150, 3],
        filters: bestParams.filters,
        kernelSize: bestParams.kernelSize,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: bestParams.l2 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Define callbacks
    const callbacks = [
        tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 }),
        tf.node.tensorBoard('./logs')
    ];

    // Train the model using the dataset
    await model.fitDataset(dataGenerator, {
        epochs: 10, // Adjust as needed
        validationSplit: 0.2, // Splits the data into 80% training and 20% validation
        callbacks: callbacks
    });

    // Save the model
    const modelPath = path.join(__dirname, '..', 'trained_models', 'weldscanner_quality_model');
    await model.save(`file://${modelPath}`);
    console.log('Model trained and saved.');

    // Dispose the model to free up memory
    model.dispose();
    tf.disposeVariables();
};

trainModel().catch(err => {
    console.error('Error during training:', err);
});

module.exports = { trainModel };
