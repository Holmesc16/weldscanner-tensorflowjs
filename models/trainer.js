const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const { createDataset } = require('../controllers/imageController.js');
const runHyperparameterOptimization = require('../utils/hyperparameterOptimizer.js');

const trainModel = async () => {
    try {
        await tf.ready();

        console.log('Starting hyperparameter optimization...');
        const bestParams = await runHyperparameterOptimization();
        console.log('Best hyperparameters:', bestParams);

        console.log('Starting data processing...');
        const { dataset: dataGenerator } = await createDataset(bestParams.batchSize);
        console.log('Data processing completed.');

        // Fetch one batch from the dataset without consuming the original dataset
        await dataGenerator.take(1).forEachAsync(({ xs, ys }) => {
            console.log('Input xs shape:', xs.shape); // Expected: [batch_size, height, width, channels]
            console.log('Input ys shape:', ys.shape); // Expected: [batch_size, 1]
            console.log('Sample ys dtype:', ys.dtype);
        });

        // Create model with best hyperparameters
        const model = tf.sequential();

        // First Convolutional Block
        model.add(tf.layers.conv2d({
            inputShape: [150, 150, 3],
            filters: bestParams.filters1,
            kernelSize: bestParams.kernelSize1,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: bestParams.l2 })
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Second Convolutional Block
        model.add(tf.layers.conv2d({
            filters: bestParams.filters2,
            kernelSize: bestParams.kernelSize2,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: bestParams.l2 })
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Third Convolutional Block
        model.add(tf.layers.conv2d({
            filters: bestParams.filters3,
            kernelSize: bestParams.kernelSize3,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: bestParams.l2 })
        }));
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Global Average Pooling instead of Flatten
        model.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));

        // Dense Layers
        model.add(tf.layers.dense({ units: bestParams.denseUnits, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: (yTrue, yPred) => {
                const loss = tf.losses.sigmoidCrossEntropy(yTrue, yPred);
                loss.data().then(value => {
                    console.log('Batch loss:', value);
                });
                return loss;
            },
            // metrics: ['accuracy']
        });

        // Define callbacks
        const callbacks = [
            tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 }),
            tf.node.tensorBoard('./logs')
        ];

        // Verify model output shape
        const testInput = tf.zeros([bestParams.batchSize, 150, 150, 3]);
        const testOutput = model.predict(testInput);
        console.log('Model output shape:', testOutput.shape); // Should be [batch_size, 1]
        testInput.dispose();
        testOutput.dispose();

        // Train the model using the dataset
        await model.fitDataset(dataGenerator, {
            epochs: 10, // Adjust as needed
            validationData: 0.2,
            // callbacks: callbacks
        });

        // Save the model
        const modelPath = path.join(__dirname, '..', 'trained_models', 'weldscanner_quality_model');
        await model.save(`file://${modelPath}`);
        console.log('Model trained and saved.');

        // Dispose the model to free up memory
        model.dispose();
    } catch (err) {
        console.error('Error during training:', err);
    }
};

trainModel().catch(err => {
    console.error('Error during training:', err);
});

module.exports = { trainModel };
