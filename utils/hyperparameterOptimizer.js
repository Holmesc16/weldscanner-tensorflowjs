const tf = require('@tensorflow/tfjs-node');
const { createDataset } = require('../controllers/imageController.js');

// Define the search space for hyperparameters
const searchSpace = {
    filters1: [16, 32],
    filters2: [32, 64],
    filters3: [64, 128],
    kernelSize1: [3],
    kernelSize2: [3],
    kernelSize3: [3],
    denseUnits: [16, 32],
    l2: [0.01, 0.001],
    batchSize: [16, 32]
};

// Generate all combinations of hyperparameters
const createHyperparameterCombinations = (space) => {
    const keys = Object.keys(space);
    const values = keys.map(key => space[key]);
    const combinations = cartesianProduct(...values);

    return combinations.map(combination => {
        const params = {};
        combination.forEach((value, index) => {
            params[keys[index]] = value;
        });
        return params;
    });
};

// Helper function to compute the cartesian product of arrays
const cartesianProduct = (...arrays) => {
    return arrays.reduce((acc, curr) => {
        return acc.flatMap(d => curr.map(e => [...d, e]));
    }, [[]]);
};

// Objective function to evaluate hyperparameters
const objective = async (params) => {
    console.log('Evaluating hyperparameters:', params);

    // Create the dataset with the specified batch size
    const { dataset: dataGenerator, totalSize } = await createDataset(params.batchSize);

    console.log(`Total size: ${totalSize}`);
    
    // Split the dataset into training and validation datasets
    const valSize = Math.floor(totalSize * 0.2);
    const trainSize = totalSize - valSize;

    console.log(`Total size: ${totalSize}, Validation size: ${valSize}, Training size: ${trainSize}`);  

    const trainDataset = dataGenerator.take(trainSize);
    const valDataset = dataGenerator.skip(trainSize);


    // Create the model with the current hyperparameters
    const model = tf.sequential();

    // First Convolutional Block
    model.add(tf.layers.conv2d({
        inputShape: [150, 150, 3],
        filters: params.filters1,
        kernelSize: params.kernelSize1,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: params.l2 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Second Convolutional Block
    model.add(tf.layers.conv2d({
        filters: params.filters2,
        kernelSize: params.kernelSize2,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: params.l2 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Third Convolutional Block
    model.add(tf.layers.conv2d({
        filters: params.filters3,
        kernelSize: params.kernelSize3,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: params.l2 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

    // Global Average Pooling instead of Flatten
    model.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));

    // Dense Layers
    model.add(tf.layers.dense({ units: params.denseUnits, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    console.log('Model summary:');
    model.summary();

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    console.log('Model compiled.');
    // Define early stopping callback
    const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 });

    try {
        console.log('Starting model training...');
        // Train the model
        const history = await model.fitDataset(trainDataset, {
            epochs: 10,
            validationData: valDataset,
            callbacks: [earlyStopping]
        });

        // Get the final validation accuracy
        const valAcc = history.history.val_accuracy[history.history.val_accuracy.length - 1];
        console.log('Validation accuracy for parameters', params, ':', valAcc);

        return valAcc; // Return the validation accuracy
    } catch (error) {
        console.error('Error during model training:', error);
        throw error; // Return a very low score to indicate failure
    } finally {
        console.log('Model training completed.');
        // Dispose of the model and variables to free up memory
        model.dispose();
        // tf.disposeVariables();
    }
};

// Function to run hyperparameter optimization
const runHyperparameterOptimization = async () => {
    const hyperparameterCombinations = createHyperparameterCombinations(searchSpace);
    console.log('Number of hyperparameter combinations:', hyperparameterCombinations.length);
    let bestScore = -Infinity;
    let bestParams = null;

    for (const params of hyperparameterCombinations) {
        try {
            console.log('Evaluating parameters:', params);
            const score = await objective(params);
            if (score > bestScore) {
                bestScore = score;
                bestParams = params;
            }
            console.log('Current best score:', bestScore);
        } catch (error) {
            console.error('Error evaluating parameters:', params, error);
        }
    }

    console.log('Best hyperparameters:', bestParams);
    return bestParams;
};

module.exports = runHyperparameterOptimization;
