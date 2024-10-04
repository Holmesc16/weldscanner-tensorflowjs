const tf = require('@tensorflow/tfjs-node');
const { createDataset } = require('../controllers/imageController');

// Define the search space for hyperparameters
const searchSpace = {
    filters: [32, 64],
    kernelSize: [3, 5],
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
    const dataGenerator = await createDataset(params.batchSize);

    // Split the dataset into training and validation datasets
    const totalSize = await dataGenerator.size().then(size => size);
    const valSize = Math.floor(totalSize * 0.2);
    const trainSize = totalSize - valSize;

    const trainDataset = dataGenerator.take(trainSize);
    const valDataset = dataGenerator.skip(trainSize);

    // Create the model with the current hyperparameters
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

    // Define early stopping callback
    const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 });

    try {
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
        return -Infinity; // Return a very low score to indicate failure
    } finally {
        // Dispose of the model and variables to free up memory
        model.dispose();
        tf.disposeVariables();
    }
};

// Function to run hyperparameter optimization
const runHyperparameterOptimization = async () => {
    const hyperparameterCombinations = createHyperparameterCombinations(searchSpace);
    let bestScore = -Infinity;
    let bestParams = null;

    for (const params of hyperparameterCombinations) {
        try {
            const score = await objective(params);
            if (score > bestScore) {
                bestScore = score;
                bestParams = params;
            }
        } catch (error) {
            console.error('Error evaluating parameters:', params, error);
        }
    }

    console.log('Best hyperparameters:', bestParams);
    return bestParams;
};

module.exports = runHyperparameterOptimization;
