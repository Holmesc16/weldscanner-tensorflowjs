const tf = require('@tensorflow/tfjs-node');
const { processDataInBatches } = require('../controllers/imageController');
const { combinations } = require('simple-statistics');

const searchSpace = {
    filters: [32, 64, 128, 256],
    kernelSize: [3, 5, 7, 9],
    l2: [0.01, 0.001, 0.0001, 0.00001],
    batchSize: [16, 32, 64, 128]
};

const createHyperparameterCombinations = (space) => {
    const keys = Object.keys(space);
    const values = keys.map(key => space[key]);
    const allCombinations = combinations(values.flat());
    return allCombinations.map(combination => {
        const params = {};
        combination.forEach((value, index) => {
            params[keys[index % keys.length]] = value;
        });
        return params;
    });
};

const objective = async (params) => {
    console.log('Evaluating hyperparameters:', params);
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
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    const { xs, ys } = await processDataInBatches(params.batchSize);

    const earlyStopping = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 });

    const history = await model.fit(xs, ys, {
        epochs: 50,  // Increase epochs to allow early stopping to take effect
        validationSplit: 0.2,
        batchSize: params.batchSize,
        callbacks: [earlyStopping]
    });

    const valAcc = history.history.val_accuracy[history.history.val_accuracy.length - 1];
    console.log('Validation accuracy for parameters', params, ':', valAcc);

    return -valAcc;
};

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