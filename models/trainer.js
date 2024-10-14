const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const path = require('path');
const { createDataset } = require('../controllers/imageController.js');

function precision() {
    return function(yTrue, yPred) {
        const truePositives = tf.sum(tf.mul(yTrue, tf.round(yPred)));
        const falsePositives = tf.sum(tf.mul(yTrue, tf.sub(1, tf.round(yPred))));
        const falseNegatives = tf.sum(tf.mul(tf.sub(1, yTrue), tf.round(yPred)));
        const predictedPositives = tf.sum(tf.round(yPred));
        const predictedNegatives = tf.sum(tf.sub(1, tf.round(yPred)));
        const precision = tf.divNoNan(truePositives, tf.add(truePositives, predictedPositives));
        return precision;
    }
}

function recall() {
    return function(yTrue, yPred) {
        const truePositives = tf.sum(tf.mul(yTrue, tf.round(yPred)));
        const possiblePositives = tf.sum(yTrue);
        const recall = tf.divNoNan(truePositives, possiblePositives);
        return recall;
    }
};

function f1Score() {
    const precision = precision();
    const recall = recall();
    return function(yTrue, yPred) {
        const p = precision(yTrue, yPred);
        const r = recall(yTrue, yPred);
        const f1 = tf.divNoNan(tf.mul(2, tf.mul(p, r)), tf.add(p, r));
        return f1;
    }
}

async function loadPretrainedModel() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const truncatedModel = tf.model({
        inputs: mobilenet.inputs,
        outputs: mobilenet.layers[mobilenet.layers.length - 4].output
    });
    return truncatedModel;
}

async function createModel(dropoutRate = 0.5, regularizerStrength = 0.001) {
    const baseModel = await loadPretrainedModel();
    baseModel.trainable = false;

    const imageInput = tf.input({ shape: [224, 224, 3], name: 'imageInput' });

    const lastConvLayerName ='conv_pw_13_relu' // name of last convolutional layer in mobilenet
    const lastConvLayer = baseModel.getLayer(lastConvLayerName);
    const lastConvLayerOutput = lastConvLayer.output;

    const baseModelOutput = baseModel.apply(imageInput);
    const baseModelFlattened = tf.layers.flatten().apply(baseModelOutput);

    const categoryInput = tf.input({ shape: [3], name: 'categoryInput' });
    
    const categoryDense = tf.layers.dense({ units: 32, activation: 'relu', name: 'categoryDense' }).apply(categoryInput);

    const concatenated = tf.layers.concatenate().apply([baseModelFlattened, categoryDense]);

    let x = concatenated;
    const regularizer = tf.regularizers.l2({ l2: regularizerStrength });
    x = tf.layers.dense({ units: 128, activation: 'relu', kernelRegularizer: regularizer, name: 'dense_1'}).apply(x);
    x = tf.layers.dropout({ rate: dropoutRate, name: 'dropout_1'}).apply(x);
    x = tf.layers.dense({ units: 64, activation: 'relu', kernelRegularizer: regularizer, name: 'dense_2'}).apply(x);
    x = tf.layers.dropout({ rate: dropoutRate, name: 'dropout_2'}).apply(x);

    const output = tf.layers.dense({ units: 1, activation: 'sigmoid', name: 'output' }).apply(x);
    const outputReshaped = tf.layers.reshape({ targetShape: [1] }).apply(output);

    const model = tf.model({ inputs: [imageInput, categoryInput], outputs: outputReshaped });
    console.log(model.summary());

    model.lastConvLayerOutput = lastConvLayerOutput;

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy', precision(), recall(), f1Score()]
    });

    return model;
}

// async function evaluateModel() {
//     const modelPath = path.join(__dirname, '..', '_trained_models', 'weldscanner_quality_model_v2');
//     const model = await tf.loadLayersModel(`file://${modelPath}`);

//     const batchSize = 16;
//     const { testDataset, totalSize} = await createTestDataset(batchSize)

//     const evalResult = await model.evaluateDataset(testDataset);
//     console.log(`Test Loss: ${evalResult?.[0]?.dataSync()?.[0]}, Test Accuracy: ${evalResult?.[1]?.dataSync()?.[0]}`);
// }

async function trainModel(learningRate, batchSize, dropoutRate, regularizerStrength) {
    const model = await createModel(dropoutRate, regularizerStrength);
    const { trainDataset, valDataset, totalSize } = await createDataset(batchSize);

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    const history = await model.fitDataset(trainDataset, {
        epochs: 10,
        validationData: valDataset,
        callbacks: [tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })]
    });

    const modelName = `model_lr${learningRate}_bs${batchSize}_dr${dropoutRate}_rs${regularizerStrength}`
    const modelPath = path.join(__dirname, '..', '_trained_models', modelName);
    await model.save(`file://${modelPath}`);
    console.log('Model trained and saved.');
}

module.exports = {
    trainModel
};
