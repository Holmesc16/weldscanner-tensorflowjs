const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const path = require('path')
const { createDataset } = require('../controllers/imageController.js')

async function loadPretrainedModel() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
    const truncatedModel = tf.model({
        inputs: mobilenet.inputs,
        outputs: mobilenet.layers[mobilenet.layers.length - 4].output
    })
    return truncatedModel
}

async function createModel() {
    const baseModel = await loadPretrainedModel()
    baseModel.trainable = false

    const imageInput = tf.input({ shape: [224, 224, 3], name: 'imageInput' })
    const baseModelOutput = baseModel.apply(imageInput)

    const baseModelFlattened = tf.layers.flatten().apply(baseModelOutput)

    const categoryInput = tf.input({ shape: [3], name: 'categoryInput' })
    
    const categoryDense = tf.layers.dense({ units: 32, activation: 'relu', name: 'categoryDense' }).apply(categoryInput)

    const concatenated = tf.layers.concatenate().apply([baseModelFlattened, categoryDense])

    let x = concatenated
    x = tf.layers.dense({ units: 128, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }).apply(x)
    x = tf.layers.batchNormalization().apply(x) 
    x = tf.layers.dropout({ rate: 0.5 }).apply(x)

    const output = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x)
    const outputReshaped = tf.layers.reshape({ targetShape: [1] }).apply(output)

    const model = tf.model({ inputs: [imageInput, categoryInput], outputs: outputReshaped })
    console.log(model.summary())

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    return model
}

class LearningRateScheduler {
    constructor(schedule) {
        this.schedule = schedule;
    }

    setModel(model) {
        this.model = model; // Ensure the callback is linked to the model
    }

    onEpochBegin(epoch, logs) {
        const newLearningRate = this.schedule(epoch, this.model.optimizer.learningRate);
        this.model.optimizer.learningRate = newLearningRate;
        console.log(`Epoch ${epoch + 1}: Learning rate updated to ${newLearningRate}`);
    }
}

async function trainModel() {
    // const model = await createModel()
    const { trainDataset, valDataset, totalSize } = await createDataset(16)

    const learningRates = [0.001, 0.0001]
    const batchSizes = [16, 32]
    const epochs = 10

    let bestModel
    let bestValAccuracy = 0

    for (const lr of learningRates) {
        for (const batchSize of batchSizes) {
            console.log(`Training with learning rate ${lr} and batch size ${batchSize}`)

            const model = await createModel()
            model.compile({
                optimizer: tf.train.adam(lr),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            })

            const learningRateScheduler = new LearningRateScheduler((epoch, currentLearningRate) => {
                return epoch > 5 ? currentLearningRate : currentLearningRate * 0.9 // reduce learning rate by 10% every epoch after the 5th
            })

            const history = await model.fitDataset(trainDataset, {
                epochs: epochs,
                validationData: valDataset,
                callbacks: [
                    tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 }),
                    learningRateScheduler
                ]
            })

            const valAccuracy = history.history.val_accuracy
                ? history.history.val_accuracy[history.history.val_accuracy.length - 1]
                : 0;
            console.log(`Validation accuracy for learning rate ${lr} and batch size ${batchSize}: ${valAccuracy}`)
            if (valAccuracy > bestValAccuracy) {
                bestValAccuracy = valAccuracy
                bestModel = model
            }   
        }
    }

    const modelPath = path.join(__dirname, '..', '_trained_models', 'weldscanner_quality_model_v2')
    await model.save(`file://${modelPath}`)
    console.log('Model trained and saved with accuracy:', bestValAccuracy)
}

module.exports = {
    trainModel
}
