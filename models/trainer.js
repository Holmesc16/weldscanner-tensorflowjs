const tf = require('@tensorflow/tfjs-node');

// Load the pre-trained MobileNet model
async function loadPretrainedModel() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    // Remove the top layers
    const truncatedModel = tf.model({
        inputs: mobilenet.inputs,
        outputs: mobilenet.layers[mobilenet.layers.length - 4].output // Adjust based on the layer you want to truncate at
    });
    return truncatedModel;
}

async function createModel() {
    const baseModel = await loadPretrainedModel();

    // Freeze the base model layers
    baseModel.trainable = false;

    // Add new layers on top of the base model
    const model = tf.sequential();
    model.add(baseModel);
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Compile the model
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

async function trainModel() {
    const model = await createModel();
    const { dataset, totalSize } = await createDataset(16); // Adjust batch size as needed

    // Train the model
    await model.fitDataset(dataset, {
        epochs: 10,
        validationSplit: 0.2,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5 })
    });

    // Save the model
    const modelPath = path.join(__dirname, '..', 'trained_models', 'weldscanner_quality_model');
    await model.save(`file://${modelPath}`);
    console.log('Model trained and saved.');
}

trainModel().catch(err => {
    console.error('Error during training:', err);
});