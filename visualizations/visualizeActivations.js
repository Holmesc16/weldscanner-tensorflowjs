const tf = require('@tensorflow/tfjs-node');

const visualizeActivations = async (modelPath, imageTensor) => {
    const model = await tf.loadLayersModel(`file://${modelPath}`);
    
    const layerOutputs = model.layers.map(layer => layer.output);
    const activationModel = tf.model({ inputs: model.inputs, outputs: layerOutputs });

    const activations = activationModel.predict(imageTensor);
    activations.forEach((activation, i) => {
        console.log(`Layer ${i + 1} ${model.layers[i].name}`);
        console.log({activation});
        activation.print();
    });
}

module.exports = visualizeActivations;
