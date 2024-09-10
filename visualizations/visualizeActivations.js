const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

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
    
    // const imageBuffer = fs.readFileSync(imagePath);
    // const image = tf.node.decodeImage(imageBuffer, 3)
    //     .resizeNearestNeighbor([150, 150])
    //     .expandDims()
    //     .toFloat()
    //     .div(tf.scalar(255));
    
    // const layerOutputs = model.layers.map(layer => layer.output);
    // const activationModel = tf.model({ inputs: model.inputs, outputs: layerOutputs });

    // const activations = activationModel.predict(image);
    // activations.forEach((activation, i) => {
    //     console.log(`Layer ${i + 1} ${model.layers[i].name}`);
    //     console.log({activation});
    //     activation.print();
    // });
}

module.exports = visualizeActivations;
