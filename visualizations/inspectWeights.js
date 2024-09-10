const tf = require('@tensorflow/tfjs-node');

const inspectWeights = async (modelPath) => {
    const model = await tf.loadLayersModel(`file://${modelPath}`);
    
    model.layers.forEach((layer, i) => {
        console.log(`Layer ${i + 1} ${layer.name}`);
        layer.getWeights().forEach((weight, j) => {
            console.log(`Weight ${j + 1}`);
            weight.print();
        })
    });
}

module.exports = inspectWeights;