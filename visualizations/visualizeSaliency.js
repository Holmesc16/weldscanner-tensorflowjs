const tf = require('@tensorflow/tfjs-node');

const visualizeSaliency = async (model, imageTensor) => {
    const classIndex = tf.scalar(0, 'int32');

    const lossFunction = (imageTensor) => (
        tf.losses.sigmoidCrossEntropy(classIndex, model.predict(imageTensor))
    )

    const grads = tf.grads(lossFunction);
    const saliency = grads(imageTensor).abs();

    console.log('Saliency Map: ');
    saliency.print();
}


module.exports = visualizeSaliency;