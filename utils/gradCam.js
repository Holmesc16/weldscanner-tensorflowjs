const tf = require('@tensorflow/tfjs-node');

async function computeGradCAM(model, imageInput, categoryInput) {
    const lastConvLayerName = 'conv_pw_13_relu'
    const lastConvLayer = model.getLayer(lastConvLayerName);
    
    if (!lastConvLayer) {
        throw new Error(`Layer ${lastConvLayerName} not found in model.`);
    }

    const lastConvLayerOutput = lastConvLayer.getOutputAt(0);

    const gradModel = tf.model({
        inputs: model.inputs,
        outputs: [lastConvLayerOutput, model.output]
    });

    const [convOutputs, predictions] = await gradModel.predictOnBatch([imageInput, categoryInput]);

    const convOutputTensor = convOutputs;
    const predictionTensor = predictions;

    const gradFunction = tf.function((inputs) => {
        return tf.tidy(() => {
            const [convOutputs, predictions] = gradModel.apply(inputs);
            const loss = predictions.mean();
            const grads = tf.grad((x) => loss)(convOutputs)
            return [convOutputs, grads]
        })
    });

    const [convOutputsValue, gradsValues] = gradFunction([imageInput, categoryInput]);

    const pooledGrads = tf.mean(gradsValues, [0, 1, 2]);

    const convOutputsMultiplied = convOutputsValue.mul(pooledGrads.expandDims(0).expandDims(0));

    const heatmap = convOutputsMultiplied.mean(-1)

    const heatmapRelu = heatmap.relu();

    const minVal = heatmapRelu.min();
    const maxVal = heatmapRelu.max();
    const heatmapNormalized = heatmapRelu.sub(minVal).div(maxVal.sub(minVal))

    return heatmapNormalized;
}

module.exports = {
    computeGradCAM
}