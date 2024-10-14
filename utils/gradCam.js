const tf = require('@tensorflow/tfjs-node');

const findLayersByName = (model, layerName) => {
    for (const layer of model.layers) {
        if (layer.name === layerName) {
            return layer;
        }
        else if(layer instanceof tf.Sequential || layer instanceof tf.LayersModel) {
            const nestedLayer = findLayersByName(layer, layerName);
            if (nestedLayer) {
                return nestedLayer;
            }
        }
    }
    return null;
};

async function computeGradCAM(model, imageInput, categoryInput) {
    const baseModel = model.getLayer('model1');
    // get the base model
    console.log('Base Model Layers:');
    baseModel.layers.forEach((layer, index) => {
        console.log(`Layer ${index}: ${layer.name}, Output Shape: ${layer.outputShape}`);
    });

    // conv_pw_13_relu is at index 81 of the base model layers, lets find it
    const lastConvLayerName = 'conv_pw_13_relu'
    const lastConvLayer = findLayersByName(baseModel, lastConvLayerName);

    if (!lastConvLayer) {
        throw new Error(`Could not find layer ${lastConvLayerName} in the base model.`);
    }

    const lastConvLayerOutput = lastConvLayer.output;
    console.log('Last Conv Layer Output Shape:', lastConvLayerOutput.shape);

    const gradModel = tf.model({
        inputs: model.inputs,
        outputs: [lastConvLayerOutput, model.output]
    });

    const [convOutputs, predictions] = await gradModel.predictOnBatch([imageInput, categoryInput]);

    const grads = tf.grad(inputs => {
        const [convOutputs, predictions] = gradModel.apply(inputs);
        const loss = predictions.mean();
        return loss;
    })([imageInput, categoryInput])[0];

    console.log('Grad Model Layers:');
    gradModel.layers.forEach((layer, index) => {
        console.log(`Layer ${index}: ${layer.name}, Output Shape: ${layer.outputShape}`);
    });

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