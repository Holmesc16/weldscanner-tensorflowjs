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
    const lastConvLayerOutput = model.lastConvLayerOutput;

    const gradModel = tf.model({
        inputs: model.inputs,
        outputs: [lastConvLayerOutput, model.output]
    });
    // record operations for automatic differentiation
    const tape = await tf.engine().startScope();
    
    const [convOutputs, predictions] = await gradModel.predictOnBatch([imageInput, categoryInput]);

    // compute gradients of predictions with respect to convOutputs / last convolutional layer
    const grads = tf.grad(inputs => {
        const [convOutputs, predictions] = gradModel.apply(inputs);
        return predictions.mean();
    })([imageInput, categoryInput])[0];

    // compute guided gradients
    const guidedGrads = grads.mul(convOutputs.greater(0))

    // compute weights using mean of guided gradients
    const weights = tf.mean(guidedGrads, [0, 1, 2]);

    // compute the grad-CAM heatmap
    const cam = convOutputs.mul(weights).mean(2);

    // apply ReLU to heatmap
    const heatmap = cam.relu();

    // normalize heatmap to [0, 1]
    const headMapNormalized = heatmap.div(tf.max(heatmap));

    // end the tape scope
    tf.engine().endScope(tape);

    return headMapNormalized;   
}

module.exports = {
    computeGradCAM,
    findLayersByName
}