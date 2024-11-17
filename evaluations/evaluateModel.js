const tfvis = require('@tensorflow/tfjs-vis');

const evaluateModel = async (model, xs, ys) => {
    try {
        // Validate and reshape inputs
        if (!xs || !xs.imageInput || !xs.categoryInput) {
            throw new Error('Input tensors xs must include imageInput and categoryInput.');
        }

        // Ensure imageInput has shape [1, 224, 224, 3]
        const imageInput = xs.imageInput.shape.length === 3 ? xs.imageInput.expandDims(0) : xs.imageInput;
        if (imageInput.shape.length !== 4 || imageInput.shape[1] !== 224 || imageInput.shape[2] !== 224 || imageInput.shape[3] !== 3) {
            throw new Error('imageInput must have shape [1, 224, 224, 3].');
        }

        // Ensure categoryInput has shape [1, 3]
        const categoryInput = xs.categoryInput.shape.length === 1 ? xs.categoryInput.expandDims(0) : xs.categoryInput;
        if (categoryInput.shape.length !== 2 || categoryInput.shape[1] !== 3) {
            throw new Error('categoryInput must have shape [1, 3].');
        }

        // Ensure ys has shape [1]
        const labels = ys.shape.length === 0 ? ys.expandDims(0) : ys;
        if (labels.shape.length !== 1) {
            throw new Error('ys must have shape [1].');
        }

        // Make predictions
        const predictions = await model.predict([imageInput, categoryInput]);
        const roundedPredictions = predictions.round();

        // Calculate confusion matrix
        const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, roundedPredictions);
        await tfvis.show.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' }, confusionMatrix);

    } catch (error) {
        console.error('Error during model evaluation:', error.message);
    }
}

module.exports = evaluateModel;