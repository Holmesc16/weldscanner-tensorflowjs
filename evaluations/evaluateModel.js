const tfvis = require('@tensorflow/tfjs-vis');

const evaluateModel = async (model, xs, ys) => {
    const predictions = await model.predict(xs);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(ys, predictions.round());
    await tfvis.show.confusionMatrix({ name: 'Confusion Matrix', tab: 'Evaluation' }, confusionMatrix);
}

module.exports = evaluateModel;