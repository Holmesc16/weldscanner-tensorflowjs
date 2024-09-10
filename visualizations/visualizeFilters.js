const tf = require('@tensorflow/tfjs-node');

const visualizeFilters = async (model) => {
    model.layers.forEach((layer, i) => {
        if (layer.getClassName === 'Conv2D') {
            const filters = layer.getWeights()[0];
            console.log(`Layer ${i + 1} ${layer.name} filters:`);
            filters.print();
        }
    })
}

module.exports = visualizeFilters;