const { trainModel, evaluateModel } = require('./models/trainer.js');

(async () => {
    try {
        await trainModel();
    } catch (error) {
        console.error('Error during model training:', error);
    } finally {
        await evaluateModel()
            .catch(err => console.error('Error during model evaluation:', err));
    }
})();