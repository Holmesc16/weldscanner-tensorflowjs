const { trainModel } = require('./models/trainer.js');

(async () => {
    try {
        await trainModel();
    } catch (error) {
        console.error('Error during model training:', error);
    }
})();