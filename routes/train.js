const { trainModel } = require('../controllers/imageController');
const { learningRates, batchSizes, dropoutRates, regularizerStrengths } = require('../utils/hyperparameters');

const router = express.Router();

router.post('/train', async (req, res) => {
    try {
        for (const learningRate of learningRates) {
            for (const batchSize of batchSizes) {
                for (const dropoutRate of dropoutRates) {
                    for (const regularizerStrength of regularizerStrengths) {
                        await trainModel(learningRate, batchSize, dropoutRate, regularizerStrength);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error during model training:', error);
    }
});
