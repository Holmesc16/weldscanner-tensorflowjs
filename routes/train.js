const { trainModel } = require('../controllers/imageController');

const router = express.Router();

router.post('/train', async (req, res) => {
    try {
        await trainModel();
    } catch (error) {
        console.error('Error during model training:', error);
    }
});
