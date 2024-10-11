const express = require('express');
const multer = require('multer');
const { handlePrediction, trainModel } = require('../controllers/imageController');

const router = express.Router();
const upload = multer({ dest: 'images/' });

router.post('/image', upload.single('image'), handlePrediction);
router.post('/train', async (req, res) => {
    try {
        await trainModel();
        res.json({ message: 'Model trained successfully' });
    }
    catch(err) {
        res.status(500).json({ error: err.message });
    }
})

module.exports = router;