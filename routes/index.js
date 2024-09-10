const express = require('express');
const multer = require('multer');
const imageController = require('../controllers/imageController');

const router = express.Router();
const upload = multer({ dest: 'images/' });

router.post('/image', upload.single('image'), imageController.handleImage);
router.post('/train', async (req, res) => {
    try {
        await imageController.trainModel();
        res.json({ message: 'Model trained successfully' });
    }
    catch(err) {
        res.status(500).json({ error: err.message });
    }
})

module.exports = router;