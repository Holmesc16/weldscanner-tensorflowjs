// routes/index.js

const express = require('express');
const multer = require('multer');
const { handlePrediction } = require('../controllers/imageController');

const router = express.Router();

// Use memory storage for multer
const storage = multer.memoryStorage();
const upload = multer({ storage });

router.post('/image', upload.single('image'), handlePrediction);

module.exports = router;
