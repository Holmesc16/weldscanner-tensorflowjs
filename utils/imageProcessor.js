// utils/imageProcessor.js

const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

const targetWidth = 224; // Update to match MobileNet input size
const targetHeight = 224;

const processImage = async (file) => {
    try {
        console.log(`Processing image with buffer length: ${file.buffer.length}`);
        if (!file || !file.buffer || file.buffer.length === 0) {
            throw new Error('Empty or invalid image buffer');
        }

        const resizedBuffer = await sharp(file.buffer)
            .resize(targetWidth, targetHeight)
            .toBuffer();

        const imgTensor = tf.node.decodeImage(resizedBuffer, 3)
            .toFloat()
            .div(255.0); // Normalize to [0, 1]

        return imgTensor;
    }
    catch (error) {
        console.error('Error processing image:', error.message);
        return null;
    } finally {
        if (file.buffer) {
            file.buffer = null;
        }
    }
};

module.exports = processImage;