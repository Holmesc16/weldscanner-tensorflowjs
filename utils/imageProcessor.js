// utils/imageProcessor.js

const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

const targetWidth = 150;
const targetHeight = 150;

const processImage = async (file) => {
    try {
        console.log(`Processing image with buffer length: ${file.buffer.length}`);
        if (!file || !file.buffer || file.buffer.length === 0) {
            throw new Error('Empty or invalid image buffer');
        }

        const resizedBuffer = await sharp(file.buffer)
            .resize(targetWidth, targetHeight)
            .toFormat('jpeg')
            .toBuffer();

        const tensor = tf.tidy(() => {
            return tf.node.decodeImage(resizedBuffer, 3)
                .expandDims()
                .toFloat()
                .div(255.0)
                .sub(0.5)
                .div(0.5);
        });

        return tensor;
    } catch (error) {
        console.error('Error processing image:', error.message);
        return null;
    } finally {
        if (file.buffer) {
            file.buffer = null;
        }
    }
};

module.exports = processImage;
