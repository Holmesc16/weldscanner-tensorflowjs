const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

module.exports = async (file) => {
    let tensor;
    try {
        console.log(`Processing image with buffer length: ${file.buffer.length}`);
        if (!file || !file.buffer || file.buffer.length === 0) {
            throw new Error('Empty or invalid image buffer');
        }

        const targetWidth = 150;
        const targetHeight = 150;

        const resizedBuffer = await sharp(file.buffer)
            .resize(targetWidth, targetHeight)
            .toFormat('jpeg')
            .toBuffer();

        tensor = tf.node.decodeImage(resizedBuffer, 3)
            .expandDims()
            .toFloat()
            .div(tf.scalar(255))
            .sub(tf.scalar(0.5))
            .div(tf.scalar(0.5));

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