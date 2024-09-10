const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

module.exports = async (file) => {
    try {
        console.log(`Processing image with buffer length: ${file.buffer.length}`);
        if (!file || !file.buffer || file.buffer.length === 0) {
            throw new Error('Empty or invalid image buffer');
        }

        const imageBuffer = await sharp(file.buffer)
            .resize(150, 150)
            .toFormat('jpeg')
            .toBuffer();

        const tensor = tf.node.decodeImage(imageBuffer, 3)
            .expandDims()
            .toFloat()
            .div(tf.scalar(255));

        return tensor;
    } catch (error) {
        console.error('Error processing image:', error.message);
        return null;
    }
};