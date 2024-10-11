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

        if (imgTensor.shape.length !== 3 || imgTensor.shape[2] !== 3) {
            console.error('Invalid image tensor shape:', imgTensor.shape);
            return null;
        }

        console.log(`Processed image tensor shape: ${imgTensor.shape}`);
        return imgTensor;
    } catch (error) {
        console.error('Error processing image:', error.message);
        return null;
    }
};

module.exports = processImage;