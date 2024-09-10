const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

module.exports = async (file) => {
    console.log(`Processing image ${file}`);

    if (!file.buffer || file.buffer.length === 0) {
        console.error('Empty image buffer, skipping this file');
        return null;
    }
    try {
        const imageBuffer = await sharp(file.buffer)
            .resize(150, 150)
            .toFormat('jpeg')
            .toBuffer();

        return tf.node.decodeImage(imageBuffer, 3)
            .expandDims()
            .toFloat()
            .div(tf.scalar(255));
    } catch (error) {
        console.error('Error processing image:', error);
        return null;
    }
}