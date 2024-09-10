const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

module.exports = async (file) => {
    console.log(`Processing image ${JSON.stringify(file)}`);

    if (!file || !file.buffer || file.buffer.length === 0) {
        console.error('Empty image buffer, skipping this file');
        return null;
    }
    try {
        console.log(`Buffer length: ${file.buffer.length}`);
        console.log(`Buffer type: ${typeof file.buffer}`);
        console.log(`Buffer content: ${file.buffer}`);  

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
        console.error('Error processing image:', error);
        return null;
    }
}