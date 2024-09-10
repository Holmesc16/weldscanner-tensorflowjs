const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

module.exports = async (file) => {
    console.log({ file })
    const imageBuffer = await sharp(file.path)
        .resize(150, 150)
        .toFormat('jpeg')
        .toBuffer();

    return tf.node.decodeImage(imageBuffer, 3)
        .expandDims()
        .toFloat()
        .div(tf.scalar(255));
}