const processImage = require('../utils/imageProcessor');
const tfModel = require('../models/tfModel');
const tf = require('@tensorflow/tfjs-node');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const { PromisePool } = require('@supercharge/promise-pool');
const sharp = require('sharp');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';

const streamToBuffer = (stream) => {
    const chunks = [];
    return new Promise((resolve, reject) => {
        stream.on('data', (chunk) => chunks.push(chunk));
        stream.on('end', () => resolve(Buffer.concat(chunks)));
        stream.on('error', (err) => reject(err));
    });
};

// Augments an image tensor with various transformations
const augmentImage = async (imageBuffer) => {
    if (!imageBuffer || imageBuffer.length === 0) {
        throw new Error('Invalid image buffer');
    }

    let sharpImage = sharp(imageBuffer);
    const rotation = Math.floor(Math.random() * 80 - 40);
    sharpImage = sharpImage.rotate(rotation);

    if (Math.random() > 0.5) {
        sharpImage = sharpImage.flip();
    }

    if (Math.random() > 0.5) {
        sharpImage = sharpImage.flop();
    }

    const zoomFactor = Math.random() * 0.4 + 0.8;
    const width = Math.round(150 * zoomFactor);
    const height = Math.round(150 * zoomFactor);
    sharpImage = sharpImage.resize(width, height);

    const augmentedBuffer = await sharpImage.toBuffer();

    const imgTensor = tf.node.decodeImage(augmentedBuffer, 3)
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(255))
        .sub(tf.scalar(0.5))
        .div(tf.scalar(0.5));

    augmentedBuffer.dispose();
    sharpImage.dispose();

    return imgTensor;
};

// Load images in batches and apply augmentation
const loadImagesInBatches = async (folderPath, label, batchSize = 16, numAugmentations = 5) => {
    const params = { Bucket: bucket, Prefix: folderPath };
    const data = await s3Client.send(new ListObjectsV2Command(params));

    if (!data.Contents || data.Contents.length === 0) {
        console.warn(`No images found in S3 folder: ${folderPath}`);
        return [];
    }

    const imageKeys = data.Contents.map(obj => obj.Key);
    const images = [];

    await PromisePool
        .for(imageKeys)
        .withConcurrency(batchSize)
        .process(async (imageKey, index) => {
            try {
                const getObjectParams = { Bucket: bucket, Key: imageKey };
                const imgData = await s3Client.send(new GetObjectCommand(getObjectParams));
                const imgBuffer = await streamToBuffer(imgData.Body);
                
                if (!imgBuffer || imgBuffer.length === 0) {
                    console.error(`Empty image buffer for S3 Key: ${imageKey}`);
                    return;
                }
                const imgTensor = await processImage({ buffer: imgBuffer });

                if (imgTensor) {
                    images.push({ tensor: imgTensor, label });

                    for (let i = 0; i < numAugmentations; i++) {
                        const augmentedTensor = await augmentImage(imgBuffer);
                        images.push({ tensor: augmentedTensor, label });
                        console.log(`Augmented image #${index + 1}-${i + 1} from S3 Key: ${imageKey}`);
                        augmentedTensor.dispose();
                    }   
                    imgTensor.dispose();
                }
            } catch (error) {
                console.error(`Error processing image from S3 Key: ${imageKey}`, error);
            }
        });

    return images;
};

// Function to process data in batches, including loading and augmenting images
exports.processDataInBatches = async (batchSize = 16, numAugmentations = 5) => {
    console.log('Starting data processing...');
    const categories = ['butt', 'saddle', 'electro'];
    const xsList = [];
    const ysList = [];

    await Promise.all(categories.map(async (category) => {
        console.log(`Processing category: ${category}`);
        const passImages = await loadImagesInBatches(`${category}/pass`, 1, batchSize, numAugmentations);
        const failImages = await loadImagesInBatches(`${category}/fail`, 0, batchSize, numAugmentations);

        if (passImages.length === 0 && failImages.length === 0) {
            console.warn(`No images found for category: ${category}`);
            return;
        }

        const data = [...passImages, ...failImages];
        if (data.length === 0) {
            console.warn(`No valid data found for category: ${category}`);
            return;
        };

        tf.util.shuffle(data);

        const tensors = data.map(d => d.tensor);
        const labels = data.map(d => d.label);

        if (tensors.length > 0) {
            xsList.push(tf.concat(tensors, 0));
            ysList.push(tf.tensor1d(labels, 'int32'));
            tensors.forEach(tensor => tensor.dispose());
        } else {
            console.warn(`No tensors generated for category: ${category}`);
        }
    }));

    if (xsList.length === 0 || ysList.length === 0) {
        throw new Error('No valid data to process. Ensure that images are available in S3.');
    }

    const xs = tf.concat(xsList);
    const ys = tf.concat(ysList);

    console.log('Data processing completed.');
    return { xs, ys };
};

// Handle image prediction
exports.handleImage = async (req, res) => {
    try {
        console.log('Handling image prediction...');
        const img = await processImage(req.file);

        if (!img) {
            return res.status(400).json({ error: 'Invalid image data' });
        }

        const prediction = await tfModel.predict(img);
        const result = prediction.dataSync()[0] > 0.5 ? 'Pass' : 'Fail';

        console.log(`Prediction: ${result}`);
        res.json({ result });

        img.dispose();
        prediction.dispose();
    } catch (err) {
        console.error('Error handling image prediction:', err);
        res.status(500).json({ error: err.message });
    }
};
