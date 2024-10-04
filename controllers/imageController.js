const processImage = require('../utils/imageProcessor.js');
const tfModel = require('../models/tfModel.js');
const tf = require('@tensorflow/tfjs-node');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const sharp = require('sharp');
const async = require('async');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';
const categories = ['butt', 'saddle', 'electro'];
const targetWidth = 150;
const targetHeight = 150;
const numAugmentations = 5;

const streamToBuffer = (stream) => {
    const chunks = [];
    return new Promise((resolve, reject) => {
        stream.on('data', (chunk) => chunks.push(chunk));
        stream.on('end', () => resolve(Buffer.concat(chunks)));
        stream.on('error', (err) => reject(err));
    });
};

// Augment an image buffer with various transformations
const augmentImage = async (imageBuffer) => {
    if (!imageBuffer || imageBuffer.length === 0) {
        throw new Error('Invalid image buffer');
    }

    try {
        let sharpImage = sharp(imageBuffer);

        // Apply random transformations
        const rotation = Math.floor(Math.random() * 80 - 40);
        sharpImage = sharpImage.rotate(rotation);

        if (Math.random() > 0.5) sharpImage = sharpImage.flip();
        if (Math.random() > 0.5) sharpImage = sharpImage.flop();

        // Resize the image
        sharpImage = sharpImage.resize(targetWidth, targetHeight);

        // Convert to buffer
        const augmentedBuffer = await sharpImage.toBuffer();

        // Decode image buffer to tensor
        const imgTensor = tf.node.decodeImage(augmentedBuffer, 3)
            .toFloat()
            .div(255.0)
            .sub(0.5)
            .div(0.5); // Shape: [height, width, channels]

        return imgTensor;
    } catch (error) {
        console.error('Error augmenting image:', error.message);
        throw error;
    }
};

// Function to create dataset using tf.data API
exports.createDataset = async (batchSize) => {
    console.log('Starting data processing...');

    // Helper function to get all image keys from S3
    const getAllImageKeys = async () => {
        const imageEntries = [];

        for (const category of categories) {
            for (const labelName of ['pass', 'fail']) {
                const label = labelName === 'pass' ? 1 : 0;
                const folderPath = `${category}/${labelName}`;
                const params = { Bucket: bucket, Prefix: folderPath };
                const data = await s3Client.send(new ListObjectsV2Command(params));

                if (!data.Contents || data.Contents.length === 0) {
                    console.warn(`No images found in S3 folder: ${folderPath}`);
                    continue;
                }

                const entries = data.Contents.map(obj => ({ Key: obj.Key, label }));
                imageEntries.push(...entries);
            }
        }

        return imageEntries;
    };

    const imageEntries = await getAllImageKeys();

    if (imageEntries.length === 0) {
        throw new Error('No valid data to process. Ensure that images are available in S3.');
    }

    // Shuffle the image entries to randomize the dataset
    tf.util.shuffle(imageEntries);

    // Create an array to store processed data
    const dataSamples = [];
    let processingError = null;

    // Process images and augmentations
    for (const entry of imageEntries) {
        try {
            const { Key: imageKey, label } = entry;

            const getObjectParams = { Bucket: bucket, Key: imageKey };
            const imgData = await s3Client.send(new GetObjectCommand(getObjectParams));
            const imgBuffer = await streamToBuffer(imgData.Body);

            if (!imgBuffer || imgBuffer.length === 0) {
                console.error(`Empty image buffer for S3 Key: ${imageKey}`);
                continue;
            }

            // Process original image
            const imgTensor = await processImage({ buffer: imgBuffer });
            dataSamples.push({ xs: imgTensor, ys: label });

            // Generate augmented images
            for (let i = 0; i < numAugmentations; i++) {
                try {
                    const augmentedTensor = await augmentImage(imgBuffer);
                    dataSamples.push({ xs: augmentedTensor, ys: label });
                } catch (error) {
                    console.error('Error during augmentation:', error);
                    continue;
                }
            }
        } catch (error) {
            console.error(`Error processing image from S3 Key: ${entry.Key}`, error);
            processingError = error;
            break; // Exit the loop if there's a critical error
        }
    }

    if (processingError) {
        throw processingError;
    }

    console.log(`Total samples after augmentation: ${dataSamples.length}`);

    let dataset = tf.data.array(dataSamples);

    dataset = dataset.map(sample => {
        if (tf.any(tf.isNaN(sample.xs)).dataSync()[0]) {
            console.log('Invalid input tensor:', sample);
            return null;
        }
        if (tf.any(tf.isNaN(sample.ys)).dataSync()[0]) {
            console.log('Invalid label tensor:', sample);
            return null;
        }
        if (typeof sample.ys !== 'number' || isNaN(sample.ys)) {
            console.log('Invalid label value:', sample.ys);
            return null;
        }
        if (!sample.xs || (!sample.ys && sample.ys !== 0)) {
            console.log('Invalid sample:', sample);
            return null;
        }
        return {
            xs: sample.xs,
            ys: tf.tensor1d([sample.ys], 'float32')
        }
    })
    .filter(sample => sample !== null);

    dataset = dataset.shuffle(1000).batch(batchSize);

    console.log('Data processing completed.');
    return { dataset, totalSize: dataSamples.length };
};

// Handle image prediction
exports.handleImage = async (req, res) => {
    try {
        console.log('Handling image prediction...');
        const img = await processImage(req.file);

        if (!img) {
            return res.status(400).json({ error: 'Invalid image data' });
        }

        const prediction = await tfModel.predict(img.expandDims(0)); // Add batch dimension
        const predictionValue = prediction.dataSync()[0];
        const result = predictionValue > 0.5 ? 'Pass' : 'Fail';

        console.log(`Prediction: ${result}`);
        res.json({ result });

        // Dispose tensors after prediction
        img.dispose();
        prediction.dispose();
    } catch (err) {
        console.error('Error handling image prediction:', err);
        res.status(500).json({ error: err.message });
    }
};
