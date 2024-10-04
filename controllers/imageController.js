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

// Augments an image tensor with various transformations
const augmentImage = async (imageBuffer) => {
    if (!imageBuffer || imageBuffer.length === 0) {
        throw new Error('Invalid image buffer');
    }

    try {
        let sharpImage = sharp(imageBuffer);
        const rotation = Math.floor(Math.random() * 80 - 40);
        sharpImage = sharpImage.rotate(rotation);

        if (Math.random() > 0.5) {
            sharpImage = sharpImage.flip();
        }

        if (Math.random() > 0.5) {
            sharpImage = sharpImage.flop();
        }

        sharpImage = sharpImage.resize(targetWidth, targetHeight);

        const augmentedBuffer = await sharpImage.toBuffer();

        const imgTensor = tf.tidy(() => {
            return tf.node.decodeImage(augmentedBuffer, 3)
                .toFloat()
                .div(255.0)
                .sub(0.5)
                .div(0.5); // Shape: [height, width, channels]
        });

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

    // Create a data queue to store processed data
    const dataQueue = [];
    let isProcessing = true;
    let processingError = null;

    // Set up the async queue with concurrency limit
    const processingQueue = async.queue(async (task) => {
        try {
            const { Key: imageKey, label } = task;

            const getObjectParams = { Bucket: bucket, Key: imageKey };
            const imgData = await s3Client.send(new GetObjectCommand(getObjectParams));
            const imgBuffer = await streamToBuffer(imgData.Body);

            if (!imgBuffer || imgBuffer.length === 0) {
                console.error(`Empty image buffer for S3 Key: ${imageKey}`);
                return;
            }

            const imgTensor = await processImage({ buffer: imgBuffer });
            const ys = tf.tensor1d([label], 'float32'); // Convert label to scalar tensor

            // Push the original image data
            dataQueue.push({ xs: imgTensor, ys });

            // Generate augmented images
            for (let i = 0; i < numAugmentations; i++) {
                try {
                    const augmentedTensor = await augmentImage(imgBuffer);
                    dataQueue.push({ xs: augmentedTensor, ys });
                } catch (error) {
                    console.error('Error during augmentation:', error);
                    continue;
                }
            }
        } catch (error) {
            console.error(`Error processing image from S3 Key: ${task.Key}`, error);
            processingError = error;
        }
    }, 15); // Set concurrency limit

    // Enqueue all image entries
    processingQueue.push(imageEntries);

    // When the queue is drained, set isProcessing to false
    processingQueue.drain(() => {
        isProcessing = false;
    });

    // Create a dataset using tf.data.generator
    const dataset = tf.data.generator(() => {
        return {
            async next() {
                // Wait for data to be available or processing to finish
                while (dataQueue.length === 0) {
                    if (!isProcessing) {
                        if (processingError) {
                            throw processingError;
                        } else {
                            return { done: true };
                        }
                    }
                    // Wait a bit before checking again
                    await new Promise(resolve => setTimeout(resolve, 100));
                }

                const data = dataQueue.shift();
                return { value: data, done: false };
            }
        };
    });

    const datasetSize = imageEntries.length * (numAugmentations + 1);
    console.log(`Dataset size: ${datasetSize}`);

    // Batch and shuffle the dataset
    const batchedDataset = dataset.shuffle(1000).batch(batchSize);

    const adjustedDataset = batchedDataset.map(({ xs, ys }) => {
        ys = ys.squeeze(); // Reshape labels to [batch_size]
        console.log('Adjusted ys shape:', ys.shape);
        return { xs, ys };
    });

    adjustedDataset.size = async () => datasetSize;

    console.log('Data processing completed.');
    return adjustedDataset;
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
