const processImage = require('../utils/imageProcessor');
const tfModel = require('../models/tfModel');
const tf = require('@tensorflow/tfjs-node');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const { PromisePool } = require('@supercharge/promise-pool');

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
const augmentImage = (image) => {
    console.log('Augmenting image...');

    const rotated = tf.image.rotate(image, Math.random() * 80 - 40, 0.5, 0.5);
    const widthShift = Math.random() * 0.4 - 0.2;
    const heightShift = Math.random() * 0.4 - 0.2;
    const shifted = tf.image.translate(rotated, [widthShift * image.shape[1], heightShift * image.shape[0]]);
    const zoom = Math.random() * 0.4 + 0.8;
    const zoomed = tf.image.resizeBilinear(shifted, [image.shape[0] * zoom, image.shape[1] * zoom]);
    const flipped = Math.random() > 0.5 ? tf.image.flipLeftRight(zoomed) : zoomed;
    const finalFlip = Math.random() > 0.5 ? tf.image.flipUpDown(flipped) : flipped;
    const brightnessAdjusted = tf.image.adjustBrightness(finalFlip, Math.random() * 0.4 - 0.2);
    const contrastAdjusted = tf.image.adjustContrast(brightnessAdjusted, Math.random() * 1.5 + 0.5);
    const noise = tf.randomNormal(image.shape, 0, 0.05);
    const noised = tf.add(contrastAdjusted, noise);

    return tf.clipByValue(noised, 0, 1);
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

                    // Augment the image multiple times
                    const augmentations = Array.from({ length: numAugmentations }, () => augmentImage(imgTensor));
                    const augmentedImages = await Promise.all(augmentations);

                    augmentedImages.forEach((augmented, i) => {
                        images.push({ tensor: augmented, label });
                        console.log(`Augmented image #${index + 1}-${i + 1} from S3 Key: ${imageKey}`);
                        augmented.dispose(); // Dispose augmented tensor
                    });

                    imgTensor.dispose(); // Dispose original tensor
                }
            } catch (error) {
                console.error(`Error processing image from S3 Key: ${imageKey}`, error);
            }
        });

    return images;
};

// Function to process data in batches, including loading and augmenting images
exports.processDataInBatches = async (batchSize = 32, numAugmentations = 5) => {
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
