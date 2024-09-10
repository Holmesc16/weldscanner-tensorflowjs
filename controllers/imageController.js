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

// Function to augment images with various transformations
const augmentImage = (image) => {
    console.log('Augmenting image...');
    
    // Random rotation between -40 and 40 degrees
    const rotated = tf.image.rotateWithOffset(image, Math.random() * 80 - 40, 0.5, 0.5);

    // Random width and height shift between -20% and 20%
    const widthShift = Math.random() * 0.4 - 0.2;
    const heightShift = Math.random() * 0.4 - 0.2;
    const shifted = tf.image.translate(rotated, [widthShift * image.shape[1], heightShift * image.shape[0]]);

    // Random zoom between 80% and 120%
    const zoom = Math.random() * 0.4 + 0.8;
    const zoomed = tf.image.resizeBilinear(shifted, [image.shape[0] * zoom, image.shape[1] * zoom]);

    // Random horizontal and vertical flip
    const flipped = Math.random() > 0.5 ? tf.image.flipLeftRight(zoomed) : zoomed;
    const finalFlip = Math.random() > 0.5 ? tf.image.flipUpDown(flipped) : flipped;

    // Random brightness adjustment
    const brightnessAdjusted = tf.image.adjustBrightness(finalFlip, Math.random() * 0.4 - 0.2);

    // Random contrast adjustment
    const contrastAdjusted = tf.image.adjustContrast(brightnessAdjusted, Math.random() * 1.5 + 0.5);

    // Add random noise
    const noise = tf.randomNormal(image.shape, 0, 0.05);
    const noised = tf.add(contrastAdjusted, noise);

    // Ensure the pixel values are still in the valid range
    return tf.clipByValue(noised, 0, 1);
};

// Function to load images in batches and augment them
const loadImagesInBatches = async (folderPath, label, batchSize = 16, numAugmentations = 5) => {
    const params = {
        Bucket: bucket,
        Prefix: folderPath
    };
    const data = await s3Client.send(new ListObjectsV2Command(params));
    const imageKeys = data.Contents.map(obj => obj.Key);
    const images = [];

    await PromisePool
        .for(imageKeys)
        .withConcurrency(batchSize)
        .process(async (imageKey, index) => {
            const getObjectParams = { Bucket: bucket, Key: imageKey };
            const imgData = await s3Client.send(new GetObjectCommand(getObjectParams));
            const imgBuffer = await streamToBuffer(imgData.Body);
            const imgTensor = await processImage({ buffer: imgBuffer });
            console.log(`Loaded image #${index + 1} (${imagePath})`);
            images.push({ tensor: imgTensor, label });

            const augmentations = Array.from({ length: numAugmentations }, () => augmentImage(imgTensor));
            const augmentedImages = await Promise.all(augmentations);

            augmentedImages.forEach((augmented, i) => {
                images.push({ tensor: augmented, label });
                console.log(`Augmented image #${index + 1}-${i + 1} (${imagePath})`);
                augmented.dispose(); // Dispose augmented tensor to free up memory
            });

            imgTensor.dispose(); // Dispose original tensor to free up memory
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
        console.log(`Loaded ${passImages.length} pass images and ${failImages.length} fail images for category: ${category}`);
        
        const data = [...passImages, ...failImages];
        if (data.length === 0) {
            console.warn(`No images found for category: ${category}`);
            return;
        }

        console.log(`Shuffling and concatenating data for category: ${category}`);
        tf.util.shuffle(data);
        const tensors = data.map(d => d.tensor);
        const labels = data.map(d => d.label);

        if (tensors.length === 0) {
            console.warn(`No tensors found for category: ${category}`);
            return;
        }

        xsList.push(tf.concat(tensors, 0));
        ysList.push(tf.tensor1d(labels, 'int32'));

        // Dispose tensors to free up memory
        tensors.forEach(tensor => tensor.dispose());
    }));

    const xs = tf.concat(xsList);
    const ys = tf.concat(ysList);

    console.log('Data processing completed.');
    return { xs, ys };
};

// Function to handle image prediction
exports.handleImage = async (req, res) => {
    try {
        console.log('Handling image prediction...');
        const img = await processImage(req.file);
        const prediction = await tfModel.predict(img);
        const result = prediction.dataSync()[0] > 0.5 ? 'Pass' : 'Fail';
        console.log(`Prediction: ${result}`);
        res.json({ result });
        img.dispose(); // Dispose image tensor to free up memory
        prediction.dispose(); // Dispose prediction tensor to free up memory
        console.log('Image prediction completed.');
    } catch (err) {
        console.error('Error handling image prediction:', err);
        res.status(500).json({ error: err.message });
    }
};