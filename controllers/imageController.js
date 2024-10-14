// imageController.js

const processImage = require('../utils/imageProcessor.js');
const { loadModel } = require('../models/tfModel.js');
const { computeGradCAM } = require('../utils/gradCam.js');
const { createCanvas, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const sharp = require('sharp');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';
const categories = ['butt', 'saddle', 'electro'];
const targetWidth = 150;
const targetHeight = 150;
const numAugmentations = 10; // increasing from 5 to 10

const streamToBuffer = (stream) => {
    const chunks = [];
    return new Promise((resolve, reject) => {
        stream.on('data', (chunk) => chunks.push(chunk));
        stream.on('end', () => resolve(Buffer.concat(chunks)));
        stream.on('error', (err) => reject(err));
    });
};

const augmentImage = async (imageBuffer) => {
    if (!imageBuffer || imageBuffer.length === 0) {
        throw new Error('Invalid image buffer');
    }

    try {
        let sharpImage = sharp(imageBuffer);

        // Apply random transformations
        const rotation = Math.floor(Math.random() * 360);
        sharpImage = sharpImage.rotate(rotation);

        if (Math.random() > 0.5) sharpImage = sharpImage.flip();
        if (Math.random() > 0.5) sharpImage = sharpImage.flop();

        // random brightness and saturation
        const brightness = Math.random() * 0.4 + 0.8;
        const saturation = Math.random() * 0.4 + 0.8;
        sharpImage = sharpImage.modulate({ brightness, saturation });

        // random cropping
        const metadata = await sharpImage.metadata();
        const cropWidth = Math.floor(metadata.width * (Math.random() * 0.2 + 0.8));
        const cropHeight = Math.floor(metadata.height * (Math.random() * 0.2 + 0.8));
        const cropX = Math.floor(Math.random() * (metadata.width - cropWidth));
        const cropY = Math.floor(Math.random() * (metadata.height - cropHeight));
        sharpImage = sharpImage.extract({ width: cropWidth, height: cropHeight, left: cropX, top: cropY });

        // Resize the image to 224x224
        sharpImage = sharpImage.resize(224, 224);

        // Convert to buffer
        const augmentedBuffer = await sharpImage.toBuffer();

        // Decode image buffer to tensor
        const imgTensor = tf.node.decodeImage(augmentedBuffer, 3)
            .toFloat()
            .div(255.0); // Normalize to [0, 1]

        if (tf.any(tf.isNaN(imgTensor)).dataSync()[0]) {
            console.error('NaN value detected in augmented image tensor');
            return null;
        }
        console.log(`Augmented image tensor shape: ${imgTensor.shape}`);
        return imgTensor;
    } catch (error) {
        console.error('Error augmenting image:', error.message);
        return null;
    }
};

// Function to create dataset using tf.data API
exports.createDataset = async (batchSize) => {
    console.log('Starting data processing...');

    const getAllImageKeys = async () => {
        const imageEntries = [];
        for (const category of categories) {
            console.log(`Processing category: ${category}`);
            for (const labelName of ['pass', 'fail']) {
                console.log(`Processing label: ${labelName}, type: ${typeof labelName}`);
                const label = labelName === 'pass' ? 1 : 0;
                console.log(`Label evaluated: ${label}, type: ${typeof label}`);
                const folderPath = `${category}/${labelName}`;
                console.log(`Folder path: ${folderPath}`);
                const params = { Bucket: bucket, Prefix: folderPath };
                const data = await s3Client.send(new ListObjectsV2Command(params));

                if (!data.Contents || data.Contents.length === 0) {
                    console.warn(`No images found in S3 folder: ${folderPath}`);
                    continue;
                }

                const entries = data.Contents.map(obj => ({
                    Key: obj.Key,
                    label,
                    category
                }));
                imageEntries.push(...entries);
            }
        }
        return imageEntries;
    };

    const imageEntries = await getAllImageKeys();

    if (imageEntries.length === 0) {
        throw new Error('No valid data to process. Ensure that images are available in S3.');
    }

    tf.util.shuffle(imageEntries);

    const dataSamples = [];
    let processingError = null;

    for (const entry of imageEntries) {
        try {
            const { Key: imageKey, label } = entry;

            // get category from image key
            const categoryMatch = imageKey.match(/^([^/]+)/);
            const category = categoryMatch ? categoryMatch[1] : null;

            console.log(`Category extracted: ${category}`);

            if (!category) {
                console.error(`Invalid image key: ${imageKey}`);
                continue;
            }

            const categoryIndex = categories.indexOf(category);
            if (categoryIndex === -1) {
                console.error(`Category not found in categories array: ${category}`);
                continue;
            }

            const categoryEncoding = tf.oneHot(categoryIndex, categories.length);

            const getObjectParams = { Bucket: bucket, Key: imageKey };
            const imgData = await s3Client.send(new GetObjectCommand(getObjectParams));
            const imgBuffer = await streamToBuffer(imgData.Body);

            if (!imgBuffer || imgBuffer.length === 0) {
                console.error(`Empty image buffer for S3 Key: ${imageKey}`);
                continue;
            }

            const imgTensor = await processImage({ buffer: imgBuffer });
            if (!imgTensor || imgTensor.shape.length !== 3) {
                console.error(`Invalid image tensor for S3 Key: ${imageKey}, shape: ${imgTensor ? imgTensor.shape : 'undefined'}`);
                continue;
            }

            const categoryTensor = tf.oneHot(categoryIndex, categories.length);

            const labelTensor = tf.scalar(Number(label), 'float32');
            dataSamples.push({ xs: { image: imgTensor, category: categoryTensor }, ys: labelTensor });

            for (let i = 0; i < numAugmentations; i++) {
                try {
                    const augmentedTensor = await augmentImage(imgBuffer);
                    if (!augmentedTensor || augmentedTensor.shape.length !== 3) {
                        console.error(`Invalid augmented image tensor for S3 Key: ${imageKey}`);
                        continue;
                    }
                    dataSamples.push({ xs: { image: augmentedTensor, category: categoryTensor }, ys: labelTensor });
                } catch (error) {
                    console.error('Error during augmentation:', error);
                    continue;
                }
            }
        } catch (error) {
            console.error(`Error processing image from S3 Key: ${entry.Key}`, error);
            processingError = error;
            continue;
        }
    }

    if (processingError) {
        console.error('Critical error during data processing:', processingError);
    }

    console.log(`Total samples after augmentation: ${dataSamples.length}`);

    
    const validDataSamples = dataSamples.filter(sample => {
        if (!sample.xs || !sample.ys) {
            console.error('Invalid sample detected, skipping...', {
                xs: sample.xs,
                ys: sample.ys
            });
            return false;
        }
        return true;
    });

    if (validDataSamples.length === 0) {
        throw new Error('No valid samples available for training.');
    }

    // Shuffle the data
    tf.util.shuffle(validDataSamples);

    // Split data into training and validation sets
    const totalSamples = validDataSamples.length;
    const valSize = Math.floor(totalSamples * 0.2);
    const trainSize = totalSamples - valSize;

    const valDataSamples = validDataSamples.slice(0, valSize);
    const trainDataSamples = validDataSamples.slice(valSize);

    const createDatasetFromSamples = (samples) => {
        return tf.data.array(samples).map(sample => {
            return {
                xs: {
                    imageInput: sample.xs.image,
                    categoryInput: sample.xs.category
                },
                ys: sample.ys.as1D()
            }
        }).batch(batchSize);
    };

    // Create datasets from the samples
    let trainDataset = createDatasetFromSamples(trainDataSamples);
    let valDataset = createDatasetFromSamples(valDataSamples);

    console.log('Data processing completed.');
    return { trainDataset, valDataset, totalSize: totalSamples };
};

exports.handlePrediction = async (req, res) => {
    try {
        const { file } = req;
        const { category } = req.body;

        console.log({ file, category });
        
        if (!file) 
            return res.status(400).json({ error: 'No image file provided' });

        if (!category)
            return res.status(400).json({ error: 'No weld category provided' });

        const imgTensor = await processImage(file.buffer);
        if (!imgTensor)
            return res.status(400).json({ error: 'Invalid image data' });

        const categoryIndex = categories.indexOf(category);
        if (categoryIndex === -1) {
            return res.status(400).json({ error: `Category not found in categories array: ${category}` });
        }
        
        const model = await loadModel();
        console.log(`Loaded model inputs: `, model.inputs.map(input => ({
            name: input.name,
            shape: input.shape
        })));
        // Ensure tensors are expanded to include batch dimension
        const categoryEncoding = tf.oneHot(categoryIndex, categories.length); // Shape: [3]
        const categoryInput = categoryEncoding.expandDims(0); // Shape: [1, 3]
        const imageInput = imgTensor.expandDims(0);
        
        console.log(`Image tensor shape: ${imageInput.shape}`);
        console.log(`Category tensor shape: ${categoryInput.shape}`);

        const prediction = model.predict({ imageInput, categoryInput });
        const predictionValue = prediction.dataSync()[0];
        const result = predictionValue > 0.5 ? 'Pass' : 'Fail';

        // Compute GradCAM
        const heatmapTensor = await computeGradCAM(model, imageInput, categoryEncoding);

        // Resize heatmap to match image dimensions
        const heatmapResized = tf.image.resizeBilinear(heatmapTensor, [224, 224]);

        const heatmapData = heatmapResized.squeeze().arraySync();

        // Create an overlay of the heatmap on the original image
        const canvas = createCanvas(224, 224)
        const ctx = canvas.getContext('2d')

        // Draw original image 
        const imgData = await loadImage(file.buffer)
        ctx.drawImage(imgData, 0, 0, 224, 224)

        // Draw heatmap
        const heatmapCanvas = createCanvas(224, 224)
        const heatmapCtx = heatmapCanvas.getContext('2d')

        const imageData = heatmapCtx.createImageData(224, 224)
        for (let i = 0; i < 224; i++) {
            for (let j = 0; j < 224; j++) {
                const alpha = Math.floor(heatmapData[i][j] * 255)
                const index = (i * 224 + j) * 4;
                imageData.data[index] = 255; // red channel
                imageData.data[index + 1] = 0; // green channel
                imageData.data[index + 2] = 0; // blue channel
                imageData.data[index + 3] = alpha; // alpha channel
            }
        }
        heatmapCtx.putImageData(imageData, 0, 0)

        ctx.drawImage(heatmapCanvas, 0, 0, 224, 224)

        // get result as base64 string
        const overlayImageBuffer = canvas.toBuffer('image/png')
        const base64Heatmap = overlayImageBuffer.toString('base64')

        // save heatmap to s3
        const s3Params = {
            Bucket: bucket,
            Key: `heatmaps/${category}/${file.originalname}${Date.now()}.png`,
            Body: base64Heatmap,
            ContentType: 'image/png'
        }
        await s3Client.send(new PutObjectCommand(s3Params))

        tf.dispose([imgTensor, categoryEncoding, prediction, heatmapTensor, heatmapResized]);
        console.log(`Prediction for ${category}: ${result} (confidence: ${predictionValue})`);
        res.json({ category, result }); // heatmap: base64Heatmap 

    } catch (err) {
        console.error('Error handling image prediction:', err);
        res.status(500).json({ error: err.message });
    }
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
