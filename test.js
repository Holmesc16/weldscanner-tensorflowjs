const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const { S3Client, ListObjectsV2Command, GetObjectCommand, PutObjectCommand } = require('@aws-sdk/client-s3');
const { Parser } = require('json2csv');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';
const categories = ['butt', 'saddle', 'electro'];
const path = require('path');

async function loadModel() {
    const modelPath = 'file:///home/ec2-user/app/_trained_models/weldscanner_quality_model_v2/model.json'; // Ensure this path is correct
    console.log(`Loading model from ${modelPath}`);
    return await tf.loadLayersModel(modelPath);
}

async function processImage(imageBuffer) {
    const resizedBuffer = await sharp(imageBuffer)
        .resize(224, 224)
        .toBuffer();
    const imgTensor = tf.node.decodeImage(resizedBuffer, 3)
        .toFloat()
        .div(255.0); // Normalize to [0, 1]
    return imgTensor.expandDims(); // Add batch dimension
}

async function testPrediction(category) {
    const predictions = [];
    try {
        // Load the model
        const model = await loadModel();

        // Get images from the S3 bucket
        const listCommand = new ListObjectsV2Command({ Bucket: bucket });
        const listResponse = await s3Client.send(listCommand);
        const images = listResponse.Contents;

        const categoryImages = images.filter(image => image.Key.includes(category));
        if (categoryImages.length === 0) {
            console.error(`No images found for category: ${category}`);
            return;
        }

        for (const image of categoryImages) {
            console.log('Processing image: ', image.Key);

            const getObjectCommand = new GetObjectCommand({ Bucket: bucket, Key: image.Key });
            const imageData = await s3Client.send(getObjectCommand);

            const chunks = [];
            for await (const chunk of imageData.Body) {
                chunks.push(chunk);
            }
            const imageBuffer = Buffer.concat(chunks);

            // Process the image
            const imgTensor = await processImage(imageBuffer);

            // Create category encoding
            const categoryIndex = categories.indexOf(category);
            const categoryEncoding = tf.oneHot(categoryIndex, categories.length).expandDims();

            // Make prediction
            const prediction = model.predict([imgTensor, categoryEncoding]);
            const predictionValue = prediction.dataSync()[0];
            const result = predictionValue > 0.5 ? 'Pass' : 'Fail';

            console.log(`Prediction for ${image.Key}: ${result} (confidence: ${predictionValue})`);

            predictions.push({
                image: image.Key,
                category,
                prediction: result,
                confidence: predictionValue
            });

            // Dispose tensors
            tf.dispose([imgTensor, categoryEncoding, prediction]);
        }

        // Create CSV
        const parser = new Parser();
        const csv = parser.parse(predictions);

        // Save CSV to S3
        const date = new Date();
        const fileName = `prediction_${date.toISOString().split('T')[0]}.csv`;

        const putObjectCommand = new PutObjectCommand({
            Bucket: bucket,
            Key: `predictions/${fileName}`,
            Body: csv,
            ContentType: 'text/csv'
        });

        await s3Client.send(putObjectCommand);
        console.log(`Predictions saved to ${fileName} in S3 bucket.`);
    } catch (error) {
        console.error('Error during prediction:', error.message);
    }
}

const category = categories[Math.floor(Math.random() * categories.length)];
testPrediction(category);