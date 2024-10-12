const axios = require('axios');
const FormData = require('form-data');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';

async function testPrediction(category) {
    try {
        // Get random image from the S3 bucket
        const listCommand = new ListObjectsV2Command({ Bucket: bucket });
        const listResponse = await s3Client.send(listCommand);
        const images = listResponse.Contents;
        const randomImage = images[Math.floor(Math.random() * images.length)];  

        console.log('Random image: ', randomImage.Key);
        
        const getObjectCommand = new GetObjectCommand({ Bucket: bucket, Key: randomImage.Key });
        const imageData = await s3Client.send(getObjectCommand);
        
        const chunks = [];
        for await (const chunk of imageData.Body) {
            chunks.push(chunk);
        }
        const imageBuffer = Buffer.concat(chunks);

        const formData = new FormData();
        formData.append('image', imageBuffer, randomImage.Key);
        formData.append('category', category);

        const predictionResponse = await axios.post('http://localhost:3000/image', formData, {
            headers: { 
                ...formData.getHeaders()
            }
        });

        console.log('Prediction: ', predictionResponse.data);
    } catch (error) {
        console.error('Error during prediction:', error.response ? error.response.data : error.message);
    }
}

const categories = ['butt', 'saddle', 'electro'];
const category = categories[Math.floor(Math.random() * categories.length)];

testPrediction(category);
