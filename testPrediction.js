const axios = require('axios');
const FormData = require('form-data');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';

async function testPrediction(category) {
    try {
        // get random image from weldscanner bucket
        const command = new ListObjectsV2Command({ Bucket: bucket });
        const response = await s3Client.send(command);
        const images = response.Contents;
        const randomImage = images[Math.floor(Math.random() * images.length)];  

        const imageBuffer = await getObjectFromS3(randomImage.Key);

        // prepare form data
        const formData = new FormData();
        formData.append('image', imageBuffer, randomImage.Key);

        // send request to prediction endpoint
        const predictionResponse = await axios.post('http://localhost:3000/image', formData, {
            headers: { 
                'Content-Type': 'multipart/form-data',
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
