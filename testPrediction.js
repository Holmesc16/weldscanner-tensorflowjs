const axios = require('axios');
const FormData = require('form-data');
const { S3Client, ListObjectsV2Command, GetObjectCommand } = require('@aws-sdk/client-s3');
const s3Client = new S3Client({ region: 'us-west-1' });
const bucket = 'weldscanner';
const chalk = require('chalk');

async function testPrediction(category) {
    try {
        // Get random image from the S3 bucket
        const listCommand = new ListObjectsV2Command({ Bucket: bucket });
        const listResponse = await s3Client.send(listCommand);
        const images = listResponse.Contents;
        
        const categoryImages = images.filter(image=> image.Key.includes(category))
        if (categoryImages.length === 0) {
            console.error(chalk.red(`No images found for category: ${category}`));
            return;
        }

        for (let img of categoryImages) {
            setTimeout(async () => {
            const getObjectCommand = new GetObjectCommand({ Bucket: bucket, Key: img.Key });
            const imageData = await s3Client.send(getObjectCommand);
        
            const chunks = [];
            for await (const chunk of imageData.Body) {
                chunks.push(chunk);
                const imageBuffer = Buffer.concat(chunks);
                
                const formData = new FormData();
                formData.append('image', imageBuffer, img.Key);
                formData.append('category', category);
                
                const predictionResponse = await axios.post('http://localhost:3000/image', formData, {
                    headers: { 
                        ...formData.getHeaders()
                    }
                });
                const { category, result } = predictionResponse.data;
                const expectedCategory = category;
                const returnedCategory = category;
                console.log(chalk.cyan.bold('Expected: '), chalk.yellow.bgBlack(expectedCategory));
                console.log(chalk.yellow.bold('Returned: '), chalk.yellow.bgBlack(returnedCategory));
                if (expectedCategory !== returnedCategory) {
                    console.log(chalk.red.bold('Inaccurate Prediction'));
                }
                console.log(chalk.green.bold('Result: '), chalk.yellow.bgBlack(`${result}`));
            }
            }, 3000);
        }
    } catch (error) {
        console.error(chalk.red('Error during prediction:'), error.response ? error.response.data : error.message);
    }
}

const categories = ['butt', 'saddle', 'electro'];
const category = categories[Math.floor(Math.random() * categories.length)];

testPrediction(category);
