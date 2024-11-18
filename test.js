const axios = require('axios')
const FormData = require('form-data')
const { S3Client, ListObjectsV2Command, GetObjectCommand, PutObjectCommand } = require('@aws-sdk/client-s3')
const s3Client = new S3Client({ region: 'us-west-1' })
const { Parser } = require('json2csv')
const bucket = 'weldscanner'

async function testAll() {
    try {
        const listCommand = new ListObjectsV2Command({ Bucket: bucket })
        const listResponse = await s3Client.send(listCommand)
        const images = listResponse.Contents

        images.sort((a, b) => new Date(a.LastModified) - new Date(b.LastModified))

        const predictions = []

        for (const image of images) {
            const category = image.Key.split('/')[0]
            console.log(`Processing image ${image.Key}`)

            const getObjectCommand = new GetObjectCommand({ Bucket: bucket, Key: image.Key })
            const imageData = await s3Client.send(getObjectCommand)

            const chunks = []
            for await (const chunk of imageData.Body) {
                chunks.push(chunk)
            }
            const imageBuffer = Buffer.concat(chunks)

            const formData = new FormData()
            formData.append('image', imageBuffer, image.Key)
            formData.append('category', category)

            const prediciton = await axios.post('http://localhost:3000/predict  ', formData, {
                headers: {
                    ...formData.getHeaders()
                }
            })

            predictions.push({
                image: image.Key,
                category,
                prediction: prediction.data.prediction
            })
        }
        const parser = new Parser()
        const csv = parser.parse(predictions)

        const date = new Date()
        const fileName = `prediction_${date.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' }).replace(/\//g, '-')}.csv`

        const putObjectCommand = new PutObjectCommand({
            Bucket: bucket,
            Key: `predictions/${fileName}`,
            Body: csv,
            ContentType: 'text/csv'

        })

        await s3Client.send(putObjectCommand)
        console.log(`Predictions saved to ${fileName}`)
    } catch (error) {
        console.error('Error during predictions:', error)
    }
}

testAll()