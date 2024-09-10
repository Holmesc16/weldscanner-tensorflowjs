require('dotenv').config();
const { main } = require('./models/trainer');
console.log('Training mode enabled. Starting training...');
main()
    .then(() => {
        console.log('Model training complete.');
        process.exit(0);
    })
    .catch(err => {
        console.error('Error during model training:', err);
        process.exit(1);
    });

