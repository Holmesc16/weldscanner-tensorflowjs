const express = require('express');
const app = express();
const routes = require('./routes/index.js');
const cors = require('cors');

app.use(cors({
    origin: [
        'http://localhost:19006',
        'exp://192.168.1.12:8081'
    ],
}));

app.use(express.json());
app.use('/', routes);

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
