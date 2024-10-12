const express = require('express');
const app = express();
const routes = require('./routes/index.js');

// Middleware to parse JSON bodies
app.use(express.json());

// Use the routes
app.use('/', routes);

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
