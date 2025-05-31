const express = require("express");
const app = express();

// Middleware
app.use(express.json());

// Routes
const inputRoutes = require("./routes/inputRoute");
app.use("/input", inputRoutes);

module.exports = app;
