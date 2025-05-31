const express = require("express");
const app = express();

// Middleware
app.use(express.json());

// Routes
const sensorsRoutes = require("./routes/sensorsRoutes");
app.use("/sensors", sensorsRoutes);
const audioRoutes = require("./routes/audioRoutes");
app.use("/audio", audioRoutes);

module.exports = app;
