const express = require("express");
const router = express.Router();
const controller = require("../controllers/sensorsController");

// Общий маршрут: тип датчика передается в URL
router.post("/:type", controller.handleSensorData);

module.exports = router;
