const express = require("express");
const router = express.Router();

const controller = require("../controllers/inputController");

router.post("/", controller.someFunction);

module.exports = router;
