const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const audioController = require("../controllers/audioController");

// Конфигурация multer
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});

const upload = multer({ storage });

// POST /audio/upload-voice
router.post(
  "/upload-voice",
  upload.single("voice"),
  audioController.uploadVoice
);

module.exports = router;
