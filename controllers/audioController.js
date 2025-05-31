const audioService = require("../services/audioService");

const uploadVoice = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "Файл не получен" });
  }

  try {
    const outputFilename = await audioService.convertToMp3(req.file.path);
    return res.status(200).json({
      message: "Файл получен и сконвертирован",
    });
  } catch (error) {
    console.error("Ошибка:", error);
    return res.status(500).json({ message: "Ошибка обработки файла" });
  }
};

module.exports = { uploadVoice };
