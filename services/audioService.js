const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("ffmpeg-static");
const path = require("path");

ffmpeg.setFfmpegPath(ffmpegPath);

const convertToMp3 = (inputPath) => {
  return new Promise((resolve, reject) => {
    const outputFilename =
      path.basename(inputPath, path.extname(inputPath)) + ".mp3";
    const outputPath = path.join("converted", outputFilename);

    ffmpeg(inputPath)
      .toFormat("mp3")
      .on("error", (err) => reject(err))
      .on("end", () => resolve(outputFilename))
      .save(outputPath);
  });
};

module.exports = { convertToMp3 };
