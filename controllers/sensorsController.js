const handleSensorData = (req, res) => {
  const { id, status, time } = req.body;
  const sensorType = req.params.type;

  if (!id || !status || !time) {
    return res
      .status(400)
      .json({ message: "ошибка", error: "Недостаточно данных" });
  }

  if (!["on", "off"].includes(status)) {
    return res
      .status(400)
      .json({ message: "ошибка", error: "Некорректный статус" });
  }

  // Здесь можно сохранить данные в БД или обработать (в будущем: запись поведения)
  console.log(`Получены данные от ${sensorType}:`, { id, status, time });

  return res.status(200).json({ message: "отлично" });
};

module.exports = { handleSensorData };
