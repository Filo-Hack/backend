require("dotenv").config();
const app = require("./app");

const PORT = process.env.PORT || 44444;

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
