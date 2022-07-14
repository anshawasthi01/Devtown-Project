const express = require("express");
const bodyParser = require("body-parser");

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));

app.get("/", function (request, response) {
  response.sendFile(__dirname + "/index.html");
});

app.post("/", function (req, res) {
  var n1 = Number(req.body.n1);
  var n2 = Number(req.body.n2);
  var ans = n1 + n2;
  res.send("The Answer is : " + ans);
});

app.get("/about", function (request, response) {
  response.send("<h1>Ansh</h1>");
});

app.listen(3000, function () {
  console.log("Port is running");
});
