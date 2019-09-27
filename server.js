var express = require('express');
var bodyParser = require('body-parser');

var app = express();

app.use(express.static('public'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


app.get('/', function (req, res) {
  res.sendFile( __dirname + "/" + "index.html" );
});

app.post('/forecast', function (req, res) {
  console.log("Got POST to /forecast");
  var body = req.body;
  
  var childArgs = [0, 0, 0, 'US', 'USD', 0, 'art/ceramics'];

  if(body.fast == 1)
    childArgs[0] = 1;
  if(body.goal)
    childArgs[1] = parseFloat(body.goal);
  if(body.disComm === "true")
    childArgs[2] = 1;
  if(body.country)
    childArgs[3] = body.country;
  if(body.curr)
    childArgs[4] = body.curr;
  if(body.staff === "true")
    childArgs[5] = 1;
  if(body.cat)
    childArgs[6] = body.cat;

  console.log(childArgs);


  console.log("Starting prediction script...")
  // Run python script
  var spawn = require("child_process").spawn;
  var child = spawn('python',["SENG474_Kickstarter_Prediction.py"].concat(childArgs));

  var output = "";
  child.stdout.on('data', function (data){
    output += data.toString('utf8');
  });

  child.on('exit', function(code){
    console.log("Completed prediction script.");

    var toReturn = output.split(/[\r\n]+/);
    toReturn.pop(); // Gets rid of the empty newline element at the end

    for (var i in toReturn){
      toReturn[i] = toReturn[i].split(", ");

      if(toReturn[i].length == 3){
        toReturn[i] = {"class" : toReturn[i][0], "estimation" : parseFloat(toReturn[i][1]), "accuracy" : parseFloat(toReturn[i][2])};
      }
    }

    toReturn = JSON.stringify(toReturn);
    
    res.send(toReturn);
  });

});

var port = 8888;
if(process.argv.length > 2){
  port = parseInt(process.argv[2]);
}

var server = app.listen(port, function () {
   var host = server.address().address;
   var port = server.address().port;
   
   console.log("App listening at http://%s:%s", host, port);
});