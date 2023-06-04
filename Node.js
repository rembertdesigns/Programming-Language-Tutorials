// NODE.JS - JavaScript runtime environment - by Richard Rembert



// ------------------------ NODE ------------------------ //

// node . // run index.js
// node // run node in terminal
// node -v // get node version

// in terminal
process.platform; // returns OS -> darwin(mac), win32(windows), linux...
process.version; // node version
process.env.NAME_OF_VARIABLE; // get .env variable value

// via .js file

// GLOBALS
console.log(__dirname); // /full/path/to/file
console.log(__filename); // /full/path/to/file/index.js

// OS
const os = require('os');
const currentOS = {
  name: os.type(), // name: 'Darwin',
  release: os.release(), // release: '21.6.0',
  totalMem: os.totalmem(), // totalMem: 17179869184,
  freeMem: os.freemem() // freeMem: 2077073408
}
console.log(currentOS);

// PATH
const path = require('path');
console.log(path.sep); // /
const filePath = path.join('create', '//any', 'path.txt');
console.log(filePath); // create/any/path.txt
console.log(path.basename(filePath)); // path.txt
console.log(path.resolve(__dirname, filePath)); // /full/path/to/create/any/path.txt

// ENV VAR
// in .env file: NAME_OF_VARIABLE=thevalue
require('dotenv').config();
const secret = process.env['NAME_OF_VARIABLE'];

// FILE SYSTEM
const {readFile, readFileSync} require('fs');
const txt = readFileSync('./text.txt', 'utf8'); // default way
readFile('./text.txt', 'uft8', (err, txt) => {console.log(txt)}; // non-blocking way
         
// OR
const { readFile } require('fs').promises;
async function hello() {const file = await readFile('./text.txt', 'utf8')};

// BODY PARSER
// handle data from a form (<form action="/name" method="post")
let bodyParser = require('body-parser');
app.use('/name', bodyParser.urlencoder({extended: false}));
app.post('/name', function(req, res) {
  const name = req.body;
  res.json({'name': name.first + ' ' + name.last});
})




// ------------------------ EXPRESS ------------------------ //

let express = require('express');
// import express from 'express';
let app = express();

const PORT = 8000;
app.listen(PORT, () => console.log('Server is running on port ' + PORT));

// METHODS
app.get('/route', function() {});
app.post('/route', function() {});
app.delete('/route', function() {});
app.put('/route', function() {});

// chain methods with this syntax
app.route('/route').get(function() {}).post(function() {});
