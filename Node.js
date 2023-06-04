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
