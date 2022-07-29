// JAVASCRIPT - Page Behavior - by Richard

    
    
// To create a comment. Shown in code only. Indent 2spaces
/* Multiple line comment */


// TODO: --> best practice to indicate things to do in the code

// FIXME: --> best practice to indicate things to fix in the code


// MAIN JS - main syntax

  // Variable
  var a = "example"; 
  console.log(a); // example

  // Function; also var example function() {};
  function example() {
    // code block;
  } 

  // Double arg Function
  function say(arg1) {
    return function(arg2) {
      return arg1 + " " + arg2;
    }
  }
  say ("hello")("world"); // hello world

  // Array (careful!-> if variable is set equal to another array, they will be linked and influence each other)
  var numbers = [one, two, three]; 
  console.log(numbers[0]); // 'one'
  console.log(numbers[0], numbers[1]); // 'one two'

  // Object
  var dog = {
    name: "Rex", 
    race: "Pitbul", 
    age: "6"
  }; 
  console.log(dog.name); // Rex
  console.log(dog["name"]); // Rex

  // Condition
  if (a === 1) {
    // code block;
  } else if (a === 2) {
    // code block;
  } else {
    // code block;
  }

  // Ternary operator
  var a = (b > 10) ? "Over ten" : "Under ten"; // if b > 10, a value will become "Over ten"; otherwise, "Under ten"

  // Short circuiting
  var a = true && true; // returns true
  var a = true && false; // returns false
  var a = false || false; // returns false
  var a = true || false; // returns true

  // Switch condition
  switch (a) {
    case 1:
    // code block;
      break;
    case 2:
    case 3:
    // code block;
      break;
    default:
    // code block;
  }

  // While loop
  while (a == 1) {
    // code block;
  } 

  // Do while loop; Will execute code block at least once
  do {
    // code block;
  } while (false); 

  // For loop
  for (i = 0; i < a.length; i++) {
    // code block;
  } 

  // ForEach loop from an array
  a.forEach(function(item) {
    // code block;
  }); 


// DOM syntax

  document.getElementsByTagNamegName("tag"); // will return tag elements as an array
  document.getElementsByClassName("class"); // will return .class elements as an array
  document.getElementById("id"); // will return #id element
  document.querySelector("tag"); // will return first tag of document; selection similar to css (or jquery)
  document.querySelectorAll("tag"); // will return all tags as an array; Note that it isn't updated live as other selectors are

  document.getElementById("id").setAttribute("src", "www.site.com"); // set/change attribute; can often be written something.src("change");
  document.getElementById("id").innerHTML = "some text"; // change #id content
  document.getElementById("id").textContent = "some text"; // similar to innerHTML without tag understanding; can use += to add some text
  document.getElementById("id").style.backgroundColor = "red"; // change style of background-color; everything is camelCased
  document.getElementById("id").classList.add("class"); // add a class

  var newDiv = document.createElement("div"); newDiv.innerHTML = "text"; document.body.appendChild(newDiv); // create div, add content and append to body
  document.body.removeChild(newDiv); // remove newDiv
  document.replaceChild(newDiv, otherDiv); // replace newDiv wth otherDiv

  newDiv.addEventListener("click", function () {/*code block;*/}); // new event click
  newDiv.removeEventListener("click", namedFunction); // remove event click
  newDiv.addEventListener("click", function once() {this.removeEventListener("click", once)}); // event click only once

  setTimeout(function (){/*code block;*/}, 1000); // run code after 1sec
  var timeout = setTimeout(func, 1000); clearTimeout(timeout); // timeout is set and then cleared
  setInterval(function (){/*code block;*/}, 1000); // run code every 1sec
  var interval = setInterval(func, 1000); clearInterval(interval); // interval is set and then cleared

  document.querySelector("form").addEventListener("submit", function (e) {e.PreventDefault();}); // prevent default (submit and refresh)
  document.querySelector(".item").addEventListener("click", someFunc); function someFunc(e) {e.target.style.color = "red"} // targets clicked element 

  JSON.stringify(objectJS); // JS objects array into JSON string
  JSON.parse(stringJSON); // JSON string into JS objects array




// CONSOLELOG
// show something in js console


  console.log("Text"); // Text

  console.log("up \n down"); // \n acts like enter

  console.log("\' \" \\ "); // to use those tricky characters

  console.log(3 + 4 - 1 * 4 / 2); // process and give answer

  console.log(10 % 3); // show the rest of division -> it is called modulo

  console.log("hel" + "lo"); // show addition of the two strings "bonjour"

  console.log("hello", "you"); // show both strings with a space between "hello you"

  console.log(isNaN(10)); // returns false (true if it is actually NotaNumber)

  console.log(typeof "string"); // can return number, boolean, string, function, object (array return object in JS)

  console.warn("Warning"); // display some content as a warning message

  console.assert(2 == 2); // returns undefined if true and returns detailed error when false

  console.table(["apples", "oranges", "bananas"]); // returns a nice table with index and values; works with objects as well !!IE

  console.count(); // returns the number of time this particular call to count() has been called

  console.log("%cStyled!", "font-weight: 800; color: red; font-size: 3rem;"); // style your log with some CSS




// ERRORS 
// error example -> ReferenceError: not defined; ends the work of JS (the code afterwards will never be processed)




// STRICT MODE 
// the strict mode changes bad syntax into actual errors; helping writing secure JS


  // usage of strict mode; must be placed on top of code (or function)
  "use strict";
  x = 3; // not allowed; you must declare every variables




// TYPES 
// Check types of elements;


  // Typeof the main type checker

  typeof 27; // number

  typeof NaN; // number; weird i know

  typeof "Hello"; // string

  typeof (typeof 1); // string; typeof always returns a string

  typeof true; // boolean

  typeof Symbol(); // symbol

  typeof undefined; // undefined

  typeof what; // undefined

  typeof {a:1}; // object

  typeof [1, 2, 3]; // object

  Array.isArray(arrayName); // true if an array

  typeof function(){}; // function

  typeof class C {}; // function

  typeof Math.PI; // function

  typeof null; // object


  // Type conversion

  String(true); // turns a value into a string --> "true"
  
  100 + ""; // turns a value into a string --> "100"

  Number("123"); // turns a value into a number --> 123
  
  +null; // turns a value into a number --> 0

  Number("hello"); // if the type isn't convertible --> NaN

  Boolean(1); // turns a value into a boolean --> true

  !!0; // turns a value into a boolean --> false




// VARIABLES
// JavaScript variables are containers for storing data values.


  var a; // creates variable without value

  var a; a = 10; // create variable with value of '10'

  var a = 10; // shortened method

  var a = 0; a += 1; a ++; // add 1 and 1 to 'a' variable to reach 2

  var a = 5; a -= 1; a --; // subtracts 1 and 1 to reach 3

  var a = 5; a *= 5; a /= 5; // multiply by 5 and divide by 5 to reach 5

  var a = 3; var b = a + 2; // 'b' will value 5 (3+2)

  var a = 2 ** 3; // a = 8 !!IE

  var a = 1e6; // 1000000

  var a = 3; a = (a++, a*=3, a-=3); // serie of expression in one line --> a = 9

  var a = "five" * 2; // 'a' is NaN; Not a Number

  var a = (b > 10) ? "Over ten" : "Under ten"; // if b>10, a will equal "Over ten"; otherwise, "Under ten"; it's called ternary operator

  var a = b > 0 ? "positive" : b < 0 ? "negative" : "zero"; // a will be positive, negative or zero according to b value

  var a = true && false; // false; returns the first falsy value; default is last value
 
  var a = true || false; // true; returns the first truthy value; default is last value




// ECMASCRIPT 6 !!IE

  // VARIABLES
  // Javascript const are containers you can only read; best use for value that will not change

  // basic const
  const a = 23;

  // exception to rewrite const -> in a block
  if (a === 23) {
    const a = 7; // here and only here a = 7
  }
  console.log(a); // 23 again

  // with arrays
  const myArray = [1, 2, 3];
  myArray[0] = 4; // can be freezed with Object.freeze()

  // with objects
  const myObject = {"key": "value"};
  myObject.key = "otherValue"; // keys are not protected; can be with Object.freeze()

  // with arrays
  const myArray = [];
  myArray = [8]; // not possible
  myArray.push(8); // possible

  // let are block-scoped variables
  // let vs var
  var a = 5, b = 10;
  if (a === 5) {
    let a = 4; // only 4 in if block
    var b = 1;
    console.log(a, b); // 4 1
  }
  console.log(a, b); //  5 1

  // const are block-scoped variables that can't be reassigned but that can be updated
  const arr = [1, 2, 3];
  // Trying to reassign a const will give an error
  arr = [5]; // ERROR
  // Updating a const will succeed
  arr.push(4); // [1, 2, 3, 4]


  // Template literals
  const easyText = `<p class="danger">
  It's way easier to write strings with "quotes" and breaks
  </p>`;


  // FUNCTIONS 
  // arrow function
  const myFunc = () => {
    const myVar = "value";
    return myVar;
  }
  
  // arrow function shorter
  const myFunc = () => "value";
  myFunc(); // "value"

  // arrow function with parameter
  const myFunc = (item) => item * 2;
  myFunc(3); // 6

  // arrow function with parameters
  const myConcat = (arr1, arr2) => arr1.concat(arr2);
  myConcat([1,2], [3,4]); // [1,2,3,4]

  // arrow function with higher order functions (.map)
  const arr = [1, 2, 3];
  const squares = arr.map(x => x * x); // 1, 4, 9

  // default parameter for function
  function greet(name = "Anonymous") {
    console.log("Hello" + name);
  }
  great(); // Hello Anonymous
  great("Einstein"); // Hello Einstein

  // Rest in function
  function howMany(...args) {
    return args.length + " arguments";
  }
  howMany("hi", 1, true); // 3 arguments


  // ARRAYS
  // Spread operator with array
  const arr = [1, 9, 3, 4];
  const arr2 = [17];
  const newArr = [...arr, ...arr2, 21]; // [1, 9, 3, 4, 17, 21]

  // Copy an array with spread
  const arr = [1, 2, 3, 4];
  const arrCopy = [...arr];

  // String to array with spread
  const str = "Aloha";
  const arr = [...str]; // [A, l, o, h, a]

  // rest for destructuring array
  const [first, ...rest] = [1, 9, 3, 4]; // 1 and [9, 3, 4]


  // some loop
  const arr = [1, 2, 3, 4, 5];
  arr.some((value) => { return (value == 3); }); // returns true; a single array value must be true to have a true output from an every loop


  // every loop
  const arr = [1, 2, 3, 4, 5];
  arr.every((value) => { return (value == 3); }); // returns false; all array values must be true to have a true output from an every loop


  // Class syntax; replaces constructor function
  class SpaceShuttle {
    constructor(targetPlanet){
      this.targetPlanet = targetPlanet;
    }
  }
  const zeus = new SpaceShuttle('Jupiter');

  // Getters and Setters
  class Thermostat {
    constructor(farenheit) {
      this.farenheit = farenheit;
    }
    get temperature() {
      return (this.farenheit - 32) * 5/9;
    }
    set temperature(celsius){
      this.farenheit = celsius * 9.0 / 5 + 32;
    }
  }
  const thermos = new Thermostat(76); // setting in Fahrenheit scale
  let temp = thermos.temperature; // 24.44 °C




// POP-UPS
// launch pop-up


  confirm("Are you sure to leave?"); // launch pop-up to confirm -> return true/false

  prompt("Enter tour name"); // launch pop-up where user can answer -> return the answer

  alert("hello"); // launch pop-up with text

  var name = prompt("Enter your name :"); alert("hello, " + name); // pop-up to type name and another one to say 'hello name'



// USEFUL TOOLS
// other useful snippet of codes


  // create audio and start it
  var audio = new Audio("file.mp3");
  audio.play();




// COMPARISONS
// Comparison and Logical operators are used to test for true or false


  // smaller than 
  console.log(8 < 10); // true
  console.log(10 < 8); // false
    
  // greater than
  console.log(10 > 8); // true
  console.log(8 > 10); // false
    
  // smaller or equal to
  console.log(10 <= 10); // true
  console.log(10 <= 8); // false
    
  // greater or equal to
  console.log(10 >= 10); // true
  console.log(8 >= 10); // false
    
  // equal to (weak)
  console.log(10 == 10); // true
  console.log(8 == 10); // false
    
  // different from (weak)
  console.log(10 != 8); // true
  console.log(8 != 8); // false
    
  // equal to and same type (strict)
  console.log(10 === 10); // true
  console.log(8 === 10); // false
    
  // different and different type from (strict)
  console.log(10 !== 8); // true
  console.log(8 !== 8); // false
    
  // 'AND'
  console.log((10 > 8) && (10 === 10)); // true
  console.log((10 > 8) && (8 === 10)); // false
    
  // 'OR'
  console.log((8 < 10) || (8 > 10)); // true
  console.log((10 < 8) || (8 > 10)); // false
    
  // opposite of instruction
  console.log(!(10 < 8)); // true
  console.log(!(8 < 10)); // false

  
  

// CONDITIONS
// Conditional statements are used to perform different actions based on different conditions


  // return the sentence because it is true
  var a = 4;
  if (a === 4) {
    console.log("a is 4"); // OK
  }


  // return the else if because if is false
  var a = 4;
  if (a > 4) {
    console.log("a is greater than 4");
  } else if (a === 4) {
    console.log ("a is 4"); // OK
  }


  // return the else because it's false
  var a = 4;
  if (a > 4) {
    console.log("a is greater than 4");
  } else if (a < 4) {
    console.log("a is less than 4");
  } else {
    console.log ("a isn't greater or less than 4 --> a is 4"); // OK
  }


  // switch is a better way for defined options // do not forget the break; !
  var a = 4; 
  switch (a) {
    case "1":
      console.log("a is 1");
      break;
    case "2":
      console.log("a is 2");
      break;
    case "4":
      console.log("a is 4"); // OK
      break;
    default: // if none of cases is true
      console.log("a is a variable");
      break;
  }


  // if you do not use breaks, cases will merge till the next break
  var a = 4;
  switch (a) {
    case 1:
    case 2:
    case 3:
      console.log("a is 1, 2, 3");
      break;
    case 4:
      console.log("a is 4"); // OK
      break;
    case 7:
    case 8:
    case 9:
      console.log("a is 7, 8, 9");
      break;
  }




// SHORT CIRCUIT
// handy way to manage short conditions

  
  // Logical &&
  var a = true && true; // true
  var a = true && false; // false
  var a = false && false; // false
  var a = "cat" && "dog"; // "dog"


  // Logical ||
  var a = true || true; // true
  var a = true || false; // true
  var a = false || false; // false
  var a = "cat" || "dog"; // "cat"


  // with a function
  function short(test) {
    var a = test || "Test is not defined"; // a = test value if defined; otherwise a = string
  }


  // chose between functions
  function short() {
    func1() || func2(); // runs func1 if it exists; otherwise runs func2
  }




// LOOPS
// while the condition is true, the code is looping


  // while loop; use it by default
  var a = 1;
  while (a <= 4) {
    console.log(a);
    a++;
  } // 1 2 3 4

  
  // /!\ careful with infinite loops --> it breaks applications
  var a = 1;
  while (a <= 4) {
    console.log(a); // the condition will always be true and will therefore lead to infinite loop
  }


  // do while loop; while alike, but will do the first block code at least once (even if while is false)
  var a = 1;
  do {
    console.log(a);
    a++;
  }
  while (a < 1); // 1


  // for loop; use when you know the number of loops
  var a;
  for (a = 1; a <= 4; a++) {
    console.log(a);
  } // 1 2 3 4


  // when variable is only used in the for loop
  for (var a = 1; a <= 4; a++) {
    console.log(a);
  } // 1 2 3 4


  // break; end of the loop
  for (var a = 1; a <= 4; a++) {
    if (a === 3) {
      break;  
    }
    console.log(a);
  } // 1 2; stops when 3 reached


  // continue; breaks one iteration in the loop
  for (var a = 1; a <= 4; a++) {
    if (a === 3) {
      continue; 
    }
    console.log(a);
  } // 1 2 4; skips 3


  // fizzbuzz (similar to dingdingbottle)
  for (var a = 1; a <= 100; a++) {
    if ((a % 3 === 0) && (a % 5 === 0)) {
      console.log("FizzBuzz");
    } else if (a % 3 === 0) {
      console.log("Fizz");
    } else if (a % 5 === 0) {
      console.log("Buzz");
    } else {
      console.log(a);
    }
  } // 1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz ...


  // forEach loop
  var a = [1, 2, 3, 4, 5];
  a.forEach(function(element) {
    console.log(element);
  }); // 1 2 3 4 5



  // recursion
  function countDownFrom(number) {
    if (number === 0) { return; }
    console.log(number);
    countDownFrom(number - 1);
  }
  countDownFrom(5); // 5 4 3 2 1




// FUNCTIONS
// A JavaScript function is a block of code designed to perform a particular task. You need to invoke it


  // simple function
  function sayHello() {
    console.log('Hello !');
  }
  sayHello(); // Hello !


  // function with return; function exits at return statement, whatever comes after a return line never outputs
  function sayHello() {
    return 'Hello !';
  }
  console.log(sayHello()); // Hello !


  // function with return and variable
  function sayHello() {
    return 'Hello !';
  }
  var resultat = sayHello();
  console.log(resultat); // Hello !


  // local variables; they can't be used outside the function
  function sayHello() {
    var message = 'Hello !';
    return message;
  }
  console.log(sayHello()); // Hello !
  console.log(message); // error, variable works only inside function


  // function with parameters
  function sayHello(name) {
    var message = 'Hello, ' + name + ' !';
    return message;
  }
  console.log(sayHello('Beumsk')); // Hello, Beumsk !


  // function with multiple parameters
  function sayHello(name, surname) {
    var message = 'Hello, ' + name + ' ' + surname + ' !';
    return message;
  }
  console.log(sayHello('Mr', 'Beumsk')); // Hello, Mr Beumsk !


  // function with for loop
  function carre(number) {
    var resultat = number * number;
    return resultat;
  }
  for (var i = 0; i <= 10; i++) {
    console.log(carre(i));
  }


  // function with if
  function min(number1, number2) {
    if (number1 > number2) {
      return number2;
    } else {
      return number1;
    }
  }
  console.log(min(9, 1));


  // function nested in function
  function first() {
    console.log("first");
    return function() {
      console.log("second");
    }
  }
  first(); // logs first
  first()(); // logs first and second


  // Double arg Function
  function say(arg1) {
    return function(arg2) {
      return arg1 + " " + arg2;
    }
  }
  say ("hello")("world"); // hello world


  // calculator
  function calculator(number1, mult, number2) {
    if (mult === '+') {
      return number1 + number2;
    } else if (mult === '-') {
      return number1 - number2;
    } else if (mult === '*') {
      return number1 * number2;
    } else if (mult === '/') {
      return number1 / number2;
    }
  }
  console.log(calculator(4, "+", 6)); // returns 10
  console.log(calculator(4, "-", 6)); // returns -2
  console.log(calculator(2, "*", 0)); // returns 0
  console.log(calculator(2, "/", 0)); // returns Infinity


  // circle perimeter and radius
  function perimeter(radius) {
    return 2 * Math.PI * radius;
  }
  function area(radius) {
    return Math.PI * radius * radius;
  }
  console.log(perimeter(10)); // 62.83185307179586
  console.log(area(10)); // 314.1592653589793




// STRING 
// methods working on text strings


  // string.length with variables
  var str = "Kangaroo";
  console.log(str.length); // 8


  // string.includes() check if string includes another stinrg
  var str = "Kangaroo";
  console.log(str.includes("anga")); // true


  // string.startsWith() check if string starts with something
  var str = "Kangaroo";
  console.log(str.startsWith("Kan")); // true


  // string.endsWith() check if string ends with something
  var str = "Kangaroo";
  console.log(str.endsWith("roo")); // true


  // string.charAt() return specific character
  var str = "Kangaroo";
  console.log(str[0]); // "K"
  console.log(str.charAt(0)); // "K"


  // string.charCodeAt() returns charCode
  var str = "Kangaroo";
  console.log(str.charCodeAt(0)); // charCode of "K" => 75


  // string.fromCharCode() returns character from charCode
  console.log(String.fromCharCode(75)); // show character of 75 charcode => "A"


  // string.indexOf() return position of a character
  var str = "Kangaroo";
  console.log(str.indexOf("a")); // 1


  // string.lastIndexOf() return position of a character
  var str = "Kangaroo";
  console.log(str.lastIndexOf("a")); // 4


  // string.trim() remove spaces at beginning and end of a string
  var str = "   Kangaroo   ";
  console.log(str.trim()); // "Kangaroo"
  console.log(str.trimStart()); // "Kangaroo   "
  console.log(str.trimEnd()); // "   Kangaroo"


  // string.slice() picks part of the string (start, end-1)
  var str = "Kangaroo";
  console.log("slice".slice(0, 3)); // "Kan"


  // string.repeat() multiply a string !!IE
  var str = "Kangaroo";
  console.log(str.repeat(3)); // "KangarooKangarooKangaroo"


  // string.toLowerCase()
  var str = "Kangaroo";
  console.log(str.toLowerCase()); // "kangaroo"


  // string.toUpperCase()
  var str = "Kangaroo";
  console.log(str.toUpperCase()); // "KANGAROO"


  // string.replace() replace a string in a string
  var str = "Salut, c'est chouette !";
  console.log(str.replace("chouette", "cool")); // "Salut, c'est cool !"


  // use replace to add spaces before uppercases; or smth else
  "myNameIsWhat".replace(/([a-z])([A-Z])/g, "$1 $2"); // my Name Is What
  "myNameIsWhat".replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase(); // my-name-is-what


  // return all character from a string
  var word = "Kangaroo";
  for (var i = 0; i < word.length; i++) {
    console.log(word[i]);
  }


  // lots of process on a word
  var word = "Kangaroo";
  function countVowels(word) {
    var vowels = 0;
    for (var i = 0; i < word.length; i++) {
      var letter = word[i].toLowerCase();
      if ((letter === "a") || (letter === "e") || (letter === "i") || (letter === "o") || (letter === "u") || (letter === "y")) {
        vowels++;
      }
    }
    return vowels;
  }
  console.log("Vowels: " + countVowels(word)); // Vowels: 4
  console.log("Consonants: " + (word.length - countVowels(word))); // Consonants: 4


  var word = "Kangaroo";
  function backWards(word) {
    var backWord = "";
    for (var i = 0; i < word.length; i++) {
      backWord = word[i] + backWord;
    }
    return backWord;
  }
  console.log("Backwards: " + backWards(word)); // Backwards: ooragnaK

  if (word === backWards(word)) {
    console.log("Palindrome: True");
  } else {
    console.log("Palindrome: False");
  } // Palindrome: False


  var word = "Kangaroo";
  function convertLeetSpeak(word) {
    var leetWord = "";
    for (var i = 0; i < word.length; i++) {
      leetWord = leetWord + findLeetLetter(word[i]);
    }
    return leetWord;
  }
  function findLeetLetter(letter) {
    var leetLetter = letter;
    switch (letter.toLowerCase()) {
    case "a" :
      leetLetter = "4";
      break;
    case "b" :
      leetLetter = "8";
      break;
    case "e" : 
      leetLetter = "3";
      break;
    case "l" :
      leetLetter = "1";
      break;
    case "o" :
      leetLetter = "0";
      break;
    case "s" :
      leetLetter = "5";
      break;
    }
    return leetLetter;
  }  
  console.log(convertLeetSpeak(word)); // K4ng4r00




// NUMBERS
// methods working on numbers


  // toString; number to string
  (123).toString(); // 123
  var x = 123; x.toString(); // 123
  (100 + 23).toString(); // 123


  // toFixed; number of decimals returned as string
  (2.343).toFixed(0); // '2'
  (2.343).toFixed(2); // '2.34'
  (2.343).toFixed(4); // '2.3430'


  // Number; turns value into number
  Number(true); // 1; 0 if false
  Number("10"); // 10
  Number("10 20"); // NaN


  // parseInt; first number returned
  parseInt("10"); // 10
  parseInt("10.33"); // 10
  parseInt("10 20"); // 10
  parseInt("10 years"); // 10
  parseInt("years 10"); // NaN


  // parseInt; first number returned (same as parseInt but with decimals)
  parseFloat("10"); // 10
  parseFloat("10.33"); // 10.33
  parseFloat("10 20"); // 10
  parseFloat("10 years"); // 10
  parseFloat("years 10"); // NaN




// MATH
// Functions working on numbers; they start with 'Math.'


  // Math.min; returns the smallest number from parameters
  Math.min(9, 7, 6, 1); // 1


  // Math.max; returns the biggest number from parameters
  Math.max(9, 1, 6, 3, 5); // 9


  // Math.random; returns a random number between 0 and 1
  Math.random();


  // Math.random; 
  Math.random() * 100; // random number 1 to 100; change '100' to change possibilities of course
  Math.floor(Math.random() * 100); // random integer


  // Math.round
  Math.round(0.6666 * 100) / 100; // 0.67
  Math.round(0.66666666 * 10000) / 10000; // 0.6667


  // Math.ceil; round value up
  Math.ceil(4.3); // 5


  // Math.floor; round value down
  Math.floor(4.7); // 4


  // Math.pow; returns first parameter powered by second parameter
  Math.pow(5, 2); // 25


  // Math.sqrt; returns square root of parameter
  Math.sqrt(16); // 4


  // Math.abs; returns absolute of parameter
  Math.abs(-12); // 12


  // Math.PI
  Math.PI // 3.141592653589793


  // eval; takes a string and do the math
  eval("10 * 5 + 10 / 2") // 55
  
  


// DATE
// Date object enables us to work with dates; dates are calculated from 01 Jan 1970 00:00:00


  // current date and time
  var d = new Date();
  console.log(d); // now


  // set date easily
  var d2 = new Date("January 2, 2015 10:42:00"); // Fri Jan 02 2015 10:42:00


  // set date with numbers
  var d3 = new Date(88,5,11,11,42,0,0); // Sat Jun 11 1988 11:42:00


  // time values
  var d = new Date();
  var hours = d.getHours(); // current hour
  var minutes = d.getMinutes(); // current minute
  var seconds = d.getSeconds(); // current second
  var milliseconds = d.getMilliseconds(); // current millisecond


  // day values
  var d = new Date();
  var day = d.getDay(); // day of week
  var date = d.getDate(); // day of month
  var month = d.getMonth()+1; // add 1 because month count starts at 0 (jan=0; dec=11)
  var year = d.getFullYear(); // current year


  // add days, months, years to a date
  var oldDate = new Date(); // today
  var newDate = new Date(oldDate.getFullYear(), oldDate.getMonth(), oldDate.getDate()+1); // tomorrow
  var newDate = new Date(oldDate.getFullYear(), oldDate.getMonth()+1, oldDate.getDate()); // next month
  var newDate = new Date(oldDate.getFullYear()+1, oldDate.getMonth(), oldDate.getDate()); // next year


  // function to change date format easily
  function convertDate(inputFormat) {
    function pad(s) { return (s < 10) ? "0" + s : s; }
    var d = new Date(inputFormat);
    return [pad(d.getDate()), pad(d.getMonth()+1), d.getFullYear()].join("."); // dd.mm.yyyy
  }




// ARRAYS
// JavaScript arrays are used to store multiple values in a single variable; they are number indexed
// Careful! -> if variable is set equals to another array variable, they will be linked and influence each other (may cause infinite loops)


  // basic
  var arr = [0, 1, 2, 3, 4];
  

  // array length
  var arr = [0, 1, 2, 3, 4];
  console.log(arr.length); // 5


  // empty an array
  var arr = [0, 1, 2, 3, 4];
  arr.length = 0;
  console.log(arr); // []


  // check if array is empty
  var arr = [0, 1, 2, 3, 4];
  if (arr.length !== 0) {
    console.log("Array is not empty !");
  } else {
    console.log("Array is empty");
  }


  // log an array element
  var arr = [0, 1, 2, 3, 4];
  console.log(arr[0]); // 0


  // log all array elements
  var arr = [0, 1, 2, 3, 4];
  for (var i = 0; i < arr.length; i++) {
    console.log(arr[i]);
  } // 0 1 2 3 4


  // log all array elements with forEach
  var arr = [0, 1, 2, 3, 4];
  arr.forEach(function(el) {
    console.log(el); // return in a forEach() doesn't return the whole function but only a single iteration
  }); // 0 1 2 3 4


  // log all array elements with forEach and a predefined function
  var arr = [0, 1, 2, 3, 4];
  function func(el) {
    console.log(el);
  }
  arr.forEach(func); // 0 1 2 3 4


  // log all array elements backwards
  var arr = [0, 1, 2, 3, 4];
  for (var i = arr.length-1; i >= 0; i--) {
    console.log(arr[i]);
  } // 4 3 2 1 0


  // push() add an element (to the end)
  var arr = [0, 1, 2, 3, 4];
  arr.push(5); // possible to add multiple elements
  console.log(arr); // [0, 1, 2, 3, 4, 5]


  // unshift() add an element (from the front)
  var arr = [0, 1, 2, 3, 4];
  arr.unshift(-1); // possible to add multiple elements
  console.log(arr); // [-1, 0, 1, 2, 3, 4]

  // pop() remove last element
  var arr = [0, 1, 2, 3, 4];
  arr.pop();
  console.log(arr); // [0, 1, 2, 3]


  // shift() remove first element
  var arr = [0, 1, 2, 3, 4];
  arr.shift();
  console.log(arr); // [1, 2, 3, 4]


  // pop() element and store the removed element
  var arr = [0, 1, 2, 3, 4];
  var deleted = arr.pop(); // 4 is no more in array and stored in 'deleted'
  console.log(deleted); // 4
  console.log(arr); // [0, 1, 2, 3]


  // shift() element and store the removed element
  var arr = [0, 1, 2, 3, 4];
  var deleted = arr.shift(); // 0 is no more in array and stored in 'deleted'
  console.log(deleted); // 0
  console.log(arr); // [1, 2, 3, 4]


  // turn arguments of a function into an array
  var arr;
  function func() {
    arr = Array.prototype.slice.call(arguments);
  }
  func(0, 1, 2, 3, 4);
  console.log(arr); // [0, 1, 2, 3, 4]


  // loop through function arguments
  function func(arr1, arr2, arr3) {
    for (i=0; i<arguments.length; i++) {
      console.log(arguments[i]);
    }
  }
  func(0, 1, 2, 3, 4); // 0 1 2 3 4


  // map() make change on arrays; iterate
  var arr = [1, 2, 3];
  var arr2 = arr.map(function(val) {
    return val * 3;
  });
  console.log(arr2); // [3, 6, 9]; they have been multiplied


  // reduce() use all values, one at a time, to get a single one; condense
  var arr = [0, 1, 2, 3, 4];
  var singleVal = arr.reduce(function(combinedValue, currentValue) {
    return combinedValue - currentValue;
  }, 0); // start at value[0]; do not use it for multiplications
  console.log(singleVal); // will result in -0-1-2-3-4 = -10


  // filter() an array
  var arr = [0, 1, 2, 3, 4];
  var arr2 = arr.filter(function(val) {
    return val < 3; // keep what is said; all values under 3
  });
  console.log(arr2); // [0, 1, 2]


  // delete some array values
  var arr = [0, 1, 2, 3, 4];
  delete arr[2];
  console.log(arr); // [0, 1, null, 3, 4]
  arr = arr.filter(Boolean); // to remove null, undefined etc.
  console.log(arr); // [0, 1, 3, 4]


  // filter() Boolean, will delete false, null, 0, "", undefined, and NaN.
  function numbersLettersOnly(arr) {
    return arr.filter(Boolean);
  }
  numbersLettersOnly([7, "abc", false, null, 0, 9, NaN, "hello", undefined, ""]); // [7, "abc", 9, "hello"]


  // sort() an array; for alphabetic sorting, just need sort()
  var arr = [0, 1, 2, 3, 4];
  arr.sort(function(a, b) {
    return b - a; // from largest; a - b to sort from smallest
  });
  console.log(arr); // [4, 3, 2, 1, 0]


  // indexOf() get index position of a value in an array
  var arr = [0, 1, 2, 3, 4];
  var i = arr.indexOf(3);
  console.log(i); // 3; position of value 3


  // indexOf() can be used to check if an array includes an element
  var arr = [0, 1, 2, 3, 4];
  var i = arr.indexOf(6);
  console.log(i); // -1; the value isn't in the array


  // lastIndexOf() get index position of a value in an array (starting from the end)
  var arr = [0, 1, 2, 3, 3, 4];
  var i = arr.lastIndexOf(30);
  console.log(i); // 4; position of last value 3


  // reverse() an array
  var arr = [0, 1, 2, 3, 4];
  arr.reverse();
  console.log(arr); // [4, 3, 2, 1, 0]


  // concat() an array with another one
  var arr = [1, 2, 3];
  var concatenateMe = [4, 5, 6];
  var arr2 = arr.concat(concatenateMe);
  console.log(arr2); // [1, 2, 3, 4, 5, 6]


  // split() a string into an array
  var str = "0 1 2 3 4";
  var arr = str.split(" "); // can be other character
  console.log(arr); // [0, 1, 2, 3, 4]


  // split() number into an array 
  var num = 1234;
  var arr = (""+num).split("");
  console.log(arr); // ["1", "2", "3", "4"]


  // join() an array into a string
  var arr = ["0", "1", "2", "3", "4"];
  var str = arr.join(" ");
  console.log(str); // "0 1 2 3 4"


  // slice() to copy parts of an array
  var arr = [0, 1, 2, 3, 4];
  var arr2 = arr.slice(0, 2); // keep from index O to 2 (not taken)
  console.log(arr); // [0, 1, 2, 3, 4]
  console.log(arr2); // [0, 1]


  // slice() to copy an array
  var arr = [0, 1, 2, 3, 4];
  var arr2 = arr.slice(); // keep all
  console.log(arr2); // [0, 1, 2, 3, 4]


  // splice() a single element
  var arr = [0, 1, 2, 3, 4];
  var arr2 = arr.splice(2, 1); // arr = from index 2, remove 1 || arr2 = from index 2, keep 1
  console.log(arr); // [0, 1, 3, 4]
  console.log(arr2); // [2]


  // splice() multiple elements
  var arr = [0, 1, 2, 3, 4];
  var arr2 = arr.splice(3, 2); // arr = from index 3, remove 2 || arr2 = from index 3, keep 2
  console.log(arr); // [0, 1, 2]
  console.log(arr2); // [3, 4]


  // array destructuring
  var arr = [0, 1, 2, 3, 4];
  var [zero, one, two, ...rest];
  // var [zero, one, two, ...rest] = [0, 1, 2, 3, 4];
  console.log(zero, one, two, rest); // 0, 1, 2, [3, 4]


  // find the right card 
  var cards = ['Diamond', 'Spade', 'Heart', 'Club'];
  var currentCard = 'Heart';
  while (currentCard !== 'Spade') {
    console.log(currentCard);
    var randomNumber = Math.floor(Math.random() * 4);
    currentCard = cards[randomNumber];
  }
  console.log('Spade found !');


  // list of films
  var Films = [];
  var Film = {
    init: function(title, year, director) {
      this.title = title;
      this.year = year;
      this.director = director;
    },
    describe: function() {
      return this.title + ' (' + this.year + ', ' + this.director + ')';
    }
  };
  var film1 = Object.create(Film);
  film1.init('Star Wars', 1977, 'Georges Lucas');
  Films.push(film1);
  var film2 = Object.create(Film);
  film2.init('LOTR', 2001, 'Peter Jackson');
  Films.push(film2);
  var film3 = Object.create(Film);
  film3.init('Indiana Jones', 1981, 'Steven Spielberg');
  Films.push(film3);
  Films.forEach(function(Film) {
    console.log(Film.describe());
  });




// OBJECTS
// they are like complexed variables; they include multiple parameters and functions; they are named indexed


  // simple object with 3 properties
  var pen = {
    type: "ball",
    color: "blue",
    brand: "Bic"
  };
  console.log(pen.type); // ball
  console.log(pen.color); // blue
  console.log(pen.brand); // Bic


  // other way to create object
  var obj = new Object();
  obj.prop = "anything";


  // change property
  pen.color = "red";


  // other way to set and get properties; useful for var or parameters, or with var containing spaces
  pen["is working"] = true;
  console.log(pen["is working"]); // true


  // add property
  pen.price = 2;
  console.log("My " + pen.brand + " " + pen.color + " " + pen.type + "pen costs " + pen.price + " euros." );
  

  // delete a property
  delete pen.price;


  // check if a property exists; true or false
  pen.hasOwnProperty("color");
  // or
  "color" in pen;


  // for in loop
  var person = {firstName: "John", lastName: "Doe", age: 25};
  for (var x in person) {
    console.log(x + ": " + person[x]); // logs all property value pairs
  }


  // copy an object (by reference); objects will always be linked to each other
  var person = {firstName: "John", lastName: "Doe", age: 25};
  var copy = person;
  person.firstName = "David";
  console.log(copy.firstName); // David and not John


  // clone an object; makes object independant;
  var person = {firstName: "John", lastName: "Doe", age: 25};
  var clone = {};
  for (var key in person) {
    clone[key] = person[key];
  }


  // object destructuring
  var obj = {zero: 0, one: 1, two: 2, three: 3, four: 4};
  var {zero, one, two, ...rest};
  // var {zero, one, two, ...rest} = {zero: 0, one: 1, two: 2, three: 3, four: 4};
  console.log(zero, one, two, rest); // 0, 1, 2, {three: 3, four: 4}


  // basic rpg with function inside object
  var char = {
    name: 'Beumsk',
    hp: 200,
    strength: 150,
    describe: function() {
      var description = this.name + ' has ' + this.hp + ' hp and ' + this.strength + ' strength.';
      return description;
    }
  };
  console.log(char.describe());
  console.log(char.name + ' is hurt by an arrow! 20 hp lost...');
  char.hp = char.hp - 20;
  console.log(char.describe());
  console.log(char.name + ' found a strength ring! 10 strength more!');
  char.strength = char.strength + 10;
  console.log(char.describe());


  // dog describe + function
  var dog = {
    name: 'Medor',
    race: 'Labrador',
    size: 88,
    bark: function() {
      return 'Grrrrr !';
    }
  };
  console.log(dog.name + ' is a ' + dog.race + ' of ' + dog.size + ' cm.');
  console.log('Watch out ! A cat ! ' + dog.name + ' Bark : ' + dog.bark());


  // prompt radius to give perimeter and area
  var r = Number(prompt("Enter the circle radius :"));
  var circle = {
    perimeter: function() {
      return 2 * Math.PI * r;
    },
    area: function() {
      return Math.PI * r * r;
    }
  };
  console.log("Perimeter = " + circle.perimeter());
  console.log("Area = " + circle.area());


  // account debit and credit
  var account = {
    holder: 'Jeanne',
    balance: 0,
    credit: function(money) {
      this.balance = this.balance + money;
    },
    debit: function(money) {
      this.balance = this.balance - money;
    },
    describe: function() {
      return 'Holder: ' + this.holder + '; Balance: ' + this.balance;
    }
  };
  console.log(account.describe());
  account.credit(200);
  account.debit(150);
  console.log(account.describe());


  // account debit and credit with prompt
  var credit = Number(prompt('Enter your credit'));
  var debit = Number(prompt('Enter your debit'));
  var account = {
    holder: 'Jeanne',
    balance: 0,
    credit: function() {
      this.balance = this.balance + credit;
    },
    debit: function() {
      this.balance = this.balance - debit;
    },
    describe: function() {
      return 'Holder: ' + this.holder + '; Balance: ' + this.balance;
    }
  };
  console.log(account.describe());
  account.credit();
  account.debit();
  console.log(account.describe());




// OOP
// Object Oriented programming


// Capitalize first letter of an object constructor
function Dog(name, sex, age) {
  this.name = name;
  this.sex = sex;
  this.age = age;
}
var example = new Dog("Dogidog", "male", 3);

// check if object is an instance of a constructor
example instanceof Dog; // true
fake instanceof Dog; // false

// check own properties
var ownProps = [];
for (var property in example) {
  if(example.hasOwnProperty(property)) {
    ownProps.push(property); // "name", "sex", "age"
  }
}




// PROTOTYPES
// they allow objects to get values from other objects


  // simple prototype
  var anObject = {
    a: 10
  };
  var anotherObject = Object.create(anObject);
  console.log(anObject.a);
  console.log(anotherObject.a);


  // multiple prototype
  var anObject = {
    a: 10
  };
  var anotherObject = Object.create(anObject);
  var stillAnotherObject = Object.create(anotherObject);
  console.log(anObject.a);
  console.log(anotherObject.a);
  console.log(stillAnotherObject.a);


  // use prototype to create new characters
  var char = {
    name: '',
    hp: 0,
    strength: 0,
    xp: 0,
    describe: function() {
      return this.name + ' has ' + this.hp + ' hp, ' + this.strength + ' strength and ' + this.xp + ' xp.';
    }
  };
  var nala = Object.create(char);
  nala.name = 'Nala';
  nala.hp = 200;
  nala.strength = 150;
  var apa = Object.create(char);
  apa.name = 'Apa';
  apa.hp = 300;
  apa.strength = 200;
  console.log(nala.describe());
  console.log(apa.describe());


  // faster way
  var char = {
    init: function(name, hp, strength) {
      this.name = name;
      this.hp = hp;
      this.strength = strength;
      this.xp = 0;
    },
    describe: function() {
      return this.name + ' has ' + this.hp + ' hp, ' + this.strength + ' strength and ' + this.xp + ' xp.';
    }
  };
  var nala = Object.create(char);
  nala.init('Nala', 200, 150);
  var apa = Object.create(char);
  apa.init('Apa', 300, 200);
  console.log(nala.describe());
  console.log(apa.describe());


  // with ennemies
  var char = {
    initChar: function(name, hp, strength) {
      this.name = name;
      this.hp = hp;
      this.strength = strength;
    }
  };
  var player = Object.create(char);
  player.initPlayer = function(name, hp, strength) {
    this.initChar(name, hp, strength);
    this.xp = 0;
  };
  player.describe = function() {
    return this.name + ' has ' + this.hp + ' hp, ' + this.strength + ' strength and ' + this.xp + ' xp.';
  };
  var ennemy = Object.create(char);
  ennemy.initEnnemy = function(name, hp, strength, race, value) {
    this.initChar(name, hp, strength);
    this.race = race;
    this.value = value;
  };
  var nala = Object.create(player);
  nala.initPlayer('Nala', 200, 150);
  var apa = Object.create(player);
  apa.initPlayer('Apa', 300, 200);
  console.log('Welcome ! Check out our heroes :');
  console.log(nala.describe());
  console.log(apa.describe());
  var scar = Object.create(ennemy);
  scar.initEnnemy('Scar', 500, 350, 'Lion', 10);
  console.log('An awful monster arrives : it is a ' + scar.race + ' named ' + scar.name);


  // with damages
  var char = {
    initChar: function(name, hp, strength) { // initiate character
      this.name = name;
      this.hp = hp;
      this.strength = strength;
    },
    attack: function (target) { // attack a target
      if (this.hp > 0) {
        var damages = this.strength;
        console.log(this.name + ' attacks ' + target.name + ' for a total of ' + damages + ' combat points!');
        target.hp = target.hp - damages;
        if (target.hp > 0) {
          console.log(target.name + ' have now ' + target.hp + ' hp.');
        } else {
          target.hp = 0;
          console.log(target.name + ' is dead!');
        }
      } else {
        console.log(this.name + ' can not attack, he is dead...');
      }
    }
  };
  var player = Object.create(char);
  player.initPlayer = function(name, hp, strength) { // initiate player (proto char)
    this.initChar(name, hp, strength);
    this.xp = 0;
  };
  player.describe = function() { // describe player
    return this.name + ' has ' + this.hp + ' hp, ' + this.strength + ' strength and ' + this.xp + ' xp.';
  };
  player.fight = function(ennemy) { // fight an ennemy (takes the prototype of attack)
    this.attack(ennemy);
    if (ennemy.hp === 0) {
      console.log(this.name + ' killed ' + ennemy.name + ' and gets ' + ennemy.value + ' xp!');
      this.xp += ennemy.value;
    }
  };
  var ennemy = Object.create(char);
  ennemy.initEnnemy = function(name, hp, strength, race, value) { // initiate ennemy (proto char)
    this.initChar(name, hp, strength);
    this.race = race;
    this.value = value;
  };
  var nala = Object.create(player);
  nala.initPlayer('Nala', 200, 150);
  var apa = Object.create(player);
  apa.initPlayer('Apa', 300, 200);
  console.log('Welcome ! Check out our heroes :');
  console.log(nala.describe());
  console.log(apa.describe());
  var scar = Object.create(ennemy);
  scar.initEnnemy('Scar', 300, 50, 'Lion', 100);
  console.log('An awful monster arrives : it is a ' + scar.race + ' named ' + scar.name);
  scar.attack(nala);
  scar.attack(apa);
  nala.fight(scar);
  apa.fight(scar);
  console.log(nala.describe());
  console.log(apa.describe());


  // 
  var Dog = {
    init: function (name, race, size) {
      this.name = name;
      this.race = race;
      this.size = size;
    },
    bark: function() {
      return 'Grrr !';
    }
  };
  var rachel = Object.create(Dog);
  rachel.init("Rachel", "labrador", 75);
  console.log(rachel.name + " is a " + rachel.race + " measuring " + rachel.size + " cm");
  console.log("Ow a cat ! " + rachel.name + " bark : " + rachel.bark());
  var pheobe = Object.create(Dog);
  pheobe.init("Pheobe", "golden", 22);
  console.log(pheobe.name + " is a " + pheobe.race + " measuring " + pheobe.size + " cm");
  console.log("Ow a cat ! " + pheobe.name + " bark : " + pheobe.bark());


  // transfer between spare and account
  var Account = {
    initCB: function (name, balance) {
      this.name = name;
      this.balance = balance;
    },
    describe: function () {
      return 'Holder : ' + this.name + ', balance : ' + this.balance;
    },
    debit: function () {
      this.balance = this.balance - amount;
    },
    credit: function() {
      this.balance = this.balance + amount;
    }
  };
  var Spare = Object.create(Account);
  Spare.initCE = function(name, balance, interest) {
    this.initCB(name, balance);
    this.interest = interest;
  };
  Spare.addInterests = function() {
    var interests = this.balance * this.interest;
    this.balance += interests;
  };
  var account1 = Object.create(Account);
  account1.initCB("Alex", 100);
  var account2 = Object.create(Spare);
  account2.initCE("Marco", 50, 0.05);
  console.log("Here is the inital state of the accounts :");
  console.log(account1.describe());
  console.log(account2.describe());
  var amount = Number(prompt("Enter the amount to transfer :"));
  account1.debit(amount);
  account2.credit(amount);
  account2.addInterests();
  console.log("Here is the total after transfer and interests :");
  console.log(account1.describe());
  console.log(account2.describe());




// CLASSES
// kind of special functions; classes cannot be called before it appears in the code (such as functions)

  
  // syntax example
  class Rectangle {
    constructor(hauteur, largeur) {
      this.hauteur = hauteur;
      this.largeur = largeur;
    }
  }

  // https://developer.mozilla.org/fr/docs/Web/JavaScript/Reference/Classes





// REGEXP 
// regular expression object for matching text with a pattern


  /abc/; // Find strings containing "abc"

  /^abc/; // Find strings starting with "abc"

  /abc$/; // Find strings ending with "abc"

  /^abc.*abc$/; // Find strings starting with "abc" and ending with "abc"
  
  /[abc]/; // Find any characters between the brackets
  
  /[^abc]/; // Find any character NOT between the brackets
  
  /[a-c]/; // Find any letters between a and c

  /[a-z]/; // Find any letters in lowercase
  
  /[A-C]/; // Find any letters between A and C

  /[A-Z]/; // Find any letters in uppercase
  
  /[0-9]/; // Find any digits between the brackets; similar to \d
  
  /[^0-9]/; // Find any non-digit between the brackets; similar to \D
  
  /(abc|xyz)/; // Find any alternatives separated with |
  
  /\d/; // Find a digit; similar to [0-9]

  /\D/; // Find a non digit; similar to [^0-9]
  
  /\s/; // Find a whitespace character

  /\S/; // Find anything but a whitespace character
  
  /\w/; // Find any alphanumeric character (and underscore); similar to [A-Za-z0-9_]

  /\W/; // Find any non-alphanumeric character; opposite of \w

  /\n/; // Find a newline

  /\t/; // Find a tab
  
  /n+/; // Matches any string that contains at least one n
    
  /n*/; // Matches any string that contains zero or more occurrences of n
    
  /n?/; // Matches any string that contains zero or one occurrences of n

  /a{2}/; // find strings with 2 a

  /a{2,5}/; // find strings with 2 to 5 a

  /a{3,}/; // find strings with 3 or more a
    
  /test/i; // Perform case-insensitive matching; meaning upper or lowercased Test, TEST, tEST, etc.
  
  /test/g; // Perform a global match (find all matches rather than stopping after the first match)
  
  /test/m; // Perform multiline matching

  /./; // anything 


  // Special characters to be escaped
  // \ / [ ] ( ) { } ? + * | . ^ $
  
  
  // The test() method searches a string and return true or false
  /me/i.test("Am I in this string?"); // false
  /am/i.test("I am in this string!"); // true
  
  
  // The exec() method searches a string and return the found text or null
  /me/i.exec("Am I in this string?"); // null
  /am/i.exec("I am in this string!"); // 'am'


  // The match() method uses an expression to search for a match and return that match
  "try to find me".match(/me/i); // "me"
  
  
  // The search() method uses an expression to search for a match, and returns the position of the match.
  "try to find me".search(/me/i); // 12
  
  
  // The replace() method returns a modified string where the pattern is replaced.
  "try to find him".replace(/him/i, "me"); // try to find me


  // replace() works also like this
  "this code is shit".replace(/[aeiou]/gi, ''); // ths cd s sht
  


  
// DOM BASIS
// nodes are linked to html tags; 2 types: element () and textual ()
  
  
  // show html part in console; here body
  console.log(document.body);

  
  // element node check
  if (document.body.nodeType === document.ELEMENT_NODE) {
    console.log("Body is an element node");
  }

  
  // textual node check
  if (document.body.nodeType === document.TEXTUAL_NODE) {
    console.log("Body is a textual node");
  }

  
  // access element node first child (here body element)
  console.log(document.body.childNodes[0]); // notice that it will return #text because it counts the space before first html tag
  console.log(document.body.childNodes[1]); // return real first html tag
  
  
  // access element relationships (here header element); can be an element, text or comment...
  console.log(getElementsByTagName("header").childNodes); // returns an array of header child nodes
  console.log(getElementsByTagName("header").firstChild); // returns an header first child  
  console.log(getElementsByTagName("header").lastChild); // returns an header last child
  console.log(getElementsByTagName("header").hasChildNodes); // returns true if childnodes; false otherwise
  console.log(getElementsByTagName("header").nextSibling); // returns next node on same level
  console.log(getElementsByTagName("header").previousSibling); // returns previous node on same level
  console.log(getElementsByTagName("header").parentNode); //returns parent node


  // access the first element node
  console.log(getElementById("parent").firstElementChild); 


  // access actual child elements
  console.log(getElementsByTagName("header").children); // returns an array of children elements; no stupid text nodes

  
  // check the child list
  for (var i = 0; i < document.body.childNodes.length; i++) {
    console.log(document.body.childNodes[i]);
  }

  
  // check node parent
  var h1 = document.body.childNodes[1]; // must be done, otherwise h1 would be undefined
  console.log(h1.parentNode); // return body node


  
  
// DOM GET/QUERY
// all ways to reach html elements

  
  // multiple childNodes; not very effective
  console.log(document.body.childNodes[1].childNodes[1].childNodes[1]);

  
  // get Elements By Tag Name
  var elts = document.getElementsByTagName("h2"); // all h2 titles; elts is common abreviation for elements
  console.log(elts[0]); // first h2 title
  console.log(elts.length); // number of titles
  console.log(document.getElementsByTagName("h2")[0]); // faster way

  
  // get Elements By Class Name
  var elts = document.getElementsByClassName("elts");
  for (var i = 0; i < elts; i++) {
    console.log(elts[i]);
  }
  console.log(document.getElementsByClassName("elts")[0]) // faster way to get the first one

  
  // get Element By Id; pay attention to Element which is singular here
  console.log(document.getElementById("elt"));

  
  // css selector
  console.log(querySelectorAll("p")[1]); // second p
  console.log(querySelectorAll("#elt p")[1]); // second p of #elt
  console.log(querySelectorAll(".elts")[1]); // second tag of .elts

  
  // css selector; first only
  console.log(querySelector("p")); // first p only

  
  // return all html code
  console.log(document.body.innerHTML); // works with all above get commands

  
  // return html text without tags
  console.log(document.body.textContent); // works with all above get commands

  
  // get Attribute
  console.log(document.querySelector("a").getAttribute("href")); // return href attribute of first a

  
  // id, href and value attributes are direct; note that href will return full URL !
  console.log(document.querySelector("a").href);

  
  // check presence of attribute
  if (document.querySelector("a").hasAttribute("target")) {
    console.log("First a has target attribute");
  } else {
    console.log("First a has no target attribute");
  }

  
  // List of classes of a DOM element
  console.log(document.getElementById("elt").classList[0]);

  
  // check presence of a class in a DOM element
  if (document.getElementById("elt").classList.contains("main")) {
    console.log("Identified element has 'main' class");
  } else {
    console.log("Identified element has not 'main' class");
  }


  
  
// DOM CHANGE
// ways to change html elements; clear, add, etc.

  
  // add some html using innerHTML
  document.querySelector("p").innerHTML += "<span>!!!</span>"; // adds !!! to p

  
  // innerHTML is more often used to clear an element content
  document.querySelector("p").innerHTML = ""; // clear p

  
  // add some text using textContent
  document.querySelector("p").textContent += " more text in p";


  // create and or set an attribute
  document.querySelector("p").setAttribute("id", "title"); // first its name and secondly its value 

  
  // id, href, src and value are easier
  document.querySelector("p").id = "title";

  
  // standalone attributes
  document.querySelector("p").disabled = false; // or true
  document.querySelector("p").setAttribute("required", "false"); // other way to achieve it

  
  // classList
  var elt = document.querySelector("p");
  elt.classList.remove("main");
  elt.classList.remove("main", "other");
  elt.classList.add("title");
  elt.classList.add("title", "other");
  elt.classList.toggle("else");
  // would be the same as below
  document.querySelector("p").setAttribute("class", "title");

  
  // adding an element
  var addedElt = document.createElement("span"); // span created
  addedElt.id = "added"; // define id
  addedElt.textContent = "Added span"; // define text content
  document.querySelector("p").appendChild(addedElt); // new element insertion

  
  // adding a node
  var addedElt = document.createElement("span"); // span created
  addedElt.id = "added"; // define id
  addedElt.appendChild(document.createTextNode("Added element")); // define text content
  document.querySelector("p").appendChild(addedElt); // new element insertion

  
  // add a node before another node
  var addedElt = document.createElement("span");
  addedElt.id = "added";
  addedElt.textContent = "Added element";
  document.querySelector("p").insertBefore(addedElt, document.getElementById("existing-element"));

  
  // more accurate node position with insertAdjdacentHTML; insertAdjacentElement as well !
  document.querySelector("p").insertAdjacentHTML('beforeBegin', '<span id="added">Added element</span>'); // before element itself (outside)
  document.querySelector("p").insertAdjacentHTML('afterBegin', '<span id="added">Added element</span>'); // just before firstchild
  document.querySelector("p").insertAdjacentHTML('beforeEnd', '<span id="added">Added element</span>'); // just after lastchild
  document.querySelector("p").insertAdjacentHTML('afterEnd', '<span id="added">Added element</span>'); // after element itself (outside)

  
  // replace a child node
  var addedElt = document.createElement("span");
  addedElt.id = "added";
  addedElt.textContent = "Added element";
  document.querySelector("p").replaceChild(addedElt, document.getElementById("existing-element")); // 'addedElt' replace existing-element

  
  // remove a child node
  document.querySelector("p").removeChild(document.getElementById("existing-element")); // remove existing-element


  // remove element
  var elt = document.querySelector("p");
  elt.parentNode.removeChild(elt);
  
  
  // swap 2 elements DOM position
  function swapElts(elt1, elt2) {
    var temp = document.createElement("p");
    elt1.parentNode.insertBefore(temp, elt1);
    elt2.parentNode.insertBefore(elt1, elt2);
    temp.parentNode.insertBefore(elt2, temp);
    temp.parentNode.removeChild(temp);
  }


  
    
// DOM STYLE
// give style to elements
  
  
  // style element
  var elt = document.querySelector("p");
  elt.style.color = "red";
  elt.style.margin = "50px";

  
  // style element with composed css properties
  var elt = document.querySelector("p");
  elt.style.backgroundColor = "red"; // used camelCase for composed properties
  elt.style.fontFamily = "Arial";

  
  // show style (only style within html, not external or head)
  var elt = document.querySelector("p");
  console.log(elt.style.color); // you can't get info about external that way

  
  // show style from external
  var eltStyle = getComputedStyle(document.querySelector("p"));
  console.log(eltStyle.color);

  
  // change multiple tags style
  var elts = document.querySelectorAll("p");
  for (var i = 0; i < elts.length; i++) {
    elts[i].style.color = "white";
    elts[i].style.backgroundColor = "red";
  }

  
  // get style external
  var content = document.querySelector("p");
  console.log(getComputedStyle(content).height);
  console.log(getComputedStyle(content).width);

  
  // add list with measures of another tag
  var styleElement = getComputedStyle(document.getElementById("content"));
  var listElt = document.createElement("ul");
  var lengthElt = document.createElement("li");
  lengthElt.textContent = "Length : " + styleElement.width;
  var heightElt = document.createElement("li");
  heightElt.textContent = "Height : " + styleElement.height;
  listElt.appendChild(heightElt);
  listElt.appendChild(lengthElt);
  document.getElementById("info").appendChild(document.createTextNode("Info about element"));
  document.getElementById("info").appendChild(listElt);


  // check media query in JS
  if (window.matchMedia("(min-width: 768px)").matches) {
    // do something when window is wider (equal to) than 768px
  }

  
  
  
// DOM EVENTS
// help to react to user's actions
  
  
  // monitor events; use window for example
  monitorEvents(); // write it in the console


  // add an event
  document.getElementById("button").addEventListener("click", function() {
    console.log("Click !");
  });
  
  
  // remove an event; function can't be anonymous
  document.getElementById("button").removeEventListener("click", namedFunction);


  // add event (remove on click) using class names
  var remove = document.getElementsByClassName("remove");
  for (i=0; i<remove.length; i++) {
    remove[i].addEventListener("click", function () {
      this.parentNode.remove();
    });
  }
  
  
  // get event type and target content of clicked element
  document.getElementById("button").addEventListener("click", function (e) {
    console.log("Event :" + e.type + ", target text :" + e.target.textContent);
  });


  // get event type and target content of event handler element
  document.getElementById("button").addEventListener("click", function (e) {
    console.log("Event :" + e.type + ", target text :" + e.currentTarget.textContent);
  });


  // add event with external function pointing to clicked element
  document.querySelector("button").addEventListener("click", functionName);
  function functionName(e) {
    e.target.style.color = "red";
  }
  
  
  // keypress event returning the pressed key (only for characters)
  document.addEventListener("keypress", function (e) {
      console.log("Key: " + String.fromCharCode(e.charCode)); // charCode deprecated
      console.log("Key: " + e.key);
  });
  
  
  // event type on pressed key; work also with "keyup"
  document.addEventListener("keydown", function (e) {
    console.log("Event: " + e.type);
  });
  
  
  // Wait until the page is fully loaded to make an action
  window.addEventListener("load", function() {
    console.log("Page fully loaded");
  });
  
  
  // warning before leaving page or tab
  window.addEventListener("beforeunload", function (e) {
    var message = "We are chill here !";
    e.returnValue = message; // usual confirmation
    return message; // confirmation for some browsers
  });
  
  
  // stop propagation; propagation goes from child to parents
  document.getElementById("propa").addEventListener("click", function (e) {
    console.log("button manager");
    e.stopPropagation();
  });

  // set propagation from parents to child (capturing); really useful for similar events for multiple DOM elements superposed
  document.getElementById("propa").addEventListener("click", function () {
    console.log("button manager")
  }, true); // default value is false (child to parents)
  
  
  // prevent default behavior
  document.getElementById("forbidden").addEventListener("click", function (e) {
    console.log("Keep on studying the course instead");
    e.preventDefault(); // cancel the link navigation
  });
  
  
  // one button count the clicks and another one disables the counting
  var count = 0;
  function e () {
    count++;
    document.getElementById("clickCounter").textContent = count;
  }
  document.getElementById("clicker").addEventListener("click", e);
  document.getElementById("disable").addEventListener("click", function () {
    document.getElementById("clicker").removeEventListener("click", e);
  });
  
  
  // get pressed key and change background color accordingly
  var Elts = document.getElementsByTagName("div");
  var color = "";
  document.addEventListener("keypress", function (e) {
    switch (e.charCode) {
      case 98:
        color = "blue";
        break;
      case 103:
        color = "green";
        break;
      case 114:
        color = "red";
        break;
      case 121:
        color = "yellow";
        break;
      default:
        console.log("Wrong key");
    }
    for (i=0; i<Elts.length; i++) {
      Elts[i].style.backgroundColor = color;
    }
  });
  

  // listen to arrow keys pressed and move accordingly
  var Elt = document.getElementById("ball");
  document.addEventListener("keydown", function (e) {
    switch (e.keyCode) {
      case 37: // left
        Elt.style.left -= 10;
        break;
      case 38: // up
        Elt.style.top -= 10;
        break;
      case 39: // right
        Elt.style.right -= 10;
        break;
      case 40: // down
        Elt.style.bottom -= 10;
        break;
      default:
        console.log("Wrong key");
    }
  });

  
  // Add li to ul with prompt and change those with click
  document.querySelector("button").addEventListener("click", function() {
    var dessert = prompt("type your dessert");
    var newElt = document.createElement("li");
    newElt.textContent = dessert;
    newElt.addEventListener("click", function() {
      var newName = prompt("Change dessert name of ", this.textContent);
      this.textContent = newName;
    });
    document.getElementById("desserts").appendChild(newElt);
  });
  



// DOM FORMS 
// Get values, check values, 


  // Prevent form submit; so no reload page anymore
  document.querySelector("form").addEventListener("submit", function (e) {
    e.preventDefault();
  });


  // Get value of an input
  var getValue = document.getElementsByTagName("input").value;
  console.log(getValue); // log input value


  // Set value of an input
  document.getElementsByTagName("input").value = "Hello"; // input value is now 'hello';


  // Check form elements
  var form = document.querySelector("form");
  console.log(form.elements.length); // number of inputs inside form
  console.log(form.elements[0].name); // first input name
  console.log(form.elements.mdp.type); // mdp input type


  // Put focus or remove it
  document.querySelector("input").focus(); 
  document.querySelector("input").blur(); 


  // Add event with focus on input element or blur
  document.querySelector("input").addEventListener("focus", function() { // just replace with "blur"
    document.querySelector("input").style.color = "red";
  });


  // Add event when a checkbox is checked or unchecked
  document.querySelector("input").addEventListener("change", function () {
    console.log("checkbox checked : " + e.target.checked); // true of false if checked or unchecked
  });


  // Event when radio is changed to return value
  for (i = 0; i < document.querySelectorAll("input").length; i++) {
    document.querySelectorAll("input")[i].addEventListener("change", function (e) {
      console.log("chosen radio input : " + e.target.value); // returns value attribute of the new chosen radio input
    });
  }


  // Event when dropping list option is changed
  document.querySelector("select").addEventListener("change", function (e) {
    console.log("Option selected from list : " + e.target.value); // returns value attribute of the new chosen option
  });


  // Validation on submitting with submit event
  document.querySelector("form").addEventListener("submit", function(e) {
    if (e.target.value.indexOf("http://") === -1) {
      console.log("not an URL !");
    }
  });


  // Validation while typing with input event
  document.querySelector("form").addEventListener("input", function (e) {
    if (e.target.value.length < 8) {
      console.log("Password too short");
    }
  });


  // Validation when input loses focus
  document.querySelector("form").addEventListener("blur", function(e) {
    if (e.target.value.indexOf("@") === -1) {
      console.log("email adress invalid");
    }
  });


  // Check passwords
  document.querySelector("form").addEventListener("submit", function (e) {
    var pass1 = document.getElementById("pass1");
    var pass2 = document.getElementById("pass2");
    var info = document.getElementById("infoPass");
    if (pass1.value !== pass2.value) { // if passwords are different
      info.innerHTML += "The passwords are not the same. Please try again. </br>";
    }
    if (pass1.value.length < 6) { // if password is shorter than 6 char
      info.innerHTML += "The password must contain at least 6 characters. </br>"
    }
    if (!/[0-9]/.test(pass1.value)) { // if no digit
      info.innerHTML += "The password must have at least one number. </br>";
    }
    if (pass1.value === pass2.value && pass1.value.length >=6 && /[0-9]/.test(pass1.value)) { // if all true
      info.innerHTML = "Correct Password !"
    }
    e.preventDefault();
  });


  // Select an option and make a list appear accordingly
  var houses = [ // array of object
    {
      code: "ST",
      name: "Stark"
    },
    {
      code: "LA",
      name: "Lannister"
    },
    {
      code: "BA",
      name: "Baratheon"
    },
    {
      code: "TA",
      name: "Targaryen"
    }
  ];
  function getChar(houseCode) { // check code and return names
    switch (houseCode) {
    case "ST":
      return ["Eddard", "Catelyn", "Robb", "Sansa", "Arya", "Jon Snow"];
    case "LA":
      return ["Tywin", "Cersei", "Jaime", "Tyrion"];
    case "BA":
      return ["Robert", "Stannis", "Renly"];
    case "TA":
      return ["Aerys", "Daenerys", "Viserys"];
    default:
      return [];
    }
  }
  for (i = 0; i < houses.length; i++) { // create form list
    var option = document.createElement("option");
    option.innerHTML = houses[i].name;
    option.setAttribute("value", houses[i].code);
    document.getElementById("house").appendChild(option);
  }
  document.getElementById("house").addEventListener("change", function (e) { // create list with names linked to code
    var char = getChar(e.target.value);
    document.getElementById("char").innerHTML = "";
    for (j = 0; j < char.length; j++) {
      var li = document.createElement("li");
      li.innerHTML = char[j];
      document.getElementById("char").appendChild(li);
    }
  });


  // Get suggestions and select one
  var countryList = [
    "Afghanistan",
    "Afrique du Sud",
    "Albanie",
    "Algérie",
    "Allemagne",
    "Andorre",
    "Angola",
    "Anguilla",
    "Antarctique",
    "Antigua-et-Barbuda",
    "Antilles néerlandaises",
    "Arabie saoudite",
    "Argentine",
    "Arménie",
    "Aruba",
    "Australie",
    "Autriche",
    "Azerbaïdjan"
];
var country = document.getElementById("pays");
var suggestions = document.getElementById("suggestions")
country.addEventListener("input", function (e) { // when typing
    suggestions.innerHTML = "";
    for (i = 0; i < countryList.length; i++) { 
        if (countryList[i].indexOf(e.target.value) !== -1) { // check if country name have typed letters
            var suggest = document.createElement("div");
            suggest.classList.add("suggestion");
            suggest.innerHTML = countryList[i];
            suggestions.appendChild(suggest); // add true elements as suggestions
            suggestions.addEventListener("click", function(f) { // on click on element, put that element in input
                country.value = f.target.textContent;
                suggestions.innerHTML = "";
            });
        }
    }
});




// DOM ANIMATIONS 
// All animations; use intervals for basics; CSS whenever you can; requestAnimationFrame() for the rest


  // Set interval; can define an end of it
  setInterval(function() {console.log("hello")}, 1000); // log "hello" every second


  // Clear interval; you must name your setInterval() in a var to be able to use clearInterval()
  var varSetInterval = setInterval(function() {console.log("hello")}, 1000);
  clearInterval(varSetInterval); // stops interval


  // Set timeout; 
  setTimeout(function() {console.log("hello")}, 2000); // starts log after 2sec
  

  // Clear timeout; you must name your setTimeout() in a var to be able to use clearTimeout()
  var varSetTimeout = setTimeout(function() {console.log("hello")}, 1000);
  clearTimeout(varSetTimeout); // stops timeout


  // Set timeout in a loop; tricky because you need to copy the i otherwise it will log only last i value
  var time = 10;
  for (var i = 1; i <= time; i++){
    (function(copy){
      setTimeout(function() { console.log(copy); }, 1000 * i);
    })(i);
  }


  // Request animation frame; 
  function animate() {
    // animation code...
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);


  // Cancel animation frame; you must name your requestAnimationFrame() in a var to be able to use cancelAnimationFrame()
  cancelAnimationFrame(varRequestAnimationFrame)


  // reload page
  location.reload(true);

  // Change url
  window.location.replace("new url");

  // Get window sizes
  window.innerWidth;
  window.innerHeight;

  // Get window scrolls
  window.scrollX;
  window.scrollY;

  // Get current url
  document.URL;

  // Highlight current page
  document.querySelector(".navbar a[href*='" + location.pathname + "']").classList.add("active");

  
  // Start timer and be able to pause/start
  var start = document.getElementById("start");
  var stop = document.getElementById("stop");
  var span = document.getElementById("time");
  var time = 0;
  span.innerHTML = time;
  start.addEventListener("click", function() { // on start
    stop.style.display = "block";
    start.style.display = "none";
    setTime = setInterval(function() {
      time ++;
      span.innerHTML = time;
    }, 1000);
  });
  stop.addEventListener("click", function() { // on stop
    start.style.display = "block";
    stop.style.display = "none";
    clearTimeout(setTime);
  });


  // Rebounding ball with start/stop animation
  var box = document.getElementById("box");
  var ball = document.getElementById("ball"); // #ball must be position: relative and left: 0;
  var start = document.getElementById("start");
  var stop = document.getElementById("stop");
  var ballWidth = parseFloat(getComputedStyle(ball).width);
  var direction = 1;
  function moveBall() {
    var xBall = parseFloat(getComputedStyle(ball).left); // get ball position
    var xMax = parseFloat(getComputedStyle(box).width); // get max width
    if ((xBall + ballWidth > xMax) || (xBall < 0)) {
      direction *= -1; // change direction
    }
    ball.style.left = (xBall + 10 * direction) + "px"; // actual moving line
    animationId = requestAnimationFrame(moveBall);
  }
  start.addEventListener("click", function() { // starts animation
    stop.disabled = false;
    start.disabled = true;
    animationId = requestAnimationFrame(moveBall);
  });
  stop.addEventListener("click", function() { // stops animation
    start.disabled = false;
    stop.disabled = true;
    cancelAnimationFrame(animationId);
  });




// CANVAS
// Draw graphics via scripting to build graphs, modify photos or create animations


  // First point to your canvas element; by default canvas will be 300x150px
  var canvas = document.querySelector("#myCanvas");

  // Update canvas area with width and height
  canvas.width = 400;
  canvas.height = 400;

  // You can check if canvas are supported by the browser
  if (canvas.getContext) {console.log("We can use canvas here !");}

  // To be able to manipulate the points, you have to refer to the context
  var ctx = canvas.getContext("2d");


  // Draw rectangles by defining their x, y, width and height
  ctx.fillRect(0, 0, 10, 10); // here it is a 10px square starting at top left corner (0,0)
  ctx.strokeRect(0, 0, 10, 10); // here it is a 10px square outline starting at top left corner (0,0)
  ctx.clearRect(0, 0, 10, 10); // here it is a 10px transparent square starting at top left corner (0,0)


  // Draw paths to create other shapes
  ctx.beginPath(); // start drawing
  ctx.moveTo(75, 50); // move to a certain pos(x, y)
  ctx.lineTo(100, 75); // draws line from pos defined by moveTo to the ones in lineTo(x, y)
  ctx.fill(); // draws a solid shape from the path content area (need enough points)
  ctx.closePath(); // add a straight line to the path to close it
  ctx.stroke(); // draws shape by stroking its outline


  // Draw arcs to create rounded shapes
  ctx.beginPath(); // start drawing
  ctx.arc(0, 0, 10, 0, Math.PI*2, false); // x, y, radius, startAngle, endAngle, antiClockwise()
  ctx.fill(); // to make it a circle
  ctx.stroke(); // to make it a circled outline


  // Styles and colors to canvas
  ctx.fillStyle = "red"; // fill with red; can use other color such as hexa, rgb, rgba
  ctx.strokeStyle = "blue"; // stroke to blue
  ctx.globalAlpha = 0.2; // 0 to 1 total transparency
  ctx.lineWidth = 4; // sets width for lines
  ctx.lineCap = "round"; // default is "butt"; "round" goes a bit off rounded and "square" does the same squared
  ctx.lineJoin = "round"; // default is "miter"; "round" makes joint rounded and "bevel" makes joint flat
  ctx.setLineDash([10, 5]); // size of dash and spaces


  // Drawing text
  ctx.font = "48px Arial"; // sets font to be used
  ctx.textAlign = "left"; // start, left, center, right, end
  ctx.textBaseline = "middle"; // top, hanging, alphabetic, ideographic, bottom
  ctx.fillText(); // draws normal text
  ctx.strokeText();// draws text stroke




// FETCH

fetch("https://type.fit/api/quotes")
  .then((response) => {
    if (!response.ok) {
      throw Error(response.statusText);
    }
    return response.json();
  })
  .then((data) => console.log(data))
  .catch((err) => console.log(err));




// ASYNC/AWAIT
// best option to fetch data !
const fetchData = async () => {
  try {
    const quotes = await fetch("https://type.fit/api/quotes");
    const response = await quotes.json();
    console.log(response);
  } catch (error) {
    console.log(error);
  }
};
fetchData();




// SERVER REQUESTS
// Manage requests to the server (HTTP, AJAX, JSON)


  // Synchronous GET requet of a doc through web server
  var req = new XMLHttpRequest(); // create HTTP request
  req.open("GET", "http://localhost/repository/file.txt", false); // synchronous GET request; could be POST or PUT
  req.send(null); // sending request; could be POST or PUT
  console.log(req.responseText);


  // Asynchronous GET request
  var req = new XMLHttpRequest();
  req.open("GET", "http://localhost/repository/file.txt"); // no false because asynchronous request
  req.addEventListener("load", function () { // use of load event to make it asynchronous
    console.log(req.responseText);
  });
  req.send(null);


  // Handling errors
  var req = new XMLHttpRequest();
  req.open("GET", "http://localhost/repository/file.txt");
  req.addEventListener("load", function () {
    if (req.status >= 200 && req.status < 400) { // server succeed with the request
      console.log(req.responseText);
    } else {
      console.error(req.status + " " + req.statusText); // error with request information
    }
  });
  req.addEventListener("error", function () {
    console.error("Network error"); // request did not reach server
  });
  req.send(null);


  // Generic AJAX function (AJAX = asynchronous HTTP); 
  // Better to define the function in another js file when you have multiple js file that will need it (link it with script tag above other script tags)
  function ajaxGet(url, callback) {
    var req = new XMLHttpRequest();
    req.open("GET", url);
    req.addEventListener("load", function () {
      if (req.status >= 200 && req.status < 400) {
        callback(req.responseText);
      } else {
        console.error(req.status + " " + req.statusText + " " + url);
      }
    });
    req.addEventListener("error", function () {
      console.error("Network error with URL " + url);
    });
    req.send(null);
  }
  function call(answer) { // function that handle the answer (aka callback)
    console.log(answer);
  }
  ajaxGet("http://localhost/repository/file.txt", call);
  // ajaxGet("http://localhost/repository/file.txt", function (answer) {console.log(answer)}); shorter way with anonym function


  // Transform from JS to JSON and from JSON to JS
  var planes = [
    {
      brand: "Airbus",
      model: "A320"
    },
    {
      brand: "Airbus",
      model: "A380"
    }
  ];
  console.log(planes);
  var textPlanes = JSON.stringify(planes); // JS objects array into JSON string
  console.log(textPlanes);
  console.log(JSON.parse(textPlanes)); // JSON string into JS objects array


  // Get data from server converting JSON to JS and display list
  ajaxGet("http://localhost/repository/file.txt", function (answer) {
    var list = JSON.parse(answer); // JS objects array
    list.forEach(function (smth) {
      console.log(smth.thing); // have thing for each smth
    });
  });


  // Split string and create list of li elements
  ajaxGet("http://localhost/javascript-web-srv/data/langages.txt", function (answer) {
    var arr = answer.split(";");
    console.log(arr);
    arr.forEach(function (element) {
      li = document.createElement("li");
      li.innerHTML = element;
      document.getElementById("languages").appendChild(li);
    });
  });


  // Get data from JSON and put it into table
  ajaxGet("http://localhost/javascript-web-srv/data/tableaux.json", function (answer) {
    arr = JSON.parse(answer);
    console.log(arr);
    arr.forEach( function(element) {
      tr = document.createElement("tr");
      tr.innerHTML = "<td>" + element.nom + "</td>" + "<td>" + element.annee + "</td>" + "<td>" + element.peintre + "</td>";
      document.getElementById("table").appendChild(tr);
    });
  });


  

// API 
// Application Programming Interface are made by people to help others go faster; use geolocation, weather, wiki, etc.


  // API with JSON (works the same but with online url); need ajax file with ajaxGet function defined
  ajaxGet("http://api-website/api/file", function (answer) {
    var arr = JSON.parse(answer);
    arr.forEach(function (element) {/*code block*/;});
  });


  // Github profile
  function ajaxGet(url, callback) { // usual ajaxGet func
    var req = new XMLHttpRequest();
    req.open("GET", url);
    req.addEventListener("load", function () {
      if (req.status >= 200 && req.status < 400) {
        callback(req.responseText);
      } else {
        console.error(req.status + " " + req.statusText + " " + url);
      }
    });
    req.addEventListener("error", function () {
      console.error("Network error with URL " + url);
    });
    req.send(null);
  }
  document.querySelector("form").addEventListener("submit", function (e) { // Get user value on click
    user = document.getElementById("user").value;
    e.preventDefault();
    ajaxGet("https://api.github.com/users/" + user, function (answer) { // Request API
      var profile = JSON.parse(answer); 
      img = document.createElement("img"); // Avatar
      img.src = profile.avatar_url;
      img.style.width = "200px";
      h1 = document.createElement("h1"); // Pseudo
      h1.innerHTML = profile.name;
      h1.style.color = "#3D7B74";
      p = document.createElement("p"); // Website
      p.innerHTML = "<a href='" + profile.blog + "'>" + profile.blog + "</a>";
      document.getElementById("profile").innerHTML = ""; // empty previous search
      document.getElementById("profile").appendChild(img);
      document.getElementById("profile").appendChild(h1);
      document.getElementById("profile").appendChild(p);
    });
  });


  // Get geolocation data; must have user authorization
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(function(position) {
      $("#data").html("latitude: " + position.coords.latitude + "<br>longitude: " + position.coords.longitude);
    }); // Insert latitude and longitude into #data
  }




// SEND DATA TO SERVER
// Use http and json to send data to servers


  // Basic data sending code
  var identity = new FormData();
  identity.append("login", "Bob"); // Adding info example
  identity.append("password", "azerty");
  var req = new XMLHttpRequest(); 
  req.open("POST", "http://localhost/repository/post_form.php"); // HTTP POST
  req.send(identity);


  // Generic data sending function
  function ajaxPost(url, data, callback) {
    var req = new XMLHttpRequest();
    req.open("POST", url);
    req.addEventListener("load", function () {
      if (req.status >= 200 && req.status < 400) {
        callback(req.responseText);
      } else {
        console.error(req.status + " " + req.statusText + " " + url);
      }
    });
    req.addEventListener("error", function () {
      console.error("Network error with URL " + url);
    });
    req.send(data);
  }
  var command = new FormData(); // Adaptation of basic code
  command.append("color", "red"); // Adding other info example (they erase previous ofc)
  command.append("size", "43");
  ajaxPost("http://localhost/repository/post_form.php", command, function (reponse) {
      console.log("Command sent to server");
  });


  // Handle form submission
  var form = document.querySelector("form");
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var data = new FormData(form);
    ajaxPost("http://localhost/javascript-web-srv/post_form.php", data, function () {}); // Callback func is empty here
  });


  // Data sending checking if JSON data
  function ajaxPost(url, data, callback, isJson) {
    var req = new XMLHttpRequest();
    req.open("POST", url);
    req.addEventListener("load", function () {
      if (req.status >= 200 && req.status < 400) {
        callback(req.responseText);
      } else {
        console.error(req.status + " " + req.statusText + " " + url);
      }
    });
    req.addEventListener("error", function () {
      console.error("Network error with URL " + url);
    });
    if (isJson) { // Check if json format
      req.setRequestHeader("Content-Type", "application/json");
      data = JSON.stringify(data);
    }
    req.send(data);
  }
  var movie = { // Creation of a movie object
    title: "Zootopie",
    year: "2016",
    director: "Byron Howard and Rich Moore"
  };
  ajaxPost("http://localhost/javascript-web-srv/post_json.php", movie, function (reponse) {
      console.log("The movie " + JSON.stringify(movie) + " has been sent to the server");
    },
    true // JSON parameter value
  );


  // send feedback from a form (function is defined before ofc)
  document.querySelector("form").addEventListener("submit", function (e) {
    e.preventDefault();
    var feedback = {
      pseudo: e.target.elements.pseudo.value,
      evaluation: e.target.elements.evaluation.value,
      message: e.target.elements.message.value,
    };
    ajaxPost("http://oc-jswebsrv.herokuapp.com/api/temoignage", feedback, function (reponse) {
      var messageElt = document.createElement("p");
      messageElt.textContent = "Feedback added";
      document.getElementById("result").appendChild(messageElt);
    }, true);
  });




// LOCALSTORAGE
// localStorage allows us to save data on the browser; it needs to be a string; window is optional


  // set item in localStorage
  localStorage.setItem("itemName", "itemValue");

  // get item in localStorage
  localStorage.getItem("itemName");

  // remove item in localStorage
  localStorage.removeItem("itemName");

  // clear localStorage
  localStorage.clear();

  // complex storage with JSON
  var obj = [1, 2, 3];
  localStorage.setItem("itemName", JSON.stringify(obj));
  JSON.parse(localStorage.getItem("itemName"));




// COOKIES
// 


  // set a cookie
  document.cookie = "cookieName=cookieValue";

  // set a cookie lasting more than a session
  document.cookie = "cookieName=cookieValue; expires=Wed, 19 Aug 2020 22:00:00 GMT"; // date must be in this format...

  // return list of cookies
  document.cookie.split(";"); // cookieName=cookieValue

  // check if a cookie is set
  document.cookie.split(";").indexOf("cookieName=cookieValue"); // true


  // create cookie function
  function createCookie(cookieName, cookieValue, daysToExpire) {
    var date = new Date();
    date.setTime(date.getTime() + (daysToExpire * 24 * 60 * 60 * 1000));
    document.cookie = cookieName + "=" + cookieValue + "; expires=" + date.toGMTString();
  }
  createCookie("name", "value", 100);