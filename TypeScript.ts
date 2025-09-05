// TYPESCRIPT - Typed JavaScript Superset - by Richard Rembert

// TypeScript is a strongly typed programming language that builds on JavaScript
// giving you better tooling at any scale. It compiles to plain JavaScript.

// SETUP AND BASICS

// Install TypeScript globally
// npm install -g typescript

// Install TypeScript for a project
// npm install --save-dev typescript
// npm install --save-dev @types/node  // Node.js type definitions

// Create tsconfig.json
// npx tsc --init

// Compile TypeScript files
// tsc filename.ts              // Compile single file
// tsc                          // Compile all files in project
// tsc --watch                  // Watch mode (recompile on changes)

// Run TypeScript directly (with ts-node)
// npm install -g ts-node
// ts-node filename.ts

// Basic tsconfig.json
/*
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020", "DOM"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
*/


// BASIC TYPES

// Primitive types
let userName: string = "Alice";
let userAge: number = 25;
let isActive: boolean = true;
let undefinedValue: undefined = undefined;
let nullValue: null = null;

// Type inference (TypeScript can infer types)
let inferredString = "Hello";        // inferred as string
let inferredNumber = 42;             // inferred as number
let inferredBoolean = true;          // inferred as boolean

// Any type (avoid when possible)
let anything: any = "could be anything";
anything = 42;
anything = true;

// Unknown type (safer alternative to any)
let userInput: unknown;
userInput = "hello";
userInput = 42;

// Type checking with unknown
if (typeof userInput === "string") {
  console.log(userInput.toUpperCase()); // OK, TypeScript knows it's a string
}

// Void type (functions that don't return a value)
function logMessage(message: string): void {
  console.log(message);
}

// Never type (functions that never return)
function throwError(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {
    // infinite loop
  }
}


// ARRAYS AND TUPLES

// Array types
let numbers: number[] = [1, 2, 3, 4, 5];
let strings: Array<string> = ["hello", "world"];
let mixed: (string | number)[] = ["hello", 42, "world"];

// Array methods with typing
let fruits: string[] = ["apple", "banana"];
fruits.push("orange");                    // OK
// fruits.push(42);                       // Error: number not assignable to string

// Readonly arrays
let readonlyNumbers: readonly number[] = [1, 2, 3];
// readonlyNumbers.push(4);               // Error: push doesn't exist on readonly array

// Tuples (fixed length arrays with specific types)
let coordinate: [number, number] = [10, 20];
let nameAge: [string, number] = ["Alice", 25];

// Named tuples (TypeScript 4.0+)
let point: [x: number, y: number] = [10, 20];

// Optional tuple elements
let optionalTuple: [string, number?] = ["hello"];

// Rest elements in tuples
let restTuple: [string, ...number[]] = ["hello", 1, 2, 3];

// Tuple destructuring
let [x, y] = coordinate;                  // x: number, y: number
let [name, age] = nameAge;               // name: string, age: number


// OBJECTS AND INTERFACES

// Object type annotations
let user: { name: string; age: number } = {
    name: "Alice",
    age: 25
  };
  
  // Optional properties
  let optionalUser: { name: string; age?: number } = {
    name: "Bob"  // age is optional
  };
  
  // Readonly properties
  let readonlyUser: { readonly id: number; name: string } = {
    id: 1,
    name: "Charlie"
  };
  // readonlyUser.id = 2;                   // Error: readonly property
  
  // Index signatures
  let scores: { [subject: string]: number } = {
    math: 95,
    science: 87,
    english: 92
  };
  
  // Interfaces (preferred way to define object shapes)
  interface Person {
    name: string;
    age: number;
    email?: string;                        // optional property
    readonly id: number;                   // readonly property
  }
  
  let person: Person = {
    id: 1,
    name: "Alice",
    age: 25
  };
  
  // Interface inheritance
  interface Employee extends Person {
    department: string;
    salary: number;
  }
  
  let employee: Employee = {
    id: 1,
    name: "Bob",
    age: 30,
    department: "Engineering",
    salary: 75000
  };
  
  // Multiple interface inheritance
  interface Timestamped {
    createdAt: Date;
    updatedAt: Date;
  }
  
  interface UserProfile extends Person, Timestamped {
    bio: string;
  }
  
  // Interface methods
  interface Calculator {
    add(a: number, b: number): number;
    subtract(a: number, b: number): number;
  }
  
  let calc: Calculator = {
    add: (a, b) => a + b,
    subtract: (a, b) => a - b
  };
  
  // Callable interfaces
  interface StringProcessor {
    (input: string): string;
  }
  
  let upperCase: StringProcessor = (input) => input.toUpperCase();


  // UNION AND INTERSECTION TYPES

// Union types (OR)
let stringOrNumber: string | number;
stringOrNumber = "hello";               // OK
stringOrNumber = 42;                    // OK
// stringOrNumber = true;               // Error

// Union with literal types
let size: "small" | "medium" | "large" = "medium";

// Type guards for unions
function processValue(value: string | number): string {
  if (typeof value === "string") {
    return value.toUpperCase();         // TypeScript knows it's a string
  } else {
    return value.toString();            // TypeScript knows it's a number
  }
}

// Intersection types (AND)
interface Colorful {
  color: string;
}

interface Circle {
  radius: number;
}

type ColorfulCircle = Colorful & Circle;

let circle: ColorfulCircle = {
  color: "red",
  radius: 10
};


// LITERAL TYPES

// String literals
let direction: "north" | "south" | "east" | "west";
direction = "north";                    // OK
// direction = "up";                    // Error

// Numeric literals
let diceRoll: 1 | 2 | 3 | 4 | 5 | 6;

// Boolean literals
let truthyOnly: true = true;
// let falsyValue: true = false;        // Error

// Template literal types (TypeScript 4.1+)
type Greeting = `hello ${string}`;
let greeting1: Greeting = "hello world";    // OK
let greeting2: Greeting = "hello there";    // OK
// let greeting3: Greeting = "hi there";    // Error
