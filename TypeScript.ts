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
