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


// ENUMS

// Numeric enums
enum Direction {
    Up,        // 0
    Down,      // 1
    Left,      // 2
    Right      // 3
  }
  
  let playerDirection: Direction = Direction.Up;
  
  // Numeric enums with custom values
  enum Status {
    Pending = 1,
    Approved = 2,
    Rejected = 3
  }
  
  // String enums
  enum Color {
    Red = "red",
    Green = "green",
    Blue = "blue"
  }
  
  let favoriteColor: Color = Color.Red;
  
  // Const enums (inlined at compile time)
  const enum Planet {
    Mercury = "mercury",
    Venus = "venus",
    Earth = "earth"
  }
  
  let homePlanet = Planet.Earth;          // Compiles to: let homePlanet = "earth";
  
  // Reverse mapping (numeric enums only)
  console.log(Direction[0]);              // "Up"
  console.log(Direction["Up"]);           // 0
  
  
  // FUNCTIONS
  
  // Function type annotations
  function add(a: number, b: number): number {
    return a + b;
  }
  
  // Arrow functions
  const multiply = (a: number, b: number): number => a * b;
  
  // Optional parameters
  function greet(name: string, greeting?: string): string {
    return greeting ? `${greeting}, ${name}!` : `Hello, ${name}!`;
  }
  
  // Default parameters
  function createUser(name: string, age: number = 18): Person {
    return { id: Date.now(), name, age };
  }
  
  // Rest parameters
  function sum(...numbers: number[]): number {
    return numbers.reduce((total, num) => total + num, 0);
  }
  
  // Function overloads
  function combine(a: string, b: string): string;
  function combine(a: number, b: number): number;
  function combine(a: any, b: any): any {
    return a + b;
  }
  
  let result1 = combine("Hello", " World");    // string
  let result2 = combine(5, 10);               // number
  
  // Function types
  type MathOperation = (a: number, b: number) => number;
  
  let operation: MathOperation = (x, y) => x * y;
  
  // Generic functions
  function identity<T>(arg: T): T {
    return arg;
  }
  
  let stringIdentity = identity<string>("hello");
  let numberIdentity = identity<number>(42);
  let inferredIdentity = identity("inferred");    // T inferred as string
  
  // Generic constraints
  interface Lengthwise {
    length: number;
  }
  
  function logLength<T extends Lengthwise>(arg: T): T {
    console.log(arg.length);
    return arg;
  }
  
  logLength("hello");                       // OK
  logLength([1, 2, 3]);                     // OK
  // logLength(42);                         // Error: number doesn't have length
  
  // Multiple generic parameters
  function swap<T, U>(a: T, b: U): [U, T] {
    return [b, a];
  }
  
  let swapped = swap("hello", 42);          // [number, string]


  // CLASSES

// Basic class
class Animal {
    name: string;
    
    constructor(name: string) {
      this.name = name;
    }
    
    speak(): void {
      console.log(`${this.name} makes a sound`);
    }
  }
  
  let dog = new Animal("Rex");
  
  // Class inheritance
  class Dog extends Animal {
    breed: string;
    
    constructor(name: string, breed: string) {
      super(name);
      this.breed = breed;
    }
    
    speak(): void {
      console.log(`${this.name} barks`);
    }
    
    wagTail(): void {
      console.log(`${this.name} wags tail`);
    }
  }
  
  // Access modifiers
  class BankAccount {
    public accountNumber: string;           // accessible everywhere
    private balance: number;                // only within this class
    protected owner: string;                // within this class and subclasses
    
    constructor(accountNumber: string, owner: string) {
      this.accountNumber = accountNumber;
      this.owner = owner;
      this.balance = 0;
    }
    
    public deposit(amount: number): void {
      this.balance += amount;
    }
    
    public getBalance(): number {
      return this.balance;
    }
  }
  
  // Readonly modifier
  class Circle {
    readonly radius: number;
    
    constructor(radius: number) {
      this.radius = radius;
    }
    
    // this.radius = 10;                    // Error: readonly property
  }
  
  // Parameter properties (shorthand)
  class User {
    constructor(
      public name: string,
      private age: number,
      readonly id: number
    ) {}
    
    getAge(): number {
      return this.age;
    }
  }
  
  // Abstract classes
  abstract class Shape {
    abstract area(): number;
    
    displayArea(): void {
      console.log(`Area: ${this.area()}`);
    }
  }
  
  class Rectangle extends Shape {
    constructor(private width: number, private height: number) {
      super();
    }
    
    area(): number {
      return this.width * this.height;
    }
  }
  
  // Static members
  class MathUtils {
    static PI = 3.14159;
    
    static calculateCircleArea(radius: number): number {
      return this.PI * radius * radius;
    }
  }
  
  let area = MathUtils.calculateCircleArea(5);
  
  // Getters and setters
  class Temperature {
    private _celsius: number = 0;
    
    get celsius(): number {
      return this._celsius;
    }
    
    set celsius(value: number) {
      if (value < -273.15) {
        throw new Error("Temperature below absolute zero");
      }
      this._celsius = value;
    }
    
    get fahrenheit(): number {
      return (this._celsius * 9/5) + 32;
    }
  }
  
  // Generic classes
  class GenericContainer<T> {
    private items: T[] = [];
    
    add(item: T): void {
      this.items.push(item);
    }
    
    get(index: number): T | undefined {
      return this.items[index];
    }
    
    getAll(): T[] {
      return [...this.items];
    }
  }
  
  let stringContainer = new GenericContainer<string>();
  stringContainer.add("hello");
  
  let numberContainer = new GenericContainer<number>();
  numberContainer.add(42);


  / GENERICS

// Generic interfaces
interface Repository<T> {
  save(entity: T): void;
  findById(id: string): T | undefined;
  findAll(): T[];
}

// Generic type aliases
type ApiResponse<T> = {
  data: T;
  status: number;
  message: string;
};

let userResponse: ApiResponse<User> = {
  data: new User("Alice", 25, 1),
  status: 200,
  message: "Success"
};

// Generic constraints with keyof
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

let person = { name: "Alice", age: 25, email: "alice@example.com" };
let name = getProperty(person, "name");        // string
let age = getProperty(person, "age");          // number
// let invalid = getProperty(person, "height"); // Error: 'height' doesn't exist

// Conditional types
type NonNullable<T> = T extends null | undefined ? never : T;

type Example1 = NonNullable<string | null>;   // string
type Example2 = NonNullable<number | undefined>; // number

// Mapped types
type Partial<T> = {
  [P in keyof T]?: T[P];
};

type Required<T> = {
  [P in keyof T]-?: T[P];
};

type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

// Using utility types
interface User {
  id: number;
  name: string;
  email: string;
}

type PartialUser = Partial<User>;             // All properties optional
type RequiredUser = Required<User>;           // All properties required
type ReadonlyUser = Readonly<User>;           // All properties readonly

// Advanced mapped types
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

type UserGetters = Getters<User>;
// Results in:
// {
//   getId: () => number;
//   getName: () => string;
//   getEmail: () => string;
// }


// TYPE ASSERTIONS AND TYPE GUARDS

// Type assertions (type casting)
let someValue: unknown = "hello world";
let strLength1 = (someValue as string).length;
let strLength2 = (<string>someValue).length;    // JSX syntax conflicts

// Type guards
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function processValue(value: unknown): void {
  if (isString(value)) {
    console.log(value.toUpperCase());     // TypeScript knows it's a string
  }
}

// instanceof type guard
class Bird {
  fly(): void {
    console.log("Flying...");
  }
}

class Fish {
  swim(): void {
    console.log("Swimming...");
  }
}

function move(animal: Bird | Fish): void {
  if (animal instanceof Bird) {
    animal.fly();
  } else {
    animal.swim();
  }
}

// in operator type guard
interface Car {
  drive(): void;
}

interface Boat {
  sail(): void;
}

function operate(vehicle: Car | Boat): void {
  if ("drive" in vehicle) {
    vehicle.drive();
  } else {
    vehicle.sail();
  }
}

// Discriminated unions
interface Square {
  kind: "square";
  size: number;
}

interface Rectangle {
  kind: "rectangle";
  width: number;
  height: number;
}

interface Circle {
  kind: "circle";
  radius: number;
}

type Shape = Square | Rectangle | Circle;

function area(shape: Shape): number {
  switch (shape.kind) {
    case "square":
      return shape.size * shape.size;
    case "rectangle":
      return shape.width * shape.height;
    case "circle":
      return Math.PI * shape.radius ** 2;
    default:
      const exhaustiveCheck: never = shape;
      return exhaustiveCheck;
  }
}


// MODULES AND NAMESPACES

// ES6 modules (preferred)
// math.ts
export function add(a: number, b: number): number {
  return a + b;
}

export function subtract(a: number, b: number): number {
  return a - b;
}

export default function multiply(a: number, b: number): number {
  return a * b;
}

export { divide as div } from './calculator';

// main.ts
import multiply, { add, subtract } from './math';
import * as math from './math';

// Re-exports
export { User } from './user';
export * from './types';

// Namespaces (legacy, use modules instead)
namespace Utilities {
  export function log(message: string): void {
    console.log(`[LOG]: ${message}`);
  }
  
  export namespace StringUtils {
    export function reverse(str: string): string {
      return str.split('').reverse().join('');
    }
  }
}

Utilities.log("Hello");
Utilities.StringUtils.reverse("TypeScript");


// DECORATORS (Experimental)

// Enable in tsconfig.json: "experimentalDecorators": true

// Class decorator
function sealed(constructor: Function) {
    Object.seal(constructor);
    Object.seal(constructor.prototype);
  }
  
  @sealed
  class BugReport {
    type = "report";
    title: string;
    
    constructor(title: string) {
      this.title = title;
    }
  }
  
  // Method decorator
  function enumerable(value: boolean) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
      descriptor.enumerable = value;
    };
  }
  
  class Greeter {
    greeting: string;
    
    constructor(message: string) {
      this.greeting = message;
    }
    
    @enumerable(false)
    greet() {
      return "Hello, " + this.greeting;
    }
  }
  
  // Property decorator
  function format(formatString: string) {
    return function (target: any, propertyKey: string): void {
      let value = target[propertyKey];
      
      const getter = () => `${formatString} ${value}`;
      const setter = (newVal: string) => {
        value = newVal;
      };
      
      Object.defineProperty(target, propertyKey, {
        get: getter,
        set: setter,
        enumerable: true,
        configurable: true,
      });
    };
  }
  
  class Person {
    @format("Hello")
    name: string;
    
    constructor(name: string) {
      this.name = name;
    }
  }
  
  
  // UTILITY TYPES
  
  // Built-in utility types
  interface User {
    id: number;
    name: string;
    email: string;
    password: string;
  }
  
  // Partial - makes all properties optional
  type PartialUser = Partial<User>;
  
  // Required - makes all properties required
  type RequiredUser = Required<PartialUser>;
  
  // Readonly - makes all properties readonly
  type ReadonlyUser = Readonly<User>;
  
  // Pick - pick specific properties
  type UserSummary = Pick<User, "id" | "name">;
  
  // Omit - exclude specific properties
  type CreateUser = Omit<User, "id">;
  
  // Record - create object type with specific key and value types
  type UserRoles = Record<string, "admin" | "user" | "guest">;
  
  // Extract - extract types from union
  type StringNumber = string | number | boolean;
  type StringOrNumber = Extract<StringNumber, string | number>;
  
  // Exclude - exclude types from union
  type NonBoolean = Exclude<StringNumber, boolean>;
  
  // NonNullable - exclude null and undefined
  type NonNullString = NonNullable<string | null | undefined>;
  
  // ReturnType - extract return type of function
  function createUser(): User {
    return { id: 1, name: "Alice", email: "alice@example.com", password: "secret" };
  }
  
  type CreateUserReturn = ReturnType<typeof createUser>; // User
  
  // Parameters - extract parameter types of function
  function updateUser(id: number, updates: Partial<User>): void {}
  
  type UpdateUserParams = Parameters<typeof updateUser>; // [number, Partial<User>]
  
  // ConstructorParameters - extract constructor parameter types
  class Rectangle {
    constructor(public width: number, public height: number) {}
  }
  
  type RectangleParams = ConstructorParameters<typeof Rectangle>; // [number, number]
  
  // InstanceType - extract instance type of constructor
  type RectangleInstance = InstanceType<typeof Rectangle>; // Rectangle
