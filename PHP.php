// PHP - HYPERTEXT PREPROCESSOR - Server-Side Scripting Language - by Richard Rembert

// PHP is a popular general-purpose scripting language especially suited for web development
// It can be embedded into HTML and runs on the server before sending content to the browser

// SETUP AND BASICS

// Install PHP (Ubuntu/Debian)
// sudo apt update
// sudo apt install php php-cli php-fpm php-mysql php-zip php-gd php-mbstring php-curl php-xml

// Install PHP (macOS with Homebrew)
// brew install php

// Install PHP (Windows)
// Download from php.net or use XAMPP/WAMP

// Check PHP version
// php --version
// php -m  // Show loaded modules
// php -i  // Show PHP configuration info

// Run PHP files
// php filename.php
// php -S localhost:8000  // Built-in development server

// BASIC SYNTAX

// PHP tags
echo "Hello, World!";  // Standard opening tag (preferred)
?>

<!-- HTML can be mixed with PHP -->
<h1><?php echo "Dynamic Content"; ?></h1>

<?php
// Short tags (if enabled in php.ini)
// <? echo "Short tag"; ?>

// Echo tags (if enabled)
// <?= "Echo tag shorthand" ?>

// Comments
// Single line comment
# Another single line comment
/* 
   Multi-line comment
   can span multiple lines
*/

/**
 * Documentation comment (PHPDoc)
 * @param string $name The name parameter
 * @return string The greeting message
 */

// Case sensitivity
// Variables are case-sensitive: $name vs $Name
// Functions, classes, keywords are case-insensitive

// Statement termination
echo "Statements end with semicolon";  // Required
?>

<?php
// VARIABLES AND DATA TYPES

// Variables start with $ and are case-sensitive
$name = "Richard";
$age = 22;
$height = 1.75;
$is_student = true;
$nothing = null;

// Variable variables
$var_name = "message";
$$var_name = "Hello World";  // Creates $message = "Hello World"
echo $message;  // Outputs: Hello World

// Constants
define("SITE_NAME", "My Website");
const API_URL = "https://api.example.com";  // PHP 5.3+
echo SITE_NAME;

// Magic constants
echo __FILE__;      // Current file path
echo __LINE__;      // Current line number
echo __DIR__;       // Current directory
echo __FUNCTION__;  // Current function name
echo __CLASS__;     // Current class name
echo __METHOD__;    // Current method name

// Data types
$string = "Hello World";           // String
$integer = 42;                     // Integer
$float = 3.14159;                  // Float/Double
$boolean = true;                   // Boolean
$array = [1, 2, 3];               // Array
$object = new stdClass();          // Object
$resource = fopen("file.txt", "r"); // Resource
$null_value = null;                // NULL

// Type checking
var_dump($string);     // Detailed type information
echo gettype($age);    // Get variable type
is_string($name);      // Check if string
is_int($age);          // Check if integer
is_array($array);      // Check if array
isset($variable);      // Check if variable is set
empty($variable);      // Check if variable is empty

// Type casting
$string_number = "123";
$int_number = (int)$string_number;     // Cast to integer
$float_number = (float)$string_number; // Cast to float
$bool_value = (bool)$string_number;    // Cast to boolean
$array_value = (array)$string_number;  // Cast to array


// STRINGS

// String creation
$single_quotes = 'Single quotes - literal string';
$double_quotes = "Double quotes - variables parsed: $name";
$heredoc = <<<EOD
Heredoc syntax
Variables are parsed: $name
Multiple lines supported
EOD;

$nowdoc = <<<'EOD'
Nowdoc syntax (like single quotes)
Variables NOT parsed: $name
Multiple lines supported
EOD;

// String concatenation
$first_name = "John";
$last_name = "Doe";
$full_name = $first_name . " " . $last_name;  // Concatenation
$greeting = "Hello, " . $full_name . "!";

// String interpolation
$message = "Welcome, $first_name!";
$complex = "User {$first_name} has {$age} years";

// String functions
$text = "  Hello World  ";
echo strlen($text);              // String length
echo trim($text);                // Remove whitespace
echo ltrim($text);               // Remove left whitespace
echo rtrim($text);               // Remove right whitespace
echo strtoupper($text);          // Convert to uppercase
echo strtolower($text);          // Convert to lowercase
echo ucfirst($text);             // Capitalize first letter
echo ucwords($text);             // Capitalize each word

// String searching and replacement
$haystack = "The quick brown fox";
echo strpos($haystack, "quick");     // Find position of substring
echo str_replace("quick", "slow", $haystack);  // Replace substring
echo substr($haystack, 4, 5);       // Extract substring
echo substr_count($haystack, "o");   // Count occurrences

// String splitting and joining
$csv = "apple,banana,orange";
$fruits = explode(",", $csv);        // Split string to array
$rejoined = implode(" | ", $fruits); // Join array to string

// Regular expressions
$pattern = "/\d+/";  // Match digits
$text = "I have 5 apples and 3 oranges";
preg_match($pattern, $text, $matches);     // Find first match
preg_match_all($pattern, $text, $matches); // Find all matches
preg_replace($pattern, "X", $text);        // Replace with pattern


// ARRAYS

// Array creation
$indexed_array = [1, 2, 3, 4, 5];
$associative_array = [
    "name" => "John",
    "age" => 30,
    "city" => "New York"
];

// Alternative syntax
$old_syntax = array(1, 2, 3);
$old_assoc = array("key" => "value");

// Multidimensional arrays
$matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];

$users = [
    ["name" => "John", "age" => 30],
    ["name" => "Jane", "age" => 25],
    ["name" => "Bob", "age" => 35]
];

// Array access
echo $indexed_array[0];           // Access by index
echo $associative_array["name"];  // Access by key
echo $matrix[1][2];               // Access multidimensional

// Array modification
$fruits = ["apple", "banana"];
$fruits[] = "orange";             // Add to end
array_push($fruits, "grape");     // Add to end (alternative)
array_unshift($fruits, "mango");  // Add to beginning
$removed = array_pop($fruits);    // Remove from end
$removed = array_shift($fruits);  // Remove from beginning

// Array functions
$numbers = [3, 1, 4, 1, 5, 9, 2, 6];
echo count($numbers);             // Array length
echo array_sum($numbers);         // Sum of elements
echo max($numbers);               // Maximum value
echo min($numbers);               // Minimum value
sort($numbers);                   // Sort array
rsort($numbers);                  // Reverse sort
array_reverse($numbers);          // Reverse array

// Array searching
$fruits = ["apple", "banana", "orange"];
echo in_array("banana", $fruits); // Check if value exists
echo array_search("orange", $fruits); // Find key of value
$keys = array_keys($associative_array); // Get all keys
$values = array_values($associative_array); // Get all values

// Array iteration
foreach ($fruits as $fruit) {
    echo $fruit . "\n";
}

foreach ($associative_array as $key => $value) {
    echo "$key: $value\n";
}

// Array filtering and mapping
$numbers = [1, 2, 3, 4, 5, 6];
$even = array_filter($numbers, function($n) {
    return $n % 2 == 0;
});

$squared = array_map(function($n) {
    return $n * $n;
}, $numbers);

// Array merging and slicing
$array1 = [1, 2, 3];
$array2 = [4, 5, 6];
$merged = array_merge($array1, $array2);
$slice = array_slice($numbers, 2, 3); // Extract portion


// CONTROL STRUCTURES

// If statements
$score = 85;

if ($score >= 90) {
    echo "Grade A";
} elseif ($score >= 80) {
    echo "Grade B";
} elseif ($score >= 70) {
    echo "Grade C";
} else {
    echo "Grade F";
}

// Ternary operator
$status = ($score >= 70) ? "Pass" : "Fail";

// Null coalescing operator (PHP 7+)
$username = $_GET['user'] ?? 'guest';
$config = $user_config ?? $default_config ?? 'fallback';

// Switch statement
$day = "Monday";
switch ($day) {
    case "Monday":
    case "Tuesday":
    case "Wednesday":
    case "Thursday":
    case "Friday":
        echo "Weekday";
        break;
    case "Saturday":
    case "Sunday":
        echo "Weekend";
        break;
    default:
        echo "Invalid day";
}

// Match expression (PHP 8+)
$result = match($day) {
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday' => 'Weekday',
    'Saturday', 'Sunday' => 'Weekend',
    default => 'Invalid day'
};

// LOOPS

// For loop
for ($i = 0; $i < 10; $i++) {
  echo $i . " ";
}

// While loop
$counter = 0;
while ($counter < 5) {
  echo $counter . " ";
  $counter++;
}

// Do-while loop
$x = 0;
do {
  echo $x . " ";
  $x++;
} while ($x < 3);

// Foreach loop
$colors = ["red", "green", "blue"];
foreach ($colors as $color) {
  echo $color . " ";
}

foreach ($colors as $index => $color) {
  echo "$index: $color ";
}

// Loop control
for ($i = 0; $i < 10; $i++) {
  if ($i == 5) {
      continue; // Skip iteration
  }
  if ($i == 8) {
      break; // Exit loop
  }
  echo $i . " ";
}

// FUNCTIONS

// Basic function
function greet($name) {
  return "Hello, " . $name . "!";
}

echo greet("World");

// Function with default parameters
function createUser($name, $age = 18, $active = true) {
  return [
      'name' => $name,
      'age' => $age,
      'active' => $active
  ];
}

$user1 = createUser("John");
$user2 = createUser("Jane", 25);
$user3 = createUser("Bob", 30, false);

// Variable number of arguments
function sum(...$numbers) {
  return array_sum($numbers);
}

echo sum(1, 2, 3, 4, 5); // 15

// Type declarations (PHP 7+)
function add(int $a, int $b): int {
  return $a + $b;
}

function getUser(string $name): array {
  return ['name' => $name, 'id' => 1];
}

// Anonymous functions (closures)
$multiply = function($a, $b) {
  return $a * $b;
};

echo $multiply(4, 5); // 20

// Closures with use keyword
$factor = 10;
$multiplyByFactor = function($number) use ($factor) {
  return $number * $factor;
};

// Arrow functions (PHP 7.4+)
$double = fn($x) => $x * 2;
$numbers = [1, 2, 3, 4, 5];
$doubled = array_map(fn($x) => $x * 2, $numbers);

// Variable functions
function sayHello() {
  return "Hello!";
}

$func_name = "sayHello";
echo $func_name(); // Calls sayHello()

// Recursive functions
function factorial($n) {
  if ($n <= 1) {
      return 1;
  }
  return $n * factorial($n - 1);
}

echo factorial(5); // 120


// OBJECT-ORIENTED PROGRAMMING

// Basic class
class User {
  // Properties
  public $name;
  public $email;
  private $password;
  protected $created_at;
  
  // Constructor
  public function __construct($name, $email) {
      $this->name = $name;
      $this->email = $email;
      $this->created_at = date('Y-m-d H:i:s');
  }
  
  // Methods
  public function getName() {
      return $this->name;
  }
  
  public function setPassword($password) {
      $this->password = password_hash($password, PASSWORD_DEFAULT);
  }
  
  public function verifyPassword($password) {
      return password_verify($password, $this->password);
  }
  
  // Magic methods
  public function __toString() {
      return $this->name . " (" . $this->email . ")";
  }
  
  public function __get($property) {
      if (property_exists($this, $property)) {
          return $this->$property;
      }
  }
}

// Create objects
$user = new User("John Doe", "john@example.com");
$user->setPassword("secretpassword");
echo $user->getName();

// Inheritance
class AdminUser extends User {
  private $permissions = [];
  
  public function __construct($name, $email, $permissions = []) {
      parent::__construct($name, $email);
      $this->permissions = $permissions;
  }
  
  public function addPermission($permission) {
      $this->permissions[] = $permission;
  }
  
  public function hasPermission($permission) {
      return in_array($permission, $this->permissions);
  }
}

$admin = new AdminUser("Admin", "admin@example.com", ["read", "write"]);
$admin->addPermission("delete");

// Abstract classes
abstract class Animal {
  protected $name;
  
  public function __construct($name) {
      $this->name = $name;
  }
  
  abstract public function makeSound();
  
  public function getName() {
      return $this->name;
  }
}

class Dog extends Animal {
  public function makeSound() {
      return "Woof!";
  }
}


// Interfaces
interface Flyable {
  public function fly();
}

interface Swimmable {
  public function swim();
}

class Duck implements Flyable, Swimmable {
  public function fly() {
      return "Flying in the sky";
  }
  
  public function swim() {
      return "Swimming in water";
  }
}

// Traits (PHP 5.4+)
trait Timestampable {
  private $created_at;
  private $updated_at;
  
  public function touch() {
      $this->updated_at = date('Y-m-d H:i:s');
  }
  
  public function getCreatedAt() {
      return $this->created_at;
  }
}

class Post {
  use Timestampable;
  
  private $title;
  private $content;
  
  public function __construct($title, $content) {
      $this->title = $title;
      $this->content = $content;
      $this->created_at = date('Y-m-d H:i:s');
  }
}

// Static properties and methods
class Counter {
  private static $count = 0;
  
  public static function increment() {
      self::$count++;
  }
  
  public static function getCount() {
      return self::$count;
  }
}

Counter::increment();
echo Counter::getCount(); // 1

// Constants in classes
class MathConstants {
  const PI = 3.14159;
  const E = 2.71828;
}

echo MathConstants::PI;

// SUPERGLOBALS

// $_GET - HTTP GET data
// URL: script.php?name=John&age=25
$name = $_GET['name'] ?? '';
$age = $_GET['age'] ?? 0;

// $_POST - HTTP POST data
if ($_POST) {
  $username = $_POST['username'] ?? '';
  $password = $_POST['password'] ?? '';
}

// $_REQUEST - GET, POST, and COOKIE data combined
$data = $_REQUEST['data'] ?? '';

// $_SESSION - Session variables
session_start();
$_SESSION['user_id'] = 123;
$_SESSION['username'] = 'john_doe';

// $_COOKIE - Cookie values
setcookie('preferences', 'dark_theme', time() + 3600);
$preferences = $_COOKIE['preferences'] ?? 'light_theme';

// $_FILES - File upload information
if (isset($_FILES['upload'])) {
  $file = $_FILES['upload'];
  $filename = $file['name'];
  $tmp_name = $file['tmp_name'];
  $size = $file['size'];
  $error = $file['error'];
}

// $_SERVER - Server and environment information
$host = $_SERVER['HTTP_HOST'];
$uri = $_SERVER['REQUEST_URI'];
$method = $_SERVER['REQUEST_METHOD'];
$user_agent = $_SERVER['HTTP_USER_AGENT'];
$ip = $_SERVER['REMOTE_ADDR'];

// $_ENV - Environment variables
$path = $_ENV['PATH'] ?? '';

// $GLOBALS - References to all global variables
$x = 100;
function test() {
  echo $GLOBALS['x'];
}

// ERROR HANDLING

// Error reporting
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Try-catch exceptions
try {
  $pdo = new PDO("mysql:host=localhost;dbname=test", $user, $pass);
  $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
  echo "Connection failed: " . $e->getMessage();
} finally {
  echo "Cleanup code here";
}

// Custom exceptions
class CustomException extends Exception {
  public function errorMessage() {
      return "Error on line {$this->getLine()} in {$this->getFile()}: {$this->getMessage()}";
  }
}

function testFunction($value) {
  if ($value < 0) {
      throw new CustomException("Value cannot be negative");
  }
  return $value;
}

try {
  testFunction(-1);
} catch (CustomException $e) {
  echo $e->errorMessage();
}

// Error handlers
function customError($errno, $errstr, $errfile, $errline) {
  echo "Error: [$errno] $errstr in $errfile on line $errline";
}
set_error_handler("customError");

// FILE HANDLING

// Reading files
$content = file_get_contents('file.txt');
$lines = file('file.txt', FILE_IGNORE_NEW_LINES);

// Writing files
file_put_contents('output.txt', 'Hello World');
file_put_contents('log.txt', date('Y-m-d H:i:s') . " - Log entry\n", FILE_APPEND);

// File operations with handles
$handle = fopen('data.txt', 'r');
if ($handle) {
  while (($line = fgets($handle)) !== false) {
      echo $line;
  }
  fclose($handle);
}

// CSV handling
$csv_file = fopen('data.csv', 'w');
fputcsv($csv_file, ['Name', 'Age', 'City']);
fputcsv($csv_file, ['John', 25, 'New York']);
fclose($csv_file);

// Reading CSV
$csv_data = [];
if (($handle = fopen('data.csv', 'r')) !== false) {
  while (($data = fgetcsv($handle)) !== false) {
      $csv_data[] = $data;
  }
  fclose($handle);
}

// File information
if (file_exists('file.txt')) {
  echo filesize('file.txt') . " bytes\n";
  echo "Last modified: " . date('Y-m-d H:i:s', filemtime('file.txt'));
  echo is_readable('file.txt') ? "Readable" : "Not readable";
  echo is_writable('file.txt') ? "Writable" : "Not writable";
}

// Directory operations
$files = scandir('.');
$files = glob('*.php'); // Get PHP files

if (!is_dir('uploads')) {
  mkdir('uploads', 0755, true);
}

// DATABASE OPERATIONS (PDO)

// Database connection
try {
  $pdo = new PDO(
      "mysql:host=localhost;dbname=myapp;charset=utf8",
      $username,
      $password,
      [
          PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
          PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
          PDO::ATTR_EMULATE_PREPARES => false
      ]
  );
} catch (PDOException $e) {
  die("Connection failed: " . $e->getMessage());
}

// Prepared statements (SELECT)
$stmt = $pdo->prepare("SELECT * FROM users WHERE age > ? AND city = ?");
$stmt->execute([25, 'New York']);
$users = $stmt->fetchAll();

// Single row
$stmt = $pdo->prepare("SELECT * FROM users WHERE id = ?");
$stmt->execute([$user_id]);
$user = $stmt->fetch();

// INSERT
$stmt = $pdo->prepare("INSERT INTO users (name, email, age) VALUES (?, ?, ?)");
$stmt->execute(['John Doe', 'john@example.com', 30]);
$user_id = $pdo->lastInsertId();

// UPDATE
$stmt = $pdo->prepare("UPDATE users SET age = ? WHERE id = ?");
$stmt->execute([31, $user_id]);
$affected_rows = $stmt->rowCount();

// DELETE
$stmt = $pdo->prepare("DELETE FROM users WHERE id = ?");
$stmt->execute([$user_id]);

// Transactions
try {
  $pdo->beginTransaction();
  
  $stmt1 = $pdo->prepare("INSERT INTO accounts (name, balance) VALUES (?, ?)");
  $stmt1->execute(['John', 1000]);
  
  $stmt2 = $pdo->prepare("INSERT INTO transactions (account_id, amount) VALUES (?, ?)");
  $stmt2->execute([$pdo->lastInsertId(), -100]);
  
  $pdo->commit();
} catch (Exception $e) {
  $pdo->rollback();
  throw $e;
}

// FORM HANDLING AND VALIDATION

// HTML form processing
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
  $name = trim($_POST['name'] ?? '');
  $email = trim($_POST['email'] ?? '');
  $age = (int)($_POST['age'] ?? 0);
  
  $errors = [];
  
  // Validation
  if (empty($name)) {
      $errors[] = "Name is required";
  }
  
  if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
      $errors[] = "Invalid email format";
  }
  
  if ($age < 18 || $age > 120) {
      $errors[] = "Age must be between 18 and 120";
  }
  
  if (empty($errors)) {
      // Process valid data
      echo "Form submitted successfully!";
  } else {
      // Display errors
      foreach ($errors as $error) {
          echo "<p class='error'>$error</p>";
      }
  }
}

// File upload handling
if (isset($_FILES['upload']) && $_FILES['upload']['error'] === UPLOAD_ERR_OK) {
  $file = $_FILES['upload'];
  $allowed_types = ['image/jpeg', 'image/png', 'image/gif'];
  $max_size = 5 * 1024 * 1024; // 5MB
  
  if (!in_array($file['type'], $allowed_types)) {
      $errors[] = "Invalid file type";
  }
  
  if ($file['size'] > $max_size) {
      $errors[] = "File too large";
  }
  
  if (empty($errors)) {
      $upload_dir = 'uploads/';
      $filename = uniqid() . '_' . $file['name'];
      $target_path = $upload_dir . $filename;
      
      if (move_uploaded_file($file['tmp_name'], $target_path)) {
          echo "File uploaded successfully: $filename";
      } else {
          echo "Upload failed";
      }
  }
}

// SECURITY

// Password hashing
$password = 'user_password';
$hashed = password_hash($password, PASSWORD_DEFAULT);

// Password verification
if (password_verify($password, $hashed)) {
  echo "Password is correct";
}

// CSRF protection
session_start();
if (empty($_SESSION['csrf_token'])) {
  $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}

// In forms: <input type="hidden" name="csrf_token" value="<?= $_SESSION['csrf_token'] ?>">

// Verify CSRF token
if ($_POST['csrf_token'] !== $_SESSION['csrf_token']) {
  die('CSRF token mismatch');
}

// Input sanitization
$name = filter_input(INPUT_POST, 'name', FILTER_SANITIZE_STRING);
$email = filter_input(INPUT_POST, 'email', FILTER_SANITIZE_EMAIL);
$age = filter_input(INPUT_POST, 'age', FILTER_VALIDATE_INT);

// XSS prevention
function escape($string) {
  return htmlspecialchars($string, ENT_QUOTES, 'UTF-8');
}

echo escape($user_input); // Always escape output

// SQL injection prevention (use prepared statements)
// BAD: "SELECT * FROM users WHERE id = " . $_GET['id']
// GOOD: Use prepared statements as shown in database section

// SESSIONS AND COOKIES

// Start session
session_start();

// Set session variables
$_SESSION['user_id'] = 123;
$_SESSION['username'] = 'john_doe';
$_SESSION['role'] = 'admin';

// Check if user is logged in
function isLoggedIn() {
  return isset($_SESSION['user_id']);
}

// Logout
function logout() {
  session_unset();
  session_destroy();
  setcookie(session_name(), '', time() - 3600, '/');
}

// Set cookie
setcookie('preference', 'dark_mode', time() + (86400 * 30), '/'); // 30 days

// Secure cookie
setcookie('secure_data', $value, [
  'expires' => time() + 3600,
  'path' => '/',
  'domain' => '.example.com',
  'secure' => true,      // HTTPS only
  'httponly' => true,    // No JavaScript access
  'samesite' => 'Strict' // CSRF protection
]);

// JSON AND API

// JSON encoding/decoding
$data = [
  'name' => 'John',
  'age' => 30,
  'active' => true
];

$json = json_encode($data);
$decoded = json_decode($json, true); // true for associative array

// API response
header('Content-Type: application/json');
http_response_code(200);
echo json_encode(['status' => 'success', 'data' => $data]);

// Simple REST API endpoint
switch ($_SERVER['REQUEST_METHOD']) {
  case 'GET':
      // Get resources
      $users = getAllUsers();
      echo json_encode($users);
      break;
      
  case 'POST':
      // Create resource
      $input = json_decode(file_get_contents('php://input'), true);
      $user = createUser($input);
      http_response_code(201);
      echo json_encode($user);
      break;
      
  case 'PUT':
      // Update resource
      $input = json_decode(file_get_contents('php://input'), true);
      $user = updateUser($_GET['id'], $input);
      echo json_encode($user);
      break;
      
  case 'DELETE':
      // Delete resource
      deleteUser($_GET['id']);
      http_response_code(204);
      break;
      
  default:
      http_response_code(405);
      echo json_encode(['error' => 'Method not allowed']);
}

// MODERN PHP FEATURES

// Null coalescing assignment (PHP 7.4+)
$config['timeout'] ??= 30;

// Match expressions (PHP 8+)
$result = match($status) {
  'pending' => 'Waiting for approval',
  'approved' => 'Ready to proceed',
  'rejected' => 'Application denied',
  default => 'Unknown status'
};

// Named arguments (PHP 8+)
function createUser($name, $email, $age = 18, $active = true) {
  return compact('name', 'email', 'age', 'active');
}

$user = createUser(
  name: 'John',
  email: 'john@example.com',
  active: false
);

// Constructor property promotion (PHP 8+)
class Point {
  public function __construct(
      public float $x,
      public float $y,
      public float $z = 0
  ) {}
}

// Union types (PHP 8+)
function process(string|int $value): string|int {
  return is_string($value) ? strtoupper($value) : $value * 2;
}

// Attributes (PHP 8+)
#[Route('/api/users', methods: ['GET', 'POST'])]
class UserController {
  #[Validate(['name' => 'required', 'email' => 'email'])]
  public function create() {
      // Method implementation
  }
}

// DEBUGGING AND DEVELOPMENT

// Debugging functions
var_dump($variable);     // Detailed variable information
print_r($array);         // Human-readable array output
debug_backtrace();       // Function call stack

// Error logging
error_log("Debug message: " . $debug_info);
error_log(print_r($array, true)); // Log array contents

// Development helpers
if (defined('DEBUG') && DEBUG) {
  ini_set('display_errors', 1);
  error_reporting(E_ALL);
} else {
  ini_set('display_errors', 0);
  error_reporting(0);
}

// Performance timing
$start_time = microtime(true);
// ... code to measure ...
$end_time = microtime(true);
$execution_time = ($end_time - $start_time) * 1000; // milliseconds

// Memory usage
echo "Memory usage: " . memory_get_usage(true) . " bytes\n";
echo "Peak memory: " . memory_get_peak_usage(true) . " bytes\n";

// COMPOSER AND AUTOLOADING

// composer.json example
/*
{
  "name": "myproject/myapp",
  "description": "My PHP Application",
  "type": "project",
  "require": {
      "php": ">=8.0",
      "monolog/monolog": "^2.0",
      "guzzlehttp/guzzle": "^7.0",
      "vlucas/phpdotenv": "^5.0"
  },
  "require-dev": {
      "phpunit/phpunit": "^9.0",
      "squizlabs/php_codesniffer": "^3.6"
  },
  "autoload": {
      "psr-4": {
          "App\\": "src/"
      }
  },
  "scripts": {
      "test": "phpunit",
      "lint": "phpcs --standard=PSR12 src/"
  }
}
*/

// Install dependencies: composer install
// Update dependencies: composer update
// Add package: composer require vendor/package

// Autoloading
require_once 'vendor/autoload.php';

// PSR-4 autoloading example
// File: src/Models/User.php
namespace App\Models;

class User {
  // Class implementation
}

// Usage
use App\Models\User;
$user = new User();

// COMMON LIBRARIES AND FRAMEWORKS

// Environment variables with vlucas/phpdotenv
require_once 'vendor/autoload.php';
$dotenv = Dotenv\Dotenv::createImmutable(__DIR__);
$dotenv->load();

$database_url = $_ENV['DATABASE_URL'];
$api_key = $_ENV['API_KEY'];

// HTTP requests with Guzzle
use GuzzleHttp\Client;
use GuzzleHttp\Exception\RequestException;

$client = new Client();

try {
  $response = $client->request('GET', 'https://api.example.com/users');
  $data = json_decode($response->getBody(), true);
} catch (RequestException $e) {
  echo "Request failed: " . $e->getMessage();
}

// POST request
$response = $client->request('POST', 'https://api.example.com/users', [
  'json' => [
      'name' => 'John Doe',
      'email' => 'john@example.com'
  ],
  'headers' => [
      'Authorization' => 'Bearer ' . $api_token
  ]
]);

// Logging with Monolog
use Monolog\Logger;
use Monolog\Handler\StreamHandler;
use Monolog\Handler\RotatingFileHandler;

$logger = new Logger('app');
$logger->pushHandler(new StreamHandler('logs/app.log', Logger::WARNING));
$logger->pushHandler(new RotatingFileHandler('logs/debug.log', 0, Logger::DEBUG));

$logger->info('User logged in', ['user_id' => 123]);
$logger->error('Database connection failed', ['exception' => $e->getMessage()]);

// TESTING WITH PHPUNIT

// Example test class
use PHPUnit\Framework\TestCase;

class UserTest extends TestCase {
  private $user;
  
  protected function setUp(): void {
      $this->user = new User('John', 'john@example.com');
  }
  
  public function testUserCanBeCreated(): void {
      $this->assertInstanceOf(User::class, $this->user);
      $this->assertEquals('John', $this->user->getName());
      $this->assertEquals('john@example.com', $this->user->getEmail());
  }
  
  public function testUserCanSetPassword(): void {
      $this->user->setPassword('secretpass');
      $this->assertTrue($this->user->verifyPassword('secretpass'));
      $this->assertFalse($this->user->verifyPassword('wrongpass'));
  }
  
  public function testEmailValidation(): void {
      $this->expectException(InvalidArgumentException::class);
      new User('John', 'invalid-email');
  }
  
  /**
   * @dataProvider validEmailProvider
   */
  public function testValidEmails($email): void {
      $user = new User('Test', $email);
      $this->assertEquals($email, $user->getEmail());
  }
  
  public function validEmailProvider(): array {
      return [
          ['test@example.com'],
          ['user.name@domain.co.uk'],
          ['user+tag@example.org']
      ];
  }
}

// Run tests: ./vendor/bin/phpunit

// DESIGN PATTERNS

// Singleton Pattern
class Database {
  private static $instance = null;
  private $connection;
  
  private function __construct() {
      $this->connection = new PDO(
          "mysql:host=localhost;dbname=myapp",
          $username,
          $password
      );
  }
  
  public static function getInstance(): self {
      if (self::$instance === null) {
          self::$instance = new self();
      }
      return self::$instance;
  }
  
  public function getConnection(): PDO {
      return $this->connection;
  }
  
  // Prevent cloning and unserialization
  private function __clone() {}
  public function __wakeup() {}
}

// Factory Pattern
interface Animal {
  public function makeSound(): string;
}

class Dog implements Animal {
  public function makeSound(): string {
      return "Woof!";
  }
}

class Cat implements Animal {
  public function makeSound(): string {
      return "Meow!";
  }
}

class AnimalFactory {
  public static function create(string $type): Animal {
      switch (strtolower($type)) {
          case 'dog':
              return new Dog();
          case 'cat':
              return new Cat();
          default:
              throw new InvalidArgumentException("Unknown animal type: $type");
      }
  }
}

$dog = AnimalFactory::create('dog');

// Observer Pattern
interface Observer {
  public function update(Subject $subject): void;
}

interface Subject {
  public function attach(Observer $observer): void;
  public function detach(Observer $observer): void;
  public function notify(): void;
}

class NewsletterSubscriber implements Observer {
  private string $email;
  
  public function __construct(string $email) {
      $this->email = $email;
  }
  
  public function update(Subject $subject): void {
      echo "Sending newsletter to {$this->email}\n";
  }
}

class Newsletter implements Subject {
  private array $observers = [];
  private string $content;
  
  public function attach(Observer $observer): void {
      $this->observers[] = $observer;
  }
  
  public function detach(Observer $observer): void {
      $key = array_search($observer, $this->observers);
      if ($key !== false) {
          unset($this->observers[$key]);
      }
  }
  
  public function notify(): void {
      foreach ($this->observers as $observer) {
          $observer->update($this);
      }
  }
  
  public function setContent(string $content): void {
      $this->content = $content;
      $this->notify();
  }
}

// Strategy Pattern
interface PaymentStrategy {
  public function pay(float $amount): bool;
}

class CreditCardPayment implements PaymentStrategy {
  private string $cardNumber;
  
  public function __construct(string $cardNumber) {
      $this->cardNumber = $cardNumber;
  }
  
  public function pay(float $amount): bool {
      echo "Paid $amount using credit card ending in " . substr($this->cardNumber, -4) . "\n";
      return true;
  }
}

class PayPalPayment implements PaymentStrategy {
  private string $email;
  
  public function __construct(string $email) {
      $this->email = $email;
  }
  
  public function pay(float $amount): bool {
      echo "Paid $amount using PayPal account $this->email\n";
      return true;
  }
}

class ShoppingCart {
  private PaymentStrategy $paymentStrategy;
  
  public function setPaymentStrategy(PaymentStrategy $strategy): void {
      $this->paymentStrategy = $strategy;
  }
  
  public function checkout(float $amount): bool {
      return $this->paymentStrategy->pay($amount);
  }
}

// WEB FRAMEWORKS BASICS

// Simple Router Example
class Router {
  private array $routes = [];
  
  public function get(string $path, callable $callback): void {
      $this->routes['GET'][$path] = $callback;
  }
  
  public function post(string $path, callable $callback): void {
      $this->routes['POST'][$path] = $callback;
  }
  
  public function dispatch(): void {
      $method = $_SERVER['REQUEST_METHOD'];
      $path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
      
      if (isset($this->routes[$method][$path])) {
          $callback = $this->routes[$method][$path];
          $callback();
      } else {
          http_response_code(404);
          echo "Not Found";
      }
  }
}

// Usage
$router = new Router();

$router->get('/', function() {
  echo "Welcome to homepage!";
});

$router->get('/users', function() {
  header('Content-Type: application/json');
  echo json_encode(['users' => getAllUsers()]);
});

$router->post('/users', function() {
  $input = json_decode(file_get_contents('php://input'), true);
  $user = createUser($input);
  header('Content-Type: application/json');
  echo json_encode($user);
});

$router->dispatch();

// MVC Pattern Example
// Model
class UserModel {
  private PDO $db;
  
  public function __construct(PDO $db) {
      $this->db = $db;
  }
  
  public function findAll(): array {
      $stmt = $this->db->query("SELECT * FROM users");
      return $stmt->fetchAll();
  }
  
  public function findById(int $id): ?array {
      $stmt = $this->db->prepare("SELECT * FROM users WHERE id = ?");
      $stmt->execute([$id]);
      return $stmt->fetch() ?: null;
  }
  
  public function create(array $data): int {
      $stmt = $this->db->prepare("INSERT INTO users (name, email) VALUES (?, ?)");
      $stmt->execute([$data['name'], $data['email']]);
      return $this->db->lastInsertId();
  }
}

// View
class View {
  private string $template;
  private array $data = [];
  
  public function __construct(string $template) {
      $this->template = $template;
  }
  
  public function assign(string $key, $value): void {
      $this->data[$key] = $value;
  }
  
  public function render(): string {
      extract($this->data);
      ob_start();
      include $this->template;
      return ob_get_clean();
  }
}

// Controller
class UserController {
  private UserModel $userModel;
  
  public function __construct(UserModel $userModel) {
      $this->userModel = $userModel;
  }
  
  public function index(): void {
      $users = $this->userModel->findAll();
      $view = new View('views/users/index.php');
      $view->assign('users', $users);
      echo $view->render();
  }
  
  public function show(int $id): void {
      $user = $this->userModel->findById($id);
      if (!$user) {
          http_response_code(404);
          echo "User not found";
          return;
      }
      
      $view = new View('views/users/show.php');
      $view->assign('user', $user);
      echo $view->render();
  }
}

// CACHING

// Simple file-based cache
class FileCache {
  private string $cacheDir;
  
  public function __construct(string $cacheDir = 'cache') {
      $this->cacheDir = $cacheDir;
      if (!is_dir($cacheDir)) {
          mkdir($cacheDir, 0755, true);
      }
  }
  
  public function get(string $key) {
      $file = $this->cacheDir . '/' . md5($key) . '.cache';
      
      if (!file_exists($file)) {
          return null;
      }
      
      $data = unserialize(file_get_contents($file));
      
      if ($data['expires'] && time() > $data['expires']) {
          unlink($file);
          return null;
      }
      
      return $data['value'];
  }
  
  public function set(string $key, $value, int $ttl = 3600): void {
      $file = $this->cacheDir . '/' . md5($key) . '.cache';
      $data = [
          'value' => $value,
          'expires' => $ttl > 0 ? time() + $ttl : null
      ];
      
      file_put_contents($file, serialize($data));
  }
  
  public function delete(string $key): void {
      $file = $this->cacheDir . '/' . md5($key) . '.cache';
      if (file_exists($file)) {
          unlink($file);
      }
  }
}

// Usage
$cache = new FileCache();

$users = $cache->get('all_users');
if ($users === null) {
  $users = $userModel->findAll(); // Expensive database query
  $cache->set('all_users', $users, 300); // Cache for 5 minutes
}

// COMMAND LINE INTERFACE

// CLI script example
if (php_sapi_name() !== 'cli') {
  die('This script can only be run from command line');
}

// Get command line arguments
$options = getopt('u:p:h', ['user:', 'password:', 'help']);

if (isset($options['h']) || isset($options['help'])) {
  echo "Usage: php script.php -u username -p password\n";
  exit(0);
}

$username = $options['u'] ?? $options['user'] ?? null;
$password = $options['p'] ?? $options['password'] ?? null;

if (!$username || !$password) {
  echo "Error: Username and password are required\n";
  exit(1);
}

// Interactive input
echo "Enter your name: ";
$name = trim(fgets(STDIN));

echo "Are you sure? (y/n): ";
$confirm = trim(fgets(STDIN));

if (strtolower($confirm) === 'y') {
  echo "Processing...\n";
  // Do something
  echo "Done!\n";
} else {
  echo "Cancelled.\n";
}

// Progress bar example
function showProgress(int $current, int $total): void {
  $percent = ($current / $total) * 100;
  $bar = str_repeat('=', (int)($percent / 2));
  $spaces = str_repeat(' ', 50 - strlen($bar));
  echo "\r[{$bar}{$spaces}] {$percent}%";
  
  if ($current === $total) {
      echo "\n";
  }
}

for ($i = 1; $i <= 100; $i++) {
  usleep(50000); // Simulate work
  showProgress($i, 100);
}

// CONFIGURATION AND BEST PRACTICES

// Configuration class
class Config {
  private static array $config = [];
  
  public static function load(string $file): void {
      if (file_exists($file)) {
          self::$config = include $file;
      }
  }
  
  public static function get(string $key, $default = null) {
      $keys = explode('.', $key);
      $value = self::$config;
      
      foreach ($keys as $k) {
          if (!isset($value[$k])) {
              return $default;
          }
          $value = $value[$k];
      }
      
      return $value;
  }
}

// config.php file
return [
  'database' => [
      'host' => 'localhost',
      'port' => 3306,
      'database' => 'myapp',
      'username' => 'user',
      'password' => 'pass'
  ],
  'app' => [
      'name' => 'My Application',
      'debug' => true,
      'timezone' => 'UTC'
  ],
  'cache' => [
      'default_ttl' => 3600
  ]
];

// Usage
Config::load('config.php');
$db_host = Config::get('database.host');
$app_name = Config::get('app.name', 'Default App');

// SECURITY BEST PRACTICES

// 1. Input Validation
function validateInput(array $data, array $rules): array {
  $errors = [];
  
  foreach ($rules as $field => $rule) {
      $value = $data[$field] ?? null;
      
      if (isset($rule['required']) && empty($value)) {
          $errors[$field] = "Field is required";
          continue;
      }
      
      if (!empty($value) && isset($rule['type'])) {
          switch ($rule['type']) {
              case 'email':
                  if (!filter_var($value, FILTER_VALIDATE_EMAIL)) {
                      $errors[$field] = "Invalid email format";
                  }
                  break;
              case 'int':
                  if (!filter_var($value, FILTER_VALIDATE_INT)) {
                      $errors[$field] = "Must be an integer";
                  }
                  break;
              case 'url':
                  if (!filter_var($value, FILTER_VALIDATE_URL)) {
                      $errors[$field] = "Invalid URL format";
                  }
                  break;
          }
      }
      
      if (!empty($value) && isset($rule['min_length']) && strlen($value) < $rule['min_length']) {
          $errors[$field] = "Minimum length is {$rule['min_length']} characters";
      }
      
      if (!empty($value) && isset($rule['max_length']) && strlen($value) > $rule['max_length']) {
          $errors[$field] = "Maximum length is {$rule['max_length']} characters";
      }
  }
  
  return $errors;
}

// 2. Rate Limiting
class RateLimiter {
  private string $cacheDir;
  
  public function __construct(string $cacheDir = 'rate_limit') {
      $this->cacheDir = $cacheDir;
      if (!is_dir($cacheDir)) {
          mkdir($cacheDir, 0755, true);
      }
  }
  
  public function isAllowed(string $identifier, int $maxRequests = 60, int $timeWindow = 3600): bool {
      $file = $this->cacheDir . '/' . md5($identifier) . '.json';
      $now = time();
      
      if (file_exists($file)) {
          $data = json_decode(file_get_contents($file), true);
          $requests = array_filter($data['requests'], fn($time) => $now - $time < $timeWindow);
      } else {
          $requests = [];
      }
      
      if (count($requests) >= $maxRequests) {
          return false;
      }
      
      $requests[] = $now;
      file_put_contents($file, json_encode(['requests' => $requests]));
      
      return true;
  }
}

// 3. Content Security Policy
function setSecurityHeaders(): void {
  header("Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'");
  header("X-Content-Type-Options: nosniff");
  header("X-Frame-Options: DENY");
  header("X-XSS-Protection: 1; mode=block");
  header("Referrer-Policy: strict-origin-when-cross-origin");
}

// COMMON UTILITIES

// String utilities
class StringHelper {
  public static function slug(string $text): string {
      $text = strtolower($text);
      $text = preg_replace('/[^a-z0-9\s-]/', '', $text);
      $text = preg_replace('/[\s-]+/', '-', $text);
      return trim($text, '-');
  }
  
  public static function truncate(string $text, int $length = 100, string $append = '...'): string {
      if (strlen($text) <= $length) {
          return $text;
      }
      
      return substr($text, 0, $length) . $append;
  }
  
  public static function randomString(int $length = 10): string {
      return substr(str_shuffle('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), 0, $length);
  }
}

// Array utilities
class ArrayHelper {
  public static function get(array $array, string $key, $default = null) {
      $keys = explode('.', $key);
      
      foreach ($keys as $k) {
          if (!is_array($array) || !array_key_exists($k, $array)) {
              return $default;
          }
          $array = $array[$k];
      }
      
      return $array;
  }
  
  public static function pluck(array $array, string $key): array {
      return array_map(fn($item) => $item[$key] ?? null, $array);
  }
  
  public static function groupBy(array $array, string $key): array {
      $result = [];
      
      foreach ($array as $item) {
          $groupKey = $item[$key] ?? 'unknown';
          $result[$groupKey][] = $item;
      }
      
      return $result;
  }
}

// Date utilities
class DateHelper {
  public static function timeAgo(string $datetime): string {
      $time = time() - strtotime($datetime);
      
      if ($time < 60) return 'just now';
      if ($time < 3600) return floor($time/60) . ' minutes ago';
      if ($time < 86400) return floor($time/3600) . ' hours ago';
      if ($time < 2592000) return floor($time/86400) . ' days ago';
      if ($time < 31536000) return floor($time/2592000) . ' months ago';
      
      return floor($time/31536000) . ' years ago';
  }
  
  public static function isWeekend(string $date = null): bool {
      $timestamp = $date ? strtotime($date) : time();
      $dayOfWeek = date('N', $timestamp);
      return $dayOfWeek >= 6;
  }
}

echo "\n" . str_repeat("=", 50) . "\n";
echo "PHP Reference Complete!\n";
echo "PHP powers the web with dynamic server-side scripting\n";
echo "Remember: Always validate input, escape output, use prepared statements\n";
echo "Start with: php -S localhost:8000 for development server\n";
echo str_repeat("=", 50) . "\n";
?>