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


