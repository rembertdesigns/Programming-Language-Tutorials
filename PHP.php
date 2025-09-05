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