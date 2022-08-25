HYPERTEXT PREPROCESSOR
  -Server Scripting
    -by Richard Rembert


// To create a comment. Shown in code only. Indent 2spaces
/* Multiple line comment */

<?php ?> or <? ?> // PHP tags

<?php echo "text that will be written"; ?>
<?php print "text that will be written"; ?>
<?php include("file.php"); ?>


// VARIABLES



<?php $name_of_visitor = "Richard"; ?> // string = "text"
<?php $age_du_visitor = 22; ?> // int = 42
<?php $visitor_weight = 73.5; ?> // float = 42.98
<?php $je_suis_un_zero = true; ?> // bool = true/false
<?php $no_value = null; ?> // NULL = nothing



<?php echo 'Visitor has '. $age_of_visitor . ' year.'; ?>



<?php $number = (3+4-2)*8/5; ?> // number is 8
<?php
$number = 10; // number is 10
$result = $number+10; // result is 20
?>
<?php $number = 10%3; ?> //number is 1 pcq modulo (%) calculates the remainder of the division


