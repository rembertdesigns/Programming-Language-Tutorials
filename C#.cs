// C# - Object Oriented Programming language - by Richard Rembert


// file > new project > web web app > basic + MVC


// useful starting lines; they are assemblies (libraries) that contain useful objects, methods, functions.
using System; // contain for example the Console object
using System.Collections.Generic; // contains the list 
using System.Linq;
using System.Text;
using System.Threading.Tasks;
// less usual assembly
using System.IO; // File methods like creating, writing, saving 
using System.Speech.Synthesis; // synthesize strings and more
using Microsoft.VisualStudio.TestTools.UnitTesting; // testing assembly
using anyClass; // could be any crafted class (Grades ie)


// without VS -> in console
$ dotnet new console
$ dotnet restore
$ dotnet run


// Hello world
namespace Hello
{
  class Program
  {
    static void Main(string[] args) // Main method is MANDATORY for the code to work and must be STATIC
    {
      Console.WriteLine("Hello, World!");
    }
  }
}


// Hello world with prompt
namespace Hello
{
  class Program
  {
    static void Main(string[] args) // void is used when there is no return in a method
    {
      Console.WriteLine("Your name?");
      string name = Console.ReadLine();
      Console.WriteLine("Hello " + name);
    }
  }
}


// use parameters from command line
namespace Hello
{
  class Program
  {
    static void Main(string[] args) // you can populate parameters through properties > debug; parameters are delimited with spaces; that can be tricked with ""
    {
      foreach (string parameters in args) 
      {
        Console.WriteLine(parameters);
      }
    }
  }
}


// Types
var anyType; // var force the compiler to find the right type for us; should be avoided for clarity
byte number; // integer 0 à 255
short number; // integer -32768 à 32767
int number; // integer -2147483648 à 2147483647
byte number; // integer -9223372036854775808 à 9223372036854775807
float number; // simple precision -3,402823e38 à 3,402823e38
double number; // double precision  -1,79769313486232e308 à 1,79769313486232e308
decimal number; // for financial values with lots of decimals
char letter; // single character
string name; // character string
bool diff; // true or false
type[] arr; // array of any type (ex int); reference type
List<type> list; // list of any type (ex int)
DayOfWeek day; // day of week enum


// Type casting
short s = 200; int i = s; // 200 becomes of int type
int i = 200; short s = i; // not possible; short could be too specific for the number
int i = 200; short s = (short)i; // possible because we cast 'explicitly'
int i = 40000; short s = (short)i; // not possible even with cast 'explicitly'
int i = 20; double d = i; // 20 becomes of double type
int age = 24; string ageStr = Convert.ToString(age); // converts int to string "24"
string ageStr = "24"; int age = int.Parse(ageStr); // converts string to int 24
int.TryParse(ageStr, out age); // checks if parse is authorized and stores value in out int; will return true here


// Boxing
int i = 5;
object o = i;
o = 6; // i = 5; o = 6


// Unboxing
int i =5;
object o = i;
o = 6;
int j = (int)o;
j = 7; // i = 5; o = 6; j = 7


// Console methods
Console.WriteLine("Hello"); // writes the string
string input = Console.ReadLine(); // ask user to type a line; needs to be stored
Console.ReadKey(true); // ask user to type on a key to continue; true prevents the write the key typed on
ConsoleKeyInfo typed = Console.ReadKey(true).key; // contains the key
Console.ForegroundColor = ConsoleColor.Black; // color of the text
Console.BackgroundColor = ConsoleColor.White; // color of background
Console.ResetColor(); // reset back and fore colors
Console.Clear(); // clear the screen
Console.SetCursorPosition(20, 5); // sets cursor at column 20 and line 5
int x = Console.CursorLeft; // current position of cursor column
int y = Console.CursorRight; // current position of cursor line
int w = Console.WindowWidth; // current width of window
Console.MoveBufferArea(5, 6, 7, 8, 9, 10); // the text in the zone ((5,6),(12,14)) will move to (9,10)


// readline of a number
static void Main(string[] args)
{
  bool inputValid = false;
  int age = -1;
  while (!inputValid)
  {
    Console.WriteLine("Please type your age");
    string input = Console.ReadLine();
    if (int.TryParse(input, out age))
    {
      inputValid = true;
    }
    else
    {
      inputValid = false;
      Console.WriteLine("Your age input is invalid !");
    }
  }
  Console.WriteLine("You are " + age + " years old");
}


// LIST; can be in 2 dimension
List<string> days = new List<string>();
days.Add("Monday"); // "Monday"
days.Add("Tuesday"); // "Monday", "Tuesday"
days.Insert(2, "Wednesday"); // "Monday", "Tuesday", "Wednesday"
days.Remove("Wednesday"); // "Monday", "Tuesday"
days.RemoveAt(1); // "Monday"
int length = days.Count; // 1
int index = days.indexOf("Monday"); // 0
// shortened way
List<string> days = new List<string> {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
// write
Console.WriteLine(days[0]); // Monday


// ARRAY; fixed length
string[] days = new string[7];
days[0] = "Monday";
days[1] = "Tuesday";
days[2] = "Wednesday";
days[3] = "Thursday";
days[4] = "Friday";
days[5] = "Saturday";
days[6] = "Sunday";
int length = days.Length; // 7
Array.Sort(days); // sort alpha
// shortened way
string[] days = new string[] {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
// write
Console.WriteLine(days[0]); // Monday


// enum
enum Days
{
  Monday,
  Tuesday,
  Wednesday,
  Thursday,
  Friday,
  Saturday,
  Sunday
}
// write
Console.WriteLine(Days.Monday); // Monday


// conditions
namespace Sleep
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine("How many hours did you sleep?");
      Int hours = Console.ReadLine();
      if(hours >= 8)
      {
        Console.WriteLine("You slept enough");
      }
      else
      {
        Console.WriteLine("You need more sleep");
      }
    }
  }
}
// ternary operator
string test = 18 > 20 ? "passed" : "failed";


// switch
string result;
switch(letterGrade)
{
  case "A":
    result = "Excellent";
    break;
  case "B":
    result = "Good";
    break;
  case "C":
    result = "Average";
    break;
  case "D":
    result = "Below Average";
    break;
  default:
    result = "Failing";
    break;
}
return result;


// foreach loop
foreach (int day in days)
{
  Console.WriteLine(day);
}


// for loop
for (int i = 0; i<grades.Count; i++) 
{
  Console.WriteLine(grade);
}


// while loop
while (age > 0)
{
  age -= 1;
  Console.WriteLine(age);
}


// do while loop 
do 
{
  age++;
  Console.WriteLine(age);
} while (age < 50);


// break
for (int i = 0; i<5; i++) 
{
  if (i == 3)
  {
    break;
  }
  Console.WriteLine(i);
} // 0, 1, 2


// continue
for (int i = 0; i<5; i++) 
{
  if (i == 3)
  {
    continue;
  }
  Console.WriteLine(i);
} // 0, 1, 2, 4


// object
static void writeResult(string description, int result) 
{
  Console.WriteLine(description + ": " + result);
  // OR
  Console.WriteLine("{0}: {1}", description, result); // c# string formatting; can add other strings and characters
  // OR
  Console.WriteLine($"{description}: {result}");
}
writeResult("My age", 24); // My age: 24


// constructor; instiancates an instance of that class, creates an object
GradeBook book = new GradeBook(); // default constructor
public GradeBook() // custom constructor
{
  // initialization code
}


// Interface
internal interface IThing
{
  // ... no access modifiers allowed (because not needed)
}