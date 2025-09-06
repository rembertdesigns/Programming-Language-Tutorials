/ SOLIDITY - Smart Contract Development Reference - by Richard Rembert

// Solidity is a statically-typed, contract-oriented programming language 
// designed for developing smart contracts that run on the Ethereum Virtual Machine (EVM)

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND DEVELOPMENT ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════════════════

/*
DEVELOPMENT ENVIRONMENT SETUP:

1. Install Node.js and npm
2. Choose a development framework:
   
   HARDHAT (Recommended):
   npm install --save-dev hardhat
   npx hardhat init
   
   FOUNDRY (Rust-based, fast):
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   forge init my-project
   
   REMIX IDE (Browser-based):
   https://remix.ethereum.org

3. Install dependencies:
   npm install @openzeppelin/contracts
   npm install --save-dev @nomicfoundation/hardhat-toolbox

4. Basic commands:
   npx hardhat compile    // Compile contracts
   npx hardhat test      // Run tests
   npx hardhat node      // Start local blockchain
   npx hardhat run scripts/deploy.js --network localhost

SPDX License Identifiers (required at top of every file):
- MIT: Most permissive
- GPL-3.0: Copyleft license
- Apache-2.0: Permissive with patent protection
- UNLICENSED: All rights reserved
*/

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. BASIC CONTRACT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

// Pragma directive specifies compiler version
pragma solidity ^0.8.19;  // Compatible with 0.8.19 and above (but below 0.9.0)
// pragma solidity >=0.8.0 <0.9.0;  // Range specification
// pragma solidity 0.8.19;  // Exact version only

// Import statements
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./interfaces/IMyInterface.sol";

// Basic contract structure
contract BasicContract {
    // State variables
    address public owner;
    uint256 public value;
    bool public isActive;
    
    // Constructor - runs once when contract is deployed
    constructor(uint256 _initialValue) {
        owner = msg.sender;
        value = _initialValue;
        isActive = true;
    }
    
    // Function that modifies state
    function setValue(uint256 _newValue) public {
        require(msg.sender == owner, "Only owner can set value");
        value = _newValue;
    }
    
    // Function that reads state (view function)
    function getValue() public view returns (uint256) {
        return value;
    }
    
    // Pure function (doesn't read or modify state)
    function add(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. DATA TYPES AND VARIABLES
// ═══════════════════════════════════════════════════════════════════════════════

contract DataTypes {
    
    // Boolean
    bool public isTrue = true;
    bool public isFalse = false;
    
    // Integers
    uint8 public smallNumber = 255;        // 0 to 255 (1 byte)
    uint16 public mediumNumber = 65535;    // 0 to 65535 (2 bytes)
    uint256 public largeNumber;            // 0 to 2^256-1 (32 bytes) - most common
    uint public defaultUint = 100;         // Same as uint256
    
    int8 public signedSmall = -128;        // -128 to 127
    int256 public signedLarge = -1000;     // Same as int
    
    // Address types
    address public walletAddress;          // 20-byte Ethereum address
    address payable public recipient;      // Can receive Ether
    
    // Fixed-size byte arrays
    bytes1 public singleByte = 0xFF;       // 1 byte
    bytes32 public hash;                   // 32 bytes (common for hashes)
    
    // Dynamic arrays
    bytes public dynamicBytes;
    string public message = "Hello, Blockchain!";
    
    // Enums
    enum Status { Pending, Approved, Rejected, Completed }
    Status public currentStatus = Status.Pending;
    
    // Arrays
    uint256[] public dynamicArray;         // Dynamic array
    uint256[5] public fixedArray;          // Fixed-size array
    string[] public names;                 // Dynamic string array
    
    // Mappings (like hash tables)
    mapping(address => uint256) public balances;           // Address to balance
    mapping(uint256 => bool) public exists;                // ID to existence check
    mapping(address => mapping(address => uint256)) public allowances; // Nested mapping
    
    // Structs
    struct User {
        string name;
        uint256 age;
        bool isActive;
        address wallet;
    }
    
    User public admin;
    mapping(address => User) public users;
    User[] public allUsers;
    
    // Constants and immutable variables
    uint256 public constant MAX_SUPPLY = 1000000 * 10**18; // Set at compile time
    address public immutable factory;                       // Set in constructor only
    
    constructor(address _factory) {
        factory = _factory;
        admin = User("Administrator", 0, true, msg.sender);
    }
    
    // Variable visibility:
    // public: Automatic getter function, accessible everywhere
    // internal: Only within this contract and inherited contracts (default for state variables)
    // private: Only within this contract
    uint256 public publicVar;      // Anyone can read
    uint256 internal internalVar;  // This contract + children
    uint256 private privateVar;    // Only this contract
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           4. FUNCTIONS AND MODIFIERS
// ═══════════════════════════════════════════════════════════════════════════════

contract Functions {
    address public owner;
    mapping(address => uint256) public balances;
    
    // Events for logging
    event Transfer(address indexed from, address indexed to, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    constructor() {
        owner = msg.sender;
        balances[msg.sender] = 1000000;
    }
    
    // Modifiers - reusable code that can be applied to functions
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;  // Function body is inserted here
    }
    
    modifier validAddress(address _addr) {
        require(_addr != address(0), "Invalid address");
        require(_addr != address(this), "Cannot be contract address");
        _;
    }
    
    modifier sufficientBalance(address _from, uint256 _amount) {
        require(balances[_from] >= _amount, "Insufficient balance");
        _;
    }
    
    // Function visibility:
    // public: Callable from anywhere, creates automatic getter for state variables
    // external: Only callable from outside the contract (more gas efficient)
    // internal: Only callable from within this contract and inherited contracts
    // private: Only callable from within this contract
    
    // Function state mutability:
    // (default): Can read and modify state
    // view: Can read state but cannot modify it
    // pure: Cannot read or modify state
    // payable: Can receive Ether
    
    // Pure function - no state access
    function add(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;
    }
    
    // View function - reads state but doesn't modify
    function getBalance(address _account) public view returns (uint256) {
        return balances[_account];
    }
    
    // State-changing function with modifiers
    function transfer(address _to, uint256 _amount) 
        public 
        validAddress(_to) 
        sufficientBalance(msg.sender, _amount) 
        returns (bool) 
    {
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
        
        emit Transfer(msg.sender, _to, _amount);
        return true;
    }
    
    // Payable function - can receive Ether
    function deposit() public payable {
        require(msg.value > 0, "Must send some Ether");
        balances[msg.sender] += msg.value;
    }
    
    // Internal function (can only be called within this contract or inherited contracts)
    function _mint(address _to, uint256 _amount) internal validAddress(_to) {
        balances[_to] += _amount;
        emit Transfer(address(0), _to, _amount);
    }
    
    // External function (can only be called from outside)
    function mint(address _to, uint256 _amount) external onlyOwner {
        _mint(_to, _amount);
    }
    
    // Function overloading (same name, different parameters)
    function withdraw() external {
        withdraw(balances[msg.sender]);
    }
    
    function withdraw(uint256 _amount) public sufficientBalance(msg.sender, _amount) {
        balances[msg.sender] -= _amount;
        payable(msg.sender).transfer(_amount);
    }
    
    // Multiple return values
    function getAccountInfo(address _account) 
        public 
        view 
        returns (uint256 balance, bool isOwner, uint256 contractBalance) 
    {
        return (
            balances[_account],
            _account == owner,
            address(this).balance
        );
    }
    
    // Transfer ownership
    function transferOwnership(address _newOwner) public onlyOwner validAddress(_newOwner) {
        address previousOwner = owner;
        owner = _newOwner;
        emit OwnershipTransferred(previousOwner, _newOwner);
    }
    
    // Fallback function - called when function doesn't exist
    fallback() external payable {
        // Handle unknown function calls
        revert("Function does not exist");
    }
    
    // Receive function - called for plain Ether transfers
    receive() external payable {
        deposit();
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           5. CONTROL STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

contract ControlStructures {
    mapping(address => uint256) public scores;
    uint256[] public numbers;
    
    // If-else statements
    function checkScore(address _user) public view returns (string memory) {
        uint256 score = scores[_user];
        
        if (score >= 90) {
            return "Excellent";
        } else if (score >= 80) {
            return "Good";
        } else if (score >= 70) {
            return "Average";
        } else {
            return "Needs Improvement";
        }
    }
    
    // Ternary operator
    function getStatus(bool _condition) public pure returns (string memory) {
        return _condition ? "Active" : "Inactive";
    }
    
    // For loops
    function calculateSum() public view returns (uint256) {
        uint256 sum = 0;
        for (uint256 i = 0; i < numbers.length; i++) {
            sum += numbers[i];
        }
        return sum;
    }
    
    // While loops (use carefully - can consume lots of gas!)
    function findFirstEven() public view returns (uint256) {
        uint256 i = 0;
        while (i < numbers.length) {
            if (numbers[i] % 2 == 0) {
                return numbers[i];
            }
            i++;
        }
        return 0; // Not found
    }
    
    // Do-while loops
    function doWhileExample() public pure returns (uint256) {
        uint256 i = 0;
        uint256 sum = 0;
        do {
            sum += i;
            i++;
        } while (i < 10);
        return sum;
    }
    
    // Break and continue in loops
    function processNumbers() public {
        for (uint256 i = 0; i < 100; i++) {
            if (i % 2 == 0) {
                continue; // Skip even numbers
            }
            
            if (i > 50) {
                break; // Exit loop when i > 50
            }
            
            numbers.push(i);
        }
    }
    
    // Try-catch (for external calls)
    function safeDivision(uint256 a, uint256 b) public pure returns (uint256, bool) {
        try this.divide(a, b) returns (uint256 result) {
            return (result, true);
        } catch {
            return (0, false);
        }
    }
    
    function divide(uint256 a, uint256 b) public pure returns (uint256) {
        require(b != 0, "Division by zero");
        return a / b;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           6. ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

contract ErrorHandling {
    mapping(address => uint256) public balances;
    
    // Custom errors (more gas efficient than require strings)
    error InsufficientBalance(uint256 available, uint256 required);
    error InvalidAddress(address addr);
    error TransferFailed();
    error Unauthorized(address caller);
    
    // Require statements
    function transfer(address _to, uint256 _amount) public {
        require(_to != address(0), "Cannot transfer to zero address");
        require(_amount > 0, "Amount must be greater than zero");
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
    }
    
    // Assert statements (for internal errors that should never happen)
    function assertExample(uint256 _value) public pure returns (uint256) {
        assert(_value != 0); // Should never be false if code is correct
        return 100 / _value;
    }
    
    // Revert with custom errors
    function transferWithCustomError(address _to, uint256 _amount) public {
        if (_to == address(0)) {
            revert InvalidAddress(_to);
        }
        
        uint256 balance = balances[msg.sender];
        if (balance < _amount) {
            revert InsufficientBalance(balance, _amount);
        }
        
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
    }
    
    // Try-catch for external calls
    interface IExternalContract {
        function riskyFunction() external returns (bool);
    }
    
    function safeExternalCall(address _contract) public returns (string memory) {
        try IExternalContract(_contract).riskyFunction() returns (bool success) {
            return success ? "Success" : "Failed";
        } catch Error(string memory reason) {
            return string(abi.encodePacked("Error: ", reason));
        } catch (bytes memory lowLevelData) {
            return "Low-level error occurred";
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. EVENTS AND LOGGING
// ═══════════════════════════════════════════════════════════════════════════════

contract Events {
    // Event declarations (up to 3 parameters can be indexed)
    event UserRegistered(address indexed user, string name, uint256 timestamp);
    event BalanceUpdated(address indexed user, uint256 oldBalance, uint256 newBalance);
    event Transfer(address indexed from, address indexed to, uint256 amount, bytes data);
    
    // Complex event with multiple indexed parameters
    event ComplexEvent(
        address indexed user,
        uint256 indexed category,
        string indexed status, // String is hashed when indexed
        uint256 amount,
        bytes data
    );
    
    mapping(address => uint256) public balances;
    mapping(address => string) public userNames;
    
    function registerUser(string memory _name) public {
        userNames[msg.sender] = _name;
        emit UserRegistered(msg.sender, _name, block.timestamp);
    }
    
    function updateBalance(uint256 _newBalance) public {
        uint256 oldBalance = balances[msg.sender];
        balances[msg.sender] = _newBalance;
        emit BalanceUpdated(msg.sender, oldBalance, _newBalance);
    }
    
    function transferWithData(address _to, uint256 _amount, bytes memory _data) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        balances[msg.sender] -= _amount;
        balances[_to] += _amount;
        
        emit Transfer(msg.sender, _to, _amount, _data);
    }
    
    // Anonymous events (cheaper gas, no event signature)
    event AnonymousEvent(address user, uint256 value) anonymous;
    
    function triggerAnonymousEvent(uint256 _value) public {
        emit AnonymousEvent(msg.sender, _value);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           8. INHERITANCE AND INTERFACES
// ═══════════════════════════════════════════════════════════════════════════════

// Interface definition
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

// Abstract contract
abstract contract AccessControl {
    mapping(bytes32 => mapping(address => bool)) private _roles;
    
    event RoleGranted(bytes32 indexed role, address indexed account);
    event RoleRevoked(bytes32 indexed role, address indexed account);
    
    modifier onlyRole(bytes32 role) {
        require(hasRole(role, msg.sender), "AccessControl: access denied");
        _;
    }
    
    function hasRole(bytes32 role, address account) public view virtual returns (bool) {
        return _roles[role][account];
    }
    
    function grantRole(bytes32 role, address account) public virtual;
    function revokeRole(bytes32 role, address account) public virtual;
}

// Base contract
contract Ownable {
    address public owner;
    
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    
    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }
    
    modifier onlyOwner() {
        require(owner == msg.sender, "Ownable: caller is not the owner");
        _;
    }
    
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}

// Contract with multiple inheritance
contract MyToken is IERC20, Ownable, AccessControl {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");
    
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 private _totalSupply;
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    constructor(string memory _name, string memory _symbol) {
        name = _name;
        symbol = _symbol;
        decimals = 18;
        _totalSupply = 1000000 * 10**decimals;
        _balances[msg.sender] = _totalSupply;
        
        // Grant roles to deployer
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(BURNER_ROLE, msg.sender);
    }
    
    // Implement IERC20 interface
    function totalSupply() public view override returns (uint256) {
        return _totalSupply;
    }
    
    function balanceOf(address account) public view override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address to, uint256 amount) public override returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }
    
    function allowance(address owner, address spender) public view override returns (uint256) {
        return _allowances[owner][spender];
    }
    
    function approve(address spender, uint256 amount) public override returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) public override returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        require(currentAllowance >= amount, "ERC20: transfer amount exceeds allowance");
        
        _transfer(from, to, amount);
        _approve(from, msg.sender, currentAllowance - amount);
        
        return true;
    }
    
    // Override AccessControl functions
    function grantRole(bytes32 role, address account) public override onlyOwner {
        _grantRole(role, account);
    }
    
    function revokeRole(bytes32 role, address account) public override onlyOwner {
        _revokeRole(role, account);
    }
    
    // Additional functions
    function mint(address to, uint256 amount) public onlyRole(MINTER_ROLE) {
        _totalSupply += amount;
        _balances[to] += amount;
        emit Transfer(address(0), to, amount);
    }
    
    function burn(uint256 amount) public onlyRole(BURNER_ROLE) {
        require(_balances[msg.sender] >= amount, "ERC20: burn amount exceeds balance");
        _balances[msg.sender] -= amount;
        _totalSupply -= amount;
        emit Transfer(msg.sender, address(0), amount);
    }
    
    // Internal functions
    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        require(_balances[from] >= amount, "ERC20: transfer amount exceeds balance");
        
        _balances[from] -= amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }
    
    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
    
    function _grantRole(bytes32 role, address account) internal {
        _roles[role][account] = true;
        emit RoleGranted(role, account);
    }
    
    function _revokeRole(bytes32 role, address account) internal {
        _roles[role][account] = false;
        emit RoleRevoked(role, account);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           9. LIBRARIES AND USING FOR
// ═══════════════════════════════════════════════════════════════════════════════

// Library for safe math operations (not needed in Solidity 0.8+ due to built-in overflow protection)
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        return a - b;
    }
    
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
    
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        return a / b;
    }
}

// String utility library
library StringUtils {
    function concat(string memory a, string memory b) internal pure returns (string memory) {
        return string(abi.encodePacked(a, b));
    }
    
    function length(string memory str) internal pure returns (uint256) {
        return bytes(str).length;
    }
    
    function equal(string memory a, string memory b) internal pure returns (bool) {
        return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }
}

// Address utility library
library AddressUtils {
    function isContract(address account) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(account)
        }
        return size > 0;
    }
    
    function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");
        
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }
}

// Contract using libraries
contract LibraryExample {
    using SafeMath for uint256;    // Attach library functions to uint256
    using StringUtils for string;  // Attach library functions to string
    using AddressUtils for address; // Attach library functions to address
    
    string public name = "Example";
    
    function safeCalculation(uint256 a, uint256 b) public pure returns (uint256) {
        return a.add(b).mul(2); // Using library functions
    }
    
    function stringOperations() public view returns (string memory, uint256, bool) {
        string memory fullName = name.concat(" Token");
        uint256 nameLength = name.length();
        bool isEmpty = name.equal("");
        
        return (fullName, nameLength, isEmpty);
    }
    
    function checkIfContract(address addr) public view returns (bool) {
        return addr.isContract();
    }
    
    function sendEther(address payable recipient, uint256 amount) public {
        recipient.sendValue(amount);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           10. SECURITY PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

// Reentrancy Guard
contract ReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    
    uint256 private _status;
    
    constructor() {
        _status = _NOT_ENTERED;
    }
    
    modifier nonReentrant() {
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

// Secure contract following best practices
contract SecureContract is ReentrancyGuard {
    mapping(address => uint256) public balances;
    address public owner;
    bool public paused = false;
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event EmergencyStop();
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() external payable whenNotPaused {
        require(msg.value > 0, "Must send some Ether");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    // Secure withdrawal using checks-effects-interactions pattern
    function withdraw(uint256 amount) external nonReentrant whenNotPaused {
        // Checks
        require(amount > 0, "Amount must be greater than 0");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Effects
        balances[msg.sender] -= amount;
        
        // Interactions
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
        
        emit Withdrawal(msg.sender, amount);
    }
    
    // Emergency stop (circuit breaker pattern)
    function emergencyStop() external onlyOwner {
        paused = true;
        emit EmergencyStop();
    }
    
    function resume() external onlyOwner {
        paused = false;
    }
    
    // Pull payment pattern (alternative to direct transfers)
    mapping(address => uint256) public pendingWithdrawals;
    
    function setPendingWithdrawal(address user, uint256 amount) internal {
        pendingWithdrawals[user] += amount;
    }
    
    function withdrawPending() external nonReentrant {
        uint256 amount = pendingWithdrawals[msg.sender];
        require(amount > 0, "No pending withdrawals");
        
        pendingWithdrawals[msg.sender] = 0;
        
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
    }
}