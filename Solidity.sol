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


// ═══════════════════════════════════════════════════════════════════════════════
//                           11. NFT CONTRACTS (ERC-721)
// ═══════════════════════════════════════════════════════════════════════════════

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract MyNFT is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    uint256 public constant MAX_SUPPLY = 10000;
    uint256 public mintPrice = 0.05 ether;
    bool public mintingEnabled = true;
    uint256 public maxMintsPerAddress = 5;
    
    mapping(address => uint256) public mintedPerAddress;
    
    event NFTMinted(address indexed to, uint256 indexed tokenId, string tokenURI);
    event MintPriceUpdated(uint256 newPrice);
    event MintingToggled(bool enabled);
    
    constructor() ERC721("MyNFT", "MNFT") {}
    
    modifier mintingActive() {
        require(mintingEnabled, "Minting is not active");
        _;
    }
    
    function mint(address to, string memory tokenURI) public payable mintingActive {
        require(_tokenIds.current() < MAX_SUPPLY, "Max supply reached");
        require(msg.value >= mintPrice, "Insufficient payment");
        require(mintedPerAddress[to] < maxMintsPerAddress, "Max mints per address reached");
        
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _mint(to, newTokenId);
        _setTokenURI(newTokenId, tokenURI);
        
        mintedPerAddress[to]++;
        
        emit NFTMinted(to, newTokenId, tokenURI);
        
        // Refund excess payment
        if (msg.value > mintPrice) {
            payable(msg.sender).transfer(msg.value - mintPrice);
        }
    }
    
    function batchMint(address[] memory recipients, string[] memory tokenURIs) 
        external onlyOwner {
        require(recipients.length == tokenURIs.length, "Arrays length mismatch");
        require(_tokenIds.current() + recipients.length <= MAX_SUPPLY, "Would exceed max supply");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            _tokenIds.increment();
            uint256 newTokenId = _tokenIds.current();
            
            _mint(recipients[i], newTokenId);
            _setTokenURI(newTokenId, tokenURIs[i]);
            
            emit NFTMinted(recipients[i], newTokenId, tokenURIs[i]);
        }
    }
    
    function setMintPrice(uint256 _newPrice) external onlyOwner {
        mintPrice = _newPrice;
        emit MintPriceUpdated(_newPrice);
    }
    
    function toggleMinting() external onlyOwner {
        mintingEnabled = !mintingEnabled;
        emit MintingToggled(mintingEnabled);
    }
    
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        payable(owner()).transfer(balance);
    }
    
    function totalSupply() public view returns (uint256) {
        return _tokenIds.current();
    }
    
    function tokenExists(uint256 tokenId) public view returns (bool) {
        return _exists(tokenId);
    }
    
    // Override required functions
    function _burn(uint256 tokenId) internal override(ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    function tokenURI(uint256 tokenId) public view override(ERC721URIStorage) 
        returns (string memory) {
        return super.tokenURI(tokenId);
    }
}

// NFT Marketplace
contract NFTMarketplace {
    struct Listing {
        address seller;
        address nftContract;
        uint256 tokenId;
        uint256 price;
        bool active;
    }
    
    mapping(uint256 => Listing) public listings;
    mapping(address => mapping(uint256 => uint256)) public nftToListing;
    
    uint256 public listingCounter;
    uint256 public marketplaceFee = 250; // 2.5% (basis points)
    address public feeRecipient;
    
    event NFTListed(
        uint256 indexed listingId,
        address indexed seller,
        address indexed nftContract,
        uint256 tokenId,
        uint256 price
    );
    
    event NFTSold(
        uint256 indexed listingId,
        address indexed buyer,
        address indexed seller,
        uint256 price
    );
    
    event ListingCanceled(uint256 indexed listingId);
    event PriceUpdated(uint256 indexed listingId, uint256 newPrice);
    
    constructor(address _feeRecipient) {
        feeRecipient = _feeRecipient;
    }
    
    function listNFT(address nftContract, uint256 tokenId, uint256 price) external {
        require(price > 0, "Price must be greater than 0");
        require(IERC721(nftContract).ownerOf(tokenId) == msg.sender, "Not the owner");
        require(
            IERC721(nftContract).isApprovedForAll(msg.sender, address(this)) ||
            IERC721(nftContract).getApproved(tokenId) == address(this), 
            "Contract not approved"
        );
        
        listingCounter++;
        
        listings[listingCounter] = Listing({
            seller: msg.sender,
            nftContract: nftContract,
            tokenId: tokenId,
            price: price,
            active: true
        });
        
        nftToListing[nftContract][tokenId] = listingCounter;
        
        emit NFTListed(listingCounter, msg.sender, nftContract, tokenId, price);
    }
    
    function buyNFT(uint256 listingId) external payable {
        Listing storage listing = listings[listingId];
        require(listing.active, "Listing not active");
        require(msg.value >= listing.price, "Insufficient payment");
        require(msg.sender != listing.seller, "Cannot buy your own NFT");
        
        listing.active = false;
        nftToListing[listing.nftContract][listing.tokenId] = 0;
        
        // Calculate fees
        uint256 fee = (listing.price * marketplaceFee) / 10000;
        uint256 sellerAmount = listing.price - fee;
        
        // Transfer NFT
        IERC721(listing.nftContract).transferFrom(listing.seller, msg.sender, listing.tokenId);
        
        // Transfer payments
        payable(listing.seller).transfer(sellerAmount);
        if (fee > 0) {
            payable(feeRecipient).transfer(fee);
        }
        
        // Refund excess
        if (msg.value > listing.price) {
            payable(msg.sender).transfer(msg.value - listing.price);
        }
        
        emit NFTSold(listingId, msg.sender, listing.seller, listing.price);
    }
    
    function cancelListing(uint256 listingId) external {
        Listing storage listing = listings[listingId];
        require(listing.seller == msg.sender, "Not the seller");
        require(listing.active, "Listing not active");
        
        listing.active = false;
        nftToListing[listing.nftContract][listing.tokenId] = 0;
        
        emit ListingCanceled(listingId);
    }
    
    function updatePrice(uint256 listingId, uint256 newPrice) external {
        require(newPrice > 0, "Price must be greater than 0");
        Listing storage listing = listings[listingId];
        require(listing.seller == msg.sender, "Not the seller");
        require(listing.active, "Listing not active");
        
        listing.price = newPrice;
        emit PriceUpdated(listingId, newPrice);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           12. DEFI PROTOCOLS
// ═══════════════════════════════════════════════════════════════════════════════

// Simple DEX (Automated Market Maker)
contract SimpleDEX {
    mapping(address => mapping(address => uint256)) public tokenBalances;
    mapping(address => uint256) public ethBalances;
    
    event TokenDeposit(address indexed user, address indexed token, uint256 amount);
    event TokenWithdraw(address indexed user, address indexed token, uint256 amount);
    event TokenPurchase(address indexed user, address indexed token, uint256 amount, uint256 cost);
    event TokenSold(address indexed user, address indexed token, uint256 amount, uint256 revenue);
    event LiquidityAdded(address indexed provider, address indexed token, uint256 tokenAmount, uint256 ethAmount);
    
    function depositToken(address _token, uint256 _amount) public {
        require(IERC20(_token).transferFrom(msg.sender, address(this), _amount), "Transfer failed");
        tokenBalances[msg.sender][_token] += _amount;
        emit TokenDeposit(msg.sender, _token, _amount);
    }
    
    function withdrawToken(address _token, uint256 _amount) public {
        require(tokenBalances[msg.sender][_token] >= _amount, "Insufficient balance");
        tokenBalances[msg.sender][_token] -= _amount;
        require(IERC20(_token).transfer(msg.sender, _amount), "Transfer failed");
        emit TokenWithdraw(msg.sender, _token, _amount);
    }
    
    function depositEther() public payable {
        ethBalances[msg.sender] += msg.value;
    }
    
    function withdrawEther(uint256 _amount) public {
        require(ethBalances[msg.sender] >= _amount, "Insufficient ETH balance");
        ethBalances[msg.sender] -= _amount;
        payable(msg.sender).transfer(_amount);
    }
    
    // Simple AMM formula: price = eth_reserve / token_reserve
    function getTokenPrice(address _token) public view returns (uint256) {
        uint256 tokenReserve = IERC20(_token).balanceOf(address(this));
        uint256 ethReserve = address(this).balance;
        
        require(tokenReserve > 0 && ethReserve > 0, "No liquidity");
        return (ethReserve * 1e18) / tokenReserve; // Price in wei per token
    }
    
    function buyTokens(address _token, uint256 _tokenAmount) public payable {
        uint256 cost = (_tokenAmount * getTokenPrice(_token)) / 1e18;
        require(msg.value >= cost, "Insufficient ETH sent");
        require(IERC20(_token).balanceOf(address(this)) >= _tokenAmount, "Insufficient token liquidity");
        
        // Transfer tokens to buyer
        require(IERC20(_token).transfer(msg.sender, _tokenAmount), "Token transfer failed");
        
        // Refund excess ETH
        if (msg.value > cost) {
            payable(msg.sender).transfer(msg.value - cost);
        }
        
        emit TokenPurchase(msg.sender, _token, _tokenAmount, cost);
    }
    
    function sellTokens(address _token, uint256 _tokenAmount) public {
        require(IERC20(_token).transferFrom(msg.sender, address(this), _tokenAmount), "Token transfer failed");
        
        uint256 revenue = (_tokenAmount * getTokenPrice(_token)) / 1e18;
        require(address(this).balance >= revenue, "Insufficient ETH liquidity");
        
        payable(msg.sender).transfer(revenue);
        emit TokenSold(msg.sender, _token, _tokenAmount, revenue);
    }
    
    function addLiquidity(address _token, uint256 _tokenAmount) public payable {
        require(_tokenAmount > 0 && msg.value > 0, "Must provide both tokens and ETH");
        require(IERC20(_token).transferFrom(msg.sender, address(this), _tokenAmount), "Token transfer failed");
        
        emit LiquidityAdded(msg.sender, _token, _tokenAmount, msg.value);
    }
}

// Staking Rewards Contract
contract StakingRewards {
    IERC20 public stakingToken;
    IERC20 public rewardsToken;
    
    address public owner;
    uint256 public rewardRate = 100; // Rewards per second
    uint256 public lastUpdateTime;
    uint256 public rewardPerTokenStored;
    
    mapping(address => uint256) public userRewardPerTokenPaid;
    mapping(address => uint256) public rewards;
    mapping(address => uint256) public balances;
    
    uint256 public totalSupply;
    
    event Staked(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event RewardPaid(address indexed user, uint256 reward);
    event RewardRateUpdated(uint256 newRate);
    
    constructor(address _stakingToken, address _rewardsToken) {
        stakingToken = IERC20(_stakingToken);
        rewardsToken = IERC20(_rewardsToken);
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier updateReward(address account) {
        rewardPerTokenStored = rewardPerToken();
        lastUpdateTime = block.timestamp;
        
        if (account != address(0)) {
            rewards[account] = earned(account);
            userRewardPerTokenPaid[account] = rewardPerTokenStored;
        }
        _;
    }
    
    function rewardPerToken() public view returns (uint256) {
        if (totalSupply == 0) {
            return rewardPerTokenStored;
        }
        return rewardPerTokenStored + 
               (((block.timestamp - lastUpdateTime) * rewardRate * 1e18) / totalSupply);
    }
    
    function earned(address account) public view returns (uint256) {
        return ((balances[account] * (rewardPerToken() - userRewardPerTokenPaid[account])) / 1e18) + 
               rewards[account];
    }
    
    function stake(uint256 amount) external updateReward(msg.sender) {
        require(amount > 0, "Cannot stake 0");
        totalSupply += amount;
        balances[msg.sender] += amount;
        stakingToken.transferFrom(msg.sender, address(this), amount);
        emit Staked(msg.sender, amount);
    }
    
    function withdraw(uint256 amount) public updateReward(msg.sender) {
        require(amount > 0, "Cannot withdraw 0");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        totalSupply -= amount;
        balances[msg.sender] -= amount;
        stakingToken.transfer(msg.sender, amount);
        emit Withdrawn(msg.sender, amount);
    }
    
    function getReward() public updateReward(msg.sender) {
        uint256 reward = rewards[msg.sender];
        if (reward > 0) {
            rewards[msg.sender] = 0;
            rewardsToken.transfer(msg.sender, reward);
            emit RewardPaid(msg.sender, reward);
        }
    }
    
    function exit() external {
        withdraw(balances[msg.sender]);
        getReward();
    }
    
    function setRewardRate(uint256 _rewardRate) external onlyOwner {
        rewardRate = _rewardRate;
        emit RewardRateUpdated(_rewardRate);
    }
}

// Simple Lending Protocol
contract SimpleLending {
    mapping(address => uint256) public deposits;
    mapping(address => uint256) public borrowed;
    mapping(address => uint256) public lastUpdateTime;
    
    uint256 public constant INTEREST_RATE = 5; // 5% annual
    uint256 public constant COLLATERAL_RATIO = 150; // 150% collateralization required
    uint256 public constant SECONDS_PER_YEAR = 365 * 24 * 60 * 60;
    
    IERC20 public token;
    
    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event Borrowed(address indexed user, uint256 amount);
    event Repaid(address indexed user, uint256 amount);
    event Liquidated(address indexed user, uint256 amount);
    
    constructor(address _token) {
        token = IERC20(_token);
    }
    
    function deposit(uint256 amount) external {
        require(amount > 0, "Cannot deposit 0");
        updateInterest(msg.sender);
        
        token.transferFrom(msg.sender, address(this), amount);
        deposits[msg.sender] += amount;
        
        emit Deposited(msg.sender, amount);
    }
    
    function withdraw(uint256 amount) external {
        updateInterest(msg.sender);
        require(deposits[msg.sender] >= amount, "Insufficient deposits");
        
        // Check if withdrawal maintains collateral ratio
        uint256 newDeposits = deposits[msg.sender] - amount;
        uint256 currentBorrowed = borrowed[msg.sender];
        
        if (currentBorrowed > 0) {
            require(newDeposits * 100 >= currentBorrowed * COLLATERAL_RATIO, 
                   "Would break collateral ratio");
        }
        
        deposits[msg.sender] -= amount;
        token.transfer(msg.sender, amount);
        
        emit Withdrawn(msg.sender, amount);
    }
    
    function borrow(uint256 amount) external {
        require(amount > 0, "Cannot borrow 0");
        updateInterest(msg.sender);
        
        uint256 maxBorrow = (deposits[msg.sender] * 100) / COLLATERAL_RATIO;
        require(borrowed[msg.sender] + amount <= maxBorrow, "Insufficient collateral");
        
        borrowed[msg.sender] += amount;
        token.transfer(msg.sender, amount);
        
        emit Borrowed(msg.sender, amount);
    }
    
    function repay(uint256 amount) external {
        require(amount > 0, "Cannot repay 0");
        updateInterest(msg.sender);
        
        uint256 debt = borrowed[msg.sender];
        uint256 repayAmount = amount > debt ? debt : amount;
        
        token.transferFrom(msg.sender, address(this), repayAmount);
        borrowed[msg.sender] -= repayAmount;
        
        emit Repaid(msg.sender, repayAmount);
    }
    
    function updateInterest(address user) internal {
        if (borrowed[user] > 0 && lastUpdateTime[user] > 0) {
            uint256 timeElapsed = block.timestamp - lastUpdateTime[user];
            uint256 interest = (borrowed[user] * INTEREST_RATE * timeElapsed) / 
                              (100 * SECONDS_PER_YEAR);
            borrowed[user] += interest;
        }
        lastUpdateTime[user] = block.timestamp;
    }
    
    function getAccountInfo(address user) external view returns (
        uint256 userDeposits,
        uint256 userBorrowed,
        uint256 availableToBorrow,
        uint256 healthFactor
    ) {
        userDeposits = deposits[user];
        userBorrowed = borrowed[user];
        availableToBorrow = userBorrowed < (userDeposits * 100) / COLLATERAL_RATIO 
            ? (userDeposits * 100) / COLLATERAL_RATIO - userBorrowed 
            : 0;
        healthFactor = userBorrowed > 0 ? (userDeposits * 100) / userBorrowed : type(uint256).max;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           13. ADVANCED PATTERNS AND ASSEMBLY
// ═══════════════════════════════════════════════════════════════════════════════

// Factory Pattern
contract TokenFactory {
    address[] public deployedTokens;
    mapping(address => address[]) public userTokens;
    
    event TokenCreated(address indexed tokenAddress, address indexed creator, string name, string symbol);
    
    function createToken(
        string memory _name,
        string memory _symbol,
        uint256 _initialSupply
    ) public returns (address) {
        MyToken newToken = new MyToken(_name, _symbol);
        
        deployedTokens.push(address(newToken));
        userTokens[msg.sender].push(address(newToken));
        
        emit TokenCreated(address(newToken), msg.sender, _name, _symbol);
        
        return address(newToken);
    }
    
    function getDeployedTokens() public view returns (address[] memory) {
        return deployedTokens;
    }
    
    function getUserTokens(address user) public view returns (address[] memory) {
        return userTokens[user];
    }
}

// State Machine Pattern
contract StateMachine {
    enum State { Created, Locked, Inactive }
    State public state = State.Created;
    
    event StateChanged(State indexed newState);
    
    modifier onlyState(State _state) {
        require(state == _state, "Invalid state");
        _;
    }
    
    modifier transitionTo(State _state) {
        _;
        state = _state;
        emit StateChanged(_state);
    }
    
    function lock() public onlyState(State.Created) transitionTo(State.Locked) {
        // Lock logic here
    }
    
    function deactivate() public onlyState(State.Locked) transitionTo(State.Inactive) {
        // Deactivation logic here
    }
    
    function reactivate() public onlyState(State.Inactive) transitionTo(State.Created) {
        // Reactivation logic here
    }
}

// Assembly Examples
contract AssemblyExamples {
    
    // Efficient addition using assembly
    function efficientAdd(uint256 a, uint256 b) public pure returns (uint256 result) {
        assembly {
            result := add(a, b)
        }
    }
    
    // Get contract code size
    function getCodeSize(address target) public view returns (uint256 size) {
        assembly {
            size := extcodesize(target)
        }
    }
    
    // Memory operations
    function memoryExample() public pure returns (bytes32 result) {
        assembly {
            // Store value at memory position 0x80
            mstore(0x80, 0x123456789abcdef0)
            // Load value from memory position 0x80
            result := mload(0x80)
        }
    }
    
    // Get message sender without using msg.sender
    function getMsgSender() public view returns (address sender) {
        assembly {
            sender := caller()
        }
    }
    
    // Get current block timestamp
    function getBlockTimestamp() public view returns (uint256 timestamp) {
        assembly {
            timestamp := timestamp()
        }
    }
    
    // Efficient string comparison
    function compareStrings(string memory a, string memory b) public pure returns (bool) {
        return keccak256(abi.encodePacked(a)) == keccak256(abi.encodePacked(b));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           14. GAS OPTIMIZATION TECHNIQUES
// ═══════════════════════════════════════════════════════════════════════════════

contract GasOptimization {
    
    // Pack structs efficiently (group by size)
    struct OptimizedStruct {
        uint128 value1;  // 16 bytes
        uint128 value2;  // 16 bytes - packed together in one storage slot
        uint256 value3;  // 32 bytes - separate storage slot
        bool flag;       // 1 byte
        uint8 smallNum;  // 1 byte - can be packed with bool
    }
    
    // Use fixed-size arrays when possible
    uint256[100] public fixedArray; // More gas efficient than dynamic arrays for known sizes
    
    // Use mappings instead of arrays for sparse data
    mapping(uint256 => address) public owners; // Better than address[] for sparse data
    
    // Custom errors instead of require strings (Solidity 0.8.4+)
    error InsufficientBalance(uint256 available, uint256 required);
    error InvalidAddress();
    
    function optimizedTransfer(address to, uint256 amount) external {
        if (to == address(0)) revert InvalidAddress();
        
        uint256 balance = address(this).balance;
        if (balance < amount) {
            revert InsufficientBalance(balance, amount);
        }
        
        // Transfer logic
    }
    
    // Batch operations to reduce transaction costs
    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) 
        external {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            // Transfer logic here
        }
    }
    
    // Use unchecked blocks for safe operations (Solidity 0.8+)
    function efficientLoop(uint256 iterations) external pure returns (uint256) {
        uint256 sum = 0;
        for (uint256 i = 0; i < iterations;) {
            sum += i;
            unchecked { i++; } // Save gas on overflow checks when safe
        }
        return sum;
    }
    
    // Use events for data storage when appropriate (much cheaper than storage)
    event DataStored(address indexed user, string data, uint256 timestamp);
    
    function storeDataInEvent(string calldata data) external {
        emit DataStored(msg.sender, data, block.timestamp);
        // Much cheaper than storing in contract storage
    }
    
    // Prefer immutable and constant variables
    uint256 public constant FIXED_VALUE = 100;  // No storage slot used
    address public immutable FACTORY;           // Set once in constructor
    
    constructor(address _factory) {
        FACTORY = _factory;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           15. COMMON VULNERABILITIES TO AVOID
// ═══════════════════════════════════════════════════════════════════════════════

contract VulnerabilityExamples {
    
    mapping(address => uint256) public balances;
    address public owner;
    
    // ❌ VULNERABLE - Reentrancy attack
    function vulnerableWithdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount);
        
        // Interaction before state change - VULNERABLE!
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        
        balances[msg.sender] -= amount; // Too late - can be re-entered!
    }
    
    // ✅ SECURE - Follow checks-effects-interactions pattern
    function secureWithdraw(uint256 amount) public {
        // Checks
        require(balances[msg.sender] >= amount);
        
        // Effects
        balances[msg.sender] -= amount;
        
        // Interactions
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
    }
    
    // ❌ VULNERABLE - tx.origin usage
    function vulnerableAuth() public view {
        require(tx.origin == owner); // Vulnerable to phishing attacks
    }
    
    // ✅ SECURE - msg.sender usage
    function secureAuth() public view {
        require(msg.sender == owner); // Correct authentication
    }
    
    // ❌ VULNERABLE - Unchecked external calls
    function vulnerableCall(address target) public {
        target.call(""); // Return value not checked
    }
    
    // ✅ SECURE - Checked external calls
    function secureCall(address target) public {
        (bool success, ) = target.call("");
        require(success, "Call failed");
    }
    
    // ❌ VULNERABLE - Timestamp dependence for critical logic
    function vulnerableRandom() public view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp))); // Miners can manipulate
    }
    
    // ✅ BETTER - Use commit-reveal or oracle for randomness
    function betterRandom() public view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(
            block.difficulty,
            block.timestamp,
            msg.sender,
            blockhash(block.number - 1)
        ))); // Still not perfect, but better
    }
    
    // ❌ VULNERABLE - Integer overflow (pre-0.8.0)
    // function vulnerableAdd(uint256 a, uint256 b) public pure returns (uint256) {
    //     return a + b; // Could overflow in older Solidity versions
    // }
    
    // ✅ SECURE - Solidity 0.8+ has built-in overflow protection
    function secureAdd(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b; // Automatically reverts on overflow in 0.8+
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           16. TESTING AND DEPLOYMENT PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

contract TestingHelpers {
    
    // Constructor for deployment setup
    address public deployer;
    uint256 public deploymentTime;
    string public version = "1.0.0";
    
    constructor() {
        deployer = msg.sender;
        deploymentTime = block.timestamp;
    }
    
    // Deployment verification
    function verifyDeployment() external view returns (bool) {
        return address(this) != address(0) && 
               address(this).code.length > 0 &&
               deployer != address(0);
    }
    
    // Get contract info for testing
    function getContractInfo() external view returns (
        address contractAddress,
        address contractDeployer,
        uint256 blockNumber,
        uint256 timestamp,
        uint256 balance,
        string memory contractVersion
    ) {
        return (
            address(this),
            deployer,
            block.number,
            block.timestamp,
            address(this).balance,
            version
        );
    }
    
    // Test helper functions
    function getBlockInfo() external view returns (
        uint256 blockNumber,
        uint256 timestamp,
        uint256 difficulty,
        address coinbase
    ) {
        return (
            block.number,
            block.timestamp,
            block.difficulty,
            block.coin

// ═══════════════════════════════════════════════════════════════════════════════
//                           17. MULTI-SIGNATURE WALLET
// ═══════════════════════════════════════════════════════════════════════════════

contract MultiSigWallet {
    event Deposit(address indexed sender, uint256 amount, uint256 balance);
    event SubmitTransaction(
        address indexed owner,
        uint256 indexed txIndex,
        address indexed to,
        uint256 value,
        bytes data
    );
    event ConfirmTransaction(address indexed owner, uint256 indexed txIndex);
    event RevokeConfirmation(address indexed owner, uint256 indexed txIndex);
    event ExecuteTransaction(address indexed owner, uint256 indexed txIndex);
    
    address[] public owners;
    mapping(address => bool) public isOwner;
    uint256 public numConfirmationsRequired;
    
    struct Transaction {
        address to;
        uint256 value;
        bytes data;
        bool executed;
        uint256 numConfirmations;
    }
    
    Transaction[] public transactions;
    
    // mapping from tx index => owner => bool
    mapping(uint256 => mapping(address => bool)) public isConfirmed;
    
    modifier onlyOwner() {
        require(isOwner[msg.sender], "Not owner");
        _;
    }
    
    modifier txExists(uint256 _txIndex) {
        require(_txIndex < transactions.length, "Transaction does not exist");
        _;
    }
    
    modifier notExecuted(uint256 _txIndex) {
        require(!transactions[_txIndex].executed, "Transaction already executed");
        _;
    }
    
    modifier notConfirmed(uint256 _txIndex) {
        require(!isConfirmed[_txIndex][msg.sender], "Transaction already confirmed");
        _;
    }
    
    constructor(address[] memory _owners, uint256 _numConfirmationsRequired) {
        require(_owners.length > 0, "Owners required");
        require(
            _numConfirmationsRequired > 0 && 
            _numConfirmationsRequired <= _owners.length,
            "Invalid number of required confirmations"
        );
        
        for (uint256 i = 0; i < _owners.length; i++) {
            address owner = _owners[i];
            require(owner != address(0), "Invalid owner");
            require(!isOwner[owner], "Owner not unique");
            
            isOwner[owner] = true;
            owners.push(owner);
        }
        
        numConfirmationsRequired = _numConfirmationsRequired;
    }
    
    receive() external payable {
        emit Deposit(msg.sender, msg.value, address(this).balance);
    }
    
    function submitTransaction(address _to, uint256 _value, bytes memory _data)
        public
        onlyOwner
    {
        uint256 txIndex = transactions.length;
        
        transactions.push(Transaction({
            to: _to,
            value: _value,
            data: _data,
            executed: false,
            numConfirmations: 0
        }));
        
        emit SubmitTransaction(msg.sender, txIndex, _to, _value, _data);
    }
    
    function confirmTransaction(uint256 _txIndex)
        public
        onlyOwner
        txExists(_txIndex)
        notExecuted(_txIndex)
        notConfirmed(_txIndex)
    {
        Transaction storage transaction = transactions[_txIndex];
        transaction.numConfirmations += 1;
        isConfirmed[_txIndex][msg.sender] = true;
        
        emit ConfirmTransaction(msg.sender, _txIndex);
    }
    
    function executeTransaction(uint256 _txIndex)
        public
        onlyOwner
        txExists(_txIndex)
        notExecuted(_txIndex)
    {
        Transaction storage transaction = transactions[_txIndex];
        
        require(
            transaction.numConfirmations >= numConfirmationsRequired,
            "Cannot execute transaction"
        );
        
        transaction.executed = true;
        
        (bool success, ) = transaction.to.call{value: transaction.value}(transaction.data);
        require(success, "Transaction failed");
        
        emit ExecuteTransaction(msg.sender, _txIndex);
    }
    
    function revokeConfirmation(uint256 _txIndex)
        public
        onlyOwner
        txExists(_txIndex)
        notExecuted(_txIndex)
    {
        Transaction storage transaction = transactions[_txIndex];
        
        require(isConfirmed[_txIndex][msg.sender], "Transaction not confirmed");
        
        transaction.numConfirmations -= 1;
        isConfirmed[_txIndex][msg.sender] = false;
        
        emit RevokeConfirmation(msg.sender, _txIndex);
    }
    
    function getOwners() public view returns (address[] memory) {
        return owners;
    }
    
    function getTransactionCount() public view returns (uint256) {
        return transactions.length;
    }
    
    function getTransaction(uint256 _txIndex)
        public
        view
        returns (
            address to,
            uint256 value,
            bytes memory data,
            bool executed,
            uint256 numConfirmations
        )
    {
        Transaction storage transaction = transactions[_txIndex];
        
        return (
            transaction.to,
            transaction.value,
            transaction.data,
            transaction.executed,
            transaction.numConfirmations
        );
    }
    
    function isTransactionConfirmed(uint256 _txIndex, address _owner) 
        public 
        view 
        returns (bool) 
    {
        return isConfirmed[_txIndex][_owner];
    }
    
    function getConfirmationCount(uint256 _txIndex) public view returns (uint256) {
        return transactions[_txIndex].numConfirmations;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           18. UPGRADEABLE CONTRACTS
// ═══════════════════════════════════════════════════════════════════════════════

// Simple Proxy Contract
contract SimpleProxy {
    address public implementation;
    address public admin;
    
    event Upgraded(address indexed newImplementation);
    event AdminChanged(address indexed newAdmin);
    
    constructor(address _implementation) {
        implementation = _implementation;
        admin = msg.sender;
    }
    
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }
    
    function upgrade(address newImplementation) external onlyAdmin {
        require(newImplementation != address(0), "Invalid implementation");
        require(newImplementation.code.length > 0, "Implementation is not a contract");
        implementation = newImplementation;
        emit Upgraded(newImplementation);
    }
    
    function changeAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "Invalid admin");
        admin = newAdmin;
        emit AdminChanged(newAdmin);
    }
    
    function getImplementation() external view returns (address) {
        return implementation;
    }
    
    fallback() external payable {
        address impl = implementation;
        assembly {
            // Copy msg.data to memory
            calldatacopy(0, 0, calldatasize())
            
            // Delegate call to implementation
            let result := delegatecall(gas(), impl, 0, calldatasize(), 0, 0)
            
            // Copy return data to memory
            returndatacopy(0, 0, returndatasize())
            
            // Return or revert based on call result
            switch result
            case 0 { revert(0, returndatasize()) }
            default { return(0, returndatasize()) }
        }
    }
    
    receive() external payable {}
}

// Implementation Contract V1
contract ImplementationV1 {
    address public owner;
    uint256 public value;
    bool public initialized;
    
    event ValueUpdated(uint256 newValue);
    event Initialized(address owner);
    
    function initialize(address _owner) external {
        require(!initialized, "Already initialized");
        owner = _owner;
        initialized = true;
        emit Initialized(_owner);
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyInitialized() {
        require(initialized, "Not initialized");
        _;
    }
    
    function setValue(uint256 _value) external onlyOwner onlyInitialized {
        value = _value;
        emit ValueUpdated(_value);
    }
    
    function getValue() external view returns (uint256) {
        return value;
    }
    
    function getVersion() external pure virtual returns (string memory) {
        return "1.0.0";
    }
}

// Upgraded Implementation V2
contract ImplementationV2 is ImplementationV1 {
    uint256 public multiplier = 1; // New state variable
    
    event MultiplierUpdated(uint256 newMultiplier);
    
    function setMultiplier(uint256 _multiplier) external onlyOwner onlyInitialized {
        require(_multiplier > 0, "Multiplier must be greater than 0");
        multiplier = _multiplier;
        emit MultiplierUpdated(_multiplier);
    }
    
    function getMultipliedValue() external view returns (uint256) {
        return value * multiplier;
    }
    
    function getVersion() external pure override returns (string memory) {
        return "2.0.0";
    }
    
    // New function in V2
    function reset() external onlyOwner onlyInitialized {
        value = 0;
        multiplier = 1;
        emit ValueUpdated(0);
        emit MultiplierUpdated(1);
    }
}

// Storage Layout Helper for Upgrades
contract StorageLayout {
    // NEVER change the order of these variables in upgrades
    // NEVER change the type of these variables
    // NEVER remove these variables
    // You can only ADD new variables at the end
    
    address public owner;        // Slot 0
    uint256 public value;        // Slot 1
    bool public initialized;     // Slot 2
    uint256 public multiplier;   // Slot 3 (added in V2)
    // New variables go here in future versions
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           19. ORACLE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

// Chainlink Price Feed Interface
interface AggregatorV3Interface {
    function decimals() external view returns (uint8);
    function description() external view returns (string memory);
    function version() external view returns (uint256);
    function getRoundData(uint80 _roundId) external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
    function latestRoundData() external view returns (
        uint80 roundId,
        int256 answer,
        uint256 startedAt,
        uint256 updatedAt,
        uint80 answeredInRound
    );
}

contract PriceOracle {
    AggregatorV3Interface internal priceFeed;
    uint256 public constant PRICE_DECIMALS = 8;
    uint256 public constant STALE_PRICE_DELAY = 3600; // 1 hour
    address public owner;
    
    event PriceUpdated(int256 price, uint256 timestamp);
    event PriceFeedUpdated(address newPriceFeed);
    
    constructor(address _priceFeed) {
        owner = msg.sender;
        priceFeed = AggregatorV3Interface(_priceFeed);
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    function updatePriceFeed(address _newPriceFeed) external onlyOwner {
        require(_newPriceFeed != address(0), "Invalid price feed");
        priceFeed = AggregatorV3Interface(_newPriceFeed);
        emit PriceFeedUpdated(_newPriceFeed);
    }
    
    function getLatestPrice() public view returns (int256, uint256) {
        (
            uint80 roundID, 
            int256 price,
            uint256 startedAt,
            uint256 timeStamp,
            uint80 answeredInRound
        ) = priceFeed.latestRoundData();
        
        require(timeStamp > 0, "Round not complete");
        require(block.timestamp - timeStamp <= STALE_PRICE_DELAY, "Price data is stale");
        require(price > 0, "Invalid price");
        
        return (price, timeStamp);
    }
    
    function getPrice() public view returns (uint256) {
        (int256 price, ) = getLatestPrice();
        return uint256(price);
    }
    
    function convertEthToUsd(uint256 ethAmount) public view returns (uint256) {
        uint256 ethPrice = getPrice(); // Price has 8 decimals
        return (ethAmount * ethPrice) / (10**PRICE_DECIMALS);
    }
    
    function convertUsdToEth(uint256 usdAmount) public view returns (uint256) {
        uint256 ethPrice = getPrice(); // Price has 8 decimals
        return (usdAmount * (10**PRICE_DECIMALS)) / ethPrice;
    }
    
    function getPriceWithValidation() external returns (uint256) {
        (int256 price, uint256 timestamp) = getLatestPrice();
        emit PriceUpdated(price, timestamp);
        return uint256(price);
    }
}

// Custom Oracle for Multiple Assets
contract CustomOracle {
    address public owner;
    mapping(string => uint256) public prices;
    mapping(string => uint256) public lastUpdateTime;
    mapping(address => bool) public authorized;
    mapping(string => uint8) public decimals;
    
    uint256 public constant PRICE_VALIDITY_PERIOD = 1 hours;
    
    event PriceUpdated(string asset, uint256 price, uint256 timestamp);
    event AuthorizedUpdater(address updater);
    event RevokedUpdater(address updater);
    event AssetAdded(string asset, uint8 decimals);
    
    constructor() {
        owner = msg.sender;
        authorized[msg.sender] = true;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorized[msg.sender], "Not authorized");
        _;
    }
    
    function addAsset(string memory asset, uint8 _decimals) external onlyOwner {
        require(_decimals > 0 && _decimals <= 18, "Invalid decimals");
        decimals[asset] = _decimals;
        emit AssetAdded(asset, _decimals);
    }
    
    function authorizeUpdater(address updater) external onlyOwner {
        require(updater != address(0), "Invalid updater");
        authorized[updater] = true;
        emit AuthorizedUpdater(updater);
    }
    
    function revokeUpdater(address updater) external onlyOwner {
        authorized[updater] = false;
        emit RevokedUpdater(updater);
    }
    
    function updatePrice(string memory asset, uint256 price) external onlyAuthorized {
        require(price > 0, "Invalid price");
        require(decimals[asset] > 0, "Asset not supported");
        
        prices[asset] = price;
        lastUpdateTime[asset] = block.timestamp;
        emit PriceUpdated(asset, price, block.timestamp);
    }
    
    function batchUpdatePrices(
        string[] memory assets, 
        uint256[] memory _prices
    ) external onlyAuthorized {
        require(assets.length == _prices.length, "Arrays length mismatch");
        
        for (uint256 i = 0; i < assets.length; i++) {
            updatePrice(assets[i], _prices[i]);
        }
    }
    
    function getPrice(string memory asset) external view returns (uint256) {
        require(lastUpdateTime[asset] > 0, "Price not set");
        require(
            block.timestamp - lastUpdateTime[asset] <= PRICE_VALIDITY_PERIOD,
            "Price data stale"
        );
        return prices[asset];
    }
    
    function isPriceValid(string memory asset) external view returns (bool) {
        return lastUpdateTime[asset] > 0 && 
               block.timestamp - lastUpdateTime[asset] <= PRICE_VALIDITY_PERIOD;
    }
    
    function getAssetInfo(string memory asset) external view returns (
        uint256 price,
        uint256 timestamp,
        uint8 assetDecimals,
        bool isValid
    ) {
        return (
            prices[asset],
            lastUpdateTime[asset],
            decimals[asset],
            isPriceValid(asset)
        );
    }
}

// Oracle Consumer Contract
contract OracleConsumer {
    PriceOracle public priceOracle;
    uint256 public priceThreshold = 2000 * 10**8; // $2000 with 8 decimals
    
    event PriceAlert(uint256 price, uint256 threshold, string alertType);
    
    constructor(address _priceOracle) {
        priceOracle = PriceOracle(_priceOracle);
    }
    
    function checkPriceThreshold() external returns (bool) {
        uint256 currentPrice = priceOracle.getPrice();
        
        if (currentPrice > priceThreshold) {
            emit PriceAlert(currentPrice, priceThreshold, "ABOVE");
            return true;
        } else if (currentPrice < priceThreshold) {
            emit PriceAlert(currentPrice, priceThreshold, "BELOW");
            return false;
        }
        
        return true;
    }
    
    function updateThreshold(uint256 _newThreshold) external {
        priceThreshold = _newThreshold;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           20. BEST PRACTICES SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @title Solidity Best Practices Contract
 * @notice This contract demonstrates comprehensive security and gas optimization practices
 * @dev Always follow these patterns in production contracts
 */
contract BestPractices is ReentrancyGuard {
    
    // 1. Use specific compiler version for production
    // pragma solidity 0.8.19; // Instead of ^0.8.19 for production
    
    // 2. Use custom errors for gas efficiency (Solidity 0.8.4+)
    error InsufficientBalance(uint256 available, uint256 required);
    error Unauthorized(address caller);
    error InvalidInput(string reason);
    error ContractPaused();
    error InvalidAddress(address addr);
    error TransferFailed();
    
    // 3. State variables with proper visibility and optimization
    address public immutable owner; // Set once in constructor, gas efficient
    uint256 public constant MAX_SUPPLY = 1000000; // Compile-time constant
    uint256 private constant PRECISION = 1e18;
    
    mapping(address => uint256) public balances;
    mapping(address => bool) public authorized;
    
    bool public paused = false;
    uint256 public totalDeposits;
    
    // 4. Events for important state changes
    event BalanceUpdated(address indexed user, uint256 oldBalance, uint256 newBalance);
    event EmergencyStop(address indexed caller);
    event ContractResumed(address indexed caller);
    event AuthorizationChanged(address indexed user, bool authorized);
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    
    constructor() {
        owner = msg.sender;
        authorized[msg.sender] = true;
    }
    
    // 5. Modifiers for access control and state validation
    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized(msg.sender);
        _;
    }
    
    modifier onlyAuthorized() {
        if (!authorized[msg.sender]) revert Unauthorized(msg.sender);
        _;
    }
    
    modifier whenNotPaused() {
        if (paused) revert ContractPaused();
        _;
    }
    
    modifier validAddress(address addr) {
        if (addr == address(0)) revert InvalidAddress(addr);
        _;
    }
    
    modifier validAmount(uint256 amount) {
        if (amount == 0) revert InvalidInput("Amount must be greater than zero");
        _;
    }
    
    // 6. Follow checks-effects-interactions pattern
    function withdraw(uint256 amount) 
        external 
        whenNotPaused 
        validAmount(amount)
        nonReentrant 
    {
        // Checks
        if (balances[msg.sender] < amount) {
            revert InsufficientBalance(balances[msg.sender], amount);
        }
        
        // Effects
        uint256 oldBalance = balances[msg.sender];
        balances[msg.sender] -= amount;
        totalDeposits -= amount;
        
        emit BalanceUpdated(msg.sender, oldBalance, balances[msg.sender]);
        emit Withdrawal(msg.sender, amount);
        
        // Interactions
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        if (!success) revert TransferFailed();
    }
    
    function deposit() external payable whenNotPaused validAmount(msg.value) {
        uint256 oldBalance = balances[msg.sender];
        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;
        
        emit BalanceUpdated(msg.sender, oldBalance, balances[msg.sender]);
        emit Deposit(msg.sender, msg.value);
    }
    
    // 7. Input validation and sanitization
    function transfer(address to, uint256 amount) 
        external 
        whenNotPaused 
        validAddress(to) 
        validAmount(amount)
    {
        if (balances[msg.sender] < amount) {
            revert InsufficientBalance(balances[msg.sender], amount);
        }
        
        uint256 senderOldBalance = balances[msg.sender];
        uint256 receiverOldBalance = balances[to];
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
        
        emit BalanceUpdated(msg.sender, senderOldBalance, balances[msg.sender]);
        emit BalanceUpdated(to, receiverOldBalance, balances[to]);
    }
    
    // 8. Circuit breaker pattern
    function emergencyStop() external onlyOwner {
        paused = true;
        emit EmergencyStop(msg.sender);
    }
    
    function resume() external onlyOwner {
        paused = false;
        emit ContractResumed(msg.sender);
    }
    
    // 9. Access control management
    function setAuthorization(address user, bool isAuthorized) 
        external 
        onlyOwner 
        validAddress(user) 
    {
        authorized[user] = isAuthorized;
        emit AuthorizationChanged(user, isAuthorized);
    }
    
    // 10. Batch operations for gas efficiency
    function batchTransfer(
        address[] calldata recipients, 
        uint256[] calldata amounts
    ) external whenNotPaused {
        if (recipients.length != amounts.length) {
            revert InvalidInput("Arrays length mismatch");
        }
        if (recipients.length > 100) {
            revert InvalidInput("Too many recipients");
        }
        
        for (uint256 i = 0; i < recipients.length; i++) {
            transfer(recipients[i], amounts[i]);
        }
    }
    
    // 11. Safe external calls with proper error handling
    function safeExternalCall(address target, bytes calldata data) 
        external 
        onlyAuthorized 
        validAddress(target)
        returns (bool success, bytes memory returnData) 
    {
        if (target.code.length == 0) {
            revert InvalidInput("Target is not a contract");
        }
        
        (success, returnData) = target.call(data);
        // Note: We return success status instead of reverting
        // This allows the caller to handle failures appropriately
    }
    
    // 12. View functions for external integrations
    function getAccountSummary(address user) 
        external 
        view 
        validAddress(user)
        returns (
            uint256 balance,
            bool isUserAuthorized,
            uint256 contractBalance,
            bool isContractPaused
        ) 
    {
        return (
            balances[user],
            authorized[user],
            address(this).balance,
            paused
        );
    }
    
    function getContractStats() external view returns (
        uint256 totalUsers,
        uint256 totalBalance,
        uint256 contractBalance,
        address contractOwner
    ) {
        return (
            0, // Would need to track this separately for gas efficiency
            totalDeposits,
            address(this).balance,
            owner
        );
    }
    
    // 13. Emergency withdrawal for owner (last resort)
    function emergencyWithdrawAll() external onlyOwner {
        require(paused, "Can only emergency withdraw when paused");
        
        uint256 contractBalance = address(this).balance;
        (bool success, ) = payable(owner).call{value: contractBalance}("");
        if (!success) revert TransferFailed();
    }
    
    // 14. Upgrade preparation function
    function prepareUpgrade() external onlyOwner returns (
        uint256 contractBalance,
        uint256 totalUsers,
        uint256 totalDeposited
    ) {
        // Return critical state information for upgrade
        return (
            address(this).balance,
            0, // Would implement user counting if needed
            totalDeposits
        );
    }
}

/*
═══════════════════════════════════════════════════════════════════════════════
                               TESTING EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

// Hardhat Test Example (JavaScript)
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("BestPractices Contract", function () {
    let BestPractices, bestPractices, owner, addr1, addr2;

    beforeEach(async function () {
        BestPractices = await ethers.getContractFactory("BestPractices");
        [owner, addr1, addr2] = await ethers.getSigners();
        bestPractices = await BestPractices.deploy();
        await bestPractices.deployed();
    });

    describe("Deployment", function () {
        it("Should set the right owner", async function () {
            expect(await bestPractices.owner()).to.equal(owner.address);
        });

        it("Should start unpaused", async function () {
            expect(await bestPractices.paused()).to.equal(false);
        });
    });

    describe("Deposits", function () {
        it("Should accept deposits", async function () {
            const depositAmount = ethers.utils.parseEther("1.0");
            
            await expect(bestPractices.connect(addr1).deposit({ value: depositAmount }))
                .to.emit(bestPractices, "Deposit")
                .withArgs(addr1.address, depositAmount);
            
            expect(await bestPractices.balances(addr1.address)).to.equal(depositAmount);
        });

        it("Should reject zero deposits", async function () {
            await expect(
                bestPractices.connect(addr1).deposit({ value: 0 })
            ).to.be.revertedWithCustomError(bestPractices, "InvalidInput");
        });
    });

    describe("Withdrawals", function () {
        beforeEach(async function () {
            const depositAmount = ethers.utils.parseEther("1.0");
            await bestPractices.connect(addr1).deposit({ value: depositAmount });
        });

        it("Should allow withdrawals", async function () {
            const withdrawAmount = ethers.utils.parseEther("0.5");
            
            await expect(bestPractices.connect(addr1).withdraw(withdrawAmount))
                .to.emit(bestPractices, "Withdrawal")
                .withArgs(addr1.address, withdrawAmount);
        });

        it("Should reject withdrawals exceeding balance", async function () {
            const withdrawAmount = ethers.utils.parseEther("2.0");
            
            await expect(
                bestPractices.connect(addr1).withdraw(withdrawAmount)
            ).to.be.revertedWithCustomError(bestPractices, "InsufficientBalance");
        });
    });

    describe("Emergency Functions", function () {
        it("Should allow owner to pause contract", async function () {
            await expect(bestPractices.emergencyStop())
                .to.emit(bestPractices, "EmergencyStop")
                .withArgs(owner.address);
            
            expect(await bestPractices.paused()).to.equal(true);
        });

        it("Should prevent non-owner from pausing", async function () {
            await expect(
                bestPractices.connect(addr1).emergencyStop()
            ).to.be.revertedWithCustomError(bestPractices, "Unauthorized");
        });

        it("Should reject operations when paused", async function () {
            await bestPractices.emergencyStop();
            
            await expect(
                bestPractices.connect(addr1).deposit({ value: ethers.utils.parseEther("1.0") })
            ).to.be.revertedWithCustomError(bestPractices, "ContractPaused");
        });
    });
});

// Foundry Test Example (Solidity)
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/BestPractices.sol";

contract BestPracticesTest is Test {
    BestPractices public bestPractices;
    address public owner;
    address public user1;
    address public user2;

    function setUp() public {
        owner = address(this);
        user1 = address(0x1);
        user2 = address(0x2);
        bestPractices = new BestPractices();
        
        // Give users some ETH
        vm.deal(user1, 10 ether);
        vm.deal(user2, 10 ether);
    }

    function testDeployment() public {
        assertEq(bestPractices.owner(), owner);
        assertEq(bestPractices.paused(), false);
    }

    function testDeposit() public {
        vm.prank(user1);
        bestPractices.deposit{value: 1 ether}();
        
        assertEq(bestPractices.balances(user1), 1 ether);
        assertEq(bestPractices.totalDeposits(), 1 ether);
    }

    function testFailDepositZero() public {
        vm.prank(user1);
        bestPractices.deposit{value: 0}();
    }

    function testWithdraw() public {
        // Setup: deposit first
        vm.prank(user1);
        bestPractices.deposit{value: 1 ether}();
        
        // Test withdrawal
        vm.prank(user1);
        bestPractices.withdraw(0.5 ether);
        
        assertEq(bestPractices.balances(user1), 0.5 ether);
    }

    function testFailWithdrawExceedsBalance() public {
        vm.prank(user1);
        bestPractices.deposit{value: 1 ether}();
        
        vm.prank(user1);
        bestPractices.withdraw(2 ether); // Should fail
    }

    function testEmergencyStop() public {
        bestPractices.emergencyStop();
        assertTrue(bestPractices.paused());
    }

    function testFailDepositWhenPaused() public {
        bestPractices.emergencyStop();
        
        vm.prank(user1);
        bestPractices.deposit{value: 1 ether}(); // Should fail
    }

    function testFuzz_Deposit(uint256 amount) public {
        vm.assume(amount > 0 && amount <= 100 ether);
        vm.deal(user1, amount);
        
        vm.prank(user1);
        bestPractices.deposit{value: amount}();
        
        assertEq(bestPractices.balances(user1), amount);
    }

    function testFuzz_WithdrawPartial(uint256 depositAmount, uint256 withdrawAmount) public {
        vm.assume(depositAmount > 0 && depositAmount <= 10 ether);
        vm.assume(withdrawAmount > 0 && withdrawAmount <= depositAmount);
        vm.deal(user1, depositAmount);
        
        vm.prank(user1);
        bestPractices.deposit{value: depositAmount}();
        
        vm.prank(user1);
        bestPractices.withdraw(withdrawAmount);
        
        assertEq(bestPractices.balances(user1), depositAmount - withdrawAmount);
    }
}

═══════════════════════════════════════════════════════════════════════════════
                               DEPLOYMENT SCRIPTS
═══════════════════════════════════════════════════════════════════════════════

// Hardhat Deployment Script (scripts/deploy.js)
const { ethers, network } = require("hardhat");

async function main() {
    console.log(`Deploying to network: ${network.name}`);
    
    const [deployer] = await ethers.getSigners();
    console.log("Deploying contracts with account:", deployer.address);
    console.log("Account balance:", (await deployer.getBalance()).toString());
    
    // Deploy BestPractices contract
    console.log("\n🚀 Deploying BestPractices...");
    const BestPractices = await ethers.getContractFactory("BestPractices");
    const bestPractices = await BestPractices.deploy();
    await bestPractices.deployed();
    console.log("✅ BestPractices deployed to:", bestPractices.address);
    
    // Deploy MultiSig Wallet
    console.log("\n🚀 Deploying MultiSig Wallet...");
    const owners = [deployer.address]; // Add more owners as needed
    const requiredConfirmations = 1;
    
    const MultiSigWallet = await ethers.getContractFactory("MultiSigWallet");
    const multiSig = await MultiSigWallet.deploy(owners, requiredConfirmations);
    await multiSig.deployed();
    console.log("✅ MultiSig Wallet deployed to:", multiSig.address);
    
    // Deploy Price Oracle (if on mainnet/testnet)
    if (network.name !== "hardhat") {
        console.log("\n🚀 Deploying Price Oracle...");
        const ETH_USD_PRICE_FEED = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"; // Mainnet
        
        const PriceOracle = await ethers.getContractFactory("PriceOracle");
        const priceOracle = await PriceOracle.deploy(ETH_USD_PRICE_FEED);
        await priceOracle.deployed();
        console.log("✅ Price Oracle deployed to:", priceOracle.address);
    }
    
    // Verify contracts on Etherscan (if not local)
    if (network.name !== "hardhat") {
        console.log("\n⏳ Waiting for block confirmations...");
        await bestPractices.deployTransaction.wait(6);
        
        try {
            console.log("\n🔍 Verifying BestPractices on Etherscan...");
            await hre.run("verify:verify", {
                address: bestPractices.address,
                constructorArguments: [],
            });
            console.log("✅ BestPractices verified!");
        } catch (error) {
            console.log("❌ Verification failed:", error.message);
        }
    }
    
    // Save deployment info
    const deploymentInfo = {
        network: network.name,
        deployer: deployer.address,
        contracts: {
            BestPractices: bestPractices.address,
            MultiSigWallet: multiSig.address
        },
        timestamp: new Date().toISOString()
    };
    
    console.log("\n📋 Deployment Summary:");
    console.log(JSON.stringify(deploymentInfo, null, 2));
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("❌ Deployment failed:", error);
        process.exit(1);
    });

// Foundry Deployment Script (script/Deploy.s.sol)
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/BestPractices.sol";
import "../src/MultiSigWallet.sol";
import "../src/PriceOracle.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);
        
        console.log("Deploying with address:", deployer);
        console.log("Balance:", deployer.balance);
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy BestPractices
        BestPractices bestPractices = new BestPractices();
        console.log("BestPractices deployed to:", address(bestPractices));
        
        // Deploy MultiSig Wallet
        address[] memory owners = new address[](1);
        owners[0] = deployer;
        MultiSigWallet multiSig = new MultiSigWallet(owners, 1);
        console.log("MultiSig Wallet deployed to:", address(multiSig));
        
        // Deploy Price Oracle (mainnet ETH/USD feed)
        address ethUsdFeed = 0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419;
        PriceOracle priceOracle = new PriceOracle(ethUsdFeed);
        console.log("Price Oracle deployed to:", address(priceOracle));
        
        vm.stopBroadcast();
        
        // Save addresses to file
        string memory json = string(abi.encodePacked(
            '{"BestPractices":"', vm.toString(address(bestPractices)), '",',
            '"MultiSigWallet":"', vm.toString(address(multiSig)), '",',
            '"PriceOracle":"', vm.toString(address(priceOracle)), '"}'
        ));
        
        vm.writeFile("./deployments.json", json);
        console.log("Deployment addresses saved to deployments.json");
    }
}

═══════════════════════════════════════════════════════════════════════════════
                               USEFUL RESOURCES & TOOLS
═══════════════════════════════════════════════════════════════════════════════

🛠️ DEVELOPMENT TOOLS:
- Remix IDE: https://remix.ethereum.org (Browser-based development)
- Hardhat: https://hardhat.org (JavaScript-based framework)
- Foundry: https://getfoundry.sh (Rust-based, fast testing)
- VS Code Solidity Extension: solidity.solidity-hardhat

🔒 SECURITY TOOLS:
- Slither: Static analysis tool by Trail of Bits
- Mythril: Security analysis platform
- Echidna: Property-based fuzzing tool
- Manticore: Symbolic execution tool
- Securify: Automated security scanner

📚 LIBRARIES & STANDARDS:
- OpenZeppelin Contracts: https://openzeppelin.com/contracts
- OpenZeppelin Defender: Security operations platform
- Chainlink: https://chain.link (Oracles and external data)
- Uniswap V3: https://uniswap.org (DEX reference implementation)

📖 DOCUMENTATION:
- Solidity Documentation: https://docs.soliditylang.org
- Ethereum.org Developer Portal: https://ethereum.org/developers
- ConsenSys Best Practices: https://consensys.github.io/smart-contract-best-practices
- Trail of Bits Guidelines: https://github.com/crytic/building-secure-contracts

🎓 LEARNING RESOURCES:
- CryptoZombies: https://cryptozombies.io (Interactive tutorial)
- Solidity by Example: https://solidity-by-example.org
- Ethernaut: https://ethernaut.openzeppelin.com (Security challenges)
- Damn Vulnerable DeFi: https://www.damnvulnerabledefi.xyz

🌐 COMMUNITY:
- Ethereum Stack Exchange: https://ethereum.stackexchange.com
- OpenZeppelin Forum: https://forum.openzeppelin.com
- Hardhat Discord: https://hardhat.org/discord
- Foundry Telegram: https://t.me/foundry_rs

🔧 DEVELOPMENT SETUP:
// package.json for Hardhat project
{
  "name": "solidity-project",
  "version": "1.0.0",
  "devDependencies": {
    "@nomicfoundation/hardhat-toolbox": "^2.0.0",
    "@openzeppelin/contracts": "^4.8.0",
    "hardhat": "^2.12.0"
  },
  "scripts": {
    "compile": "hardhat compile",
    "test": "hardhat test",
    "deploy": "hardhat run scripts/deploy.js",
    "verify": "hardhat verify"
  }
}

// foundry.toml for Foundry project
[profile.default]
src = "src"
out = "out"
libs = ["lib"]
solc = "0.8.19"
optimizer = true
optimizer_runs = 200
via_ir = true

[rpc_endpoints]
mainnet = "${MAINNET_RPC_URL}"
goerli = "${GOERLI_RPC_URL}"
sepolia = "${SEPOLIA_RPC_URL}"

[etherscan]
mainnet = { key = "${ETHERSCAN_API_KEY}" }
goerli = { key = "${ETHERSCAN_API_KEY}" }
sepolia = { key = "${ETHERSCAN_API_KEY}" }

═══════════════════════════════════════════════════════════════════════════════
*/

/**
 * @title Solidity Smart Contract Development Reference - COMPLETE
 * @author Richard Rembert
 * @notice Comprehensive reference covering Solidity from fundamentals to production deployment
 * @dev This reference serves as both learning material and production guide
 * 
 * ✅ COMPLETE COVERAGE:
 * 1. ✅ Setup and Development Environment
 * 2. ✅ Basic Contract Structure  
 * 3. ✅ Data Types and Variables
 * 4. ✅ Functions and Modifiers
 * 5. ✅ Control Structures
 * 6. ✅ Error Handling
 * 7. ✅ Events and Logging
 * 8. ✅ Inheritance and Interfaces
 * 9. ✅ Libraries and Using For
 * 10. ✅ Security Patterns
 * 11. ✅ NFT Contracts (ERC-721)
 * 12. ✅ DeFi Protocols (DEX, Staking, Lending)
 * 13. ✅ Advanced Patterns and Assembly
 * 14. ✅ Gas Optimization Techniques
 * 15. ✅ Common Vulnerabilities to Avoid
 * 16. ✅ Testing and Deployment Patterns
 * 17. ✅ Multi-Signature Wallets
 * 18. ✅ Upgradeable Contracts
 * 19. ✅ Oracle Integration
 * 20. ✅ Best Practices Summary
 * 
 * 🎯 PRODUCTION READY:
 * - Complete testing examples (Hardhat & Foundry)
 * - Deployment scripts and verification
 * - Security patterns and vulnerability prevention
 * - Gas optimization techniques
 * - Real-world contract examples
 * 
 * 🔐 SECURITY CHECKLIST:
 * ✅ Reentrancy protection (ReentrancyGuard)
 * ✅ Access controls (onlyOwner, role-based)
 * ✅ Input validation (custom errors)
 * ✅ Circuit breakers (emergency stops)
 * ✅ Checks-effects-interactions pattern
 * ✅ Safe external calls
 * ✅ Integer overflow protection (Solidity 0.8+)
 * ✅ Proper event emission
 * ✅ Gas optimization
 * ✅ Upgrade safety considerations
 * 
 * 🚀 NEXT STEPS:
 * 1. Practice with the examples provided
 * 2. Deploy to testnets (Goerli, Sepolia)
 * 3. Get contracts audited before mainnet
 * 4. Stay updated with latest Solidity versions
 * 5. Join the community and contribute
 * 
 * Remember: Security first, test extensively, deploy carefully! 🛡️
 * 
 * Happy building the decentralized future! 🌟⛓️
 */