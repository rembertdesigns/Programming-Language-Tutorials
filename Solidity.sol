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