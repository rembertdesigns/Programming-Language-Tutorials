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