# BLOCKCHAIN PYTHON - Comprehensive Development Reference - by Richard Rembert

# Python is extensively used in blockchain development for analysis, automation,
# DApp backends, trading bots, data science, and blockchain infrastructure

# ═══════════════════════════════════════════════════════════════════════════════
#                           1. SETUP AND ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

"""
BLOCKCHAIN PYTHON DEVELOPMENT SETUP:

1. Install Python 3.8+ and pip
2. Create virtual environment:
   python -m venv blockchain_env
   source blockchain_env/bin/activate  # Linux/Mac
   blockchain_env\Scripts\activate     # Windows

3. Essential blockchain packages:
   pip install web3                    # Ethereum interaction
   pip install bitcoin                 # Bitcoin utilities
   pip install requests               # HTTP requests
   pip install pandas numpy           # Data analysis
   pip install matplotlib seaborn     # Visualization
   pip install asyncio aiohttp        # Async programming
   pip install websockets             # Real-time data
   pip install python-dotenv          # Environment variables
   pip install click                  # CLI applications
   pip install celery redis           # Task queues
   pip install sqlalchemy            # Database ORM
   pip install pytest                # Testing

4. Advanced packages:
   pip install brownie                # Smart contract development
   pip install eth-brownie           # Ethereum development framework
   pip install py-solc-x             # Solidity compiler
   pip install rlp                   # RLP encoding/decoding
   pip install eth-hash eth-keys     # Cryptographic utilities
   pip install coincurve             # Elliptic curve cryptography
   pip install pysha3                # Keccak hashing

5. API clients:
   pip install python-binance        # Binance API
   pip install ccxt                  # Cryptocurrency exchange APIs
   pip install alpha-vantage         # Financial data
   pip install cryptocompare         # Crypto market data
   pip install ta                    # Technical analysis

6. Development tools:
   pip install black flake8          # Code formatting and linting
   pip install mypy                  # Type checking
   pip install jupyter               # Interactive notebooks
   pip install mnemonic              # BIP39 mnemonic generation
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                           2. WEB3 AND ETHEREUM INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════

from web3 import Web3, HTTPProvider, WebsocketProvider
from web3.contract import Contract
from web3.exceptions import TransactionNotFound, BlockNotFound
from web3.middleware import geth_poa_middleware
import requests

class EthereumClient:
    """Comprehensive Ethereum blockchain client"""
    
    def __init__(self, node_url: str, private_key: Optional[str] = None):
        """
        Initialize Ethereum client
        
        Args:
            node_url: RPC endpoint (Infura, Alchemy, local node)
            private_key: Private key for transaction signing
        """
        self.w3 = Web3(HTTPProvider(node_url))
        
        # Add PoA middleware for testnets like Goerli
        if 'goerli' in node_url.lower() or 'sepolia' in node_url.lower():
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.private_key = private_key
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
        else:
            self.account = None
            
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")
        
        logger.info(f"Connected to Ethereum network: {self.get_network_info()}")
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        try:
            chain_id = self.w3.eth.chain_id
            block_number = self.w3.eth.block_number
            gas_price = self.w3.eth.gas_price
            
            return {
                'chain_id': chain_id,
                'latest_block': block_number,
                'gas_price_gwei': self.w3.from_wei(gas_price, 'gwei'),
                'is_syncing': self.w3.eth.syncing
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {}
    
    def get_balance(self, address: str, block: str = 'latest') -> float:
        """Get ETH balance for an address"""
        try:
            balance_wei = self.w3.eth.get_balance(
                self.w3.to_checksum_address(address), 
                block
            )
            return self.w3.from_wei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction details"""
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            return {
                'hash': tx['hash'].hex(),
                'from': tx['from'],
                'to': tx['to'],
                'value': self.w3.from_wei(tx['value'], 'ether'),
                'gas': tx['gas'],
                'gas_price': self.w3.from_wei(tx['gasPrice'], 'gwei'),
                'nonce': tx['nonce'],
                'block_number': receipt['blockNumber'],
                'gas_used': receipt['gasUsed'],
                'status': receipt['status'],
                'confirmations': self.w3.eth.block_number - receipt['blockNumber']
            }
        except TransactionNotFound:
            logger.warning(f"Transaction not found: {tx_hash}")
            return None
        except Exception as e:
            logger.error(f"Error getting transaction: {e}")
            return None
    
    def get_block(self, block_number: Union[int, str] = 'latest') -> Optional[Dict]:
        """Get block information"""
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)
            
            return {
                'number': block['number'],
                'hash': block['hash'].hex(),
                'parent_hash': block['parentHash'].hex(),
                'timestamp': datetime.fromtimestamp(block['timestamp']),
                'gas_limit': block['gasLimit'],
                'gas_used': block['gasUsed'],
                'transactions_count': len(block['transactions']),
                'size': block['size'],
                'miner': block['miner']
            }
        except BlockNotFound:
            logger.warning(f"Block not found: {block_number}")
            return None
        except Exception as e:
            logger.error(f"Error getting block: {e}")
            return None
    
    def send_transaction(self, to_address: str, value_eth: float, 
                        gas_limit: int = 21000, gas_price_gwei: Optional[float] = None) -> Optional[str]:
        """Send ETH transaction"""
        if not self.account:
            raise ValueError("Private key required for sending transactions")
        
        try:
            # Get current gas price if not specified
            if gas_price_gwei is None:
                gas_price_gwei = self.w3.from_wei(self.w3.eth.gas_price, 'gwei')
            
            # Build transaction
            transaction = {
                'to': self.w3.to_checksum_address(to_address),
                'value': self.w3.to_wei(value_eth, 'ether'),
                'gas': gas_limit,
                'gasPrice': self.w3.to_wei(gas_price_gwei, 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'chainId': self.w3.eth.chain_id
            }
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            logger.info(f"Transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return None
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 300) -> Optional[Dict]:
        """Wait for transaction confirmation"""
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            logger.info(f"Transaction confirmed: {tx_hash}")
            return dict(receipt)
        except Exception as e:
            logger.error(f"Error waiting for transaction: {e}")
            return None

# Smart Contract Interaction
class ContractInteractor:
    """Interact with smart contracts"""
    
    def __init__(self, w3: Web3, contract_address: str, abi: List[Dict]):
        """
        Initialize contract interactor
        
        Args:
            w3: Web3 instance
            contract_address: Contract address
            abi: Contract ABI (Application Binary Interface)
        """
        self.w3 = w3
        self.contract = w3.eth.contract(
            address=w3.to_checksum_address(contract_address),
            abi=abi
        )
    
    def call_function(self, function_name: str, *args, **kwargs) -> Any:
        """Call a read-only contract function"""
        try:
            function = getattr(self.contract.functions, function_name)
            result = function(*args, **kwargs).call()
            return result
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            return None
    
    def send_transaction(self, function_name: str, private_key: str, 
                        *args, gas_limit: int = 200000, **kwargs) -> Optional[str]:
        """Send a transaction to a contract function"""
        try:
            account = self.w3.eth.account.from_key(private_key)
            function = getattr(self.contract.functions, function_name)
            
            # Build transaction
            transaction = function(*args, **kwargs).build_transaction({
                'from': account.address,
                'gas': gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(account.address),
                'chainId': self.w3.eth.chain_id
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error sending transaction to {function_name}: {e}")
            return None
    
    def get_events(self, event_name: str, from_block: int = 0, to_block: str = 'latest') -> List[Dict]:
        """Get contract events"""
        try:
            event_filter = getattr(self.contract.events, event_name).create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            events = event_filter.get_all_entries()
            
            return [dict(event) for event in events]
            
        except Exception as e:
            logger.error(f"Error getting events {event_name}: {e}")
            return []

# ERC-20 Token interaction
class ERC20Token(ContractInteractor):
    """ERC-20 token interaction class"""
    
    # Standard ERC-20 ABI (minimal)
    ERC20_ABI = [
        {
            "constant": True,
            "inputs": [],
            "name": "name",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "symbol",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [],
            "name": "totalSupply",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function"
        },
        {
            "constant": True,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "type": "function"
        },
        {
            "constant": False,
            "inputs": [
                {"name": "_to", "type": "address"},
                {"name": "_value", "type": "uint256"}
            ],
            "name": "transfer",
            "outputs": [{"name": "", "type": "bool"}],
            "type": "function"
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "from", "type": "address"},
                {"indexed": True, "name": "to", "type": "address"},
                {"indexed": False, "name": "value", "type": "uint256"}
            ],
            "name": "Transfer",
            "type": "event"
        }
    ]
    
    def __init__(self, w3: Web3, token_address: str):
        super().__init__(w3, token_address, self.ERC20_ABI)
        
        # Get token info
        self.name = self.call_function('name')
        self.symbol = self.call_function('symbol')
        self.decimals = self.call_function('decimals')
        self.total_supply = self.call_function('totalSupply')
        
        logger.info(f"Initialized ERC-20 token: {self.name} ({self.symbol})")
    
    def get_balance(self, address: str) -> float:
        """Get token balance for an address"""
        balance = self.call_function('balanceOf', self.w3.to_checksum_address(address))
        if balance is not None:
            return balance / (10 ** self.decimals)
        return 0.0
    
    def transfer(self, private_key: str, to_address: str, amount: float) -> Optional[str]:
        """Transfer tokens"""
        amount_wei = int(amount * (10 ** self.decimals))
        return self.send_transaction('transfer', private_key, 
                                   self.w3.to_checksum_address(to_address), amount_wei)
    
    def get_transfer_events(self, from_block: int = 0) -> List[Dict]:
        """Get transfer events"""
        events = self.get_events('Transfer', from_block=from_block)
        
        processed_events = []
        for event in events:
            processed_events.append({
                'from': event['args']['from'],
                'to': event['args']['to'],
                'value': event['args']['value'] / (10 ** self.decimals),
                'transaction_hash': event['transactionHash'].hex(),
                'block_number': event['blockNumber']
            })
        
        return processed_events