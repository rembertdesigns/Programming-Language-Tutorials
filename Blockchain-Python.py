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


# ═══════════════════════════════════════════════════════════════════════════════
#                           3. BITCOIN AND CRYPTOCURRENCY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

import hashlib
import hmac
import base64
from bitcoin import *
from Crypto.Hash import SHA256, RIPEMD160
from Crypto.Cipher import AES
import secrets

class BitcoinWallet:
    """Bitcoin wallet utilities"""
    
    def __init__(self, private_key: Optional[str] = None):
        """Initialize Bitcoin wallet"""
        if private_key:
            self.private_key = private_key
        else:
            self.private_key = self.generate_private_key()
        
        self.public_key = self.private_key_to_public_key(self.private_key)
        self.address = self.public_key_to_address(self.public_key)
        
        logger.info(f"Bitcoin wallet initialized: {self.address}")
    
    @staticmethod
    def generate_private_key() -> str:
        """Generate a random private key"""
        return secrets.token_hex(32)
    
    @staticmethod
    def private_key_to_public_key(private_key: str) -> str:
        """Convert private key to public key"""
        return privkey_to_pubkey(private_key)
    
    @staticmethod
    def public_key_to_address(public_key: str) -> str:
        """Convert public key to Bitcoin address"""
        return pubkey_to_address(public_key)
    
    @staticmethod
    def validate_address(address: str) -> bool:
        """Validate Bitcoin address"""
        try:
            # Simple validation - in production use more robust validation
            if len(address) < 26 or len(address) > 35:
                return False
            
            # Check if address starts with valid prefixes
            valid_prefixes = ['1', '3', 'bc1']
            return any(address.startswith(prefix) for prefix in valid_prefixes)
        except:
            return False
    
    def sign_message(self, message: str) -> str:
        """Sign a message with the private key"""
        try:
            signature = ecdsa_sign(message, self.private_key)
            return signature
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            return ""
    
    @staticmethod
    def verify_signature(message: str, signature: str, public_key: str) -> bool:
        """Verify a message signature"""
        try:
            return ecdsa_verify(message, signature, public_key)
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False

class CryptographicUtils:
    """Cryptographic utilities for blockchain development"""
    
    @staticmethod
    def hash_sha256(data: Union[str, bytes]) -> str:
        """SHA-256 hash"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def hash_keccak256(data: Union[str, bytes]) -> str:
        """Keccak-256 hash (used by Ethereum)"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        import sha3
        return sha3.keccak_256(data).hexdigest()
    
    @staticmethod
    def hash_ripemd160(data: Union[str, bytes]) -> str:
        """RIPEMD-160 hash"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        ripemd160 = RIPEMD160.new()
        ripemd160.update(data)
        return ripemd160.hexdigest()
    
    @staticmethod
    def generate_mnemonic(strength: int = 128) -> str:
        """Generate BIP39 mnemonic phrase"""
        from mnemonic import Mnemonic
        mnemo = Mnemonic("english")
        return mnemo.generate(strength=strength)
    
    @staticmethod
    def mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to seed"""
        from mnemonic import Mnemonic
        mnemo = Mnemonic("english")
        return mnemo.to_seed(mnemonic, passphrase)
    
    @staticmethod
    def encrypt_data(data: str, password: str) -> str:
        """Encrypt data with AES"""
        try:
            # Generate salt and key
            salt = secrets.token_bytes(16)
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            
            # Encrypt data
            cipher = AES.new(key, AES.MODE_GCM)
            ciphertext, tag = cipher.encrypt_and_digest(data.encode())
            
            # Combine salt, nonce, tag, and ciphertext
            encrypted = salt + cipher.nonce + tag + ciphertext
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return ""
    
    @staticmethod
    def decrypt_data(encrypted_data: str, password: str) -> str:
        """Decrypt AES encrypted data"""
        try:
            # Decode and extract components
            encrypted = base64.b64decode(encrypted_data.encode())
            salt = encrypted[:16]
            nonce = encrypted[16:32]
            tag = encrypted[32:48]
            ciphertext = encrypted[48:]
            
            # Derive key
            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            
            # Decrypt
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            data = cipher.decrypt_and_verify(ciphertext, tag)
            
            return data.decode()
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return ""


# ═══════════════════════════════════════════════════════════════════════════════
#                           4. BLOCKCHAIN DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import sqlite3

class BlockchainAnalyzer:
    """Blockchain data analysis and visualization"""
    
    def __init__(self, ethereum_client: EthereumClient):
        self.eth_client = ethereum_client
        self.w3 = ethereum_client.w3
        
    def analyze_address_activity(self, address: str, blocks_to_analyze: int = 1000) -> Dict:
        """Analyze activity for a specific address"""
        try:
            address = self.w3.to_checksum_address(address)
            current_block = self.w3.eth.block_number
            start_block = max(0, current_block - blocks_to_analyze)
            
            transactions = []
            eth_received = 0
            eth_sent = 0
            gas_used = 0
            
            logger.info(f"Analyzing {address} from block {start_block} to {current_block}")
            
            for block_num in range(start_block, current_block + 1):
                if block_num % 100 == 0:
                    logger.info(f"Processing block {block_num}")
                
                try:
                    block = self.w3.eth.get_block(block_num, full_transactions=True)
                    
                    for tx in block['transactions']:
                        if tx['from'] == address or tx['to'] == address:
                            value_eth = self.w3.from_wei(tx['value'], 'ether')
                            
                            # Get transaction receipt for gas info
                            try:
                                receipt = self.w3.eth.get_transaction_receipt(tx['hash'])
                                gas_cost = receipt['gasUsed'] * tx['gasPrice']
                            except:
                                gas_cost = 0
                            
                            tx_data = {
                                'hash': tx['hash'].hex(),
                                'block_number': block_num,
                                'from': tx['from'],
                                'to': tx['to'],
                                'value_eth': float(value_eth),
                                'gas_used': tx['gas'],
                                'gas_price_gwei': self.w3.from_wei(tx['gasPrice'], 'gwei'),
                                'gas_cost_eth': self.w3.from_wei(gas_cost, 'ether'),
                                'timestamp': datetime.fromtimestamp(block['timestamp'])
                            }
                            
                            transactions.append(tx_data)
                            
                            if tx['to'] == address:
                                eth_received += float(value_eth)
                            if tx['from'] == address:
                                eth_sent += float(value_eth)
                                gas_used += self.w3.from_wei(gas_cost, 'ether')
                
                except Exception as e:
                    logger.warning(f"Error processing block {block_num}: {e}")
                    continue
            
            # Analysis results
            analysis = {
                'address': address,
                'blocks_analyzed': blocks_to_analyze,
                'total_transactions': len(transactions),
                'eth_received': eth_received,
                'eth_sent': eth_sent,
                'net_eth_flow': eth_received - eth_sent,
                'total_gas_cost': gas_used,
                'current_balance': self.eth_client.get_balance(address),
                'transactions': transactions
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing address: {e}")
            return {}
    
    def create_transaction_graph(self, transactions: List[Dict]) -> None:
        """Create transaction flow visualization"""
        if not transactions:
            logger.warning("No transactions to visualize")
            return
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Time series analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Transaction volume over time
        daily_volume = df.resample('D')['value_eth'].sum()
        axes[0, 0].plot(daily_volume.index, daily_volume.values)
        axes[0, 0].set_title('Daily Transaction Volume (ETH)')
        axes[0, 0].set_ylabel('ETH')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Transaction count over time
        daily_count = df.resample('D').size()
        axes[0, 1].plot(daily_count.index, daily_count.values)
        axes[0, 1].set_title('Daily Transaction Count')
        axes[0, 1].set_ylabel('Transactions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Gas price distribution
        axes[1, 0].hist(df['gas_price_gwei'], bins=30, alpha=0.7)
        axes[1, 0].set_title('Gas Price Distribution')
        axes[1, 0].set_xlabel('Gas Price (Gwei)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Value distribution
        axes[1, 1].hist(df[df['value_eth'] > 0]['value_eth'], bins=30, alpha=0.7)
        axes[1, 1].set_title('Transaction Value Distribution')
        axes[1, 1].set_xlabel('Value (ETH)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_gas_trends(self, blocks_to_analyze: int = 1000) -> Dict:
        """Analyze gas price trends"""
        try:
            current_block = self.w3.eth.block_number
            start_block = max(0, current_block - blocks_to_analyze)
            
            gas_data = []
            
            for block_num in range(start_block, current_block + 1, 10):  # Sample every 10 blocks
                try:
                    block = self.w3.eth.get_block(block_num, full_transactions=True)
                    
                    if block['transactions']:
                        gas_prices = [self.w3.from_wei(tx['gasPrice'], 'gwei') 
                                    for tx in block['transactions']]
                        
                        gas_data.append({
                            'block_number': block_num,
                            'timestamp': datetime.fromtimestamp(block['timestamp']),
                            'avg_gas_price': np.mean(gas_prices),
                            'median_gas_price': np.median(gas_prices),
                            'max_gas_price': np.max(gas_prices),
                            'min_gas_price': np.min(gas_prices),
                            'tx_count': len(block['transactions']),
                            'gas_used': block['gasUsed'],
                            'gas_limit': block['gasLimit'],
                            'gas_utilization': block['gasUsed'] / block['gasLimit']
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing block {block_num}: {e}")
                    continue
            
            return {
                'gas_data': gas_data,
                'blocks_analyzed': len(gas_data),
                'avg_gas_price': np.mean([d['avg_gas_price'] for d in gas_data]),
                'avg_gas_utilization': np.mean([d['gas_utilization'] for d in gas_data])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gas trends: {e}")
            return {}
    
    def token_holder_analysis(self, token_address: str, top_n: int = 100) -> Dict:
        """Analyze token distribution among holders"""
        try:
            token = ERC20Token(self.w3, token_address)
            
            # Get transfer events to find all holders
            transfer_events = token.get_transfer_events()
            
            # Track balances
            balances = defaultdict(float)
            
            for event in transfer_events:
                from_addr = event['from']
                to_addr = event['to']
                value = event['value']
                
                # Handle minting (from zero address)
                if from_addr != '0x0000000000000000000000000000000000000000':
                    balances[from_addr] -= value
                
                balances[to_addr] += value
            
            # Remove zero balances and get current balances
            active_holders = {}
            for address, balance in balances.items():
                if balance > 0:
                    # Verify current balance on-chain
                    current_balance = token.get_balance(address)
                    if current_balance > 0:
                        active_holders[address] = current_balance
            
            # Sort by balance
            sorted_holders = sorted(active_holders.items(), key=lambda x: x[1], reverse=True)
            top_holders = sorted_holders[:top_n]
            
            total_supply = token.total_supply / (10 ** token.decimals)
            total_held = sum(active_holders.values())
            
            # Calculate distribution metrics
            gini_coefficient = self._calculate_gini_coefficient([balance for _, balance in sorted_holders])
            
            return {
                'token_name': token.name,
                'token_symbol': token.symbol,
                'total_supply': total_supply,
                'total_holders': len(active_holders),
                'top_holders': top_holders,
                'concentration_ratio': {
                    'top_10': sum([balance for _, balance in sorted_holders[:10]]) / total_supply * 100,
                    'top_50': sum([balance for _, balance in sorted_holders[:50]]) / total_supply * 100,
                    'top_100': sum([balance for _, balance in sorted_holders[:100]]) / total_supply * 100
                },
                'gini_coefficient': gini_coefficient
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token holders: {e}")
            return {}
    
    @staticmethod
    def _calculate_gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for wealth distribution"""
        if not values:
            return 0
        
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(values, 1))) / (n * sum(values))

class BlockchainDatabase:
    """SQLite database for storing blockchain data"""
    
    def __init__(self, db_path: str = "blockchain_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE NOT NULL,
                block_number INTEGER,
                from_address TEXT,
                to_address TEXT,
                value_eth REAL,
                gas_used INTEGER,
                gas_price_gwei REAL,
                timestamp DATETIME,
                status INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Blocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number INTEGER UNIQUE NOT NULL,
                hash TEXT UNIQUE NOT NULL,
                parent_hash TEXT,
                timestamp DATETIME,
                gas_limit INTEGER,
                gas_used INTEGER,
                miner TEXT,
                transaction_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Token transfers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_transfers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT,
                from_address TEXT,
                to_address TEXT,
                value REAL,
                transaction_hash TEXT,
                block_number INTEGER,
                log_index INTEGER,
                timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tx_hash ON transactions(hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_address)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_address)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_block_number ON blocks(number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_address ON token_transfers(token_address)')
        
        self.conn.commit()
    
    def insert_transaction(self, tx_data: Dict):
        """Insert transaction data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO transactions 
            (hash, block_number, from_address, to_address, value_eth, gas_used, gas_price_gwei, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tx_data['hash'],
            tx_data['block_number'],
            tx_data['from'],
            tx_data['to'],
            tx_data['value'],
            tx_data['gas_used'],
            tx_data['gas_price_gwei'],
            tx_data['timestamp'],
            tx_data['status']
        ))
        self.conn.commit()
    
    def insert_block(self, block_data: Dict):
        """Insert block data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO blocks 
            (number, hash, parent_hash, timestamp, gas_limit, gas_used, miner, transaction_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            block_data['number'],
            block_data['hash'],
            block_data['parent_hash'],
            block_data['timestamp'],
            block_data['gas_limit'],
            block_data['gas_used'],
            block_data['miner'],
            block_data['transactions_count']
        ))
        self.conn.commit()
    
    def get_address_transactions(self, address: str, limit: int = 100) -> List[Dict]:
        """Get transactions for an address"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE from_address = ? OR to_address = ?
            ORDER BY block_number DESC
            LIMIT ?
        ''', (address, address, limit))
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_gas_statistics(self, hours: int = 24) -> Dict:
        """Get gas statistics for the last N hours"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                AVG(gas_price_gwei) as avg_gas_price,
                MIN(gas_price_gwei) as min_gas_price,
                MAX(gas_price_gwei) as max_gas_price,
                COUNT(*) as transaction_count
            FROM transactions 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours))
        
        row = cursor.fetchone()
        return {
            'avg_gas_price': row[0] or 0,
            'min_gas_price': row[1] or 0,
            'max_gas_price': row[2] or 0,
            'transaction_count': row[3] or 0
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           5. CRYPTOCURRENCY TRADING AND APIS
# ═══════════════════════════════════════════════════════════════════════════════

import ccxt
import pandas as pd
from typing import Tuple
import ta  # Technical analysis library

class CryptocurrencyTrader:
    """Cryptocurrency trading utilities and API integration"""
    
    def __init__(self, exchange_name: str = 'binance', api_key: str = None, 
                 api_secret: str = None, sandbox: bool = True):
        """
        Initialize cryptocurrency trader
        
        Args:
            exchange_name: Exchange name (binance, coinbase, kraken, etc.)
            api_key: API key for authenticated endpoints
            api_secret: API secret for authenticated endpoints
            sandbox: Use sandbox/testnet environment
        """
        self.exchange_name = exchange_name
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': sandbox,
            'enableRateLimit': True,
        })
        
        logger.info(f"Initialized {exchange_name} exchange (sandbox: {sandbox})")
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current price ticker for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': ticker['symbol'],
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change_24h': ticker['change'],
                'percentage_change_24h': ticker['percentage'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Optional[Dict]:
        """Get order book for a symbol"""
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return {
                'symbol': symbol,
                'bids': order_book['bids'][:limit],
                'asks': order_book['asks'][:limit],
                'timestamp': datetime.fromtimestamp(order_book['timestamp'] / 1000)
            }
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', 
                  limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV (candlestick) data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return None
    
    def place_order(self, symbol: str, side: str, amount: float, 
                   price: Optional[float] = None, order_type: str = 'market') -> Optional[Dict]:
        """Place a trading order"""
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            logger.info(f"Order placed: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict]:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            
            # Filter out zero balances
            non_zero_balances = {
                currency: info for currency, info in balance.items() 
                if isinstance(info, dict) and info.get('total', 0) > 0
            }
            
            return non_zero_balances
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
    
    def get_my_trades(self, symbol: str, limit: int = 50) -> Optional[List[Dict]]:
        """Get user's trade history"""
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return None

class TechnicalAnalysis:
    """Technical analysis utilities for trading"""
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data"""
        df = df.copy()
        
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        
        # Volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        return df
    
    @staticmethod
    def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Detect trading signals"""
        df = df.copy()
        
        # Golden Cross (SMA 20 crosses above SMA 50)
        df['golden_cross'] = (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        
        # Death Cross (SMA 20 crosses below SMA 50)
        df['death_cross'] = (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
        
        # RSI Overbought/Oversold
        df['rsi_overbought'] = df['rsi'] > 70
        df['rsi_oversold'] = df['rsi'] < 30
        
        # MACD Bullish/Bearish
        df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Bollinger Band signals
        df['bb_squeeze'] = (df['close'] > df['bb_upper'])
        df['bb_bounce'] = (df['close'] < df['bb_lower'])
        
        return df
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        recent_data = df.tail(window)
        
        # Simple approach: use recent highs and lows
        resistance = recent_data['high'].max()
        support = recent_data['low'].min()
        
        return support, resistance
    
    @staticmethod
    def backtest_strategy(df: pd.DataFrame, buy_signal_col: str, 
                         sell_signal_col: str, initial_capital: float = 10000) -> Dict:
        """Simple backtesting framework"""
        position = 0  # 0 = no position, 1 = long
        capital = initial_capital
        trades = []
        
        for i, row in df.iterrows():
            if row[buy_signal_col] and position == 0:
                # Buy signal
                position = 1
                shares = capital / row['close']
                capital = 0
                trades.append({
                    'type': 'buy',
                    'date': i,
                    'price': row['close'],
                    'shares': shares
                })
            
            elif row[sell_signal_col] and position == 1:
                # Sell signal
                position = 0
                capital = shares * row['close']
                trades.append({
                    'type': 'sell',
                    'date': i,
                    'price': row['close'],
                    'shares': shares
                })
        
        # Calculate final value
        if position == 1:
            final_value = shares * df['close'].iloc[-1]
        else:
            final_value = capital
        
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'trades': trades
        }

class PriceAlert:
    """Price monitoring and alert system"""
    
    def __init__(self, trader: CryptocurrencyTrader):
        self.trader = trader
        self.alerts = []
        self.running = False
    
    def add_alert(self, symbol: str, condition: str, price: float, 
                  message: str = None, email: str = None):
        """
        Add price alert
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            condition: 'above' or 'below'
            price: Trigger price
            message: Custom alert message
            email: Email to send alert (optional)
        """
        alert = {
            'id': len(self.alerts),
            'symbol': symbol,
            'condition': condition,
            'price': price,
            'message': message or f"{symbol} {condition} {price}",
            'email': email,
            'triggered': False,
            'created_at': datetime.now()
        }
        
        self.alerts.append(alert)
        logger.info(f"Alert added: {alert['message']}")
    
    def check_alerts(self):
        """Check all active alerts"""
        for alert in self.alerts:
            if alert['triggered']:
                continue
            
            ticker = self.trader.get_ticker(alert['symbol'])
            if not ticker:
                continue
            
            current_price = ticker['last_price']
            
            should_trigger = False
            if alert['condition'] == 'above' and current_price > alert['price']:
                should_trigger = True
            elif alert['condition'] == 'below' and current_price < alert['price']:
                should_trigger = True
            
            if should_trigger:
                self.trigger_alert(alert, current_price)
    
    def trigger_alert(self, alert: Dict, current_price: float):
        """Trigger an alert"""
        alert['triggered'] = True
        alert['triggered_at'] = datetime.now()
        alert['triggered_price'] = current_price
        
        message = f"ALERT: {alert['message']} (Current: {current_price})"
        logger.info(message)
        
        # Send email if configured
        if alert['email']:
            self.send_email_alert(alert['email'], message)
    
    def send_email_alert(self, email: str, message: str):
        """Send email alert (implement with your preferred email service)"""
        # Placeholder for email sending
        logger.info(f"Email alert sent to {email}: {message}")
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous price monitoring"""
        self.running = True
        logger.info(f"Starting price monitoring (interval: {interval}s)")
        
        while self.running:
            try:
                self.check_alerts()
                time.sleep(interval)
            except KeyboardInterrupt:
                self.stop_monitoring()
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop price monitoring"""
        self.running = False
        logger.info("Price monitoring stopped")


# ═══════════════════════════════════════════════════════════════════════════════
#                           6. DEFI AND YIELD FARMING
# ═══════════════════════════════════════════════════════════════════════════════

class DeFiProtocol:
    """DeFi protocol interaction utilities"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        
        # Common DeFi protocol addresses (Ethereum mainnet)
        self.protocols = {
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v2_factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
            'sushiswap_router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'compound_comptroller': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B',
            'aave_lending_pool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'
        }
    
    def get_uniswap_pair_price(self, token0_address: str, token1_address: str) -> Optional[float]:
        """Get price from Uniswap V2 pair"""
        try:
            # Uniswap V2 Factory ABI (simplified)
            factory_abi = [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "", "type": "address"},
                        {"name": "", "type": "address"}
                    ],
                    "name": "getPair",
                    "outputs": [{"name": "", "type": "address"}],
                    "type": "function"
                }
            ]
            
            # Uniswap V2 Pair ABI (simplified)
            pair_abi = [
                {
                    "constant": True,
                    "inputs": [],
                    "name": "getReserves",
                    "outputs": [
                        {"name": "_reserve0", "type": "uint112"},
                        {"name": "_reserve1", "type": "uint112"},
                        {"name": "_blockTimestampLast", "type": "uint32"}
                    ],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"name": "", "type": "address"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"name": "", "type": "address"}],
                    "type": "function"
                }
            ]
            
            # Get factory contract
            factory_contract = self.w3.eth.contract(
                address=self.protocols['uniswap_v2_factory'],
                abi=factory_abi
            )
            
            # Get pair address
            pair_address = factory_contract.functions.getPair(
                self.w3.to_checksum_address(token0_address),
                self.w3.to_checksum_address(token1_address)
            ).call()
            
            if pair_address == '0x0000000000000000000000000000000000000000':
                logger.warning("Pair not found")
                return None
            
            # Get pair contract
            pair_contract = self.w3.eth.contract(
                address=pair_address,
                abi=pair_abi
            )
            
            # Get reserves
            reserves = pair_contract.functions.getReserves().call()
            reserve0, reserve1, _ = reserves
            
            # Get token order
            pair_token0 = pair_contract.functions.token0().call()
            
            # Calculate price (token1/token0)
            if pair_token0.lower() == token0_address.lower():
                price = reserve1 / reserve0
            else:
                price = reserve0 / reserve1
            
            return price
            
        except Exception as e:
            logger.error(f"Error getting Uniswap price: {e}")
            return None
    
    def calculate_impermanent_loss(self, initial_price: float, current_price: float) -> Dict:
        """Calculate impermanent loss for LP position"""
        try:
            price_ratio = current_price / initial_price
            
            # IL formula: IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
            il_multiplier = 2 * (price_ratio ** 0.5) / (1 + price_ratio)
            impermanent_loss = (il_multiplier - 1) * 100
            
            # Value if held vs LP
            hold_value = (1 + price_ratio) / 2  # Average of both tokens
            lp_value = il_multiplier
            
            return {
                'price_change_pct': (price_ratio - 1) * 100,
                'impermanent_loss_pct': impermanent_loss,
                'hold_value_ratio': hold_value,
                'lp_value_ratio': lp_value,
                'hold_vs_lp_pct': (hold_value - lp_value) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating impermanent loss: {e}")
            return {}
    
    def estimate_yield_farming_returns(self, principal: float, apr: float, 
                                     days: int = 365, compound_frequency: int = 365) -> Dict:
        """Estimate yield farming returns with compounding"""
        try:
            # Simple interest
            simple_interest = principal * (apr / 100) * (days / 365)
            
            # Compound interest
            periods = compound_frequency * (days / 365)
            compound_interest = principal * ((1 + (apr / 100) / compound_frequency) ** periods) - principal
            
            return {
                'principal': principal,
                'days': days,
                'apr_pct': apr,
                'simple_interest': simple_interest,
                'compound_interest': compound_interest,
                'simple_total': principal + simple_interest,
                'compound_total': principal + compound_interest,
                'compound_advantage': compound_interest - simple_interest
            }
            
        except Exception as e:
            logger.error(f"Error calculating yield farming returns: {e}")
            return {}

class LiquidityPoolAnalyzer:
    """Analyze liquidity pools and farming opportunities"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.defi = DeFiProtocol(w3)
    
    def analyze_pool(self, token0_address: str, token1_address: str, 
                    pool_address: str = None) -> Dict:
        """Analyze a liquidity pool"""
        try:
            analysis = {}
            
            # Get current price
            current_price = self.defi.get_uniswap_pair_price(token0_address, token1_address)
            if current_price:
                analysis['current_price'] = current_price
            
            # Get token information
            token0 = ERC20Token(self.w3, token0_address)
            token1 = ERC20Token(self.w3, token1_address)
            
            analysis['token0'] = {
                'address': token0_address,
                'name': token0.name,
                'symbol': token0.symbol,
                'decimals': token0.decimals
            }
            
            analysis['token1'] = {
                'address': token1_address,
                'name': token1.name,
                'symbol': token1.symbol,
                'decimals': token1.decimals
            }
            
            # Historical price analysis would require additional data sources
            # This is a simplified version
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pool: {e}")
            return {}
    
    def find_arbitrage_opportunities(self, token_pairs: List[Tuple[str, str]], 
                                   exchanges: List[str] = None) -> List[Dict]:
        """Find arbitrage opportunities between exchanges"""
        # This would require implementing multiple exchange price feeds
        # Simplified placeholder implementation
        opportunities = []
        
        for token0, token1 in token_pairs:
            try:
                # Get prices from different sources
                uniswap_price = self.defi.get_uniswap_pair_price(token0, token1)
                
                # In a real implementation, you would compare prices across
                # multiple exchanges and identify profitable arbitrage opportunities
                
                if uniswap_price:
                    opportunities.append({
                        'pair': f"{token0}/{token1}",
                        'uniswap_price': uniswap_price,
                        'potential_profit': 0  # Placeholder
                    })
                    
            except Exception as e:
                logger.error(f"Error checking arbitrage for {token0}/{token1}: {e}")
                continue
        
        return opportunities


# ═══════════════════════════════════════════════════════════════════════════════
#                           7. NFT AND MARKETPLACE INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════

class NFTAnalyzer:
    """NFT analysis and marketplace interaction"""
    
    def __init__(self, w3: Web3, opensea_api_key: str = None):
        self.w3 = w3
        self.opensea_api_key = opensea_api_key
        self.opensea_base_url = "https://api.opensea.io/api/v1"
        
        # ERC-721 ABI (simplified)
        self.erc721_abi = [
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
                "inputs": [{"name": "_tokenId", "type": "uint256"}],
                "name": "tokenURI",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_tokenId", "type": "uint256"}],
                "name": "ownerOf",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "from", "type": "address"},
                    {"indexed": True, "name": "to", "type": "address"},
                    {"indexed": True, "name": "tokenId", "type": "uint256"}
                ],
                "name": "Transfer",
                "type": "event"
            }
        ]
    
    def get_nft_metadata(self, contract_address: str, token_id: int) -> Optional[Dict]:
        """Get NFT metadata from contract"""
        try:
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(contract_address),
                abi=self.erc721_abi
            )
            
            # Get basic info
            name = contract.functions.name().call()
            symbol = contract.functions.symbol().call()
            owner = contract.functions.ownerOf(token_id).call()
            token_uri = contract.functions.tokenURI(token_id).call()
            
            # Fetch metadata from URI
            metadata = {}
            if token_uri:
                try:
                    # Handle IPFS URIs
                    if token_uri.startswith('ipfs://'):
                        token_uri = token_uri.replace('ipfs://', 'https://ipfs.io/ipfs/')
                    
                    response = requests.get(token_uri, timeout=10)
                    if response.status_code == 200:
                        metadata = response.json()
                except Exception as e:
                    logger.warning(f"Could not fetch metadata from {token_uri}: {e}")
            
            return {
                'contract_address': contract_address,
                'token_id': token_id,
                'collection_name': name,
                'collection_symbol': symbol,
                'owner': owner,
                'token_uri': token_uri,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting NFT metadata: {e}")
            return None
    
    def get_collection_stats(self, contract_address: str) -> Optional[Dict]:
        """Get collection statistics from OpenSea"""
        if not self.opensea_api_key:
            logger.warning("OpenSea API key required for collection stats")
            return None
        
        try:
            headers = {"X-API-KEY": self.opensea_api_key}
            url = f"{self.opensea_base_url}/collection/{contract_address}/stats"
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"OpenSea API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return None
    
    def analyze_nft_transfers(self, contract_address: str, from_block: int = 0) -> Dict:
        """Analyze NFT transfer activity"""
        try:
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(contract_address),
                abi=self.erc721_abi
            )
            
            # Get transfer events
            transfer_filter = contract.events.Transfer.create_filter(
                fromBlock=from_block,
                toBlock='latest'
            )
            events = transfer_filter.get_all_entries()
            
            # Analyze transfers
            transfers = []
            holders = set()
            mint_count = 0
            
            for event in events:
                from_addr = event['args']['from']
                to_addr = event['args']['to']
                token_id = event['args']['tokenId']
                
                # Check if it's a mint (from zero address)
                if from_addr == '0x0000000000000000000000000000000000000000':
                    mint_count += 1
                
                holders.add(to_addr)
                
                transfers.append({
                    'from': from_addr,
                    'to': to_addr,
                    'token_id': token_id,
                    'transaction_hash': event['transactionHash'].hex(),
                    'block_number': event['blockNumber']
                })
            
            return {
                'contract_address': contract_address,
                'total_transfers': len(transfers),
                'total_mints': mint_count,
                'unique_holders': len(holders),
                'transfers': transfers
            }
            
        except Exception as e:
            logger.error(f"Error analyzing NFT transfers: {e}")
            return {}
    
    def check_nft_rarity(self, contract_address: str, token_id: int) -> Dict:
        """Check NFT rarity based on traits (simplified implementation)"""
        try:
            metadata = self.get_nft_metadata(contract_address, token_id)
            if not metadata or 'metadata' not in metadata:
                return {}
            
            traits = metadata['metadata'].get('attributes', [])
            if not traits:
                return {}
            
            # This is a simplified rarity calculation
            # In practice, you'd need to analyze the entire collection
            rarity_score = 0
            trait_rarities = []
            
            for trait in traits:
                trait_type = trait.get('trait_type', '')
                trait_value = trait.get('value', '')
                
                # Placeholder rarity calculation
                # In reality, you'd calculate based on frequency across collection
                estimated_rarity = 1.0 / (len(str(trait_value)) + 1)  # Simple estimation
                rarity_score += estimated_rarity
                
                trait_rarities.append({
                    'trait_type': trait_type,
                    'value': trait_value,
                    'estimated_rarity': estimated_rarity
                })
            
            return {
                'token_id': token_id,
                'total_traits': len(traits),
                'rarity_score': rarity_score,
                'trait_rarities': trait_rarities
            }
            
        except Exception as e:
            logger.error(f"Error checking NFT rarity: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
#                           8. LAYER 2 AND SCALING SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Layer2Client:
    """Layer 2 scaling solutions client"""
    
    def __init__(self, network: str = 'polygon'):
        """
        Initialize Layer 2 client
        
        Args:
            network: Layer 2 network (polygon, arbitrum, optimism, etc.)
        """
        self.network = network
        self.rpc_urls = {
            'polygon': 'https://polygon-rpc.com',
            'arbitrum': 'https://arb1.arbitrum.io/rpc',
            'optimism': 'https://mainnet.optimism.io',
            'base': 'https://mainnet.base.org',
            'avalanche': 'https://api.avax.network/ext/bc/C/rpc'
        }
        
        if network not in self.rpc_urls:
            raise ValueError(f"Unsupported network: {network}")
        
        self.w3 = Web3(HTTPProvider(self.rpc_urls[network]))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {network}")
        
        logger.info(f"Connected to {network} network")
    
    def get_gas_price(self) -> Dict:
        """Get current gas price for Layer 2"""
        try:
            gas_price = self.w3.eth.gas_price
            return {
                'network': self.network,
                'gas_price_wei': gas_price,
                'gas_price_gwei': self.w3.from_wei(gas_price, 'gwei')
            }
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            return {}
    
    def bridge_cost_estimate(self, amount_eth: float, destination: str = 'ethereum') -> Dict:
        """Estimate bridging costs (simplified)"""
        try:
            # This is a simplified estimation
            # Real bridging costs depend on many factors
            
            base_costs = {
                'polygon': {'to_ethereum': 0.001, 'from_ethereum': 0.01},
                'arbitrum': {'to_ethereum': 0.002, 'from_ethereum': 0.005},
                'optimism': {'to_ethereum': 0.002, 'from_ethereum': 0.005}
            }
            
            if self.network not in base_costs:
                return {}
            
            direction = f"to_{destination}" if destination != 'ethereum' else 'to_ethereum'
            base_cost = base_costs[self.network].get(direction, 0.001)
            
            # Estimate based on amount (simplified)
            estimated_cost = base_cost + (amount_eth * 0.001)
            
            return {
                'network': self.network,
                'destination': destination,
                'amount': amount_eth,
                'estimated_cost_eth': estimated_cost,
                'cost_percentage': (estimated_cost / amount_eth) * 100 if amount_eth > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error estimating bridge cost: {e}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
#                           9. BLOCKCHAIN MONITORING AND ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

class BlockchainMonitor:
    """Comprehensive blockchain monitoring system"""
    
    def __init__(self, ethereum_client: EthereumClient):
        self.eth_client = ethereum_client
        self.w3 = ethereum_client.w3
        self.monitors = []
        self.running = False
        
    def add_address_monitor(self, address: str, alert_threshold: float = 0):
        """Monitor an address for transactions"""
        monitor = {
            'type': 'address',
            'address': self.w3.to_checksum_address(address),
            'threshold': alert_threshold,
            'last_checked_block': self.w3.eth.block_number
        }
        self.monitors.append(monitor)
        logger.info(f"Added address monitor for {address}")
    
    def add_contract_monitor(self, contract_address: str, event_name: str):
        """Monitor a contract for specific events"""
        monitor = {
            'type': 'contract_event',
            'address': self.w3.to_checksum_address(contract_address),
            'event': event_name,
            'last_checked_block': self.w3.eth.block_number
        }
        self.monitors.append(monitor)
        logger.info(f"Added contract monitor for {contract_address} - {event_name}")
    
    def add_gas_monitor(self, threshold_gwei: float):
        """Monitor gas prices"""
        monitor = {
            'type': 'gas_price',
            'threshold': threshold_gwei
        }
        self.monitors.append(monitor)
        logger.info(f"Added gas price monitor (threshold: {threshold_gwei} Gwei)")
    
    def check_monitors(self):
        """Check all active monitors"""
        current_block = self.w3.eth.block_number
        
        for monitor in self.monitors:
            try:
                if monitor['type'] == 'address':
                    self._check_address_monitor(monitor, current_block)
                elif monitor['type'] == 'contract_event':
                    self._check_contract_monitor(monitor, current_block)
                elif monitor['type'] == 'gas_price':
                    self._check_gas_monitor(monitor)
            except Exception as e:
                logger.error(f"Error checking monitor {monitor}: {e}")
    
    def _check_address_monitor(self, monitor: Dict, current_block: int):
        """Check address monitor"""
        address = monitor['address']
        last_block = monitor['last_checked_block']
        
        # Check for new transactions
        for block_num in range(last_block + 1, current_block + 1):
            try:
                block = self.w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block['transactions']:
                    if tx['from'] == address or tx['to'] == address:
                        value_eth = self.w3.from_wei(tx['value'], 'ether')
                        
                        if value_eth >= monitor['threshold']:
                            self._trigger_alert(
                                f"Address {address} transaction: {value_eth} ETH",
                                {
                                    'type': 'address_transaction',
                                    'address': address,
                                    'transaction_hash': tx['hash'].hex(),
                                    'value': value_eth,
                                    'block': block_num
                                }
                            )
            except Exception as e:
                logger.warning(f"Error checking block {block_num}: {e}")
        
        monitor['last_checked_block'] = current_block
    
    def _check_contract_monitor(self, monitor: Dict, current_block: int):
        """Check contract event monitor"""
        # This would require the contract ABI
        # Simplified implementation
        logger.info(f"Checking contract monitor for {monitor['address']}")
        monitor['last_checked_block'] = current_block
    
    def _check_gas_monitor(self, monitor: Dict):
        """Check gas price monitor"""
        try:
            current_gas = self.w3.from_wei(self.w3.eth.gas_price, 'gwei')
            
            if current_gas >= monitor['threshold']:
                self._trigger_alert(
                    f"High gas price alert: {current_gas} Gwei",
                    {
                        'type': 'gas_price',
                        'current_price': current_gas,
                        'threshold': monitor['threshold']
                    }
                )
        except Exception as e:
            logger.error(f"Error checking gas price: {e}")
    
    def _trigger_alert(self, message: str, data: Dict):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'data': data
        }
        
        logger.info(f"ALERT: {message}")
        
        # Here you could add email, webhook, or other notification methods
        self._send_notification(alert)
    
    def _send_notification(self, alert: Dict):
        """Send notification (implement your preferred method)"""
        # Placeholder for notification sending
        # Could integrate with email, Slack, Discord, etc.
        pass
    
    def start_monitoring(self, interval: int = 30):
        """Start monitoring loop"""
        self.running = True
        logger.info(f"Starting blockchain monitoring (interval: {interval}s)")
        
        while self.running:
            try:
                self.check_monitors()
                time.sleep(interval)
            except KeyboardInterrupt:
                self.stop_monitoring()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Blockchain monitoring stopped")


# ═══════════════════════════════════════════════════════════════════════════════
#                           10. TESTING AND DEVELOPMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

import pytest
import unittest
from unittest.mock import Mock, patch

class BlockchainTestCase(unittest.TestCase):
    """Base test case for blockchain applications"""
    
    def setUp(self):
        """Set up test environment"""
        # Use a local test network or mock
        self.test_rpc_url = "http://localhost:8545"  # Ganache/Hardhat
        self.test_private_key = "0x" + "0" * 64  # Test private key
        
        try:
            self.eth_client = EthereumClient(self.test_rpc_url, self.test_private_key)
            self.connected = True
        except:
            self.connected = False
            logger.warning("No test network available, using mocks")
    
    def create_mock_web3(self):
        """Create a mock Web3 instance for testing"""
        mock_w3 = Mock()
        mock_w3.is_connected.return_value = True
        mock_w3.eth.block_number = 1000000
        mock_w3.eth.gas_price = 20000000000  # 20 Gwei
        mock_w3.eth.chain_id = 1
        return mock_w3
    
    def test_ethereum_client_connection(self):
        """Test Ethereum client connection"""
        if self.connected:
            self.assertTrue(self.eth_client.w3.is_connected())
        else:
            self.skipTest("No test network available")
    
    def test_balance_retrieval(self):
        """Test balance retrieval"""
        if self.connected:
            # Test with a known address
            test_address = "0x742d35Cc6634C0532925a3b8D0C9E785E1B85F3A"
            balance = self.eth_client.get_balance(test_address)
            self.assertIsInstance(balance, float)
            self.assertGreaterEqual(balance, 0)
        else:
            self.skipTest("No test network available")

class MockBlockchainData:
    """Mock blockchain data for testing"""
    
    @staticmethod
    def mock_transaction_data():
        """Generate mock transaction data"""
        return {
            'hash': '0x' + 'a' * 64,
            'from': '0x742d35Cc6634C0532925a3b8D0C9E785E1B85F3A',
            'to': '0x8ba1f109551bD432803012645Hac136c5a8d3De8',
            'value': 1.5,
            'gas': 21000,
            'gas_price': 20.0,
            'block_number': 1000000,
            'timestamp': datetime.now()
        }
    
    @staticmethod
    def mock_price_data():
        """Generate mock price data"""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        prices = np.random.normal(50000, 5000, len(dates))  # Mock BTC prices
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.normal(1000000, 100000, len(dates))
        })

# Test utilities
def run_blockchain_tests():
    """Run all blockchain tests"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(BlockchainTestCase)
    unittest.TextTestRunner(verbosity=2).run(test_suite)