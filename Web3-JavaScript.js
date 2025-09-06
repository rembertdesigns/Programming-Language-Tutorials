// WEB3 JAVASCRIPT - Comprehensive Frontend/dApp Development Reference - by Richard Rembert
// JavaScript is the essential language for Web3 frontend development, powering dApps,
// wallet integrations, and blockchain user interfaces across all major networks

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════════════════

/*
WEB3 JAVASCRIPT DEVELOPMENT SETUP:

1. Node.js and npm/yarn:
   - Install Node.js 18+ (LTS recommended)
   - Use npm or yarn for package management
   - Consider using nvm for Node version management

2. Essential Web3 libraries:
   # Core Web3 libraries
   npm install web3@latest ethers@^6.0.0
   npm install @web3-react/core @web3-react/injected-connector
   npm install wagmi viem
   
   # Wallet connectors
   npm install @walletconnect/web3-provider
   npm install @coinbase/wallet-sdk
   npm install @metamask/sdk
   
   # Framework integrations
   npm install react react-dom next.js
   npm install vue@next nuxt3
   npm install svelte @sveltejs/kit
   
   # UI libraries
   npm install @rainbow-me/rainbowkit
   npm install @chakra-ui/react
   npm install tailwindcss
   
   # Utility libraries
   npm install axios
   npm install lodash
   npm install moment
   npm install big.js bignumber.js
   npm install crypto-js
   
   # Development tools
   npm install --save-dev hardhat @nomiclabs/hardhat-ethers
   npm install --save-dev @testing-library/react jest
   npm install --save-dev eslint prettier

3. Development environment:
   # Create new project
   npx create-react-app my-dapp
   # or
   npx create-next-app@latest my-dapp
   # or
   npm create vue@latest my-dapp

4. Browser extensions for testing:
   - MetaMask
   - Coinbase Wallet
   - WalletConnect
   - Rainbow Wallet

5. Testing networks:
   - Ethereum Goerli/Sepolia testnet
   - Polygon Mumbai testnet
   - Arbitrum Goerli
   - Optimism Goerli
*/

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. CORE WEB3 INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

// Web3 Provider Management
class Web3Manager {
    constructor() {
        this.web3 = null;
        this.provider = null;
        this.account = null;
        this.chainId = null;
        this.isConnected = false;
        this.listeners = [];
        
        this.init();
    }
    
    async init() {
        // Check if Web3 is already injected
        if (typeof window !== 'undefined' && window.ethereum) {
            this.provider = window.ethereum;
            await this.setupEventListeners();
            
            // Check if already connected
            const accounts = await this.provider.request({ method: 'eth_accounts' });
            if (accounts.length > 0) {
                this.account = accounts[0];
                this.isConnected = true;
                this.chainId = await this.provider.request({ method: 'eth_chainId' });
                this.notifyListeners('accountsChanged', accounts);
            }
        }
    }
    
    async connectWallet(walletType = 'metamask') {
        try {
            switch (walletType) {
                case 'metamask':
                    return await this.connectMetaMask();
                case 'walletconnect':
                    return await this.connectWalletConnect();
                case 'coinbase':
                    return await this.connectCoinbase();
                default:
                    throw new Error(`Unsupported wallet type: ${walletType}`);
            }
        } catch (error) {
            console.error('Wallet connection failed:', error);
            throw error;
        }
    }
    
    async connectMetaMask() {
        if (!window.ethereum || !window.ethereum.isMetaMask) {
            throw new Error('MetaMask not detected');
        }
        
        try {
            const accounts = await window.ethereum.request({
                method: 'eth_requestAccounts'
            });
            
            this.provider = window.ethereum;
            this.account = accounts[0];
            this.isConnected = true;
            this.chainId = await this.provider.request({ method: 'eth_chainId' });
            
            await this.setupEventListeners();
            this.notifyListeners('connected', { account: this.account, chainId: this.chainId });
            
            return {
                account: this.account,
                chainId: this.chainId,
                provider: this.provider
            };
        } catch (error) {
            throw new Error(`MetaMask connection failed: ${error.message}`);
        }
    }
    
    async connectWalletConnect() {
        const WalletConnectProvider = (await import('@walletconnect/web3-provider')).default;
        
        const provider = new WalletConnectProvider({
            infuraId: process.env.REACT_APP_INFURA_ID,
            rpc: {
                1: `https://mainnet.infura.io/v3/${process.env.REACT_APP_INFURA_ID}`,
                5: `https://goerli.infura.io/v3/${process.env.REACT_APP_INFURA_ID}`,
            }
        });
        
        try {
            await provider.enable();
            
            this.provider = provider;
            this.account = provider.accounts[0];
            this.isConnected = true;
            this.chainId = provider.chainId;
            
            this.notifyListeners('connected', { account: this.account, chainId: this.chainId });
            
            return {
                account: this.account,
                chainId: this.chainId,
                provider: this.provider
            };
        } catch (error) {
            throw new Error(`WalletConnect connection failed: ${error.message}`);
        }
    }
    
    async connectCoinbase() {
        const { CoinbaseWalletSDK } = await import('@coinbase/wallet-sdk');
        
        const coinbaseWallet = new CoinbaseWalletSDK({
            appName: 'My dApp',
            appLogoUrl: 'https://example.com/logo.png',
            darkMode: false
        });
        
        const provider = coinbaseWallet.makeWeb3Provider(
            `https://mainnet.infura.io/v3/${process.env.REACT_APP_INFURA_ID}`,
            1
        );
        
        try {
            const accounts = await provider.request({
                method: 'eth_requestAccounts'
            });
            
            this.provider = provider;
            this.account = accounts[0];
            this.isConnected = true;
            this.chainId = await provider.request({ method: 'eth_chainId' });
            
            this.notifyListeners('connected', { account: this.account, chainId: this.chainId });
            
            return {
                account: this.account,
                chainId: this.chainId,
                provider: this.provider
            };
        } catch (error) {
            throw new Error(`Coinbase Wallet connection failed: ${error.message}`);
        }
    }
    
    async disconnect() {
        if (this.provider && this.provider.disconnect) {
            await this.provider.disconnect();
        }
        
        this.provider = null;
        this.account = null;
        this.isConnected = false;
        this.chainId = null;
        
        this.notifyListeners('disconnected', {});
    }
    
    async setupEventListeners() {
        if (!this.provider) return;
        
        this.provider.on('accountsChanged', (accounts) => {
            if (accounts.length === 0) {
                this.disconnect();
            } else {
                this.account = accounts[0];
                this.notifyListeners('accountsChanged', accounts);
            }
        });
        
        this.provider.on('chainChanged', (chainId) => {
            this.chainId = chainId;
            this.notifyListeners('chainChanged', chainId);
        });
        
        this.provider.on('disconnect', () => {
            this.disconnect();
        });
    }
    
    async switchNetwork(chainId) {
        if (!this.provider) {
            throw new Error('No provider connected');
        }
        
        try {
            await this.provider.request({
                method: 'wallet_switchEthereumChain',
                params: [{ chainId }],
            });
        } catch (switchError) {
            // Chain not added, try to add it
            if (switchError.code === 4902) {
                const networkConfig = this.getNetworkConfig(chainId);
                if (networkConfig) {
                    await this.provider.request({
                        method: 'wallet_addEthereumChain',
                        params: [networkConfig],
                    });
                }
            } else {
                throw switchError;
            }
        }
    }
    
    getNetworkConfig(chainId) {
        const networks = {
            '0x1': { // Ethereum Mainnet
                chainName: 'Ethereum Mainnet',
                nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
                rpcUrls: ['https://mainnet.infura.io/v3/YOUR_INFURA_ID'],
                blockExplorerUrls: ['https://etherscan.io/']
            },
            '0x89': { // Polygon Mainnet
                chainName: 'Polygon Mainnet',
                nativeCurrency: { name: 'MATIC', symbol: 'MATIC', decimals: 18 },
                rpcUrls: ['https://polygon-rpc.com/'],
                blockExplorerUrls: ['https://polygonscan.com/']
            },
            '0xa4b1': { // Arbitrum One
                chainName: 'Arbitrum One',
                nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
                rpcUrls: ['https://arb1.arbitrum.io/rpc'],
                blockExplorerUrls: ['https://arbiscan.io/']
            }
        };
        
        return networks[chainId];
    }
    
    addListener(callback) {
        this.listeners.push(callback);
    }
    
    removeListener(callback) {
        this.listeners = this.listeners.filter(listener => listener !== callback);
    }
    
    notifyListeners(event, data) {
        this.listeners.forEach(callback => {
            try {
                callback(event, data);
            } catch (error) {
                console.error('Listener error:', error);
            }
        });
    }
    
    getWeb3Instance() {
        if (!this.provider) {
            throw new Error('No provider connected');
        }
        
        if (!this.web3) {
            const Web3 = require('web3');
            this.web3 = new Web3(this.provider);
        }
        
        return this.web3;
    }
    
    getEthersProvider() {
        if (!this.provider) {
            throw new Error('No provider connected');
        }
        
        const { ethers } = require('ethers');
        return new ethers.providers.Web3Provider(this.provider);
    }
}

// Singleton instance
const web3Manager = new Web3Manager();

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. SMART CONTRACT INTERACTION
// ═══════════════════════════════════════════════════════════════════════════════

// Contract interaction utilities
class ContractManager {
    constructor(web3Manager) {
        this.web3Manager = web3Manager;
        this.contracts = new Map();
        this.abis = new Map();
    }
    
    // Register contract ABI
    registerContract(name, address, abi) {
        this.abis.set(name, { address, abi });
        
        // Create contract instance
        const web3 = this.web3Manager.getWeb3Instance();
        const contract = new web3.eth.Contract(abi, address);
        this.contracts.set(name, contract);
        
        return contract;
    }
    
    // Get contract instance
    getContract(name) {
        if (!this.contracts.has(name)) {
            throw new Error(`Contract ${name} not registered`);
        }
        return this.contracts.get(name);
    }
    
    // Call read-only contract method
    async callMethod(contractName, methodName, ...args) {
        try {
            const contract = this.getContract(contractName);
            const result = await contract.methods[methodName](...args).call();
            return result;
        } catch (error) {
            console.error(`Error calling ${contractName}.${methodName}:`, error);
            throw error;
        }
    }
    
    // Send transaction to contract
    async sendTransaction(contractName, methodName, args = [], options = {}) {
        try {
            const contract = this.getContract(contractName);
            const account = this.web3Manager.account;
            
            if (!account) {
                throw new Error('No account connected');
            }
            
            const method = contract.methods[methodName](...args);
            
            // Estimate gas
            const gasEstimate = await method.estimateGas({ from: account });
            const gasPrice = await this.web3Manager.getWeb3Instance().eth.getGasPrice();
            
            const txOptions = {
                from: account,
                gas: Math.floor(gasEstimate * 1.1), // Add 10% buffer
                gasPrice: gasPrice,
                ...options
            };
            
            // Send transaction
            const receipt = await method.send(txOptions);
            
            return {
                transactionHash: receipt.transactionHash,
                blockNumber: receipt.blockNumber,
                gasUsed: receipt.gasUsed,
                receipt
            };
        } catch (error) {
            console.error(`Error sending transaction to ${contractName}.${methodName}:`, error);
            throw error;
        }
    }
    
    // Listen to contract events
    addEventListener(contractName, eventName, callback, filter = {}) {
        try {
            const contract = this.getContract(contractName);
            
            const eventListener = contract.events[eventName](filter)
                .on('data', callback)
                .on('error', (error) => {
                    console.error(`Event listener error for ${contractName}.${eventName}:`, error);
                });
            
            return eventListener;
        } catch (error) {
            console.error(`Error setting up event listener for ${contractName}.${eventName}:`, error);
            throw error;
        }
    }
    
    // Get past events
    async getPastEvents(contractName, eventName, filter = {}) {
        try {
            const contract = this.getContract(contractName);
            const events = await contract.getPastEvents(eventName, {
                fromBlock: filter.fromBlock || 0,
                toBlock: filter.toBlock || 'latest',
                filter: filter.filter || {}
            });
            
            return events;
        } catch (error) {
            console.error(`Error getting past events for ${contractName}.${eventName}:`, error);
            throw error;
        }
    }
}

// ERC-20 Token utilities
class ERC20Manager extends ContractManager {
    constructor(web3Manager) {
        super(web3Manager);
        
        // Standard ERC-20 ABI
        this.erc20ABI = [
            {
                "constant": true,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": false,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": false,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ];
    }
    
    // Add ERC-20 token
    addToken(symbol, address) {
        return this.registerContract(`ERC20_${symbol}`, address, this.erc20ABI);
    }
    
    // Get token info
    async getTokenInfo(symbol) {
        const contractName = `ERC20_${symbol}`;
        
        try {
            const [name, tokenSymbol, decimals, totalSupply] = await Promise.all([
                this.callMethod(contractName, 'name'),
                this.callMethod(contractName, 'symbol'),
                this.callMethod(contractName, 'decimals'),
                this.callMethod(contractName, 'totalSupply')
            ]);
            
            return {
                name,
                symbol: tokenSymbol,
                decimals: parseInt(decimals),
                totalSupply: totalSupply.toString()
            };
        } catch (error) {
            console.error(`Error getting token info for ${symbol}:`, error);
            throw error;
        }
    }
    
    // Get token balance
    async getBalance(symbol, address) {
        const contractName = `ERC20_${symbol}`;
        
        try {
            const balance = await this.callMethod(contractName, 'balanceOf', address);
            const decimals = await this.callMethod(contractName, 'decimals');
            
            return {
                raw: balance.toString(),
                formatted: this.formatTokenAmount(balance, parseInt(decimals))
            };
        } catch (error) {
            console.error(`Error getting balance for ${symbol}:`, error);
            throw error;
        }
    }
    
    // Transfer tokens
    async transfer(symbol, to, amount, decimals = 18) {
        const contractName = `ERC20_${symbol}`;
        const amountWei = this.parseTokenAmount(amount, decimals);
        
        try {
            return await this.sendTransaction(contractName, 'transfer', [to, amountWei]);
        } catch (error) {
            console.error(`Error transferring ${symbol}:`, error);
            throw error;
        }
    }
    
    // Approve tokens
    async approve(symbol, spender, amount, decimals = 18) {
        const contractName = `ERC20_${symbol}`;
        const amountWei = this.parseTokenAmount(amount, decimals);
        
        try {
            return await this.sendTransaction(contractName, 'approve', [spender, amountWei]);
        } catch (error) {
            console.error(`Error approving ${symbol}:`, error);
            throw error;
        }
    }
    
    // Get allowance
    async getAllowance(symbol, owner, spender) {
        const contractName = `ERC20_${symbol}`;
        
        try {
            const allowance = await this.callMethod(contractName, 'allowance', owner, spender);
            const decimals = await this.callMethod(contractName, 'decimals');
            
            return {
                raw: allowance.toString(),
                formatted: this.formatTokenAmount(allowance, parseInt(decimals))
            };
        } catch (error) {
            console.error(`Error getting allowance for ${symbol}:`, error);
            throw error;
        }
    }
    
    // Utility functions
    parseTokenAmount(amount, decimals) {
        const BigNumber = require('bignumber.js');
        return new BigNumber(amount).multipliedBy(new BigNumber(10).pow(decimals)).toString();
    }
    
    formatTokenAmount(amount, decimals) {
        const BigNumber = require('bignumber.js');
        return new BigNumber(amount).dividedBy(new BigNumber(10).pow(decimals)).toString();
    }
}