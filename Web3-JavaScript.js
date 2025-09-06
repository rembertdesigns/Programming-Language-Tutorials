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


// ═══════════════════════════════════════════════════════════════════════════════
//                           4. DEFI PROTOCOL INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

// Uniswap V3 integration
class UniswapV3Manager extends ContractManager {
    constructor(web3Manager) {
        super(web3Manager);
        
        // Uniswap V3 contract addresses (Ethereum mainnet)
        this.addresses = {
            factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            quoter: '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
            nftManager: '0xC36442b4a4522E871399CD717aBDD847Ab11FE88'
        };
        
        // Common token addresses
        this.tokens = {
            WETH: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            USDC: '0xA0b86a33E6eA0dcdeA23e5B4a73c36e9c5b8a22F',
            USDT: '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            DAI: '0x6B175474E89094C44Da98b954EedeAC495271d0F'
        };
        
        this.initializeContracts();
    }
    
    async initializeContracts() {
        // Router ABI (simplified)
        const routerABI = [
            {
                "inputs": [
                    {
                        "components": [
                            {"name": "tokenIn", "type": "address"},
                            {"name": "tokenOut", "type": "address"},
                            {"name": "fee", "type": "uint24"},
                            {"name": "recipient", "type": "address"},
                            {"name": "deadline", "type": "uint256"},
                            {"name": "amountIn", "type": "uint256"},
                            {"name": "amountOutMinimum", "type": "uint256"},
                            {"name": "sqrtPriceLimitX96", "type": "uint160"}
                        ],
                        "name": "params",
                        "type": "tuple"
                    }
                ],
                "name": "exactInputSingle",
                "outputs": [{"name": "amountOut", "type": "uint256"}],
                "type": "function"
            }
        ];
        
        // Quoter ABI (simplified)
        const quoterABI = [
            {
                "inputs": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"}
                ],
                "name": "quoteExactInputSingle",
                "outputs": [{"name": "amountOut", "type": "uint256"}],
                "type": "function"
            }
        ];
        
        this.registerContract('UniswapRouter', this.addresses.router, routerABI);
        this.registerContract('UniswapQuoter', this.addresses.quoter, quoterABI);
    }
    
    // Get quote for token swap
    async getQuote(tokenIn, tokenOut, amountIn, fee = 3000) {
        try {
            const amountOut = await this.callMethod(
                'UniswapQuoter',
                'quoteExactInputSingle',
                tokenIn,
                tokenOut,
                fee,
                amountIn,
                0
            );
            
            return amountOut.toString();
        } catch (error) {
            console.error('Error getting Uniswap quote:', error);
            throw error;
        }
    }
    
    // Execute token swap
    async swapTokens(tokenIn, tokenOut, amountIn, amountOutMinimum, fee = 3000, deadline = null) {
        try {
            const account = this.web3Manager.account;
            if (!account) {
                throw new Error('No account connected');
            }
            
            const swapDeadline = deadline || Math.floor(Date.now() / 1000) + 1800; // 30 minutes
            
            const params = {
                tokenIn,
                tokenOut,
                fee,
                recipient: account,
                deadline: swapDeadline,
                amountIn,
                amountOutMinimum,
                sqrtPriceLimitX96: 0
            };
            
            return await this.sendTransaction('UniswapRouter', 'exactInputSingle', [params]);
        } catch (error) {
            console.error('Error executing Uniswap swap:', error);
            throw error;
        }
    }
}

// Aave lending protocol integration
class AaveManager extends ContractManager {
    constructor(web3Manager) {
        super(web3Manager);
        
        // Aave V3 contract addresses (Ethereum mainnet)
        this.addresses = {
            pool: '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
            dataProvider: '0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3'
        };
        
        this.initializeContracts();
    }
    
    async initializeContracts() {
        // Simplified Aave Pool ABI
        const poolABI = [
            {
                "inputs": [
                    {"name": "asset", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "onBehalfOf", "type": "address"},
                    {"name": "referralCode", "type": "uint16"}
                ],
                "name": "supply",
                "outputs": [],
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "asset", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "to", "type": "address"}
                ],
                "name": "withdraw",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "asset", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "interestRateMode", "type": "uint256"},
                    {"name": "referralCode", "type": "uint16"},
                    {"name": "onBehalfOf", "type": "address"}
                ],
                "name": "borrow",
                "outputs": [],
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "asset", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "rateMode", "type": "uint256"},
                    {"name": "onBehalfOf", "type": "address"}
                ],
                "name": "repay",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ];
        
        this.registerContract('AavePool', this.addresses.pool, poolABI);
    }
    
    // Supply tokens to Aave
    async supply(asset, amount) {
        try {
            const account = this.web3Manager.account;
            if (!account) {
                throw new Error('No account connected');
            }
            
            return await this.sendTransaction('AavePool', 'supply', [
                asset,
                amount,
                account,
                0 // referral code
            ]);
        } catch (error) {
            console.error('Error supplying to Aave:', error);
            throw error;
        }
    }
    
    // Withdraw tokens from Aave
    async withdraw(asset, amount) {
        try {
            const account = this.web3Manager.account;
            if (!account) {
                throw new Error('No account connected');
            }
            
            return await this.sendTransaction('AavePool', 'withdraw', [
                asset,
                amount,
                account
            ]);
        } catch (error) {
            console.error('Error withdrawing from Aave:', error);
            throw error;
        }
    }
    
    // Borrow tokens from Aave
    async borrow(asset, amount, interestRateMode = 2) {
        try {
            const account = this.web3Manager.account;
            if (!account) {
                throw new Error('No account connected');
            }
            
            return await this.sendTransaction('AavePool', 'borrow', [
                asset,
                amount,
                interestRateMode, // 1 = stable, 2 = variable
                0, // referral code
                account
            ]);
        } catch (error) {
            console.error('Error borrowing from Aave:', error);
            throw error;
        }
    }
    
    // Repay borrowed tokens
    async repay(asset, amount, rateMode = 2) {
        try {
            const account = this.web3Manager.account;
            if (!account) {
                throw new Error('No account connected');
            }
            
            return await this.sendTransaction('AavePool', 'repay', [
                asset,
                amount,
                rateMode, // 1 = stable, 2 = variable
                account
            ]);
        } catch (error) {
            console.error('Error repaying to Aave:', error);
            throw error;
        }
    }
}

// Compound protocol integration
class CompoundManager extends ContractManager {
    constructor(web3Manager) {
        super(web3Manager);
        
        // Compound token addresses (Ethereum mainnet)
        this.cTokens = {
            cETH: '0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5',
            cUSDC: '0x39AA39c021dfbaE8faC545936693aC917d5E7563',
            cDAI: '0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643'
        };
        
        this.initializeContracts();
    }
    
    async initializeContracts() {
        // Simplified cToken ABI
        const cTokenABI = [
            {
                "inputs": [],
                "name": "mint",
                "outputs": [{"name": "", "type": "uint256"}],
                "payable": true,
                "type": "function"
            },
            {
                "inputs": [{"name": "mintAmount", "type": "uint256"}],
                "name": "mint",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "redeemTokens", "type": "uint256"}],
                "name": "redeem",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "borrowAmount", "type": "uint256"}],
                "name": "borrow",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "repayAmount", "type": "uint256"}],
                "name": "repayBorrow",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ];
        
        // Register all cTokens
        Object.entries(this.cTokens).forEach(([symbol, address]) => {
            this.registerContract(symbol, address, cTokenABI);
        });
    }
    
    // Supply tokens to Compound
    async supply(cTokenSymbol, amount = null) {
        try {
            if (cTokenSymbol === 'cETH') {
                // ETH supply
                return await this.sendTransaction('cETH', 'mint', [], {
                    value: amount
                });
            } else {
                // ERC-20 token supply
                return await this.sendTransaction(cTokenSymbol, 'mint', [amount]);
            }
        } catch (error) {
            console.error('Error supplying to Compound:', error);
            throw error;
        }
    }
    
    // Redeem tokens from Compound
    async redeem(cTokenSymbol, cTokenAmount) {
        try {
            return await this.sendTransaction(cTokenSymbol, 'redeem', [cTokenAmount]);
        } catch (error) {
            console.error('Error redeeming from Compound:', error);
            throw error;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. NFT MARKETPLACE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

// NFT utilities and marketplace integration
class NFTManager extends ContractManager {
    constructor(web3Manager) {
        super(web3Manager);
        
        // OpenSea API configuration
        this.openseaConfig = {
            apiKey: process.env.REACT_APP_OPENSEA_API_KEY,
            baseURL: 'https://api.opensea.io/api/v1',
            testnetURL: 'https://testnets-api.opensea.io/api/v1'
        };
        
        // ERC-721 ABI
        this.erc721ABI = [
            {
                "inputs": [{"name": "tokenId", "type": "uint256"}],
                "name": "tokenURI",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "tokenId", "type": "uint256"}],
                "name": "ownerOf",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "tokenId", "type": "uint256"}
                ],
                "name": "transferFrom",
                "outputs": [],
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "approved", "type": "bool"}
                ],
                "name": "setApprovalForAll",
                "outputs": [],
                "type": "function"
            }
        ];
    }
    
    // Add NFT collection
    addCollection(symbol, address) {
        return this.registerContract(`NFT_${symbol}`, address, this.erc721ABI);
    }
    
    // Get NFT metadata
    async getNFTMetadata(collectionSymbol, tokenId) {
        try {
            const contractName = `NFT_${collectionSymbol}`;
            const tokenURI = await this.callMethod(contractName, 'tokenURI', tokenId);
            
            // Fetch metadata from URI
            if (tokenURI.startsWith('ipfs://')) {
                const ipfsHash = tokenURI.replace('ipfs://', '');
                const metadataURL = `https://ipfs.io/ipfs/${ipfsHash}`;
                const response = await fetch(metadataURL);
                return await response.json();
            } else if (tokenURI.startsWith('http')) {
                const response = await fetch(tokenURI);
                return await response.json();
            } else {
                throw new Error('Unsupported URI format');
            }
        } catch (error) {
            console.error('Error getting NFT metadata:', error);
            throw error;
        }
    }
    
    // Get NFT owner
    async getNFTOwner(collectionSymbol, tokenId) {
        try {
            const contractName = `NFT_${collectionSymbol}`;
            return await this.callMethod(contractName, 'ownerOf', tokenId);
        } catch (error) {
            console.error('Error getting NFT owner:', error);
            throw error;
        }
    }
    
    // Get user's NFT balance
    async getNFTBalance(collectionSymbol, address) {
        try {
            const contractName = `NFT_${collectionSymbol}`;
            const balance = await this.callMethod(contractName, 'balanceOf', address);
            return parseInt(balance);
        } catch (error) {
            console.error('Error getting NFT balance:', error);
            throw error;
        }
    }
    
    // Transfer NFT
    async transferNFT(collectionSymbol, from, to, tokenId) {
        try {
            const contractName = `NFT_${collectionSymbol}`;
            return await this.sendTransaction(contractName, 'transferFrom', [from, to, tokenId]);
        } catch (error) {
            console.error('Error transferring NFT:', error);
            throw error;
        }
    }
    
    // Approve NFT marketplace
    async approveMarketplace(collectionSymbol, marketplaceAddress) {
        try {
            const contractName = `NFT_${collectionSymbol}`;
            return await this.sendTransaction(contractName, 'setApprovalForAll', [marketplaceAddress, true]);
        } catch (error) {
            console.error('Error approving marketplace:', error);
            throw error;
        }
    }
    
    // Get collection from OpenSea
    async getOpenSeaCollection(slug) {
        try {
            const url = `${this.openseaConfig.baseURL}/collection/${slug}`;
            const response = await fetch(url, {
                headers: {
                    'X-API-KEY': this.openseaConfig.apiKey
                }
            });
            
            if (!response.ok) {
                throw new Error(`OpenSea API error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching OpenSea collection:', error);
            throw error;
        }
    }
    
    // Get NFT listings from OpenSea
    async getOpenSeaListings(contractAddress, tokenId = null) {
        try {
            let url = `${this.openseaConfig.baseURL}/orders?asset_contract_address=${contractAddress}&order_direction=desc&order_by=created_date`;
            
            if (tokenId) {
                url += `&token_id=${tokenId}`;
            }
            
            const response = await fetch(url, {
                headers: {
                    'X-API-KEY': this.openseaConfig.apiKey
                }
            });
            
            if (!response.ok) {
                throw new Error(`OpenSea API error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching OpenSea listings:', error);
            throw error;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. REACT HOOKS AND COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

// React hooks for Web3 integration
import React, { useState, useEffect, useContext, createContext, useCallback } from 'react';

// Web3 Context
const Web3Context = createContext();

// Web3 Provider Component
export const Web3Provider = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [account, setAccount] = useState(null);
    const [chainId, setChainId] = useState(null);
    const [provider, setProvider] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        // Initialize Web3Manager
        web3Manager.addListener(handleWeb3Event);
        
        return () => {
            web3Manager.removeListener(handleWeb3Event);
        };
    }, []);
    
    const handleWeb3Event = useCallback((event, data) => {
        switch (event) {
            case 'connected':
                setIsConnected(true);
                setAccount(data.account);
                setChainId(data.chainId);
                setProvider(data.provider);
                setError(null);
                break;
            case 'disconnected':
                setIsConnected(false);
                setAccount(null);
                setChainId(null);
                setProvider(null);
                break;
            case 'accountsChanged':
                setAccount(data[0] || null);
                break;
            case 'chainChanged':
                setChainId(data);
                break;
            default:
                break;
        }
        setLoading(false);
    }, []);
    
    const connect = useCallback(async (walletType = 'metamask') => {
        setLoading(true);
        setError(null);
        
        try {
            await web3Manager.connectWallet(walletType);
        } catch (error) {
            setError(error.message);
            setLoading(false);
        }
    }, []);
    
    const disconnect = useCallback(async () => {
        setLoading(true);
        try {
            await web3Manager.disconnect();
        } catch (error) {
            setError(error.message);
        }
        setLoading(false);
    }, []);
    
    const switchNetwork = useCallback(async (chainId) => {
        setLoading(true);
        setError(null);
        
        try {
            await web3Manager.switchNetwork(chainId);
        } catch (error) {
            setError(error.message);
        }
        setLoading(false);
    }, []);
    
    const value = {
        isConnected,
        account,
        chainId,
        provider,
        loading,
        error,
        connect,
        disconnect,
        switchNetwork,
        web3Manager
    };
    
    return (
        <Web3Context.Provider value={value}>
            {children}
        </Web3Context.Provider>
    );
};

// useWeb3 hook
export const useWeb3 = () => {
    const context = useContext(Web3Context);
    if (!context) {
        throw new Error('useWeb3 must be used within a Web3Provider');
    }
    return context;
};

// useContract hook
export const useContract = (contractName, address, abi) => {
    const { web3Manager } = useWeb3();
    const [contract, setContract] = useState(null);
    
    useEffect(() => {
        if (web3Manager && address && abi) {
            const contractManager = new ContractManager(web3Manager);
            const contractInstance = contractManager.registerContract(contractName, address, abi);
            setContract(contractManager);
        }
    }, [web3Manager, contractName, address, abi]);
    
    return contract;
};

// useERC20 hook
export const useERC20 = (symbol, address) => {
    const { web3Manager } = useWeb3();
    const [erc20Manager, setERC20Manager] = useState(null);
    const [tokenInfo, setTokenInfo] = useState(null);
    const [balance, setBalance] = useState(null);
    const [loading, setLoading] = useState(false);
    
    useEffect(() => {
        if (web3Manager && address) {
            const manager = new ERC20Manager(web3Manager);
            manager.addToken(symbol, address);
            setERC20Manager(manager);
        }
    }, [web3Manager, symbol, address]);
    
    const getTokenInfo = useCallback(async () => {
        if (!erc20Manager) return;
        
        setLoading(true);
        try {
            const info = await erc20Manager.getTokenInfo(symbol);
            setTokenInfo(info);
        } catch (error) {
            console.error('Error getting token info:', error);
        }
        setLoading(false);
    }, [erc20Manager, symbol]);
    
    const getBalance = useCallback(async (address) => {
        if (!erc20Manager || !address) return;
        
        setLoading(true);
        try {
            const balanceData = await erc20Manager.getBalance(symbol, address);
            setBalance(balanceData);
        } catch (error) {
            console.error('Error getting balance:', error);
        }
        setLoading(false);
    }, [erc20Manager, symbol]);
    
    const transfer = useCallback(async (to, amount) => {
        if (!erc20Manager) throw new Error('ERC20 manager not initialized');
        
        return await erc20Manager.transfer(symbol, to, amount, tokenInfo?.decimals);
    }, [erc20Manager, symbol, tokenInfo]);
    
    const approve = useCallback(async (spender, amount) => {
        if (!erc20Manager) throw new Error('ERC20 manager not initialized');
        
        return await erc20Manager.approve(symbol, spender, amount, tokenInfo?.decimals);
    }, [erc20Manager, symbol, tokenInfo]);
    
    return {
        tokenInfo,
        balance,
        loading,
        getTokenInfo,
        getBalance,
        transfer,
        approve
    };
};

// Wallet Connection Component
export const WalletConnector = () => {
    const { isConnected, account, chainId, loading, error, connect, disconnect } = useWeb3();
    const [showWalletOptions, setShowWalletOptions] = useState(false);
    
    const handleConnect = async (walletType) => {
        await connect(walletType);
        setShowWalletOptions(false);
    };
    
    const formatAddress = (address) => {
        if (!address) return '';
        return `${address.slice(0, 6)}...${address.slice(-4)}`;
    };
    
    const getNetworkName = (chainId) => {
        const networks = {
            '0x1': 'Ethereum',
            '0x89': 'Polygon',
            '0xa4b1': 'Arbitrum',
            '0xa': 'Optimism'
        };
        return networks[chainId] || 'Unknown';
    };
    
    if (loading) {
        return (
            <button disabled className="wallet-button loading">
                Connecting...
            </button>
        );
    }
    
    if (isConnected) {
        return (
            <div className="wallet-info">
                <div className="account-info">
                    <span className="network">{getNetworkName(chainId)}</span>
                    <span className="address">{formatAddress(account)}</span>
                </div>
                <button onClick={disconnect} className="disconnect-button">
                    Disconnect
                </button>
            </div>
        );
    }
    
    return (
        <div className="wallet-connector">
            {!showWalletOptions ? (
                <button 
                    onClick={() => setShowWalletOptions(true)}
                    className="connect-wallet-button"
                >
                    Connect Wallet
                </button>
            ) : (
                <div className="wallet-options">
                    <button onClick={() => handleConnect('metamask')}>
                        MetaMask
                    </button>
                    <button onClick={() => handleConnect('walletconnect')}>
                        WalletConnect
                    </button>
                    <button onClick={() => handleConnect('coinbase')}>
                        Coinbase Wallet
                    </button>
                    <button onClick={() => setShowWalletOptions(false)}>
                        Cancel
                    </button>
                </div>
            )}
            {error && <div className="error">{error}</div>}
        </div>
    );
};

// Token Balance Component
export const TokenBalance = ({ symbol, address, tokenAddress }) => {
    const { account } = useWeb3();
    const { balance, loading, getBalance } = useERC20(symbol, tokenAddress);
    
    useEffect(() => {
        if (account) {
            getBalance(address || account);
        }
    }, [account, address, getBalance]);
    
    if (loading) {
        return <div className="token-balance loading">Loading...</div>;
    }
    
    return (
        <div className="token-balance">
            <span className="amount">{balance?.formatted || '0'}</span>
            <span className="symbol">{symbol}</span>
        </div>
    );
};

// Transaction Component
export const TransactionButton = ({ 
    children, 
    onClick, 
    disabled, 
    requiresConnection = true,
    className = '' 
}) => {
    const { isConnected, loading } = useWeb3();
    const [txLoading, setTxLoading] = useState(false);
    const [txError, setTxError] = useState(null);
    const [txSuccess, setTxSuccess] = useState(false);
    
    const handleClick = async () => {
        if (requiresConnection && !isConnected) {
            setTxError('Please connect your wallet');
            return;
        }
        
        setTxLoading(true);
        setTxError(null);
        setTxSuccess(false);
        
        try {
            await onClick();
            setTxSuccess(true);
            setTimeout(() => setTxSuccess(false), 3000);
        } catch (error) {
            setTxError(error.message);
            setTimeout(() => setTxError(null), 5000);
        }
        
        setTxLoading(false);
    };
    
    const isDisabled = disabled || loading || txLoading || (requiresConnection && !isConnected);
    
    return (
        <div className="transaction-button-container">
            <button 
                onClick={handleClick}
                disabled={isDisabled}
                className={`transaction-button ${className} ${txSuccess ? 'success' : ''}`}
            >
                {txLoading ? 'Processing...' : children}
            </button>
            {txError && <div className="tx-error">{txError}</div>}
            {txSuccess && <div className="tx-success">Transaction successful!</div>}
        </div>
    );
};

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. TRANSACTION MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

// Transaction manager for handling complex transactions
class TransactionManager {
    constructor(web3Manager) {
        this.web3Manager = web3Manager;
        this.pendingTransactions = new Map();
        this.transactionHistory = [];
        this.listeners = [];
    }
    
    // Send transaction with detailed tracking
    async sendTransaction(txData, options = {}) {
        const txId = this.generateTxId();
        
        try {
            const web3 = this.web3Manager.getWeb3Instance();
            const account = this.web3Manager.account;
            
            if (!account) {
                throw new Error('No account connected');
            }
            
            // Prepare transaction
            const transaction = {
                from: account,
                gas: options.gasLimit || await this.estimateGas(txData),
                gasPrice: options.gasPrice || await web3.eth.getGasPrice(),
                ...txData
            };
            
            // Add to pending transactions
            this.pendingTransactions.set(txId, {
                id: txId,
                status: 'pending',
                transaction,
                timestamp: Date.now(),
                confirmations: 0
            });
            
            this.notifyListeners('transactionPending', { id: txId, transaction });
            
            // Send transaction
            const receipt = await web3.eth.sendTransaction(transaction);
            
            // Update status
            this.updateTransactionStatus(txId, 'confirmed', receipt);
            
            // Wait for confirmations
            this.waitForConfirmations(txId, receipt.transactionHash, options.confirmations || 1);
            
            return {
                id: txId,
                hash: receipt.transactionHash,
                receipt
            };
            
        } catch (error) {
            this.updateTransactionStatus(txId, 'failed', null, error);
            throw error;
        }
    }
    
    // Estimate gas for transaction
    async estimateGas(txData) {
        try {
            const web3 = this.web3Manager.getWeb3Instance();
            const gasEstimate = await web3.eth.estimateGas(txData);
            return Math.floor(gasEstimate * 1.1); // Add 10% buffer
        } catch (error) {
            console.warn('Gas estimation failed, using default:', error);
            return 100000; // Default gas limit
        }
    }
    
    // Wait for transaction confirmations
    async waitForConfirmations(txId, txHash, requiredConfirmations) {
        const web3 = this.web3Manager.getWeb3Instance();
        
        const checkConfirmations = async () => {
            try {
                const receipt = await web3.eth.getTransactionReceipt(txHash);
                if (!receipt) return;
                
                const currentBlock = await web3.eth.getBlockNumber();
                const confirmations = currentBlock - receipt.blockNumber;
                
                // Update transaction with confirmation count
                const tx = this.pendingTransactions.get(txId);
                if (tx) {
                    tx.confirmations = confirmations;
                    this.notifyListeners('transactionConfirmation', { 
                        id: txId, 
                        confirmations,
                        requiredConfirmations 
                    });
                    
                    if (confirmations >= requiredConfirmations) {
                        this.updateTransactionStatus(txId, 'finalized', receipt);
                        return;
                    }
                }
                
                // Check again after next block
                setTimeout(checkConfirmations, 15000); // ~15 second block time
                
            } catch (error) {
                console.error('Error checking confirmations:', error);
                setTimeout(checkConfirmations, 15000);
            }
        };
        
        checkConfirmations();
    }
    
    // Update transaction status
    updateTransactionStatus(txId, status, receipt = null, error = null) {
        const tx = this.pendingTransactions.get(txId);
        if (!tx) return;
        
        tx.status = status;
        tx.receipt = receipt;
        tx.error = error;
        tx.updatedAt = Date.now();
        
        if (status === 'finalized' || status === 'failed') {
            // Move to history
            this.transactionHistory.push({ ...tx });
            this.pendingTransactions.delete(txId);
        }
        
        this.notifyListeners('transactionStatusChanged', { id: txId, status, receipt, error });
    }
    
    // Get transaction status
    getTransactionStatus(txId) {
        return this.pendingTransactions.get(txId) || 
               this.transactionHistory.find(tx => tx.id === txId);
    }
    
    // Get all pending transactions
    getPendingTransactions() {
        return Array.from(this.pendingTransactions.values());
    }
    
    // Get transaction history
    getTransactionHistory(limit = 50) {
        return this.transactionHistory
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, limit);
    }
    
    // Generate unique transaction ID
    generateTxId() {
        return `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    // Add listener for transaction events
    addListener(callback) {
        this.listeners.push(callback);
    }
    
    // Remove listener
    removeListener(callback) {
        this.listeners = this.listeners.filter(listener => listener !== callback);
    }
    
    // Notify listeners
    notifyListeners(event, data) {
        this.listeners.forEach(callback => {
            try {
                callback(event, data);
            } catch (error) {
                console.error('Transaction listener error:', error);
            }
        });
    }
}

// React hook for transaction management
export const useTransactionManager = () => {
    const { web3Manager } = useWeb3();
    const [txManager] = useState(() => new TransactionManager(web3Manager));
    const [pendingTxs, setPendingTxs] = useState([]);
    const [txHistory, setTxHistory] = useState([]);
    
    useEffect(() => {
        const handleTxEvent = (event, data) => {
            switch (event) {
                case 'transactionPending':
                case 'transactionConfirmation':
                case 'transactionStatusChanged':
                    setPendingTxs(txManager.getPendingTransactions());
                    setTxHistory(txManager.getTransactionHistory());
                    break;
            }
        };
        
        txManager.addListener(handleTxEvent);
        
        return () => {
            txManager.removeListener(handleTxEvent);
        };
    }, [txManager]);
    
    const sendTransaction = useCallback(async (txData, options) => {
        return await txManager.sendTransaction(txData, options);
    }, [txManager]);
    
    return {
        sendTransaction,
        pendingTransactions: pendingTxs,
        transactionHistory: txHistory,
        getTransactionStatus: txManager.getTransactionStatus.bind(txManager)
    };
};