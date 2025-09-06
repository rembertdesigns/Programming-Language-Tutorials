// RUST BLOCKCHAIN - Comprehensive Development Reference - by Richard Rembert
// Rust is the leading language for high-performance blockchain development,
// powering Solana, Polkadot, NEAR, and many Layer 2 solutions

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════════════════

/*
RUST BLOCKCHAIN DEVELOPMENT SETUP:

1. Install Rust and Cargo:
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   rustc --version

2. Install Solana CLI:
   sh -c "$(curl -sSfL https://release.solana.com/v1.17.0/install)"
   export PATH="/home/user/.local/share/solana/install/active_release/bin:$PATH"
   solana --version

3. Essential Rust crates for blockchain development:
   [dependencies]
   solana-program = "~1.17.0"
   solana-sdk = "~1.17.0"
   solana-client = "~1.17.0"
   anchor-lang = "0.29.0"
   anchor-spl = "0.29.0"
   spl-token = "4.0.0"
   borsh = "0.10.3"
   serde = { version = "1.0", features = ["derive"] }
   tokio = { version = "1.0", features = ["full"] }
   anyhow = "1.0"
   thiserror = "1.0"
   clap = { version = "4.0", features = ["derive"] }
   reqwest = { version = "0.11", features = ["json"] }
   base64 = "0.21"
   bs58 = "0.4"
   sha2 = "0.10"
   rand = "0.8"
   hex = "0.4"
   
4. Development tools:
   cargo install cargo-expand    # Macro expansion
   cargo install cargo-audit     # Security auditing
   cargo install cargo-criterion # Benchmarking
   cargo install solana-cli      # Solana toolchain
   
5. Anchor Framework (recommended for Solana):
   cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
   avm install latest
   avm use latest
   anchor --version

6. Create new project structure:
   anchor init my_blockchain_project
   cd my_blockchain_project
   anchor build
   anchor test
*/

use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    system_instruction,
    sysvar::Sysvar,
};

use solana_sdk::{
    signature::{Keypair, Signer},
    transaction::Transaction,
    commitment_config::CommitmentConfig,
    rpc_client::RpcClient,
};

use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use anyhow::{Result, anyhow};
use tokio;

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. CORE BLOCKCHAIN PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════════

/// Core blockchain data structures and cryptographic primitives
pub mod primitives {
    use super::*;
    use sha2::{Sha256, Digest};
    use std::fmt;

    /// Transaction structure for blockchain operations
    #[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
    pub struct BlockchainTransaction {
        pub from: Pubkey,
        pub to: Pubkey,
        pub amount: u64,
        pub fee: u64,
        pub nonce: u64,
        pub timestamp: i64,
        pub signature: Option<String>,
    }

    impl BlockchainTransaction {
        pub fn new(from: Pubkey, to: Pubkey, amount: u64, fee: u64, nonce: u64) -> Self {
            Self {
                from,
                to,
                amount,
                fee,
                nonce,
                timestamp: chrono::Utc::now().timestamp(),
                signature: None,
            }
        }

        pub fn hash(&self) -> String {
            let mut hasher = Sha256::new();
            hasher.update(self.from.to_bytes());
            hasher.update(self.to.to_bytes());
            hasher.update(&self.amount.to_le_bytes());
            hasher.update(&self.fee.to_le_bytes());
            hasher.update(&self.nonce.to_le_bytes());
            hasher.update(&self.timestamp.to_le_bytes());
            
            hex::encode(hasher.finalize())
        }

        pub fn verify_signature(&self) -> bool {
            // Signature verification logic would go here
            // For now, we'll assume it's valid if signature exists
            self.signature.is_some()
        }
    }

    /// Block structure containing multiple transactions
    #[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
    pub struct Block {
        pub index: u64,
        pub timestamp: i64,
        pub previous_hash: String,
        pub merkle_root: String,
        pub nonce: u64,
        pub difficulty: u32,
        pub transactions: Vec<BlockchainTransaction>,
    }

    impl Block {
        pub fn new(index: u64, previous_hash: String, transactions: Vec<BlockchainTransaction>) -> Self {
            let timestamp = chrono::Utc::now().timestamp();
            let merkle_root = Self::calculate_merkle_root(&transactions);
            
            Self {
                index,
                timestamp,
                previous_hash,
                merkle_root,
                nonce: 0,
                difficulty: 4, // Default difficulty
                transactions,
            }
        }

        pub fn hash(&self) -> String {
            let mut hasher = Sha256::new();
            hasher.update(&self.index.to_le_bytes());
            hasher.update(&self.timestamp.to_le_bytes());
            hasher.update(self.previous_hash.as_bytes());
            hasher.update(self.merkle_root.as_bytes());
            hasher.update(&self.nonce.to_le_bytes());
            hasher.update(&self.difficulty.to_le_bytes());
            
            hex::encode(hasher.finalize())
        }

        pub fn mine_block(&mut self) {
            let target = "0".repeat(self.difficulty as usize);
            
            loop {
                let hash = self.hash();
                if hash.starts_with(&target) {
                    println!("Block mined: {}", hash);
                    break;
                }
                self.nonce += 1;
            }
        }

        fn calculate_merkle_root(transactions: &[BlockchainTransaction]) -> String {
            if transactions.is_empty() {
                return String::new();
            }

            let mut hashes: Vec<String> = transactions.iter()
                .map(|tx| tx.hash())
                .collect();

            while hashes.len() > 1 {
                let mut next_level = Vec::new();
                
                for chunk in hashes.chunks(2) {
                    let combined = if chunk.len() == 2 {
                        format!("{}{}", chunk[0], chunk[1])
                    } else {
                        format!("{}{}", chunk[0], chunk[0])
                    };
                    
                    let mut hasher = Sha256::new();
                    hasher.update(combined.as_bytes());
                    next_level.push(hex::encode(hasher.finalize()));
                }
                
                hashes = next_level;
            }

            hashes[0].clone()
        }
    }

    /// Blockchain state management
    #[derive(Debug, Clone)]
    pub struct Blockchain {
        pub chain: Vec<Block>,
        pub difficulty: u32,
        pub pending_transactions: Vec<BlockchainTransaction>,
        pub mining_reward: u64,
    }

    impl Blockchain {
        pub fn new() -> Self {
            let mut blockchain = Self {
                chain: Vec::new(),
                difficulty: 4,
                pending_transactions: Vec::new(),
                mining_reward: 100,
            };
            
            // Create genesis block
            blockchain.create_genesis_block();
            blockchain
        }

        fn create_genesis_block(&mut self) {
            let genesis_block = Block::new(0, "0".to_string(), Vec::new());
            self.chain.push(genesis_block);
        }

        pub fn get_latest_block(&self) -> &Block {
            self.chain.last().unwrap()
        }

        pub fn add_transaction(&mut self, transaction: BlockchainTransaction) -> Result<()> {
            if !transaction.verify_signature() {
                return Err(anyhow!("Invalid transaction signature"));
            }
            
            self.pending_transactions.push(transaction);
            Ok(())
        }

        pub fn mine_pending_transactions(&mut self, mining_reward_address: Pubkey) {
            // Add mining reward transaction
            let reward_tx = BlockchainTransaction::new(
                Pubkey::default(), // From system
                mining_reward_address,
                self.mining_reward,
                0,
                0,
            );
            self.pending_transactions.push(reward_tx);

            let block = Block::new(
                self.chain.len() as u64,
                self.get_latest_block().hash(),
                self.pending_transactions.clone(),
            );

            self.chain.push(block);
            self.pending_transactions.clear();
        }

        pub fn get_balance(&self, address: &Pubkey) -> u64 {
            let mut balance = 0;

            for block in &self.chain {
                for tx in &block.transactions {
                    if tx.from == *address {
                        balance = balance.saturating_sub(tx.amount + tx.fee);
                    }
                    if tx.to == *address {
                        balance += tx.amount;
                    }
                }
            }

            balance
        }

        pub fn is_chain_valid(&self) -> bool {
            for i in 1..self.chain.len() {
                let current_block = &self.chain[i];
                let previous_block = &self.chain[i - 1];

                if current_block.hash() != current_block.hash() {
                    return false;
                }

                if current_block.previous_hash != previous_block.hash() {
                    return false;
                }
            }
            true
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. SOLANA PROGRAM DEVELOPMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Solana program (smart contract) development
pub mod solana_programs {
    use super::*;

    /// Custom error types for Solana programs
    #[derive(Debug, Clone, thiserror::Error)]
    pub enum CustomError {
        #[error("Invalid instruction")]
        InvalidInstruction,
        #[error("Invalid account")]
        InvalidAccount,
        #[error("Insufficient funds")]
        InsufficientFunds,
        #[error("Unauthorized")]
        Unauthorized,
        #[error("Account already initialized")]
        AccountAlreadyInitialized,
        #[error("Account not initialized")]
        AccountNotInitialized,
    }

    impl From<CustomError> for ProgramError {
        fn from(e: CustomError) -> Self {
            ProgramError::Custom(e as u32)
        }
    }

    /// Program instructions enum
    #[derive(Debug, BorshSerialize, BorshDeserialize)]
    pub enum Instruction {
        /// Initialize a new account
        /// Accounts: [signer, writable] user, [writable] new_account, [] system_program
        Initialize { initial_value: u64 },
        
        /// Transfer tokens between accounts
        /// Accounts: [signer, writable] from, [writable] to
        Transfer { amount: u64 },
        
        /// Update account data
        /// Accounts: [signer, writable] account
        Update { new_value: u64 },
        
        /// Close account and reclaim rent
        /// Accounts: [signer, writable] account, [writable] destination
        Close,
    }

    /// Account data structure
    #[derive(Debug, BorshSerialize, BorshDeserialize)]
    pub struct AccountData {
        pub owner: Pubkey,
        pub balance: u64,
        pub data: Vec<u8>,
        pub is_initialized: bool,
    }

    impl AccountData {
        pub const LEN: usize = 32 + 8 + 4 + 1000 + 1; // pubkey + u64 + vec len + data + bool

        pub fn new(owner: Pubkey, balance: u64) -> Self {
            Self {
                owner,
                balance,
                data: Vec::new(),
                is_initialized: true,
            }
        }
    }

    /// Token account structure
    #[derive(Debug, BorshSerialize, BorshDeserialize)]
    pub struct TokenAccount {
        pub mint: Pubkey,
        pub owner: Pubkey,
        pub amount: u64,
        pub delegate: Option<Pubkey>,
        pub state: u8, // 0 = uninitialized, 1 = initialized, 2 = frozen
        pub is_native: bool,
        pub delegated_amount: u64,
        pub close_authority: Option<Pubkey>,
    }

    impl TokenAccount {
        pub const LEN: usize = 165; // Standard SPL token account size
    }

    /// Main program entry point
    entrypoint!(process_instruction);

    pub fn process_instruction(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        instruction_data: &[u8],
    ) -> ProgramResult {
        let instruction = Instruction::try_from_slice(instruction_data)
            .map_err(|_| CustomError::InvalidInstruction)?;

        match instruction {
            Instruction::Initialize { initial_value } => {
                process_initialize(program_id, accounts, initial_value)
            }
            Instruction::Transfer { amount } => {
                process_transfer(program_id, accounts, amount)
            }
            Instruction::Update { new_value } => {
                process_update(program_id, accounts, new_value)
            }
            Instruction::Close => {
                process_close(program_id, accounts)
            }
        }
    }

    fn process_initialize(
        program_id: &Pubkey,
        accounts: &[AccountInfo],
        initial_value: u64,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let user_account = next_account_info(account_info_iter)?;
        let new_account = next_account_info(account_info_iter)?;
        let system_program = next_account_info(account_info_iter)?;

        // Verify user is signer
        if !user_account.is_signer {
            return Err(CustomError::Unauthorized.into());
        }

        // Verify account ownership
        if new_account.owner != program_id {
            return Err(CustomError::InvalidAccount.into());
        }

        // Check if account is already initialized
        if new_account.data_len() > 0 {
            let account_data = AccountData::try_from_slice(&new_account.data.borrow())?;
            if account_data.is_initialized {
                return Err(CustomError::AccountAlreadyInitialized.into());
            }
        }

        // Initialize account data
        let account_data = AccountData::new(*user_account.key, initial_value);
        account_data.serialize(&mut &mut new_account.data.borrow_mut()[..])?;

        msg!("Account initialized with value: {}", initial_value);
        Ok(())
    }

    fn process_transfer(
        _program_id: &Pubkey,
        accounts: &[AccountInfo],
        amount: u64,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let from_account = next_account_info(account_info_iter)?;
        let to_account = next_account_info(account_info_iter)?;

        // Verify from_account is signer
        if !from_account.is_signer {
            return Err(CustomError::Unauthorized.into());
        }

        // Deserialize account data
        let mut from_data = AccountData::try_from_slice(&from_account.data.borrow())?;
        let mut to_data = AccountData::try_from_slice(&to_account.data.borrow())?;

        // Check if accounts are initialized
        if !from_data.is_initialized || !to_data.is_initialized {
            return Err(CustomError::AccountNotInitialized.into());
        }

        // Check sufficient balance
        if from_data.balance < amount {
            return Err(CustomError::InsufficientFunds.into());
        }

        // Perform transfer
        from_data.balance -= amount;
        to_data.balance += amount;

        // Serialize updated data
        from_data.serialize(&mut &mut from_account.data.borrow_mut()[..])?;
        to_data.serialize(&mut &mut to_account.data.borrow_mut()[..])?;

        msg!("Transferred {} from {} to {}", amount, from_account.key, to_account.key);
        Ok(())
    }

    fn process_update(
        _program_id: &Pubkey,
        accounts: &[AccountInfo],
        new_value: u64,
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let account = next_account_info(account_info_iter)?;

        // Verify account is signer
        if !account.is_signer {
            return Err(CustomError::Unauthorized.into());
        }

        // Deserialize and update account data
        let mut account_data = AccountData::try_from_slice(&account.data.borrow())?;
        
        if !account_data.is_initialized {
            return Err(CustomError::AccountNotInitialized.into());
        }

        account_data.balance = new_value;
        account_data.serialize(&mut &mut account.data.borrow_mut()[..])?;

        msg!("Account updated with new value: {}", new_value);
        Ok(())
    }

    fn process_close(
        _program_id: &Pubkey,
        accounts: &[AccountInfo],
    ) -> ProgramResult {
        let account_info_iter = &mut accounts.iter();
        let account_to_close = next_account_info(account_info_iter)?;
        let destination_account = next_account_info(account_info_iter)?;

        // Verify account is signer
        if !account_to_close.is_signer {
            return Err(CustomError::Unauthorized.into());
        }

        // Transfer lamports to destination
        let dest_starting_lamports = destination_account.lamports();
        **destination_account.lamports.borrow_mut() = dest_starting_lamports
            .checked_add(account_to_close.lamports())
            .ok_or(ProgramError::ArithmeticOverflow)?;

        // Zero out the closed account
        **account_to_close.lamports.borrow_mut() = 0;
        
        // Clear account data
        let mut data = account_to_close.data.borrow_mut();
        data.fill(0);

        msg!("Account closed and rent reclaimed");
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. ANCHOR FRAMEWORK DEVELOPMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Anchor framework for simplified Solana development
pub mod anchor_programs {
    use anchor_lang::prelude::*;
    use anchor_spl::token::{self, Token, TokenAccount, Mint, Transfer};

    declare_id!("YourProgramIDHere111111111111111111111111111");

    #[program]
    pub mod anchor_example {
        use super::*;

        /// Initialize a new counter program
        pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
            let counter = &mut ctx.accounts.counter;
            counter.authority = ctx.accounts.authority.key();
            counter.count = 0;
            
            msg!("Counter initialized!");
            Ok(())
        }

        /// Increment the counter
        pub fn increment(ctx: Context<Increment>) -> Result<()> {
            let counter = &mut ctx.accounts.counter;
            counter.count = counter.count.checked_add(1).unwrap();
            
            msg!("Counter incremented to: {}", counter.count);
            Ok(())
        }

        /// Decrement the counter
        pub fn decrement(ctx: Context<Decrement>) -> Result<()> {
            let counter = &mut ctx.accounts.counter;
            counter.count = counter.count.checked_sub(1).unwrap();
            
            msg!("Counter decremented to: {}", counter.count);
            Ok(())
        }

        /// Transfer tokens between accounts
        pub fn transfer_tokens(
            ctx: Context<TransferTokens>,
            amount: u64,
        ) -> Result<()> {
            let cpi_accounts = Transfer {
                from: ctx.accounts.from.to_account_info(),
                to: ctx.accounts.to.to_account_info(),
                authority: ctx.accounts.authority.to_account_info(),
            };
            
            let cpi_program = ctx.accounts.token_program.to_account_info();
            let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
            
            token::transfer(cpi_ctx, amount)?;
            
            msg!("Transferred {} tokens", amount);
            Ok(())
        }

        /// Create a new token mint
        pub fn create_mint(
            ctx: Context<CreateMint>,
            decimals: u8,
        ) -> Result<()> {
            msg!("Created new token mint with {} decimals", decimals);
            Ok(())
        }

        /// Stake tokens in the program
        pub fn stake_tokens(
            ctx: Context<StakeTokens>,
            amount: u64,
        ) -> Result<()> {
            let stake_account = &mut ctx.accounts.stake_account;
            let clock = Clock::get()?;
            
            // Transfer tokens to program
            let cpi_accounts = Transfer {
                from: ctx.accounts.user_token_account.to_account_info(),
                to: ctx.accounts.program_token_account.to_account_info(),
                authority: ctx.accounts.user.to_account_info(),
            };
            
            let cpi_program = ctx.accounts.token_program.to_account_info();
            let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
            
            token::transfer(cpi_ctx, amount)?;
            
            // Update stake account
            stake_account.user = ctx.accounts.user.key();
            stake_account.amount += amount;
            stake_account.stake_time = clock.unix_timestamp;
            
            msg!("Staked {} tokens", amount);
            Ok(())
        }

        /// Unstake tokens from the program
        pub fn unstake_tokens(
            ctx: Context<UnstakeTokens>,
            amount: u64,
        ) -> Result<()> {
            let stake_account = &mut ctx.accounts.stake_account;
            
            require!(stake_account.amount >= amount, CustomError::InsufficientStake);
            
            // Transfer tokens back to user
            let seeds = &[
                b"program_authority",
                &[ctx.bumps.program_authority],
            ];
            let signer = &[&seeds[..]];
            
            let cpi_accounts = Transfer {
                from: ctx.accounts.program_token_account.to_account_info(),
                to: ctx.accounts.user_token_account.to_account_info(),
                authority: ctx.accounts.program_authority.to_account_info(),
            };
            
            let cpi_program = ctx.accounts.token_program.to_account_info();
            let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
            
            token::transfer(cpi_ctx, amount)?;
            
            // Update stake account
            stake_account.amount -= amount;
            
            msg!("Unstaked {} tokens", amount);
            Ok(())
        }
    }

    // Account structures
    #[account]
    pub struct Counter {
        pub authority: Pubkey,
        pub count: u64,
    }

    #[account]
    pub struct StakeAccount {
        pub user: Pubkey,
        pub amount: u64,
        pub stake_time: i64,
        pub reward_debt: u64,
    }

    // Context structures
    #[derive(Accounts)]
    pub struct Initialize<'info> {
        #[account(
            init,
            payer = authority,
            space = 8 + 32 + 8,
            seeds = [b"counter"],
            bump
        )]
        pub counter: Account<'info, Counter>,
        
        #[account(mut)]
        pub authority: Signer<'info>,
        
        pub system_program: Program<'info, System>,
    }

    #[derive(Accounts)]
    pub struct Increment<'info> {
        #[account(
            mut,
            seeds = [b"counter"],
            bump,
            has_one = authority
        )]
        pub counter: Account<'info, Counter>,
        
        pub authority: Signer<'info>,
    }

    #[derive(Accounts)]
    pub struct Decrement<'info> {
        #[account(
            mut,
            seeds = [b"counter"],
            bump,
            has_one = authority
        )]
        pub counter: Account<'info, Counter>,
        
        pub authority: Signer<'info>,
    }

    #[derive(Accounts)]
    pub struct TransferTokens<'info> {
        #[account(mut)]
        pub from: Account<'info, TokenAccount>,
        
        #[account(mut)]
        pub to: Account<'info, TokenAccount>,
        
        pub authority: Signer<'info>,
        
        pub token_program: Program<'info, Token>,
    }

    #[derive(Accounts)]
    pub struct CreateMint<'info> {
        #[account(
            init,
            payer = payer,
            mint::decimals = decimals,
            mint::authority = mint_authority,
        )]
        pub mint: Account<'info, Mint>,
        
        #[account(mut)]
        pub payer: Signer<'info>,
        
        /// CHECK: This is not dangerous because we don't read or write from this account
        pub mint_authority: AccountInfo<'info>,
        
        pub token_program: Program<'info, Token>,
        pub system_program: Program<'info, System>,
        pub rent: Sysvar<'info, Rent>,
    }

    #[derive(Accounts)]
    pub struct StakeTokens<'info> {
        #[account(
            init_if_needed,
            payer = user,
            space = 8 + 32 + 8 + 8 + 8,
            seeds = [b"stake", user.key().as_ref()],
            bump
        )]
        pub stake_account: Account<'info, StakeAccount>,
        
        #[account(mut)]
        pub user: Signer<'info>,
        
        #[account(mut)]
        pub user_token_account: Account<'info, TokenAccount>,
        
        #[account(mut)]
        pub program_token_account: Account<'info, TokenAccount>,
        
        pub token_program: Program<'info, Token>,
        pub system_program: Program<'info, System>,
    }

    #[derive(Accounts)]
    pub struct UnstakeTokens<'info> {
        #[account(
            mut,
            seeds = [b"stake", user.key().as_ref()],
            bump,
            has_one = user
        )]
        pub stake_account: Account<'info, StakeAccount>,
        
        #[account(mut)]
        pub user: Signer<'info>,
        
        #[account(mut)]
        pub user_token_account: Account<'info, TokenAccount>,
        
        #[account(mut)]
        pub program_token_account: Account<'info, TokenAccount>,
        
        #[account(
            seeds = [b"program_authority"],
            bump
        )]
        /// CHECK: This is a PDA used as authority
        pub program_authority: AccountInfo<'info>,
        
        pub token_program: Program<'info, Token>,
    }

    // Custom errors
    #[error_code]
    pub enum CustomError {
        #[msg("Insufficient stake amount")]
        InsufficientStake,
        #[msg("Unauthorized access")]
        Unauthorized,
        #[msg("Invalid calculation")]
        InvalidCalculation,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. SOLANA CLIENT AND RPC INTERACTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Solana client for interacting with the blockchain
pub mod solana_client {
    use super::*;
    use solana_client::rpc_client::RpcClient;
    use solana_sdk::{
        commitment_config::CommitmentConfig,
        signature::{read_keypair_file, Signature},
        transaction::Transaction,
        instruction::{AccountMeta, Instruction},
        system_instruction,
        sysvar,
    };
    use spl_token::{
        instruction as token_instruction,
        state::{Account as TokenAccountState, Mint},
    };

    pub struct SolanaClient {
        pub client: RpcClient,
        pub commitment: CommitmentConfig,
    }

    impl SolanaClient {
        pub fn new(rpc_url: &str) -> Self {
            Self {
                client: RpcClient::new_with_commitment(
                    rpc_url.to_string(),
                    CommitmentConfig::confirmed(),
                ),
                commitment: CommitmentConfig::confirmed(),
            }
        }

        pub fn new_with_commitment(rpc_url: &str, commitment: CommitmentConfig) -> Self {
            Self {
                client: RpcClient::new_with_commitment(rpc_url.to_string(), commitment),
                commitment,
            }
        }

        pub async fn get_balance(&self, pubkey: &Pubkey) -> Result<u64> {
            let balance = self.client.get_balance(pubkey)?;
            Ok(balance)
        }

        pub async fn get_account_info(&self, pubkey: &Pubkey) -> Result<Option<solana_sdk::account::Account>> {
            let account = self.client.get_account(pubkey).ok();
            Ok(account)
        }

        pub async fn send_transaction(&self, transaction: &Transaction) -> Result<Signature> {
            let signature = self.client.send_and_confirm_transaction(transaction)?;
            Ok(signature)
        }

        pub async fn create_account(
            &self,
            from: &Keypair,
            to: &Keypair,
            lamports: u64,
            space: u64,
            owner: &Pubkey,
        ) -> Result<Signature> {
            let instruction = system_instruction::create_account(
                &from.pubkey(),
                &to.pubkey(),
                lamports,
                space,
                owner,
            );

            let transaction = Transaction::new_signed_with_payer(
                &[instruction],
                Some(&from.pubkey()),
                &[from, to],
                self.client.get_latest_blockhash()?,
            );

            self.send_transaction(&transaction).await
        }

        pub async fn transfer_sol(
            &self,
            from: &Keypair,
            to: &Pubkey,
            lamports: u64,
        ) -> Result<Signature> {
            let instruction = system_instruction::transfer(&from.pubkey(), to, lamports);

            let transaction = Transaction::new_signed_with_payer(
                &[instruction],
                Some(&from.pubkey()),
                &[from],
                self.client.get_latest_blockhash()?,
            );

            self.send_transaction(&transaction).await
        }

        pub async fn create_token_mint(
            &self,
            payer: &Keypair,
            mint: &Keypair,
            mint_authority: &Pubkey,
            freeze_authority: Option<&Pubkey>,
            decimals: u8,
        ) -> Result<Signature> {
            let rent = self.client.get_minimum_balance_for_rent_exemption(Mint::LEN)?;

            let instructions = vec![
                system_instruction::create_account(
                    &payer.pubkey(),
                    &mint.pubkey(),
                    rent,
                    Mint::LEN as u64,
                    &spl_token::id(),
                ),
                token_instruction::initialize_mint(
                    &spl_token::id(),
                    &mint.pubkey(),
                    mint_authority,
                    freeze_authority,
                    decimals,
                )?,
            ];

            let transaction = Transaction::new_signed_with_payer(
                &instructions,
                Some(&payer.pubkey()),
                &[payer, mint],
                self.client.get_latest_blockhash()?,
            );

            self.send_transaction(&transaction).await
        }

        pub async fn create_token_account(
            &self,
            payer: &Keypair,
            account: &Keypair,
            mint: &Pubkey,
            owner: &Pubkey,
        ) -> Result<Signature> {
            let rent = self.client.get_minimum_balance_for_rent_exemption(TokenAccountState::LEN)?;

            let instructions = vec![
                system_instruction::create_account(
                    &payer.pubkey(),
                    &account.pubkey(),
                    rent,
                    TokenAccountState::LEN as u64,
                    &spl_token::id(),
                ),
                token_instruction::initialize_account(
                    &spl_token::id(),
                    &account.pubkey(),
                    mint,
                    owner,
                )?,
            ];

            let transaction = Transaction::new_signed_with_payer(
                &instructions,
                Some(&payer.pubkey()),
                &[payer, account],
                self.client.get_latest_blockhash()?,
            );

            self.send_transaction(&transaction).await
        }

        pub async fn mint_tokens(
            &self,
            mint_authority: &Keypair,
            mint: &Pubkey,
            to: &Pubkey,
            amount: u64,
        ) -> Result<Signature> {
            let instruction = token_instruction::mint_to(
                &spl_token::id(),
                mint,
                to,
                &mint_authority.pubkey(),
                &[],
                amount,
            )?;

            let transaction = Transaction::new_signed_with_payer(
                &[instruction],
                Some(&mint_authority.pubkey()),
                &[mint_authority],
                self.client.get_latest_blockhash()?,
            );

            self.send_transaction(&transaction).await
        }

        pub async fn transfer_tokens(
            &self,
            owner: &Keypair,
            from: &Pubkey,
            to: &Pubkey,
            amount: u64,
        ) -> Result<Signature> {
            let instruction = token_instruction::transfer(
                &spl_token::id(),
                from,
                to,
                &owner.pubkey(),
                &[],
                amount,
            )?;

            let transaction = Transaction::new_signed_with_payer(
                &[instruction],
                Some(&owner.pubkey()),
                &[owner],
                self.client.get_latest_blockhash()?,
            );

            self.send_transaction(&transaction).await
        }

        pub async fn get_token_account_balance(&self, account: &Pubkey) -> Result<u64> {
            let account_info = self.client.get_account(account)?;
            let token_account = TokenAccountState::unpack(&account_info.data)?;
            Ok(token_account.amount)
        }

        pub async fn get_token_supply(&self, mint: &Pubkey) -> Result<u64> {
            let supply = self.client.get_token_supply(mint)?;
            Ok(supply.amount.parse()?)
        }

        pub async fn airdrop(&self, pubkey: &Pubkey, lamports: u64) -> Result<Signature> {
            let signature = self.client.request_airdrop(pubkey, lamports)?;
            self.client.confirm_transaction(&signature)?;
            Ok(signature)
        }
    }

    /// Wallet management utilities
    pub struct Wallet {
        pub keypair: Keypair,
        pub client: SolanaClient,
    }

    impl Wallet {
        pub fn new(client: SolanaClient) -> Self {
            Self {
                keypair: Keypair::new(),
                client,
            }
        }

        pub fn from_keypair(keypair: Keypair, client: SolanaClient) -> Self {
            Self { keypair, client }
        }

        pub fn from_file(path: &str, client: SolanaClient) -> Result<Self> {
            let keypair = read_keypair_file(path)?;
            Ok(Self { keypair, client })
        }

        pub fn pubkey(&self) -> Pubkey {
            self.keypair.pubkey()
        }

        pub async fn balance(&self) -> Result<u64> {
            self.client.get_balance(&self.pubkey()).await
        }

        pub async fn airdrop(&self, amount: u64) -> Result<Signature> {
            self.client.airdrop(&self.pubkey(), amount).await
        }

        pub async fn transfer(&self, to: &Pubkey, amount: u64) -> Result<Signature> {
            self.client.transfer_sol(&self.keypair, to, amount).await
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           6. DeFi AND TOKEN PROTOCOLS
// ═══════════════════════════════════════════════════════════════════════════════

/// DeFi protocols and token management
pub mod defi_protocols {
    use super::*;
    use anchor_lang::prelude::*;
    use anchor_spl::token::{Token, TokenAccount, Mint};

    /// Automated Market Maker (AMM) implementation
    #[account]
    pub struct LiquidityPool {
        pub token_a_mint: Pubkey,
        pub token_b_mint: Pubkey,
        pub token_a_account: Pubkey,
        pub token_b_account: Pubkey,
        pub lp_token_mint: Pubkey,
        pub fee_rate: u64, // basis points (e.g., 30 = 0.3%)
        pub total_liquidity: u64,
        pub bump: u8,
    }

    impl LiquidityPool {
        pub const LEN: usize = 8 + 32 * 5 + 8 + 8 + 1;

        pub fn calculate_swap_amount(
            &self,
            input_amount: u64,
            input_reserve: u64,
            output_reserve: u64,
        ) -> Result<u64> {
            // Constant product formula: x * y = k
            // With fees: output = (input * 997 * output_reserve) / (input_reserve * 1000 + input * 997)
            
            let input_with_fee = input_amount
                .checked_mul(1000 - self.fee_rate)
                .ok_or(ProgramError::ArithmeticOverflow)?;
            
            let numerator = input_with_fee
                .checked_mul(output_reserve)
                .ok_or(ProgramError::ArithmeticOverflow)?;
            
            let denominator = input_reserve
                .checked_mul(1000)
                .ok_or(ProgramError::ArithmeticOverflow)?
                .checked_add(input_with_fee)
                .ok_or(ProgramError::ArithmeticOverflow)?;
            
            if denominator == 0 {
                return Err(ProgramError::DivisionByZero.into());
            }
            
            Ok(numerator / denominator)
        }

        pub fn calculate_liquidity_amount(
            &self,
            token_a_amount: u64,
            token_b_amount: u64,
            token_a_reserve: u64,
            token_b_reserve: u64,
        ) -> Result<u64> {
            if self.total_liquidity == 0 {
                // Initial liquidity
                return Ok((token_a_amount * token_b_amount).sqrt());
            }

            let liquidity_a = token_a_amount
                .checked_mul(self.total_liquidity)
                .ok_or(ProgramError::ArithmeticOverflow)?
                .checked_div(token_a_reserve)
                .ok_or(ProgramError::DivisionByZero)?;

            let liquidity_b = token_b_amount
                .checked_mul(self.total_liquidity)
                .ok_or(ProgramError::ArithmeticOverflow)?
                .checked_div(token_b_reserve)
                .ok_or(ProgramError::DivisionByZero)?;

            Ok(liquidity_a.min(liquidity_b))
        }
    }

    /// Staking pool for token rewards
    #[account]
    pub struct StakingPool {
        pub stake_token_mint: Pubkey,
        pub reward_token_mint: Pubkey,
        pub stake_token_account: Pubkey,
        pub reward_token_account: Pubkey,
        pub reward_rate: u64, // tokens per second
        pub total_staked: u64,
        pub last_update_time: i64,
        pub accumulated_reward_per_token: u128,
        pub bump: u8,
    }

    impl StakingPool {
        pub const LEN: usize = 8 + 32 * 4 + 8 * 3 + 16 + 1;

        pub fn update_rewards(&mut self) -> Result<()> {
            let clock = Clock::get()?;
            let current_time = clock.unix_timestamp;
            
            if self.total_staked > 0 {
                let time_diff = current_time - self.last_update_time;
                let reward_amount = self.reward_rate * time_diff as u64;
                let reward_per_token = (reward_amount as u128 * 1e18 as u128) / self.total_staked as u128;
                
                self.accumulated_reward_per_token += reward_per_token;
            }
            
            self.last_update_time = current_time;
            Ok(())
        }

        pub fn calculate_pending_rewards(
            &self,
            user_stake: u64,
            user_reward_debt: u128,
        ) -> u128 {
            let user_accumulated = (user_stake as u128 * self.accumulated_reward_per_token) / 1e18 as u128;
            user_accumulated.saturating_sub(user_reward_debt)
        }
    }

    /// User stake information
    #[account]
    pub struct UserStake {
        pub user: Pubkey,
        pub pool: Pubkey,
        pub amount: u64,
        pub reward_debt: u128,
        pub last_stake_time: i64,
        pub bump: u8,
    }

    impl UserStake {
        pub const LEN: usize = 8 + 32 * 2 + 8 + 16 + 8 + 1;
    }

    /// Lending protocol structures
    #[account]
    pub struct LendingMarket {
        pub admin: Pubkey,
        pub reserves: Vec<Pubkey>,
        pub oracle: Pubkey,
        pub liquidation_threshold: u64, // basis points
        pub liquidation_bonus: u64,     // basis points
        pub bump: u8,
    }

    #[account]
    pub struct Reserve {
        pub market: Pubkey,
        pub token_mint: Pubkey,
        pub supply_token_account: Pubkey,
        pub borrow_token_account: Pubkey,
        pub collateral_mint: Pubkey,
        pub debt_mint: Pubkey,
        pub total_supply: u64,
        pub total_borrow: u64,
        pub supply_rate: u64,
        pub borrow_rate: u64,
        pub last_update_time: i64,
        pub reserve_factor: u64, // percentage of interest that goes to reserves
        pub bump: u8,
    }

    impl Reserve {
        pub const LEN: usize = 8 + 32 * 6 + 8 * 6 + 1;

        pub fn calculate_supply_rate(&self, utilization_rate: u64) -> u64 {
            // Simple linear interest rate model
            let base_rate = 200; // 2% base rate
            let multiplier = 1000; // 10% multiplier
            
            base_rate + (utilization_rate * multiplier / 10000)
        }

        pub fn calculate_borrow_rate(&self, utilization_rate: u64) -> u64 {
            let supply_rate = self.calculate_supply_rate(utilization_rate);
            supply_rate + 200 // 2% spread
        }

        pub fn get_utilization_rate(&self) -> u64 {
            if self.total_supply == 0 {
                0
            } else {
                (self.total_borrow * 10000) / self.total_supply
            }
        }
    }

    /// Yield farming implementation
    #[account]
    pub struct Farm {
        pub lp_token_mint: Pubkey,
        pub reward_token_mint: Pubkey,
        pub lp_token_account: Pubkey,
        pub reward_token_account: Pubkey,
        pub reward_per_block: u64,
        pub total_staked: u64,
        pub last_reward_block: u64,
        pub accumulated_reward_per_share: u128,
        pub start_block: u64,
        pub end_block: u64,
        pub bump: u8,
    }

    impl Farm {
        pub const LEN: usize = 8 + 32 * 4 + 8 * 6 + 16 + 1;

        pub fn update_pool(&mut self, current_block: u64) -> Result<()> {
            if current_block <= self.last_reward_block {
                return Ok(());
            }

            if self.total_staked == 0 {
                self.last_reward_block = current_block;
                return Ok(());
            }

            let blocks_passed = current_block - self.last_reward_block;
            let reward_amount = blocks_passed * self.reward_per_block;
            
            self.accumulated_reward_per_share += 
                (reward_amount as u128 * 1e12 as u128) / self.total_staked as u128;
            
            self.last_reward_block = current_block;
            Ok(())
        }

        pub fn calculate_pending_reward(&self, user_amount: u64, user_reward_debt: u128) -> u128 {
            let user_accumulated = (user_amount as u128 * self.accumulated_reward_per_share) / 1e12 as u128;
            user_accumulated.saturating_sub(user_reward_debt)
        }
    }

    /// Options protocol for derivatives
    #[account]
    pub struct Option {
        pub underlying_mint: Pubkey,
        pub strike_price: u64,
        pub expiry_time: i64,
        pub option_type: OptionType, // Call or Put
        pub premium: u64,
        pub writer: Pubkey,
        pub holder: Option<Pubkey>,
        pub collateral_amount: u64,
        pub is_exercised: bool,
        pub bump: u8,
    }

    #[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
    pub enum OptionType {
        Call,
        Put,
    }

    impl Option {
        pub const LEN: usize = 8 + 32 * 3 + 1 + 8 * 3 + 33 + 1 + 1;

        pub fn is_in_the_money(&self, current_price: u64) -> bool {
            match self.option_type {
                OptionType::Call => current_price > self.strike_price,
                OptionType::Put => current_price < self.strike_price,
            }
        }

        pub fn calculate_intrinsic_value(&self, current_price: u64) -> u64 {
            if !self.is_in_the_money(current_price) {
                return 0;
            }

            match self.option_type {
                OptionType::Call => current_price.saturating_sub(self.strike_price),
                OptionType::Put => self.strike_price.saturating_sub(current_price),
            }
        }

        pub fn is_expired(&self) -> Result<bool> {
            let clock = Clock::get()?;
            Ok(clock.unix_timestamp >= self.expiry_time)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. NFT AND DIGITAL ASSETS
// ═══════════════════════════════════════════════════════════════════════════════

/// NFT and digital asset management
pub mod nft_protocols {
    use super::*;
    use anchor_lang::prelude::*;
    use mpl_token_metadata::state::{Metadata, TokenMetadataAccount};

    /// NFT Collection structure
    #[account]
    pub struct Collection {
        pub authority: Pubkey,
        pub name: String,
        pub symbol: String,
        pub description: String,
        pub image: String,
        pub external_url: Option<String>,
        pub total_supply: u64,
        pub max_supply: u64,
        pub mint_price: u64,
        pub royalty_percentage: u16, // basis points
        pub is_mutable: bool,
        pub is_verified: bool,
        pub bump: u8,
    }

    impl Collection {
        pub const MAX_NAME_LENGTH: usize = 32;
        pub const MAX_SYMBOL_LENGTH: usize = 10;
        pub const MAX_DESCRIPTION_LENGTH: usize = 200;
        pub const MAX_URL_LENGTH: usize = 200;
        
        pub const LEN: usize = 8 + 32 + 4 + Self::MAX_NAME_LENGTH + 4 + Self::MAX_SYMBOL_LENGTH + 
            4 + Self::MAX_DESCRIPTION_LENGTH + 4 + Self::MAX_URL_LENGTH + 
            1 + 4 + Self::MAX_URL_LENGTH + 8 + 8 + 8 + 2 + 1 + 1 + 1;
    }

    /// Individual NFT data
    #[account]
    pub struct Nft {
        pub collection: Pubkey,
        pub mint: Pubkey,
        pub owner: Pubkey,
        pub name: String,
        pub description: String,
        pub image: String,
        pub attributes: Vec<NftAttribute>,
        pub rarity_score: u32,
        pub is_listed: bool,
        pub list_price: Option<u64>,
        pub bump: u8,
    }

    #[derive(AnchorSerialize, AnchorDeserialize, Clone)]
    pub struct NftAttribute {
        pub trait_type: String,
        pub value: String,
        pub rarity: Option<f32>,
    }

    impl Nft {
        pub const MAX_ATTRIBUTES: usize = 10;
        pub const MAX_ATTRIBUTE_LENGTH: usize = 50;
        
        pub fn calculate_rarity_score(&mut self) {
            let mut score = 0u32;
            for attr in &self.attributes {
                if let Some(rarity) = attr.rarity {
                    score += (1.0 / rarity * 10000.0) as u32;
                }
            }
            self.rarity_score = score;
        }
    }

    /// NFT Marketplace
    #[account]
    pub struct Marketplace {
        pub authority: Pubkey,
        pub treasury: Pubkey,
        pub fee_percentage: u16, // basis points
        pub total_volume: u64,
        pub total_sales: u64,
        pub is_active: bool,
        pub bump: u8,
    }

    impl Marketplace {
        pub const LEN: usize = 8 + 32 * 2 + 2 + 8 * 2 + 1 + 1;
    }

    /// Listing for NFT sales
    #[account]
    pub struct Listing {
        pub marketplace: Pubkey,
        pub nft_mint: Pubkey,
        pub seller: Pubkey,
        pub price: u64,
        pub created_at: i64,
        pub expires_at: Option<i64>,
        pub is_active: bool,
        pub bump: u8,
    }

    impl Listing {
        pub const LEN: usize = 8 + 32 * 3 + 8 + 8 + 1 + 8 + 1 + 1;

        pub fn is_expired(&self) -> Result<bool> {
            if let Some(expires_at) = self.expires_at {
                let clock = Clock::get()?;
                Ok(clock.unix_timestamp >= expires_at)
            } else {
                Ok(false)
            }
        }
    }

    /// Auction system for NFTs
    #[account]
    pub struct Auction {
        pub nft_mint: Pubkey,
        pub seller: Pubkey,
        pub starting_price: u64,
        pub current_bid: u64,
        pub highest_bidder: Option<Pubkey>,
        pub end_time: i64,
        pub min_bid_increment: u64,
        pub is_active: bool,
        pub is_settled: bool,
        pub bump: u8,
    }

    impl Auction {
        pub const LEN: usize = 8 + 32 * 2 + 8 * 3 + 33 + 8 + 8 + 1 + 1 + 1;

        pub fn is_ended(&self) -> Result<bool> {
            let clock = Clock::get()?;
            Ok(clock.unix_timestamp >= self.end_time)
        }

        pub fn calculate_next_min_bid(&self) -> u64 {
            if self.current_bid == 0 {
                self.starting_price
            } else {
                self.current_bid + self.min_bid_increment
            }
        }
    }

    /// Bid information
    #[account]
    pub struct Bid {
        pub auction: Pubkey,
        pub bidder: Pubkey,
        pub amount: u64,
        pub timestamp: i64,
        pub is_winning: bool,
        pub bump: u8,
    }

    impl Bid {
        pub const LEN: usize = 8 + 32 * 2 + 8 + 8 + 1 + 1;
    }

    /// Royalty management
    #[account]
    pub struct Royalty {
        pub nft_mint: Pubkey,
        pub creator: Pubkey,
        pub percentage: u16, // basis points
        pub total_earned: u64,
        pub is_active: bool,
        pub bump: u8,
    }

    impl Royalty {
        pub const LEN: usize = 8 + 32 * 2 + 2 + 8 + 1 + 1;

        pub fn calculate_royalty_amount(&self, sale_price: u64) -> u64 {
            (sale_price * self.percentage as u64) / 10000
        }
    }

    /// Fractionalized NFT
    #[account]
    pub struct FractionalNft {
        pub original_nft_mint: Pubkey,
        pub fraction_mint: Pubkey,
        pub total_fractions: u64,
        pub price_per_fraction: u64,
        pub vault_authority: Pubkey,
        pub is_redeemable: bool,
        pub redemption_price: Option<u64>,
        pub bump: u8,
    }

    impl FractionalNft {
        pub const LEN: usize = 8 + 32 * 3 + 8 * 2 + 1 + 1 + 8 + 1;

        pub fn calculate_total_value(&self) -> u64 {
            self.total_fractions * self.price_per_fraction
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           8. CROSS-CHAIN AND INTEROPERABILITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-chain protocols and bridges
pub mod cross_chain {
    use super::*;

    /// Bridge for cross-chain asset transfers
    #[account]
    pub struct Bridge {
        pub admin: Pubkey,
        pub source_chain: String,
        pub destination_chain: String,
        pub supported_tokens: Vec<Pubkey>,
        pub fee_percentage: u16,
        pub min_transfer_amount: u64,
        pub max_transfer_amount: u64,
        pub total_locked: u64,
        pub is_active: bool,
        pub bump: u8,
    }

    impl Bridge {
        pub const MAX_CHAIN_NAME_LENGTH: usize = 32;
        pub const MAX_SUPPORTED_TOKENS: usize = 50;
        
        pub const LEN: usize = 8 + 32 + 4 + Self::MAX_CHAIN_NAME_LENGTH * 2 + 
            4 + 32 * Self::MAX_SUPPORTED_TOKENS + 2 + 8 * 3 + 1 + 1;
    }

    /// Bridge transfer record
    #[account]
    pub struct BridgeTransfer {
        pub bridge: Pubkey,
        pub sender: Pubkey,
        pub recipient: String, // Address on destination chain
        pub token_mint: Pubkey,
        pub amount: u64,
        pub fee: u64,
        pub source_tx_hash: Option<String>,
        pub destination_tx_hash: Option<String>,
        pub status: TransferStatus,
        pub initiated_at: i64,
        pub completed_at: Option<i64>,
        pub bump: u8,
    }

    #[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq)]
    pub enum TransferStatus {
        Initiated,
        Locked,
        InTransit,
        Completed,
        Failed,
        Refunded,
    }

    impl BridgeTransfer {
        pub const MAX_ADDRESS_LENGTH: usize = 64;
        pub const MAX_HASH_LENGTH: usize = 64;
        
        pub const LEN: usize = 8 + 32 * 3 + 4 + Self::MAX_ADDRESS_LENGTH + 8 * 2 + 
            1 + 4 + Self::MAX_HASH_LENGTH + 1 + 4 + Self::MAX_HASH_LENGTH + 
            1 + 8 + 1 + 8 + 1;
    }

    /// Oracle for price feeds and external data
    #[account]
    pub struct Oracle {
        pub authority: Pubkey,
        pub name: String,
        pub description: String,
        pub data_sources: Vec<DataSource>,
        pub update_frequency: u64, // seconds
        pub last_update: i64,
        pub is_active: bool,
        pub bump: u8,
    }

    #[derive(AnchorSerialize, AnchorDeserialize, Clone)]
    pub struct DataSource {
        pub name: String,
        pub url: String,
        pub weight: u16, // percentage
    }

    impl Oracle {
        pub const MAX_NAME_LENGTH: usize = 32;
        pub const MAX_DESCRIPTION_LENGTH: usize = 200;
        pub const MAX_DATA_SOURCES: usize = 10;
    }

    /// Price feed data
    #[account]
    pub struct PriceFeed {
        pub oracle: Pubkey,
        pub symbol: String,
        pub price: u64, // Price in lamports or smallest unit
        pub confidence: u64,
        pub last_update: i64,
        pub historical_prices: Vec<HistoricalPrice>,
        pub bump: u8,
    }

    #[derive(AnchorSerialize, AnchorDeserialize, Clone)]
    pub struct HistoricalPrice {
        pub price: u64,
        pub timestamp: i64,
    }

    impl PriceFeed {
        pub const MAX_SYMBOL_LENGTH: usize = 16;
        pub const MAX_HISTORICAL_ENTRIES: usize = 100;
        
        pub fn calculate_average_price(&self, duration: i64) -> Option<u64> {
            let current_time = Clock::