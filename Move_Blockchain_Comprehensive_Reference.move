// Move Language Blockchain & Smart Contracts - Comprehensive Reference
// Focus: Meta's Move ecosystem (Aptos/Sui), DeFi protocols, NFTs, and enterprise blockchain
// Platforms: Aptos, Sui, Diem (historical), Move-based Layer 1 blockchains

/*
═══════════════════════════════════════════════════════════════════════════════
                           PROJECT STRUCTURE AND SETUP
═══════════════════════════════════════════════════════════════════════════════

Move Project Structure:
move-dapp/
├── Move.toml                    # Package configuration
├── sources/                     # Smart contract source files
│   ├── defi/
│   │   ├── dex.move            # Decentralized exchange
│   │   ├── lending.move        # Lending protocol
│   │   ├── yield_farming.move  # Yield farming contracts
│   │   └── governance.move     # DAO governance
│   ├── nft/
│   │   ├── collection.move     # NFT collections
│   │   ├── marketplace.move    # NFT marketplace
│   │   └── royalty.move        # Royalty management
│   ├── tokens/
│   │   ├── fungible_token.move # ERC20-like tokens
│   │   └── wrapped_coin.move   # Wrapped native tokens
│   ├── security/
│   │   ├── access_control.move # Role-based access
│   │   ├── multi_sig.move      # Multi-signature wallet
│   │   └── timelock.move       # Time-locked operations
│   └── utils/
│       ├── math.move           # Mathematical operations
│       ├── events.move         # Event handling
│       └── oracle.move         # Price oracle integration
├── tests/                      # Unit and integration tests
│   ├── defi_tests.move
│   ├── nft_tests.move
│   └── integration_tests.move
├── scripts/                    # Deployment and interaction scripts
│   ├── deploy.move
│   └── initialize.move
└── docs/                       # Documentation
    ├── architecture.md
    └── api-reference.md

Move.toml Configuration:
[package]
name = "enterprise-move-dapp"
version = "1.0.0"
license = "Apache-2.0"
authors = ["Enterprise Blockchain Team"]

[addresses]
std = "0x1"
aptos_framework = "0x1"
aptos_token = "0x3"
enterprise_dapp = "0x42"
admin = "0x1337"

[dev-addresses]
enterprise_dapp = "0xCAFE"
admin = "0xDEAD"

[dependencies.AptosFramework]
git = "https://github.com/aptos-labs/aptos-core.git"
rev = "mainnet"
subdir = "aptos-move/framework/aptos-framework"

[dependencies.AptosToken]
git = "https://github.com/aptos-labs/aptos-core.git"
rev = "mainnet"
subdir = "aptos-move/framework/aptos-token"

Setup Commands:
aptos init --profile mainnet
aptos move compile --package-dir .
aptos move test --package-dir .
aptos move publish --profile mainnet
*/

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. CORE MOVE LANGUAGE FUNDAMENTALS
// ═══════════════════════════════════════════════════════════════════════════════

module enterprise_dapp::core_fundamentals {
    use std::string::{Self, String};
    use std::vector;
    use std::option::{Self, Option};
    use std::error;
    use std::signer;
    use aptos_framework::coin::{Self, Coin};
    use aptos_framework::aptos_coin::AptosCoin;
    use aptos_framework::timestamp;
    use aptos_framework::account;
    use aptos_framework::event::{Self, EventHandle};

    // ═══════════════════════════════════════════════════════════════════════════
    //                           ERROR CODES
    // ═══════════════════════════════════════════════════════════════════════════

    /// Caller is not authorized to perform this operation
    const E_NOT_AUTHORIZED: u64 = 1;
    /// Insufficient balance for the operation
    const E_INSUFFICIENT_BALANCE: u64 = 2;
    /// Invalid amount provided
    const E_INVALID_AMOUNT: u64 = 3;
    /// Resource already exists
    const E_ALREADY_EXISTS: u64 = 4;
    /// Resource does not exist
    const E_NOT_EXISTS: u64 = 5;
    /// Operation not allowed at this time
    const E_NOT_ALLOWED: u64 = 6;
    /// Invalid configuration
    const E_INVALID_CONFIG: u64 = 7;

    // ═══════════════════════════════════════════════════════════════════════════
    //                           CORE DATA STRUCTURES
    // ═══════════════════════════════════════════════════════════════════════════

    /// Global configuration for the enterprise DApp
    struct GlobalConfig has key {
        /// Admin address with full privileges
        admin: address,
        /// Protocol fee in basis points (1/10000)
        protocol_fee_bps: u64,
        /// Emergency pause flag
        paused: bool,
        /// Minimum transaction amount
        min_amount: u64,
        /// Maximum transaction amount per user
        max_amount_per_user: u64,
        /// Protocol treasury address
        treasury: address,
        /// Configuration update events
        config_events: EventHandle<ConfigUpdateEvent>,
    }

    /// User profile with enterprise features
    struct UserProfile has key {
        /// User's display name
        name: String,
        /// KYC verification status
        kyc_verified: bool,
        /// User tier (0=basic, 1=premium, 2=institutional)
        tier: u8,
        /// Total transaction volume
        total_volume: u64,
        /// Account creation timestamp
        created_at: u64,
        /// Last activity timestamp
        last_activity: u64,
        /// User referral code
        referral_code: String,
        /// Referred by address
        referred_by: Option<address>,
        /// User activity events
        activity_events: EventHandle<UserActivityEvent>,
    }

    /// Asset metadata for any token/coin
    struct AssetMetadata has store, copy, drop {
        /// Asset name
        name: String,
        /// Asset symbol
        symbol: String,
        /// Number of decimal places
        decimals: u8,
        /// Asset description
        description: String,
        /// Logo URL
        logo_url: String,
        /// Project website
        website: String,
        /// Asset type (coin, token, nft)
        asset_type: String,
        /// Verification status
        verified: bool,
    }

    /// Vault for secure asset storage with enterprise controls
    struct AssetVault<phantom CoinType> has key {
        /// Stored coins
        balance: Coin<CoinType>,
        /// Vault owner
        owner: address,
        /// Authorized managers
        managers: vector<address>,
        /// Minimum withdrawal amount
        min_withdrawal: u64,
        /// Daily withdrawal limit
        daily_limit: u64,
        /// Amount withdrawn today
        withdrawn_today: u64,
        /// Last withdrawal day (timestamp / 86400)
        last_withdrawal_day: u64,
        /// Withdrawal delay in seconds
        withdrawal_delay: u64,
        /// Pending withdrawals
        pending_withdrawals: vector<PendingWithdrawal>,
        /// Vault events
        vault_events: EventHandle<VaultEvent>,
    }

    /// Pending withdrawal with time delay
    struct PendingWithdrawal has store, copy, drop {
        /// Withdrawal amount
        amount: u64,
        /// Destination address
        to: address,
        /// Requested timestamp
        requested_at: u64,
        /// Available timestamp
        available_at: u64,
        /// Withdrawal ID
        id: u64,
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                           EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    struct ConfigUpdateEvent has drop, store {
        field: String,
        old_value: String,
        new_value: String,
        updated_by: address,
        timestamp: u64,
    }

    struct UserActivityEvent has drop, store {
        user: address,
        activity_type: String,
        amount: Option<u64>,
        details: String,
        timestamp: u64,
    }

    struct VaultEvent has drop, store {
        vault_owner: address,
        event_type: String,
        amount: u64,
        from_to: Option<address>,
        timestamp: u64,
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                           CORE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Initialize the enterprise DApp with global configuration
    public entry fun initialize(
        admin: &signer,
        protocol_fee_bps: u64,
        min_amount: u64,
        max_amount_per_user: u64,
        treasury: address
    ) {
        let admin_addr = signer::address_of(admin);
        
        // Validate inputs
        assert!(protocol_fee_bps <= 1000, error::invalid_argument(E_INVALID_CONFIG)); // Max 10%
        assert!(min_amount > 0, error::invalid_argument(E_INVALID_AMOUNT));
        assert!(max_amount_per_user >= min_amount, error::invalid_argument(E_INVALID_CONFIG));
        
        // Ensure not already initialized
        assert!(!exists<GlobalConfig>(admin_addr), error::already_exists(E_ALREADY_EXISTS));
        
        // Create global configuration
        let config = GlobalConfig {
            admin: admin_addr,
            protocol_fee_bps,
            paused: false,
            min_amount,
            max_amount_per_user,
            treasury,
            config_events: account::new_event_handle<ConfigUpdateEvent>(admin),
        };
        
        move_to(admin, config);
        
        // Emit initialization event
        let config_ref = borrow_global_mut<GlobalConfig>(admin_addr);
        event::emit_event(
            &mut config_ref.config_events,
            ConfigUpdateEvent {
                field: string::utf8(b"initialized"),
                old_value: string::utf8(b"false"),
                new_value: string::utf8(b"true"),
                updated_by: admin_addr,
                timestamp: timestamp::now_seconds(),
            }
        );
    }

    /// Create or update user profile
    public entry fun create_user_profile(
        user: &signer,
        name: String,
        referral_code: Option<address>
    ) acquires UserProfile {
        let user_addr = signer::address_of(user);
        let current_time = timestamp::now_seconds();
        
        if (exists<UserProfile>(user_addr)) {
            // Update existing profile
            let profile = borrow_global_mut<UserProfile>(user_addr);
            profile.name = name;
            profile.last_activity = current_time;
            
            // Emit activity event
            event::emit_event(
                &mut profile.activity_events,
                UserActivityEvent {
                    user: user_addr,
                    activity_type: string::utf8(b"profile_updated"),
                    amount: option::none(),
                    details: name,
                    timestamp: current_time,
                }
            );
        } else {
            // Create new profile
            let profile = UserProfile {
                name,
                kyc_verified: false,
                tier: 0, // Basic tier
                total_volume: 0,
                created_at: current_time,
                last_activity: current_time,
                referral_code: generate_referral_code(user_addr),
                referred_by: referral_code,
                activity_events: account::new_event_handle<UserActivityEvent>(user),
            };
            
            move_to(user, profile);
        }
    }

    /// Create asset vault with enterprise security features
    public entry fun create_vault<CoinType>(
        owner: &signer,
        managers: vector<address>,
        min_withdrawal: u64,
        daily_limit: u64,
        withdrawal_delay: u64
    ) {
        let owner_addr = signer::address_of(owner);
        
        // Validate inputs
        assert!(min_withdrawal > 0, error::invalid_argument(E_INVALID_AMOUNT));
        assert!(daily_limit >= min_withdrawal, error::invalid_argument(E_INVALID_CONFIG));
        assert!(withdrawal_delay <= 7 * 24 * 3600, error::invalid_argument(E_INVALID_CONFIG)); // Max 7 days
        
        // Ensure vault doesn't exist
        assert!(!exists<AssetVault<CoinType>>(owner_addr), error::already_exists(E_ALREADY_EXISTS));
        
        let vault = AssetVault<CoinType> {
            balance: coin::zero<CoinType>(),
            owner: owner_addr,
            managers,
            min_withdrawal,
            daily_limit,
            withdrawn_today: 0,
            last_withdrawal_day: timestamp::now_seconds() / 86400,
            withdrawal_delay,
            pending_withdrawals: vector::empty(),
            vault_events: account::new_event_handle<VaultEvent>(owner),
        };
        
        move_to(owner, vault);
        
        // Emit vault creation event
        let vault_ref = borrow_global_mut<AssetVault<CoinType>>(owner_addr);
        event::emit_event(
            &mut vault_ref.vault_events,
            VaultEvent {
                vault_owner: owner_addr,
                event_type: string::utf8(b"vault_created"),
                amount: 0,
                from_to: option::none(),
                timestamp: timestamp::now_seconds(),
            }
        );
    }

    /// Deposit coins into vault
    public entry fun deposit_to_vault<CoinType>(
        depositor: &signer,
        vault_owner: address,
        amount: u64
    ) acquires AssetVault, GlobalConfig {
        let depositor_addr = signer::address_of(depositor);
        
        // Check global pause
        assert_not_paused();
        
        // Validate amount
        assert!(amount > 0, error::invalid_argument(E_INVALID_AMOUNT));
        
        // Withdraw coins from depositor
        let coins = coin::withdraw<CoinType>(depositor, amount);
        
        // Add to vault
        let vault = borrow_global_mut<AssetVault<CoinType>>(vault_owner);
        coin::merge(&mut vault.balance, coins);
        
        // Emit deposit event
        event::emit_event(
            &mut vault.vault_events,
            VaultEvent {
                vault_owner,
                event_type: string::utf8(b"deposit"),
                amount,
                from_to: option::some(depositor_addr),
                timestamp: timestamp::now_seconds(),
            }
        );
    }

    /// Request withdrawal from vault (with time delay)
    public entry fun request_withdrawal<CoinType>(
        requester: &signer,
        vault_owner: address,
        amount: u64,
        to: address
    ) acquires AssetVault, GlobalConfig {
        let requester_addr = signer::address_of(requester);
        
        // Check global pause
        assert_not_paused();
        
        // Get vault
        let vault = borrow_global_mut<AssetVault<CoinType>>(vault_owner);
        
        // Check authorization
        assert!(
            requester_addr == vault.owner || vector::contains(&vault.managers, &requester_addr),
            error::permission_denied(E_NOT_AUTHORIZED)
        );
        
        // Validate withdrawal amount
        assert!(amount >= vault.min_withdrawal, error::invalid_argument(E_INVALID_AMOUNT));
        assert!(coin::value(&vault.balance) >= amount, error::invalid_state(E_INSUFFICIENT_BALANCE));
        
        // Check daily limit
        let current_day = timestamp::now_seconds() / 86400;
        if (current_day > vault.last_withdrawal_day) {
            vault.withdrawn_today = 0;
            vault.last_withdrawal_day = current_day;
        };
        
        assert!(
            vault.withdrawn_today + amount <= vault.daily_limit,
            error::resource_exhausted(E_NOT_ALLOWED)
        );
        
        let current_time = timestamp::now_seconds();
        let withdrawal_id = vector::length(&vault.pending_withdrawals);
        
        // Create pending withdrawal
        let pending = PendingWithdrawal {
            amount,
            to,
            requested_at: current_time,
            available_at: current_time + vault.withdrawal_delay,
            id: withdrawal_id,
        };
        
        vector::push_back(&mut vault.pending_withdrawals, pending);
        
        // Emit withdrawal request event
        event::emit_event(
            &mut vault.vault_events,
            VaultEvent {
                vault_owner,
                event_type: string::utf8(b"withdrawal_requested"),
                amount,
                from_to: option::some(to),
                timestamp: current_time,
            }
        );
    }

    /// Execute pending withdrawal after delay period
    public entry fun execute_withdrawal<CoinType>(
        executor: &signer,
        vault_owner: address,
        withdrawal_id: u64
    ) acquires AssetVault {
        let executor_addr = signer::address_of(executor);
        let vault = borrow_global_mut<AssetVault<CoinType>>(vault_owner);
        
        // Check authorization
        assert!(
            executor_addr == vault.owner || vector::contains(&vault.managers, &executor_addr),
            error::permission_denied(E_NOT_AUTHORIZED)
        );
        
        // Find and validate withdrawal
        assert!(withdrawal_id < vector::length(&vault.pending_withdrawals), error::not_found(E_NOT_EXISTS));
        
        let pending = vector::borrow(&vault.pending_withdrawals, withdrawal_id);
        let current_time = timestamp::now_seconds();
        
        assert!(current_time >= pending.available_at, error::invalid_state(E_NOT_ALLOWED));
        
        let amount = pending.amount;
        let to = pending.to;
        
        // Remove from pending withdrawals
        vector::remove(&mut vault.pending_withdrawals, withdrawal_id);
        
        // Update daily withdrawal tracking
        vault.withdrawn_today = vault.withdrawn_today + amount;
        
        // Execute withdrawal
        let coins = coin::extract(&mut vault.balance, amount);
        coin::deposit(to, coins);
        
        // Emit withdrawal executed event
        event::emit_event(
            &mut vault.vault_events,
            VaultEvent {
                vault_owner,
                event_type: string::utf8(b"withdrawal_executed"),
                amount,
                from_to: option::some(to),
                timestamp: current_time,
            }
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                           UTILITY FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Check if protocol is not paused
    fun assert_not_paused() acquires GlobalConfig {
        let config = borrow_global<GlobalConfig>(@enterprise_dapp);
        assert!(!config.paused, error::unavailable(E_NOT_ALLOWED));
    }

    /// Generate referral code for user
    fun generate_referral_code(user_addr: address): String {
        // Simple referral code generation (in production, use more sophisticated method)
        let addr_bytes = std::bcs::to_bytes(&user_addr);
        let hash = std::hash::sha3_256(addr_bytes);
        let code_bytes = vector::slice(&hash, 0, 8);
        string::utf8(code_bytes)
    }

    /// Calculate protocol fee
    public fun calculate_protocol_fee(amount: u64): u64 acquires GlobalConfig {
        let config = borrow_global<GlobalConfig>(@enterprise_dapp);
        (amount * config.protocol_fee_bps) / 10000
    }

    /// Check if user is KYC verified
    public fun is_kyc_verified(user: address): bool acquires UserProfile {
        if (!exists<UserProfile>(user)) {
            return false
        };
        let profile = borrow_global<UserProfile>(user);
        profile.kyc_verified
    }

    /// Get user tier
    public fun get_user_tier(user: address): u8 acquires UserProfile {
        if (!exists<UserProfile>(user)) {
            return 0
        };
        let profile = borrow_global<UserProfile>(user);
        profile.tier
    }

    /// Update user activity
    public fun update_user_activity(user: address, amount: u64) acquires UserProfile {
        if (!exists<UserProfile>(user)) {
            return
        };
        let profile = borrow_global_mut<UserProfile>(user);
        profile.last_activity = timestamp::now_seconds();
        profile.total_volume = profile.total_volume + amount;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                           VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    #[view]
    public fun get_vault_balance<CoinType>(vault_owner: address): u64 acquires AssetVault {
        let vault = borrow_global<AssetVault<CoinType>>(vault_owner);
        coin::value(&vault.balance)
    }

    #[view]
    public fun get_pending_withdrawals<CoinType>(vault_owner: address): vector<PendingWithdrawal> acquires AssetVault {
        let vault = borrow_global<AssetVault<CoinType>>(vault_owner);
        vault.pending_withdrawals
    }

    #[view]
    public fun get_user_profile(user: address): (String, bool, u8, u64) acquires UserProfile {
        assert!(exists<UserProfile>(user), error::not_found(E_NOT_EXISTS));
        let profile = borrow_global<UserProfile>(user);
        (profile.name, profile.kyc_verified, profile.tier, profile.total_volume)
    }

    #[view]
    public fun get_global_config(): (address, u64, bool, u64, u64, address) acquires GlobalConfig {
        let config = borrow_global<GlobalConfig>(@enterprise_dapp);
        (config.admin, config.protocol_fee_bps, config.paused, config.min_amount, config.max_amount_per_user, config.treasury)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                           ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Update global configuration (admin only)
    public entry fun update_config(
        admin: &signer,
        protocol_fee_bps: Option<u64>,
        paused: Option<bool>,
        min_amount: Option<u64>,
        max_amount_per_user: Option<u64>,
        treasury: Option<address>
    ) acquires GlobalConfig {
        let admin_addr = signer::address_of(admin);
        let config = borrow_global_mut<GlobalConfig>(@enterprise_dapp);
        
        assert!(admin_addr == config.admin, error::permission_denied(E_NOT_AUTHORIZED));
        
        let current_time = timestamp::now_seconds();
        
        if (option::is_some(&protocol_fee_bps)) {
            let new_fee = option::extract(&mut protocol_fee_bps);
            assert!(new_fee <= 1000, error::invalid_argument(E_INVALID_CONFIG));
            
            event::emit_event(
                &mut config.config_events,
                ConfigUpdateEvent {
                    field: string::utf8(b"protocol_fee_bps"),
                    old_value: string::utf8(*string::bytes(&string::to_string(&config.protocol_fee_bps))),
                    new_value: string::utf8(*string::bytes(&string::to_string(&new_fee))),
                    updated_by: admin_addr,
                    timestamp: current_time,
                }
            );
            
            config.protocol_fee_bps = new_fee;
        };
        
        if (option::is_some(&paused)) {
            let new_paused = option::extract(&mut paused);
            config.paused = new_paused;
        };
        
        if (option::is_some(&min_amount)) {
            let new_min = option::extract(&mut min_amount);
            assert!(new_min > 0, error::invalid_argument(E_INVALID_AMOUNT));
            config.min_amount = new_min;
        };
        
        if (option::is_some(&max_amount_per_user)) {
            let new_max = option::extract(&mut max_amount_per_user);
            assert!(new_max >= config.min_amount, error::invalid_argument(E_INVALID_CONFIG));
            config.max_amount_per_user = new_max;
        };
        
        if (option::is_some(&treasury)) {
            let new_treasury = option::extract(&mut treasury);
            config.treasury = new_treasury;
        };
    }

    /// Verify user KYC (admin only)
    public entry fun verify_kyc(
        admin: &signer,
        user: address,
        verified: bool
    ) acquires GlobalConfig, UserProfile {
        let admin_addr = signer::address_of(admin);
        let config = borrow_global<GlobalConfig>(@enterprise_dapp);
        
        assert!(admin_addr == config.admin, error::permission_denied(E_NOT_AUTHORIZED));
        assert!(exists<UserProfile>(user), error::not_found(E_NOT_EXISTS));
        
        let profile = borrow_global_mut<UserProfile>(user);
        profile.kyc_verified = verified;
        
        // Emit activity event
        event::emit_event(
            &mut profile.activity_events,
            UserActivityEvent {
                user,
                activity_type: string::utf8(b"kyc_updated"),
                amount: option::none(),
                details: if (verified) string::utf8(b"verified") else string::utf8(b"unverified"),
                timestamp: timestamp::now_seconds(),
            }
        );
    }

    /// Upgrade user tier (admin only)
    public entry fun upgrade_user_tier(
        admin: &signer,
        user: address,
        new_tier: u8
    ) acquires GlobalConfig, UserProfile {
        let admin_addr = signer::address_of(admin);
        let config = borrow_global<GlobalConfig>(@enterprise_dapp);
        
        assert!(admin_addr == config.admin, error::permission_denied(E_NOT_AUTHORIZED));
        assert!(exists<UserProfile>(user), error::not_found(E_NOT_EXISTS));
        assert!(new_tier <= 2, error::invalid_argument(E_INVALID_CONFIG)); // Max tier 2
        
        let profile = borrow_global_mut<UserProfile>(user);
        let old_tier = profile.tier;
        profile.tier = new_tier;
        
        // Emit activity event
        event::emit_event(
            &mut profile.activity_events,
            UserActivityEvent {
                user,
                activity_type: string::utf8(b"tier_upgraded"),
                amount: option::some((old_tier as u64)),
                details: string::utf8(*string::bytes(&string::to_string(&(new_tier as u64)))),
                timestamp: timestamp::now_seconds(),
            }
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. DEFI PROTOCOLS - DECENTRALIZED EXCHANGE
// ═══════════════════════════════════════════════════════════════════════════════

module enterprise_dapp::dex {
    use std::string::{Self, String};
    use std::vector;
    use std::option::{Self, Option};
    use std::error;
    use std::signer;
    use aptos_framework::coin::{Self, Coin};
    use aptos_framework::timestamp;
    use aptos_framework::account;
    use aptos_framework::event::{Self, EventHandle};
    use enterprise_dapp::core_fundamentals::{Self, assert_not_paused, calculate_protocol_fee, update_user_activity};

    // Error codes
    const E_POOL_NOT_EXISTS: u64 = 101;
    const E_POOL_EXISTS: u64 = 102;
    const E_INSUFFICIENT_LIQUIDITY: u64 = 103;
    const E_INVALID_SLIPPAGE: u64 = 104;
    const E_DEADLINE_EXCEEDED: u64 = 105;
    const E_INSUFFICIENT_OUTPUT: u64 = 106;
    const E_INVALID_PATH: u64 = 107;
    const E_ZERO_AMOUNT: u64 = 108;

    // ═══════════════════════════════════════════════════════════════════════════
    //                           DEX DATA STRUCTURES
    // ═══════════════════════════════════════════════════════════════════════════

    /// Liquidity pool for token pair
    struct LiquidityPool<phantom X, phantom Y> has key {
        /// Reserve of token X
        reserve_x: Coin<X>,
        /// Reserve of token Y  
        reserve_y: Coin<Y>,
        /// Total LP token supply
        lp_token_supply: u128,
        /// LP token mint capability
        lp_mint_cap: coin::MintCapability<LPToken<X, Y>>,
        /// LP token burn capability
        lp_burn_cap: coin::BurnCapability<LPToken<X, Y>>,
        /// Pool fee in basis points (default: 30 = 0.3%)
        fee_bps: u64,
        /// Pool creation timestamp
        created_at: u64,
        /// Pool events
        pool_events: EventHandle<PoolEvent>,
    }

    /// LP Token for liquidity providers
    struct LPToken<phantom X, phantom Y> {}

    /// LP Token metadata and capabilities
    struct LPTokenInfo<phantom X, phantom Y> has key {
        name: String,
        symbol: String,
        decimals: u8,
    }

    /// Price oracle for tokens
    struct PriceOracle has key {
        /// Token price data (token_address -> price in USD * 1e8)
        prices: vector<TokenPrice>,
        /// Last update timestamp
        last_updated: u64,
        /// Oracle admin
        oracle_admin: address,
        /// Price events
        price_events: EventHandle<PriceUpdateEvent>,
    }

    struct Token