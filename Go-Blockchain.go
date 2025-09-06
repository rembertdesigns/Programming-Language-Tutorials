// GO BLOCKCHAIN - Comprehensive Infrastructure Development Reference - by Richard Rembert
// Go is the foundation language for blockchain infrastructure, powering Ethereum,
// Cosmos, Hyperledger, and many other blockchain networks

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════════════════

/*
GO BLOCKCHAIN DEVELOPMENT SETUP:

1. Install Go 1.21+:
   - Download from https://golang.org/dl/
   - Set GOPATH and GOROOT environment variables
   - Verify: go version

2. Essential Go modules for blockchain development:
   go mod init blockchain-go
   
   // Core blockchain libraries
   go get github.com/ethereum/go-ethereum@latest
   go get github.com/btcsuite/btcd@latest
   go get github.com/cosmos/cosmos-sdk@latest
   
   // Cryptography
   go get golang.org/x/crypto@latest
   go get github.com/btcsuite/btcutil@latest
   
   // Networking
   go get github.com/libp2p/go-libp2p@latest
   go get github.com/gorilla/websocket@latest
   go get github.com/grpc-ecosystem/grpc-gateway/v2@latest
   
   // Database
   go get github.com/syndtr/goleveldb@latest
   go get github.com/dgraph-io/badger/v3@latest
   go get go.etcd.io/bbolt@latest
   
   // Consensus
   go get github.com/tendermint/tendermint@latest
   
   // Utilities
   go get github.com/spf13/cobra@latest
   go get github.com/spf13/viper@latest
   go get github.com/sirupsen/logrus@latest
   go get github.com/stretchr/testify@latest

3. Development tools:
   go install golang.org/x/tools/cmd/goimports@latest
   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
   go install github.com/securecodewarrior/sast-scan@latest

4. Project structure:
   blockchain-go/
   ├── cmd/               # CLI applications
   ├── pkg/               # Public packages
   ├── internal/          # Private packages
   ├── api/               # API definitions
   ├── configs/           # Configuration files
   ├── deployments/       # Deployment configs
   ├── docs/              # Documentation
   ├── scripts/           # Build and deployment scripts
   ├── test/              # Test files
   └── go.mod             # Module definition
*/

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"
)

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. CORE BLOCKCHAIN PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════════

// Transaction represents a blockchain transaction
type Transaction struct {
	ID        string    `json:"id"`
	From      string    `json:"from"`
	To        string    `json:"to"`
	Amount    int64     `json:"amount"`
	Fee       int64     `json:"fee"`
	Nonce     uint64    `json:"nonce"`
	Timestamp time.Time `json:"timestamp"`
	Signature string    `json:"signature"`
	Data      []byte    `json:"data,omitempty"`
}

// Hash calculates the hash of a transaction
func (tx *Transaction) Hash() string {
	record := fmt.Sprintf("%s%s%s%d%d%d%d",
		tx.From, tx.To, tx.ID, tx.Amount, tx.Fee, tx.Nonce, tx.Timestamp.Unix())
	
	if len(tx.Data) > 0 {
		record += string(tx.Data)
	}
	
	h := sha256.New()
	h.Write([]byte(record))
	return hex.EncodeToString(h.Sum(nil))
}

// Validate performs basic validation on the transaction
func (tx *Transaction) Validate() error {
	if tx.From == "" {
		return fmt.Errorf("transaction must have a sender")
	}
	if tx.To == "" {
		return fmt.Errorf("transaction must have a recipient")
	}
	if tx.Amount <= 0 {
		return fmt.Errorf("transaction amount must be positive")
	}
	if tx.Fee < 0 {
		return fmt.Errorf("transaction fee cannot be negative")
	}
	return nil
}

// Block represents a block in the blockchain
type Block struct {
	Index        int64         `json:"index"`
	Timestamp    time.Time     `json:"timestamp"`
	Transactions []Transaction `json:"transactions"`
	PrevHash     string        `json:"prev_hash"`
	Hash         string        `json:"hash"`
	Nonce        int64         `json:"nonce"`
	Difficulty   int           `json:"difficulty"`
	MerkleRoot   string        `json:"merkle_root"`
	Validator    string        `json:"validator,omitempty"`
	Signature    string        `json:"signature,omitempty"`
}

// CalculateHash calculates the hash of a block
func (b *Block) CalculateHash() string {
	record := fmt.Sprintf("%d%s%s%d%d%s",
		b.Index, b.Timestamp.String(), b.PrevHash, b.Nonce, b.Difficulty, b.MerkleRoot)
	
	h := sha256.New()
	h.Write([]byte(record))
	return hex.EncodeToString(h.Sum(nil))
}

// CalculateMerkleRoot calculates the Merkle root of transactions
func (b *Block) CalculateMerkleRoot() string {
	if len(b.Transactions) == 0 {
		return ""
	}
	
	var hashes []string
	for _, tx := range b.Transactions {
		hashes = append(hashes, tx.Hash())
	}
	
	return calculateMerkleRoot(hashes)
}

// calculateMerkleRoot helper function to calculate Merkle root
func calculateMerkleRoot(hashes []string) string {
	if len(hashes) == 0 {
		return ""
	}
	
	if len(hashes) == 1 {
		return hashes[0]
	}
	
	var newHashes []string
	for i := 0; i < len(hashes); i += 2 {
		if i+1 < len(hashes) {
			combined := hashes[i] + hashes[i+1]
			h := sha256.Sum256([]byte(combined))
			newHashes = append(newHashes, hex.EncodeToString(h[:]))
		} else {
			// Odd number of hashes, duplicate the last one
			combined := hashes[i] + hashes[i]
			h := sha256.Sum256([]byte(combined))
			newHashes = append(newHashes, hex.EncodeToString(h[:]))
		}
	}
	
	return calculateMerkleRoot(newHashes)
}

// Validate performs validation on the block
func (b *Block) Validate() error {
	if b.Index < 0 {
		return fmt.Errorf("block index cannot be negative")
	}
	
	if b.Hash != b.CalculateHash() {
		return fmt.Errorf("block hash is invalid")
	}
	
	calculatedMerkleRoot := b.CalculateMerkleRoot()
	if b.MerkleRoot != calculatedMerkleRoot {
		return fmt.Errorf("merkle root mismatch")
	}
	
	for _, tx := range b.Transactions {
		if err := tx.Validate(); err != nil {
			return fmt.Errorf("invalid transaction: %v", err)
		}
	}
	
	return nil
}

// Blockchain represents the blockchain
type Blockchain struct {
	Blocks       []Block           `json:"blocks"`
	Difficulty   int               `json:"difficulty"`
	PendingTxs   []Transaction     `json:"pending_transactions"`
	MiningReward int64             `json:"mining_reward"`
	Balances     map[string]int64  `json:"balances"`
	mutex        sync.RWMutex
}

// NewBlockchain creates a new blockchain with genesis block
func NewBlockchain() *Blockchain {
	bc := &Blockchain{
		Difficulty:   4,
		MiningReward: 100,
		Balances:     make(map[string]int64),
	}
	
	// Create genesis block
	genesis := Block{
		Index:        0,
		Timestamp:    time.Now(),
		Transactions: []Transaction{},
		PrevHash:     "0",
		Difficulty:   bc.Difficulty,
	}
	
	genesis.MerkleRoot = genesis.CalculateMerkleRoot()
	genesis.Hash = genesis.CalculateHash()
	
	bc.Blocks = append(bc.Blocks, genesis)
	return bc
}

// GetLatestBlock returns the latest block in the chain
func (bc *Blockchain) GetLatestBlock() Block {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	return bc.Blocks[len(bc.Blocks)-1]
}

// AddTransaction adds a transaction to the pending pool
func (bc *Blockchain) AddTransaction(tx Transaction) error {
	bc.mutex.Lock()
	defer bc.mutex.Unlock()
	
	if err := tx.Validate(); err != nil {
		return err
	}
	
	// Check if sender has sufficient balance
	senderBalance := bc.Balances[tx.From]
	totalCost := tx.Amount + tx.Fee
	
	if senderBalance < totalCost {
		return fmt.Errorf("insufficient balance: have %d, need %d", senderBalance, totalCost)
	}
	
	tx.ID = tx.Hash()
	tx.Timestamp = time.Now()
	bc.PendingTxs = append(bc.PendingTxs, tx)
	
	return nil
}

// MinePendingTransactions mines a new block with pending transactions
func (bc *Blockchain) MinePendingTransactions(minerAddress string) error {
	bc.mutex.Lock()
	defer bc.mutex.Unlock()
	
	// Add mining reward transaction
	rewardTx := Transaction{
		From:      "system",
		To:        minerAddress,
		Amount:    bc.MiningReward,
		Fee:       0,
		Timestamp: time.Now(),
	}
	rewardTx.ID = rewardTx.Hash()
	
	transactions := append(bc.PendingTxs, rewardTx)
	
	block := Block{
		Index:        int64(len(bc.Blocks)),
		Timestamp:    time.Now(),
		Transactions: transactions,
		PrevHash:     bc.GetLatestBlock().Hash,
		Difficulty:   bc.Difficulty,
	}
	
	block.MerkleRoot = block.CalculateMerkleRoot()
	
	// Mine the block (Proof of Work)
	bc.mineBlock(&block)
	
	// Update balances
	for _, tx := range transactions {
		if tx.From != "system" {
			bc.Balances[tx.From] -= (tx.Amount + tx.Fee)
		}
		bc.Balances[tx.To] += tx.Amount
	}
	
	// Add block to chain and clear pending transactions
	bc.Blocks = append(bc.Blocks, block)
	bc.PendingTxs = []Transaction{}
	
	return nil
}

// mineBlock performs proof of work mining
func (bc *Blockchain) mineBlock(block *Block) {
	target := make([]byte, bc.Difficulty)
	for i := range target {
		target[i] = '0'
	}
	targetStr := string(target)
	
	for {
		hash := block.CalculateHash()
		if hash[:bc.Difficulty] == targetStr {
			block.Hash = hash
			break
		}
		block.Nonce++
	}
	
	logrus.Infof("Block mined: %s", block.Hash)
}

// GetBalance returns the balance for an address
func (bc *Blockchain) GetBalance(address string) int64 {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	return bc.Balances[address]
}

// IsChainValid validates the entire blockchain
func (bc *Blockchain) IsChainValid() bool {
	bc.mutex.RLock()
	defer bc.mutex.RUnlock()
	
	for i := 1; i < len(bc.Blocks); i++ {
		currentBlock := bc.Blocks[i]
		previousBlock := bc.Blocks[i-1]
		
		if err := currentBlock.Validate(); err != nil {
			logrus.Errorf("Block %d validation failed: %v", i, err)
			return false
		}
		
		if currentBlock.PrevHash != previousBlock.Hash {
			logrus.Errorf("Block %d has invalid previous hash", i)
			return false
		}
	}
	
	return true
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. CONSENSUS MECHANISMS
// ═══════════════════════════════════════════════════════════════════════════════

// ConsensusEngine interface for different consensus mechanisms
type ConsensusEngine interface {
	ValidateBlock(block *Block, prevBlock *Block) error
	CreateBlock(transactions []Transaction, prevBlock *Block, validator string) (*Block, error)
	SelectValidator(validators []string, round int64) string
}

// ProofOfWork consensus implementation
type ProofOfWork struct {
	Difficulty int
}

// ValidateBlock validates a block using PoW
func (pow *ProofOfWork) ValidateBlock(block *Block, prevBlock *Block) error {
	if block.PrevHash != prevBlock.Hash {
		return fmt.Errorf("invalid previous hash")
	}
	
	target := make([]byte, pow.Difficulty)
	for i := range target {
		target[i] = '0'
	}
	targetStr := string(target)
	
	if block.Hash[:pow.Difficulty] != targetStr {
		return fmt.Errorf("block hash doesn't meet difficulty requirement")
	}
	
	return block.Validate()
}

// CreateBlock creates a new block using PoW
func (pow *ProofOfWork) CreateBlock(transactions []Transaction, prevBlock *Block, validator string) (*Block, error) {
	block := &Block{
		Index:        prevBlock.Index + 1,
		Timestamp:    time.Now(),
		Transactions: transactions,
		PrevHash:     prevBlock.Hash,
		Difficulty:   pow.Difficulty,
	}
	
	block.MerkleRoot = block.CalculateMerkleRoot()
	
	// Mine the block
	target := make([]byte, pow.Difficulty)
	for i := range target {
		target[i] = '0'
	}
	targetStr := string(target)
	
	for {
		hash := block.CalculateHash()
		if hash[:pow.Difficulty] == targetStr {
			block.Hash = hash
			break
		}
		block.Nonce++
	}
	
	return block, nil
}

// SelectValidator not applicable for PoW
func (pow *ProofOfWork) SelectValidator(validators []string, round int64) string {
	return "" // Not applicable for PoW
}

// ProofOfStake consensus implementation
type ProofOfStake struct {
	Stakes map[string]int64 // validator address -> stake amount
}

// NewProofOfStake creates a new PoS consensus engine
func NewProofOfStake() *ProofOfStake {
	return &ProofOfStake{
		Stakes: make(map[string]int64),
	}
}

// ValidateBlock validates a block using PoS
func (pos *ProofOfStake) ValidateBlock(block *Block, prevBlock *Block) error {
	if block.PrevHash != prevBlock.Hash {
		return fmt.Errorf("invalid previous hash")
	}
	
	if block.Validator == "" {
		return fmt.Errorf("block must have a validator")
	}
	
	// Check if validator has sufficient stake
	if pos.Stakes[block.Validator] <= 0 {
		return fmt.Errorf("validator has no stake")
	}
	
	return block.Validate()
}

// CreateBlock creates a new block using PoS
func (pos *ProofOfStake) CreateBlock(transactions []Transaction, prevBlock *Block, validator string) (*Block, error) {
	if pos.Stakes[validator] <= 0 {
		return nil, fmt.Errorf("validator has no stake")
	}
	
	block := &Block{
		Index:        prevBlock.Index + 1,
		Timestamp:    time.Now(),
		Transactions: transactions,
		PrevHash:     prevBlock.Hash,
		Validator:    validator,
	}
	
	block.MerkleRoot = block.CalculateMerkleRoot()
	block.Hash = block.CalculateHash()
	
	return block, nil
}

// SelectValidator selects a validator based on stake
func (pos *ProofOfStake) SelectValidator(validators []string, round int64) string {
	if len(validators) == 0 {
		return ""
	}
	
	totalStake := int64(0)
	for _, validator := range validators {
		totalStake += pos.Stakes[validator]
	}
	
	if totalStake == 0 {
		return validators[round%int64(len(validators))]
	}
	
	// Weighted random selection based on stake
	randomValue := round % totalStake
	currentWeight := int64(0)
	
	for _, validator := range validators {
		currentWeight += pos.Stakes[validator]
		if randomValue < currentWeight {
			return validator
		}
	}
	
	return validators[0]
}

// AddStake adds or updates a validator's stake
func (pos *ProofOfStake) AddStake(validator string, amount int64) {
	pos.Stakes[validator] += amount
}

// RemoveStake removes stake from a validator
func (pos *ProofOfStake) RemoveStake(validator string, amount int64) error {
	if pos.Stakes[validator] < amount {
		return fmt.Errorf("insufficient stake to remove")
	}
	pos.Stakes[validator] -= amount
	return nil
}

// DelegatedProofOfStake consensus implementation
type DelegatedProofOfStake struct {
	Delegates    []string          // List of elected delegates
	Votes        map[string]string // voter -> delegate
	DelegateVotes map[string]int64 // delegate -> vote count
}

// NewDelegatedProofOfStake creates a new DPoS consensus engine
func NewDelegatedProofOfStake() *DelegatedProofOfStake {
	return &DelegatedProofOfStake{
		Delegates:     make([]string, 0),
		Votes:         make(map[string]string),
		DelegateVotes: make(map[string]int64),
	}
}

// ValidateBlock validates a block using DPoS
func (dpos *DelegatedProofOfStake) ValidateBlock(block *Block, prevBlock *Block) error {
	if block.PrevHash != prevBlock.Hash {
		return fmt.Errorf("invalid previous hash")
	}
	
	if block.Validator == "" {
		return fmt.Errorf("block must have a validator")
	}
	
	// Check if validator is an elected delegate
	isDelegate := false
	for _, delegate := range dpos.Delegates {
		if delegate == block.Validator {
			isDelegate = true
			break
		}
	}
	
	if !isDelegate {
		return fmt.Errorf("validator is not an elected delegate")
	}
	
	return block.Validate()
}

// CreateBlock creates a new block using DPoS
func (dpos *DelegatedProofOfStake) CreateBlock(transactions []Transaction, prevBlock *Block, validator string) (*Block, error) {
	// Check if validator is an elected delegate
	isDelegate := false
	for _, delegate := range dpos.Delegates {
		if delegate == validator {
			isDelegate = true
			break
		}
	}
	
	if !isDelegate {
		return nil, fmt.Errorf("validator is not an elected delegate")
	}
	
	block := &Block{
		Index:        prevBlock.Index + 1,
		Timestamp:    time.Now(),
		Transactions: transactions,
		PrevHash:     prevBlock.Hash,
		Validator:    validator,
	}
	
	block.MerkleRoot = block.CalculateMerkleRoot()
	block.Hash = block.CalculateHash()
	
	return block, nil
}

// SelectValidator selects the next delegate in round-robin fashion
func (dpos *DelegatedProofOfStake) SelectValidator(validators []string, round int64) string {
	if len(dpos.Delegates) == 0 {
		return ""
	}
	
	return dpos.Delegates[round%int64(len(dpos.Delegates))]
}

// Vote allows a user to vote for a delegate
func (dpos *DelegatedProofOfStake) Vote(voter, delegate string) {
	// Remove previous vote if exists
	if previousDelegate, exists := dpos.Votes[voter]; exists {
		dpos.DelegateVotes[previousDelegate]--
	}
	
	// Add new vote
	dpos.Votes[voter] = delegate
	dpos.DelegateVotes[delegate]++
}

// UpdateDelegates updates the list of elected delegates
func (dpos *DelegatedProofOfStake) UpdateDelegates(maxDelegates int) {
	type delegateVote struct {
		delegate string
		votes    int64
	}
	
	var sortedDelegates []delegateVote
	for delegate, votes := range dpos.DelegateVotes {
		sortedDelegates = append(sortedDelegates, delegateVote{delegate, votes})
	}
	
	// Sort by votes (descending)
	for i := 0; i < len(sortedDelegates)-1; i++ {
		for j := i + 1; j < len(sortedDelegates); j++ {
			if sortedDelegates[i].votes < sortedDelegates[j].votes {
				sortedDelegates[i], sortedDelegates[j] = sortedDelegates[j], sortedDelegates[i]
			}
		}
	}
	
	// Update delegates list
	dpos.Delegates = make([]string, 0)
	for i := 0; i < len(sortedDelegates) && i < maxDelegates; i++ {
		dpos.Delegates = append(dpos.Delegates, sortedDelegates[i].delegate)
	}
}