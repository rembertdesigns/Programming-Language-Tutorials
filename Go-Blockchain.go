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


// ═══════════════════════════════════════════════════════════════════════════════
//                           4. NETWORKING AND P2P COMMUNICATION
// ═══════════════════════════════════════════════════════════════════════════════

import (
	"bufio"
	"net"
	"strings"
	"sync"
)

// Node represents a blockchain network node
type Node struct {
	ID          string
	Address     string
	Port        int
	Blockchain  *Blockchain
	Peers       map[string]*Peer
	PeersMutex  sync.RWMutex
	MessageChan chan Message
	listener    net.Listener
	ctx         context.Context
	cancel      context.CancelFunc
}

// Peer represents a connected peer
type Peer struct {
	ID       string
	Address  string
	Port     int
	Conn     net.Conn
	LastSeen time.Time
}

// Message represents a network message
type Message struct {
	Type      string      `json:"type"`
	From      string      `json:"from"`
	To        string      `json:"to"`
	Timestamp time.Time   `json:"timestamp"`
	Data      interface{} `json:"data"`
}

// MessageType constants
const (
	MessageTypeBlock       = "block"
	MessageTypeTransaction = "transaction"
	MessageTypePeerList    = "peer_list"
	MessageTypeHandshake   = "handshake"
	MessageTypeHeartbeat   = "heartbeat"
	MessageTypeSync        = "sync"
)

// NewNode creates a new blockchain node
func NewNode(id, address string, port int, blockchain *Blockchain) *Node {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &Node{
		ID:          id,
		Address:     address,
		Port:        port,
		Blockchain:  blockchain,
		Peers:       make(map[string]*Peer),
		MessageChan: make(chan Message, 1000),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start starts the node and begins listening for connections
func (n *Node) Start() error {
	addr := fmt.Sprintf("%s:%d", n.Address, n.Port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to start listener: %v", err)
	}
	
	n.listener = listener
	logrus.Infof("Node %s started listening on %s", n.ID, addr)
	
	// Start message processor
	go n.processMessages()
	
	// Start peer maintenance
	go n.maintainPeers()
	
	// Accept incoming connections
	go n.acceptConnections()
	
	return nil
}

// Stop stops the node
func (n *Node) Stop() {
	n.cancel()
	
	if n.listener != nil {
		n.listener.Close()
	}
	
	n.PeersMutex.Lock()
	for _, peer := range n.Peers {
		peer.Conn.Close()
	}
	n.PeersMutex.Unlock()
	
	logrus.Infof("Node %s stopped", n.ID)
}

// acceptConnections accepts incoming peer connections
func (n *Node) acceptConnections() {
	for {
		select {
		case <-n.ctx.Done():
			return
		default:
			conn, err := n.listener.Accept()
			if err != nil {
				if n.ctx.Err() != nil {
					return
				}
				logrus.Errorf("Failed to accept connection: %v", err)
				continue
			}
			
			go n.handleConnection(conn)
		}
	}
}

// handleConnection handles a new peer connection
func (n *Node) handleConnection(conn net.Conn) {
	defer conn.Close()
	
	// Send handshake
	handshake := Message{
		Type:      MessageTypeHandshake,
		From:      n.ID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"node_id": n.ID,
			"version": "1.0.0",
		},
	}
	
	if err := n.sendMessage(conn, handshake); err != nil {
		logrus.Errorf("Failed to send handshake: %v", err)
		return
	}
	
	// Read messages from peer
	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		var msg Message
		if err := json.Unmarshal(scanner.Bytes(), &msg); err != nil {
			logrus.Errorf("Failed to unmarshal message: %v", err)
			continue
		}
		
		// Add to message queue
		select {
		case n.MessageChan <- msg:
		default:
			logrus.Warn("Message channel full, dropping message")
		}
	}
}

// sendMessage sends a message to a connection
func (n *Node) sendMessage(conn net.Conn, msg Message) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	
	_, err = conn.Write(append(data, '\n'))
	return err
}

// ConnectToPeer connects to a peer
func (n *Node) ConnectToPeer(address string, port int) error {
	addr := fmt.Sprintf("%s:%d", address, port)
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to connect to peer %s: %v", addr, err)
	}
	
	peer := &Peer{
		Address:  address,
		Port:     port,
		Conn:     conn,
		LastSeen: time.Now(),
	}
	
	n.PeersMutex.Lock()
	n.Peers[addr] = peer
	n.PeersMutex.Unlock()
	
	// Handle the connection
	go n.handleConnection(conn)
	
	logrus.Infof("Connected to peer %s", addr)
	return nil
}

// processMessages processes incoming messages
func (n *Node) processMessages() {
	for {
		select {
		case <-n.ctx.Done():
			return
		case msg := <-n.MessageChan:
			n.handleMessage(msg)
		}
	}
}

// handleMessage handles a received message
func (n *Node) handleMessage(msg Message) {
	logrus.Debugf("Received message type %s from %s", msg.Type, msg.From)
	
	switch msg.Type {
	case MessageTypeHandshake:
		n.handleHandshake(msg)
	case MessageTypeBlock:
		n.handleBlock(msg)
	case MessageTypeTransaction:
		n.handleTransaction(msg)
	case MessageTypePeerList:
		n.handlePeerList(msg)
	case MessageTypeHeartbeat:
		n.handleHeartbeat(msg)
	case MessageTypeSync:
		n.handleSync(msg)
	default:
		logrus.Warnf("Unknown message type: %s", msg.Type)
	}
}

// handleHandshake handles handshake messages
func (n *Node) handleHandshake(msg Message) {
	data := msg.Data.(map[string]interface{})
	peerID := data["node_id"].(string)
	
	// Update peer information
	n.PeersMutex.Lock()
	for _, peer := range n.Peers {
		if peer.ID == "" {
			peer.ID = peerID
			peer.LastSeen = time.Now()
			break
		}
	}
	n.PeersMutex.Unlock()
	
	logrus.Infof("Completed handshake with peer %s", peerID)
}

// handleBlock handles block messages
func (n *Node) handleBlock(msg Message) {
	blockData := msg.Data.(map[string]interface{})
	
	// Convert to Block struct (simplified)
	var block Block
	if err := json.Unmarshal([]byte(fmt.Sprintf("%v", blockData)), &block); err != nil {
		logrus.Errorf("Failed to unmarshal block: %v", err)
		return
	}
	
	// Validate the block
	if len(n.Blockchain.Blocks) > 0 {
		latestBlock := n.Blockchain.GetLatestBlock()
		if err := block.Validate(); err != nil {
			logrus.Errorf("Invalid block received: %v", err)
			return
		}
		
		if block.PrevHash != latestBlock.Hash {
			logrus.Errorf("Block has invalid previous hash")
			return
		}
	}
	
	// Add block to blockchain
	n.Blockchain.mutex.Lock()
	n.Blockchain.Blocks = append(n.Blockchain.Blocks, block)
	n.Blockchain.mutex.Unlock()
	
	// Broadcast to other peers
	n.broadcastMessage(msg, msg.From)
	
	logrus.Infof("Added new block %d to blockchain", block.Index)
}

// handleTransaction handles transaction messages
func (n *Node) handleTransaction(msg Message) {
	txData := msg.Data.(map[string]interface{})
	
	// Convert to Transaction struct (simplified)
	var tx Transaction
	if err := json.Unmarshal([]byte(fmt.Sprintf("%v", txData)), &tx); err != nil {
		logrus.Errorf("Failed to unmarshal transaction: %v", err)
		return
	}
	
	// Add to pending transactions
	if err := n.Blockchain.AddTransaction(tx); err != nil {
		logrus.Errorf("Failed to add transaction: %v", err)
		return
	}
	
	// Broadcast to other peers
	n.broadcastMessage(msg, msg.From)
	
	logrus.Infof("Added transaction %s to pending pool", tx.ID)
}

// handlePeerList handles peer list messages
func (n *Node) handlePeerList(msg Message) {
	peers := msg.Data.([]interface{})
	
	for _, peerData := range peers {
		peer := peerData.(map[string]interface{})
		address := peer["address"].(string)
		port := int(peer["port"].(float64))
		
		// Try to connect to new peers
		addr := fmt.Sprintf("%s:%d", address, port)
		if _, exists := n.Peers[addr]; !exists {
			go func() {
				if err := n.ConnectToPeer(address, port); err != nil {
					logrus.Debugf("Failed to connect to peer %s: %v", addr, err)
				}
			}()
		}
	}
}

// handleHeartbeat handles heartbeat messages
func (n *Node) handleHeartbeat(msg Message) {
	// Update peer last seen time
	n.PeersMutex.Lock()
	for _, peer := range n.Peers {
		if peer.ID == msg.From {
			peer.LastSeen = time.Now()
			break
		}
	}
	n.PeersMutex.Unlock()
}

// handleSync handles blockchain sync requests
func (n *Node) handleSync(msg Message) {
	// Send our blockchain to the requesting peer
	syncResponse := Message{
		Type:      "sync_response",
		From:      n.ID,
		To:        msg.From,
		Timestamp: time.Now(),
		Data:      n.Blockchain.Blocks,
	}
	
	n.sendMessageToPeer(msg.From, syncResponse)
}

// broadcastMessage broadcasts a message to all peers except the sender
func (n *Node) broadcastMessage(msg Message, excludePeer string) {
	n.PeersMutex.RLock()
	defer n.PeersMutex.RUnlock()
	
	for _, peer := range n.Peers {
		if peer.ID != excludePeer {
			go n.sendMessageToPeer(peer.ID, msg)
		}
	}
}

// sendMessageToPeer sends a message to a specific peer
func (n *Node) sendMessageToPeer(peerID string, msg Message) {
	n.PeersMutex.RLock()
	defer n.PeersMutex.RUnlock()
	
	for _, peer := range n.Peers {
		if peer.ID == peerID {
			if err := n.sendMessage(peer.Conn, msg); err != nil {
				logrus.Errorf("Failed to send message to peer %s: %v", peerID, err)
			}
			break
		}
	}
}

// maintainPeers maintains peer connections
func (n *Node) maintainPeers() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-n.ctx.Done():
			return
		case <-ticker.C:
			n.cleanupDeadPeers()
			n.sendHeartbeats()
		}
	}
}

// cleanupDeadPeers removes peers that haven't been seen recently
func (n *Node) cleanupDeadPeers() {
	n.PeersMutex.Lock()
	defer n.PeersMutex.Unlock()
	
	now := time.Now()
	for addr, peer := range n.Peers {
		if now.Sub(peer.LastSeen) > 5*time.Minute {
			peer.Conn.Close()
			delete(n.Peers, addr)
			logrus.Infof("Removed dead peer %s", addr)
		}
	}
}

// sendHeartbeats sends heartbeat messages to all peers
func (n *Node) sendHeartbeats() {
	heartbeat := Message{
		Type:      MessageTypeHeartbeat,
		From:      n.ID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"status": "alive"},
	}
	
	n.broadcastMessage(heartbeat, "")
}

// SyncWithNetwork syncs the blockchain with the network
func (n *Node) SyncWithNetwork() {
	syncRequest := Message{
		Type:      MessageTypeSync,
		From:      n.ID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"request": "blockchain"},
	}
	
	n.broadcastMessage(syncRequest, "")
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. STORAGE AND PERSISTENCE
// ═══════════════════════════════════════════════════════════════════════════════

import (
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"path/filepath"
)

// Storage interface for blockchain data persistence
type Storage interface {
	SaveBlock(block *Block) error
	GetBlock(hash string) (*Block, error)
	GetBlockByIndex(index int64) (*Block, error)
	SaveTransaction(tx *Transaction) error
	GetTransaction(id string) (*Transaction, error)
	SaveState(key string, value []byte) error
	GetState(key string) ([]byte, error)
	Close() error
}

// LevelDBStorage implements Storage using LevelDB
type LevelDBStorage struct {
	db       *leveldb.DB
	blockDB  *leveldb.DB
	txDB     *leveldb.DB
	stateDB  *leveldb.DB
}

// NewLevelDBStorage creates a new LevelDB storage
func NewLevelDBStorage(dataDir string) (*LevelDBStorage, error) {
	// Create directories
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}
	
	// Open databases
	blockDB, err := leveldb.OpenFile(filepath.Join(dataDir, "blocks"), nil)
	if err != nil {
		return nil, err
	}
	
	txDB, err := leveldb.OpenFile(filepath.Join(dataDir, "transactions"), nil)
	if err != nil {
		return nil, err
	}
	
	stateDB, err := leveldb.OpenFile(filepath.Join(dataDir, "state"), nil)
	if err != nil {
		return nil, err
	}
	
	return &LevelDBStorage{
		blockDB: blockDB,
		txDB:    txDB,
		stateDB: stateDB,
	}, nil
}

// SaveBlock saves a block to storage
func (s *LevelDBStorage) SaveBlock(block *Block) error {
	data, err := json.Marshal(block)
	if err != nil {
		return err
	}
	
	// Save by hash
	if err := s.blockDB.Put([]byte(block.Hash), data, nil); err != nil {
		return err
	}
	
	// Save by index for quick lookup
	indexKey := fmt.Sprintf("index_%d", block.Index)
	return s.blockDB.Put([]byte(indexKey), []byte(block.Hash), nil)
}

// GetBlock retrieves a block by hash
func (s *LevelDBStorage) GetBlock(hash string) (*Block, error) {
	data, err := s.blockDB.Get([]byte(hash), nil)
	if err != nil {
		return nil, err
	}
	
	var block Block
	if err := json.Unmarshal(data, &block); err != nil {
		return nil, err
	}
	
	return &block, nil
}

// GetBlockByIndex retrieves a block by index
func (s *LevelDBStorage) GetBlockByIndex(index int64) (*Block, error) {
	indexKey := fmt.Sprintf("index_%d", index)
	hash, err := s.blockDB.Get([]byte(indexKey), nil)
	if err != nil {
		return nil, err
	}
	
	return s.GetBlock(string(hash))
}

// SaveTransaction saves a transaction to storage
func (s *LevelDBStorage) SaveTransaction(tx *Transaction) error {
	data, err := json.Marshal(tx)
	if err != nil {
		return err
	}
	
	return s.txDB.Put([]byte(tx.ID), data, nil)
}

// GetTransaction retrieves a transaction by ID
func (s *LevelDBStorage) GetTransaction(id string) (*Transaction, error) {
	data, err := s.txDB.Get([]byte(id), nil)
	if err != nil {
		return nil, err
	}
	
	var tx Transaction
	if err := json.Unmarshal(data, &tx); err != nil {
		return nil, err
	}
	
	return &tx, nil
}

// SaveState saves state data
func (s *LevelDBStorage) SaveState(key string, value []byte) error {
	return s.stateDB.Put([]byte(key), value, nil)
}

// GetState retrieves state data
func (s *LevelDBStorage) GetState(key string) ([]byte, error) {
	return s.stateDB.Get([]byte(key), nil)
}

// Close closes all databases
func (s *LevelDBStorage) Close() error {
	if s.blockDB != nil {
		s.blockDB.Close()
	}
	if s.txDB != nil {
		s.txDB.Close()
	}
	if s.stateDB != nil {
		s.stateDB.Close()
	}
	return nil
}

// StateManager manages blockchain state
type StateManager struct {
	storage   Storage
	balances  map[string]int64
	nonces    map[string]uint64
	contracts map[string]*Contract
	mutex     sync.RWMutex
}

// Contract represents a smart contract
type Contract struct {
	Address string            `json:"address"`
	Code    []byte            `json:"code"`
	Storage map[string][]byte `json:"storage"`
	Owner   string            `json:"owner"`
}

// NewStateManager creates a new state manager
func NewStateManager(storage Storage) *StateManager {
	return &StateManager{
		storage:   storage,
		balances:  make(map[string]int64),
		nonces:    make(map[string]uint64),
		contracts: make(map[string]*Contract),
	}
}

// ApplyTransaction applies a transaction to the state
func (sm *StateManager) ApplyTransaction(tx *Transaction) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	// Validate transaction
	if err := tx.Validate(); err != nil {
		return err
	}
	
	// Check nonce
	if tx.Nonce != sm.nonces[tx.From] {
		return fmt.Errorf("invalid nonce: expected %d, got %d", sm.nonces[tx.From], tx.Nonce)
	}
	
	// Check balance
	senderBalance := sm.balances[tx.From]
	totalCost := tx.Amount + tx.Fee
	if senderBalance < totalCost {
		return fmt.Errorf("insufficient balance")
	}
	
	// Apply transaction
	sm.balances[tx.From] -= totalCost
	sm.balances[tx.To] += tx.Amount
	sm.nonces[tx.From]++
	
	// Save transaction
	return sm.storage.SaveTransaction(tx)
}

// GetBalance returns the balance for an address
func (sm *StateManager) GetBalance(address string) int64 {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	return sm.balances[address]
}

// GetNonce returns the nonce for an address
func (sm *StateManager) GetNonce(address string) uint64 {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	return sm.nonces[address]
}

// SetBalance sets the balance for an address
func (sm *StateManager) SetBalance(address string, balance int64) {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	sm.balances[address] = balance
}

// DeployContract deploys a smart contract
func (sm *StateManager) DeployContract(owner, address string, code []byte) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	if _, exists := sm.contracts[address]; exists {
		return fmt.Errorf("contract already exists at address %s", address)
	}
	
	contract := &Contract{
		Address: address,
		Code:    code,
		Storage: make(map[string][]byte),
		Owner:   owner,
	}
	
	sm.contracts[address] = contract
	
	// Save to storage
	data, err := json.Marshal(contract)
	if err != nil {
		return err
	}
	
	return sm.storage.SaveState(fmt.Sprintf("contract_%s", address), data)
}

// GetContract retrieves a contract
func (sm *StateManager) GetContract(address string) (*Contract, error) {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	
	if contract, exists := sm.contracts[address]; exists {
		return contract, nil
	}
	
	// Try to load from storage
	data, err := sm.storage.GetState(fmt.Sprintf("contract_%s", address))
	if err != nil {
		return nil, err
	}
	
	var contract Contract
	if err := json.Unmarshal(data, &contract); err != nil {
		return nil, err
	}
	
	sm.contracts[address] = &contract
	return &contract, nil
}

// SaveState saves the current state to storage
func (sm *StateManager) SaveState() error {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	
	// Save balances
	balancesData, err := json.Marshal(sm.balances)
	if err != nil {
		return err
	}
	if err := sm.storage.SaveState("balances", balancesData); err != nil {
		return err
	}
	
	// Save nonces
	noncesData, err := json.Marshal(sm.nonces)
	if err != nil {
		return err
	}
	if err := sm.storage.SaveState("nonces", noncesData); err != nil {
		return err
	}
	
	return nil
}

// LoadState loads state from storage
func (sm *StateManager) LoadState() error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	
	// Load balances
	if data, err := sm.storage.GetState("balances"); err == nil {
		json.Unmarshal(data, &sm.balances)
	}
	
	// Load nonces
	if data, err := sm.storage.GetState("nonces"); err == nil {
		json.Unmarshal(data, &sm.nonces)
	}
	
	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. VIRTUAL MACHINE AND SMART CONTRACTS
// ═══════════════════════════════════════════════════════════════════════════════

// VirtualMachine interface for executing smart contracts
type VirtualMachine interface {
	Execute(code []byte, input []byte, context *ExecutionContext) (*ExecutionResult, error)
	ValidateCode(code []byte) error
}

// ExecutionContext provides context for contract execution
type ExecutionContext struct {
	Caller    string
	Origin    string
	GasLimit  uint64
	GasPrice  uint64
	Value     int64
	BlockInfo *Block
	State     *StateManager
}

// ExecutionResult contains the result of contract execution
type ExecutionResult struct {
	Success     bool
	ReturnData  []byte
	GasUsed     uint64
	Error       error
	StateChanges map[string][]byte
}

// SimpleVM implements a basic virtual machine
type SimpleVM struct {
	gasTable map[string]uint64
}

// NewSimpleVM creates a new simple virtual machine
func NewSimpleVM() *SimpleVM {
	return &SimpleVM{
		gasTable: map[string]uint64{
			"ADD":    3,
			"SUB":    3,
			"MUL":    5,
			"DIV":    5,
			"STORE":  20,
			"LOAD":   5,
			"CALL":   700,
			"RETURN": 0,
		},
	}
}

// Execute executes bytecode
func (vm *SimpleVM) Execute(code []byte, input []byte, context *ExecutionContext) (*ExecutionResult, error) {
	result := &ExecutionResult{
		Success:      true,
		StateChanges: make(map[string][]byte),
	}
	
	// Simple stack-based execution (simplified example)
	stack := make([]int64, 0)
	memory := make(map[string][]byte)
	pc := 0 // program counter
	
	for pc < len(code) && result.GasUsed < context.GasLimit {
		opcode := code[pc]
		
		switch opcode {
		case 0x01: // ADD
			if len(stack) < 2 {
				result.Success = false
				result.Error = fmt.Errorf("stack underflow")
				return result, nil
			}
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			stack = append(stack, a+b)
			result.GasUsed += vm.gasTable["ADD"]
			
		case 0x02: // SUB
			if len(stack) < 2 {
				result.Success = false
				result.Error = fmt.Errorf("stack underflow")
				return result, nil
			}
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			stack = append(stack, a-b)
			result.GasUsed += vm.gasTable["SUB"]
			
		case 0x03: // MUL
			if len(stack) < 2 {
				result.Success = false
				result.Error = fmt.Errorf("stack underflow")
				return result, nil
			}
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			stack = append(stack, a*b)
			result.GasUsed += vm.gasTable["MUL"]
			
		case 0x50: // PUSH (simplified - pushes next byte as value)
			if pc+1 >= len(code) {
				result.Success = false
				result.Error = fmt.Errorf("unexpected end of code")
				return result, nil
			}
			pc++
			value := int64(code[pc])
			stack = append(stack, value)
			result.GasUsed += 3
			
		case 0x55: // STORE
			if len(stack) < 2 {
				result.Success = false
				result.Error = fmt.Errorf("stack underflow")
				return result, nil
			}
			value := stack[len(stack)-1]
			key := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			
			keyStr := fmt.Sprintf("%d", key)
			valueBytes := []byte(fmt.Sprintf("%d", value))
			memory[keyStr] = valueBytes
			result.StateChanges[keyStr] = valueBytes
			result.GasUsed += vm.gasTable["STORE"]
			
		case 0x54: // LOAD
			if len(stack) < 1 {
				result.Success = false
				result.Error = fmt.Errorf("stack underflow")
				return result, nil
			}
			key := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			
			keyStr := fmt.Sprintf("%d", key)
			if valueBytes, exists := memory[keyStr]; exists {
				value, _ := strconv.ParseInt(string(valueBytes), 10, 64)
				stack = append(stack, value)
			} else {
				stack = append(stack, 0)
			}
			result.GasUsed += vm.gasTable["LOAD"]
			
		case 0xF3: // RETURN
			if len(stack) > 0 {
				returnValue := stack[len(stack)-1]
				result.ReturnData = []byte(fmt.Sprintf("%d", returnValue))
			}
			result.GasUsed += vm.gasTable["RETURN"]
			return result, nil
			
		default:
			result.Success = false
			result.Error = fmt.Errorf("unknown opcode: 0x%02x", opcode)
			return result, nil
		}
		
		pc++
	}
	
	if result.GasUsed >= context.GasLimit {
		result.Success = false
		result.Error = fmt.Errorf("out of gas")
	}
	
	return result, nil
}

// ValidateCode validates smart contract bytecode
func (vm *SimpleVM) ValidateCode(code []byte) error {
	if len(code) == 0 {
		return fmt.Errorf("empty code")
	}
	
	if len(code) > 24576 { // Max code size (like Ethereum)
		return fmt.Errorf("code too large")
	}
	
	// Basic validation - check for valid opcodes
	for i := 0; i < len(code); i++ {
		opcode := code[i]
		switch opcode {
		case 0x01, 0x02, 0x03, 0x50, 0x54, 0x55, 0xF3:
			// Valid opcodes
		default:
			return fmt.Errorf("invalid opcode 0x%02x at position %d", opcode, i)
		}
	}
	
	return nil
}

// ContractManager manages smart contract deployment and execution
type ContractManager struct {
	vm           VirtualMachine
	stateManager *StateManager
	gasLimit     uint64
}

// NewContractManager creates a new contract manager
func NewContractManager(vm VirtualMachine, stateManager *StateManager) *ContractManager {
	return &ContractManager{
		vm:           vm,
		stateManager: stateManager,
		gasLimit:     1000000, // Default gas limit
	}
}

// DeployContract deploys a new smart contract
func (cm *ContractManager) DeployContract(owner string, code []byte, gasLimit uint64) (*ExecutionResult, error) {
	// Validate code
	if err := cm.vm.ValidateCode(code); err != nil {
		return nil, fmt.Errorf("invalid contract code: %v", err)
	}
	
	// Generate contract address (simplified)
	nonce := cm.stateManager.GetNonce(owner)
	addressData := fmt.Sprintf("%s%d", owner, nonce)
	hash := sha256.Sum256([]byte(addressData))
	contractAddress := hex.EncodeToString(hash[:20])
	
	// Deploy contract to state
	if err := cm.stateManager.DeployContract(owner, contractAddress, code); err != nil {
		return nil, err
	}
	
	// Execute constructor if present (simplified)
	context := &ExecutionContext{
		Caller:   owner,
		Origin:   owner,
		GasLimit: gasLimit,
		GasPrice: 1,
		Value:    0,
		State:    cm.stateManager,
	}
	
	result, err := cm.vm.Execute(code, []byte{}, context)
	if err != nil {
		return nil, err
	}
	
	result.ReturnData = []byte(contractAddress)
	return result, nil
}

// CallContract calls a smart contract function
func (cm *ContractManager) CallContract(caller, contractAddress string, input []byte, value int64, gasLimit uint64) (*ExecutionResult, error) {
	// Get contract
	contract, err := cm.stateManager.GetContract(contractAddress)
	if err != nil {
		return nil, fmt.Errorf("contract not found: %v", err)
	}
	
	// Create execution context
	context := &ExecutionContext{
		Caller:   caller,
		Origin:   caller,
		GasLimit: gasLimit,
		GasPrice: 1,
		Value:    value,
		State:    cm.stateManager,
	}
	
	// Execute contract
	result, err := cm.vm.Execute(contract.Code, input, context)
	if err != nil {
		return nil, err
	}
	
	// Apply state changes
	if result.Success {
		for key, value := range result.StateChanges {
			contract.Storage[key] = value
		}
		
		// Save updated contract
		contractData, _ := json.Marshal(contract)
		cm.stateManager.storage.SaveState(fmt.Sprintf("contract_%s", contractAddress), contractData)
	}
	
	return result, nil
}