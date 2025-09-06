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


// ═══════════════════════════════════════════════════════════════════════════════
//                           7. TRANSACTION POOL AND MEMPOOL
// ═══════════════════════════════════════════════════════════════════════════════

// TransactionPool manages pending transactions
type TransactionPool struct {
	pending      map[string]*Transaction // txID -> transaction
	queued       map[string][]*Transaction // address -> transactions
	all          map[string]*Transaction // all transactions
	priced       *TransactionsByPrice    // price-sorted transactions
	mutex        sync.RWMutex
	maxPoolSize  int
	maxPerSender int
}

// TransactionsByPrice implements sorting interface for transactions by gas price
type TransactionsByPrice []*Transaction

func (t TransactionsByPrice) Len() int { return len(t) }
func (t TransactionsByPrice) Less(i, j int) bool {
	return t[i].Fee > t[j].Fee // Higher fee first
}
func (t TransactionsByPrice) Swap(i, j int) { t[i], t[j] = t[j], t[i] }

// NewTransactionPool creates a new transaction pool
func NewTransactionPool(maxPoolSize, maxPerSender int) *TransactionPool {
	return &TransactionPool{
		pending:      make(map[string]*Transaction),
		queued:       make(map[string][]*Transaction),
		all:          make(map[string]*Transaction),
		priced:       &TransactionsByPrice{},
		maxPoolSize:  maxPoolSize,
		maxPerSender: maxPerSender,
	}
}

// AddTransaction adds a transaction to the pool
func (tp *TransactionPool) AddTransaction(tx *Transaction) error {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()
	
	// Validate transaction
	if err := tx.Validate(); err != nil {
		return fmt.Errorf("invalid transaction: %v", err)
	}
	
	// Check if transaction already exists
	if _, exists := tp.all[tx.ID]; exists {
		return fmt.Errorf("transaction already exists")
	}
	
	// Check pool size limit
	if len(tp.all) >= tp.maxPoolSize {
		// Remove lowest fee transaction
		tp.evictTransaction()
	}
	
	// Check per-sender limit
	senderTxs := len(tp.queued[tx.From])
	if senderTxs >= tp.maxPerSender {
		return fmt.Errorf("too many pending transactions for sender")
	}
	
	// Add to pool
	tp.all[tx.ID] = tx
	tp.pending[tx.ID] = tx
	
	// Add to queued transactions for sender
	tp.queued[tx.From] = append(tp.queued[tx.From], tx)
	
	// Add to price-sorted list
	*tp.priced = append(*tp.priced, tx)
	
	logrus.Debugf("Added transaction %s to pool", tx.ID)
	return nil
}

// RemoveTransaction removes a transaction from the pool
func (tp *TransactionPool) RemoveTransaction(txID string) {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()
	
	tx, exists := tp.all[txID]
	if !exists {
		return
	}
	
	// Remove from all maps
	delete(tp.all, txID)
	delete(tp.pending, txID)
	
	// Remove from sender's queued transactions
	senderTxs := tp.queued[tx.From]
	for i, queuedTx := range senderTxs {
		if queuedTx.ID == txID {
			tp.queued[tx.From] = append(senderTxs[:i], senderTxs[i+1:]...)
			break
		}
	}
	
	// Remove from price-sorted list
	for i, pricedTx := range *tp.priced {
		if pricedTx.ID == txID {
			*tp.priced = append((*tp.priced)[:i], (*tp.priced)[i+1:]...)
			break
		}
	}
	
	logrus.Debugf("Removed transaction %s from pool", txID)
}

// GetTransactions returns transactions for mining
func (tp *TransactionPool) GetTransactions(maxTxs int) []*Transaction {
	tp.mutex.RLock()
	defer tp.mutex.RUnlock()
	
	var transactions []*Transaction
	count := 0
	
	// Sort by fee (highest first)
	sort.Sort(*tp.priced)
	
	for _, tx := range *tp.priced {
		if count >= maxTxs {
			break
		}
		transactions = append(transactions, tx)
		count++
	}
	
	return transactions
}

// GetPendingCount returns the number of pending transactions
func (tp *TransactionPool) GetPendingCount() int {
	tp.mutex.RLock()
	defer tp.mutex.RUnlock()
	return len(tp.pending)
}

// GetTransaction retrieves a transaction by ID
func (tp *TransactionPool) GetTransaction(txID string) (*Transaction, bool) {
	tp.mutex.RLock()
	defer tp.mutex.RUnlock()
	
	tx, exists := tp.all[txID]
	return tx, exists
}

// evictTransaction removes the transaction with the lowest fee
func (tp *TransactionPool) evictTransaction() {
	if len(*tp.priced) == 0 {
		return
	}
	
	// Sort by fee (ascending)
	sort.Slice(*tp.priced, func(i, j int) bool {
		return (*tp.priced)[i].Fee < (*tp.priced)[j].Fee
	})
	
	// Remove the lowest fee transaction
	toRemove := (*tp.priced)[0]
	tp.RemoveTransaction(toRemove.ID)
}

// CleanExpiredTransactions removes expired transactions
func (tp *TransactionPool) CleanExpiredTransactions(maxAge time.Duration) {
	tp.mutex.Lock()
	defer tp.mutex.Unlock()
	
	now := time.Now()
	var toRemove []string
	
	for txID, tx := range tp.all {
		if now.Sub(tx.Timestamp) > maxAge {
			toRemove = append(toRemove, txID)
		}
	}
	
	for _, txID := range toRemove {
		delete(tp.all, txID)
		delete(tp.pending, txID)
		// Also remove from other maps...
	}
	
	logrus.Infof("Cleaned %d expired transactions", len(toRemove))
}

// Mempool represents a transaction mempool with additional features
type Mempool struct {
	*TransactionPool
	priorityQueue   *PriorityQueue
	feeEstimator    *FeeEstimator
	antiSpam        *AntiSpamManager
	reorgProtection *ReorgProtection
}

// PriorityQueue manages transaction prioritization
type PriorityQueue struct {
	transactions []*Transaction
	mutex        sync.RWMutex
}

// FeeEstimator estimates optimal transaction fees
type FeeEstimator struct {
	recentBlocks []*Block
	mutex        sync.RWMutex
}

// AntiSpamManager prevents spam transactions
type AntiSpamManager struct {
	ipLimits     map[string]*RateLimit
	addressLimits map[string]*RateLimit
	mutex        sync.RWMutex
}

// RateLimit tracks rate limiting information
type RateLimit struct {
	Count     int
	LastReset time.Time
	Limit     int
	Window    time.Duration
}

// ReorgProtection protects against blockchain reorganizations
type ReorgProtection struct {
	confirmedTxs map[string]int // txID -> confirmation count
	mutex        sync.RWMutex
}

// NewMempool creates a new mempool
func NewMempool(maxSize, maxPerSender int) *Mempool {
	return &Mempool{
		TransactionPool: NewTransactionPool(maxSize, maxPerSender),
		priorityQueue:   &PriorityQueue{},
		feeEstimator:    &FeeEstimator{},
		antiSpam:        &AntiSpamManager{
			ipLimits:      make(map[string]*RateLimit),
			addressLimits: make(map[string]*RateLimit),
		},
		reorgProtection: &ReorgProtection{
			confirmedTxs: make(map[string]int),
		},
	}
}

// EstimateFee estimates the optimal fee for a transaction
func (fe *FeeEstimator) EstimateFee(priority string) int64 {
	fe.mutex.RLock()
	defer fe.mutex.RUnlock()
	
	if len(fe.recentBlocks) == 0 {
		return 1000 // Default fee
	}
	
	var fees []int64
	for _, block := range fe.recentBlocks {
		for _, tx := range block.Transactions {
			fees = append(fees, tx.Fee)
		}
	}
	
	if len(fees) == 0 {
		return 1000
	}
	
	sort.Slice(fees, func(i, j int) bool { return fees[i] < fees[j] })
	
	switch priority {
	case "low":
		return fees[len(fees)/4] // 25th percentile
	case "medium":
		return fees[len(fees)/2] // 50th percentile (median)
	case "high":
		return fees[len(fees)*3/4] // 75th percentile
	default:
		return fees[len(fees)/2]
	}
}

// UpdateFeeEstimation updates fee estimation with new block
func (fe *FeeEstimator) UpdateFeeEstimation(block *Block) {
	fe.mutex.Lock()
	defer fe.mutex.Unlock()
	
	fe.recentBlocks = append(fe.recentBlocks, block)
	
	// Keep only last 100 blocks
	if len(fe.recentBlocks) > 100 {
		fe.recentBlocks = fe.recentBlocks[1:]
	}
}

// CheckRateLimit checks if an IP or address is rate limited
func (asm *AntiSpamManager) CheckRateLimit(ip, address string) error {
	asm.mutex.Lock()
	defer asm.mutex.Unlock()
	
	now := time.Now()
	
	// Check IP rate limit
	if ipLimit, exists := asm.ipLimits[ip]; exists {
		if now.Sub(ipLimit.LastReset) > ipLimit.Window {
			ipLimit.Count = 0
			ipLimit.LastReset = now
		}
		
		if ipLimit.Count >= ipLimit.Limit {
			return fmt.Errorf("IP rate limit exceeded")
		}
		
		ipLimit.Count++
	} else {
		asm.ipLimits[ip] = &RateLimit{
			Count:     1,
			LastReset: now,
			Limit:     100, // 100 transactions per hour per IP
			Window:    time.Hour,
		}
	}
	
	// Check address rate limit
	if addrLimit, exists := asm.addressLimits[address]; exists {
		if now.Sub(addrLimit.LastReset) > addrLimit.Window {
			addrLimit.Count = 0
			addrLimit.LastReset = now
		}
		
		if addrLimit.Count >= addrLimit.Limit {
			return fmt.Errorf("address rate limit exceeded")
		}
		
		addrLimit.Count++
	} else {
		asm.addressLimits[address] = &RateLimit{
			Count:     1,
			LastReset: now,
			Limit:     1000, // 1000 transactions per hour per address
			Window:    time.Hour,
		}
	}
	
	return nil
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           8. MINING AND BLOCK PRODUCTION
// ═══════════════════════════════════════════════════════════════════════════════

// Miner represents a blockchain miner
type Miner struct {
	id           string
	blockchain   *Blockchain
	mempool      *Mempool
	consensus    ConsensusEngine
	isActive     bool
	stopChan     chan bool
	newBlockChan chan *Block
	difficulty   int
	address      string
	workers      int
	mutex        sync.RWMutex
}

// NewMiner creates a new miner
func NewMiner(id string, blockchain *Blockchain, mempool *Mempool, consensus ConsensusEngine, address string) *Miner {
	return &Miner{
		id:           id,
		blockchain:   blockchain,
		mempool:      mempool,
		consensus:    consensus,
		address:      address,
		workers:      4, // Number of mining threads
		stopChan:     make(chan bool),
		newBlockChan: make(chan *Block, 10),
		difficulty:   4,
	}
}

// Start starts the mining process
func (m *Miner) Start() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	if m.isActive {
		return
	}
	
	m.isActive = true
	
	// Start mining workers
	for i := 0; i < m.workers; i++ {
		go m.miningWorker(i)
	}
	
	// Start block producer
	go m.blockProducer()
	
	logrus.Infof("Miner %s started with %d workers", m.id, m.workers)
}

// Stop stops the mining process
func (m *Miner) Stop() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	if !m.isActive {
		return
	}
	
	m.isActive = false
	close(m.stopChan)
	
	logrus.Infof("Miner %s stopped", m.id)
}

// miningWorker performs the actual mining work
func (m *Miner) miningWorker(workerID int) {
	logrus.Infof("Mining worker %d started", workerID)
	
	for {
		select {
		case <-m.stopChan:
			logrus.Infof("Mining worker %d stopped", workerID)
			return
		default:
			m.attemptMining(workerID)
		}
	}
}

// blockProducer creates new blocks to mine
func (m *Miner) blockProducer() {
	ticker := time.NewTicker(10 * time.Second) // Create new block template every 10 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.createBlockTemplate()
		}
	}
}

// createBlockTemplate creates a new block template for mining
func (m *Miner) createBlockTemplate() {
	// Get pending transactions
	transactions := m.mempool.GetTransactions(1000) // Max 1000 transactions per block
	
	if len(transactions) == 0 {
		return // No transactions to mine
	}
	
	// Get previous block
	prevBlock := m.blockchain.GetLatestBlock()
	
	// Create new block
	block, err := m.consensus.CreateBlock(transactionSliceToArray(transactions), &prevBlock, m.address)
	if err != nil {
		logrus.Errorf("Failed to create block template: %v", err)
		return
	}
	
	// Send to workers
	select {
	case m.newBlockChan <- block:
	default:
		// Channel full, skip this template
	}
}

// attemptMining attempts to mine a block
func (m *Miner) attemptMining(workerID int) {
	select {
	case block := <-m.newBlockChan:
		m.mineBlock(block, workerID)
	case <-time.After(1 * time.Second):
		// Timeout, continue loop
	}
}

// mineBlock performs proof-of-work mining on a block
func (m *Miner) mineBlock(block *Block, workerID int) {
	startNonce := int64(workerID * 1000000)
	block.Nonce = startNonce
	
	target := strings.Repeat("0", m.difficulty)
	attempts := 0
	maxAttempts := 1000000 // Limit attempts per worker
	
	for attempts < maxAttempts {
		select {
		case <-m.stopChan:
			return
		default:
			hash := block.CalculateHash()
			if strings.HasPrefix(hash, target) {
				// Found valid hash!
				block.Hash = hash
				m.submitBlock(block, workerID)
				return
			}
			
			block.Nonce++
			attempts++
		}
	}
}

// submitBlock submits a successfully mined block
func (m *Miner) submitBlock(block *Block, workerID int) {
	// Validate the block
	if err := block.Validate(); err != nil {
		logrus.Errorf("Mined block is invalid: %v", err)
		return
	}
	
	// Add to blockchain
	m.blockchain.mutex.Lock()
	defer m.blockchain.mutex.Unlock()
	
	// Double-check that this block is still valid (no competing block)
	currentLatest := m.blockchain.GetLatestBlock()
	if block.PrevHash != currentLatest.Hash {
		logrus.Warnf("Block is stale, discarding")
		return
	}
	
	// Add block to chain
	m.blockchain.Blocks = append(m.blockchain.Blocks, *block)
	
	// Remove mined transactions from mempool
	for _, tx := range block.Transactions {
		m.mempool.RemoveTransaction(tx.ID)
	}
	
	// Update balances
	for _, tx := range block.Transactions {
		if tx.From != "system" {
			m.blockchain.Balances[tx.From] -= (tx.Amount + tx.Fee)
		}
		m.blockchain.Balances[tx.To] += tx.Amount
	}
	
	logrus.Infof("Worker %d successfully mined block %d with hash %s", 
		workerID, block.Index, block.Hash[:16])
}

// transactionSliceToArray converts transaction pointers to values
func transactionSliceToArray(txs []*Transaction) []Transaction {
	result := make([]Transaction, len(txs))
	for i, tx := range txs {
		result[i] = *tx
	}
	return result
}

// MiningPool represents a mining pool
type MiningPool struct {
	name         string
	miners       map[string]*PoolMiner
	rewards      map[string]int64
	totalShares  int64
	difficulty   int
	mutex        sync.RWMutex
}

// PoolMiner represents a miner in a pool
type PoolMiner struct {
	ID          string
	Address     string
	HashRate    int64
	Shares      int64
	LastActive  time.Time
	IsConnected bool
}

// NewMiningPool creates a new mining pool
func NewMiningPool(name string) *MiningPool {
	return &MiningPool{
		name:     name,
		miners:   make(map[string]*PoolMiner),
		rewards:  make(map[string]int64),
		difficulty: 4,
	}
}

// AddMiner adds a miner to the pool
func (mp *MiningPool) AddMiner(minerID, address string) {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	mp.miners[minerID] = &PoolMiner{
		ID:          minerID,
		Address:     address,
		LastActive:  time.Now(),
		IsConnected: true,
	}
	
	logrus.Infof("Miner %s joined pool %s", minerID, mp.name)
}

// RemoveMiner removes a miner from the pool
func (mp *MiningPool) RemoveMiner(minerID string) {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	if miner, exists := mp.miners[minerID]; exists {
		miner.IsConnected = false
		logrus.Infof("Miner %s left pool %s", minerID, mp.name)
	}
}

// SubmitShare submits a mining share
func (mp *MiningPool) SubmitShare(minerID string, nonce int64, hash string) bool {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	miner, exists := mp.miners[minerID]
	if !exists {
		return false
	}
	
	// Validate share (simplified)
	if mp.isValidShare(hash) {
		miner.Shares++
		miner.LastActive = time.Now()
		mp.totalShares++
		return true
	}
	
	return false
}

// isValidShare validates a mining share
func (mp *MiningPool) isValidShare(hash string) bool {
	// Simplified validation - check if hash meets pool difficulty
	target := strings.Repeat("0", mp.difficulty-1) // Pool difficulty is lower than network
	return strings.HasPrefix(hash, target)
}

// DistributeRewards distributes mining rewards to pool members
func (mp *MiningPool) DistributeRewards(blockReward int64) {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	if mp.totalShares == 0 {
		return
	}
	
	// Distribute based on shares (PPLNS - Pay Per Last N Shares)
	for minerID, miner := range mp.miners {
		if miner.Shares > 0 {
			reward := (blockReward * miner.Shares) / mp.totalShares
			mp.rewards[minerID] += reward
			
			logrus.Infof("Miner %s earned %d for %d shares", minerID, reward, miner.Shares)
		}
	}
	
	// Reset shares for next round
	for _, miner := range mp.miners {
		miner.Shares = 0
	}
	mp.totalShares = 0
}

// GetMinerReward returns a miner's accumulated reward
func (mp *MiningPool) GetMinerReward(minerID string) int64 {
	mp.mutex.RLock()
	defer mp.mutex.RUnlock()
	
	return mp.rewards[minerID]
}

// PayoutReward pays out a miner's reward
func (mp *MiningPool) PayoutReward(minerID string) int64 {
	mp.mutex.Lock()
	defer mp.mutex.Unlock()
	
	reward := mp.rewards[minerID]
	mp.rewards[minerID] = 0
	
	logrus.Infof("Paid out %d to miner %s", reward, minerID)
	return reward
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           9. API AND RPC SERVICES
// ═══════════════════════════════════════════════════════════════════════════════

import (
	"encoding/json"
	"net/http"
	"strconv"
	
	"github.com/gorilla/mux"
)

// APIServer provides HTTP API for the blockchain
type APIServer struct {
	blockchain *Blockchain
	mempool    *Mempool
	miner      *Miner
	node       *Node
	port       int
	server     *http.Server
}

// NewAPIServer creates a new API server
func NewAPIServer(blockchain *Blockchain, mempool *Mempool, miner *Miner, node *Node, port int) *APIServer {
	return &APIServer{
		blockchain: blockchain,
		mempool:    mempool,
		miner:      miner,
		node:       node,
		port:       port,
	}
}

// Start starts the API server
func (api *APIServer) Start() error {
	router := mux.NewRouter()
	
	// Blockchain endpoints
	router.HandleFunc("/api/blockchain", api.getBlockchain).Methods("GET")
	router.HandleFunc("/api/blocks", api.getBlocks).Methods("GET")
	router.HandleFunc("/api/blocks/{index}", api.getBlock).Methods("GET")
	router.HandleFunc("/api/blocks/latest", api.getLatestBlock).Methods("GET")
	
	// Transaction endpoints
	router.HandleFunc("/api/transactions", api.getTransactions).Methods("GET")
	router.HandleFunc("/api/transactions", api.submitTransaction).Methods("POST")
	router.HandleFunc("/api/transactions/{id}", api.getTransaction).Methods("GET")
	router.HandleFunc("/api/transactions/pending", api.getPendingTransactions).Methods("GET")
	
	// Balance endpoints
	router.HandleFunc("/api/balance/{address}", api.getBalance).Methods("GET")
	
	// Mining endpoints
	router.HandleFunc("/api/mining/start", api.startMining).Methods("POST")
	router.HandleFunc("/api/mining/stop", api.stopMining).Methods("POST")
	router.HandleFunc("/api/mining/status", api.getMiningStatus).Methods("GET")
	
	// Network endpoints
	router.HandleFunc("/api/network/peers", api.getPeers).Methods("GET")
	router.HandleFunc("/api/network/status", api.getNetworkStatus).Methods("GET")
	
	// Health check
	router.HandleFunc("/health", api.healthCheck).Methods("GET")
	
	// Enable CORS
	router.Use(api.corsMiddleware)
	
	api.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", api.port),
		Handler: router,
	}
	
	logrus.Infof("API server starting on port %d", api.port)
	return api.server.ListenAndServe()
}

// Stop stops the API server
func (api *APIServer) Stop() error {
	if api.server != nil {
		return api.server.Shutdown(context.Background())
	}
	return nil
}

// corsMiddleware adds CORS headers
func (api *APIServer) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		next.ServeHTTP(w, r)
	})
}

// getBlockchain returns the entire blockchain
func (api *APIServer) getBlockchain(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(api.blockchain)
}

// getBlocks returns blocks with pagination
func (api *APIServer) getBlocks(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	// Parse query parameters
	limitStr := r.URL.Query().Get("limit")
	offsetStr := r.URL.Query().Get("offset")
	
	limit := 10 // default
	offset := 0 // default
	
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}
	
	if offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil {
			offset = o
		}
	}
	
	api.blockchain.mutex.RLock()
	defer api.blockchain.mutex.RUnlock()
	
	blocks := api.blockchain.Blocks
	total := len(blocks)
	
	// Calculate slice bounds
	start := offset
	end := offset + limit
	
	if start >= total {
		start = total
		end = total
	} else if end > total {
		end = total
	}
	
	result := map[string]interface{}{
		"blocks": blocks[start:end],
		"total":  total,
		"limit":  limit,
		"offset": offset,
	}
	
	json.NewEncoder(w).Encode(result)
}

// getBlock returns a specific block by index
func (api *APIServer) getBlock(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	vars := mux.Vars(r)
	indexStr := vars["index"]
	
	index, err := strconv.ParseInt(indexStr, 10, 64)
	if err != nil {
		http.Error(w, "Invalid block index", http.StatusBadRequest)
		return
	}
	
	api.blockchain.mutex.RLock()
	defer api.blockchain.mutex.RUnlock()
	
	if index < 0 || int(index) >= len(api.blockchain.Blocks) {
		http.Error(w, "Block not found", http.StatusNotFound)
		return
	}
	
	block := api.blockchain.Blocks[index]
	json.NewEncoder(w).Encode(block)
}

// getLatestBlock returns the latest block
func (api *APIServer) getLatestBlock(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	latestBlock := api.blockchain.GetLatestBlock()
	json.NewEncoder(w).Encode(latestBlock)
}

// submitTransaction submits a new transaction
func (api *APIServer) submitTransaction(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	var tx Transaction
	if err := json.NewDecoder(r.Body).Decode(&tx); err != nil {
		http.Error(w, "Invalid transaction format", http.StatusBadRequest)
		return
	}
	
	if err := api.mempool.AddTransaction(&tx); err != nil {
		http.Error(w, fmt.Sprintf("Failed to add transaction: %v", err), http.StatusBadRequest)
		return
	}
	
	result := map[string]interface{}{
		"success": true,
		"tx_id":   tx.ID,
		"message": "Transaction added to mempool",
	}
	
	json.NewEncoder(w).Encode(result)
}

// getTransaction returns a specific transaction
func (api *APIServer) getTransaction(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	vars := mux.Vars(r)
	txID := vars["id"]
	
	// First check mempool
	if tx, exists := api.mempool.GetTransaction(txID); exists {
		json.NewEncoder(w).Encode(tx)
		return
	}
	
	// Then check blockchain
	api.blockchain.mutex.RLock()
	defer api.blockchain.mutex.RUnlock()
	
	for _, block := range api.blockchain.Blocks {
		for _, tx := range block.Transactions {
			if tx.ID == txID {
				json.NewEncoder(w).Encode(tx)
				return
			}
		}
	}
	
	http.Error(w, "Transaction not found", http.StatusNotFound)
}

// getPendingTransactions returns pending transactions
func (api *APIServer) getPendingTransactions(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	transactions := api.mempool.GetTransactions(100)
	
	result := map[string]interface{}{
		"transactions": transactions,
		"count":        len(transactions),
	}
	
	json.NewEncoder(w).Encode(result)
}

// getBalance returns the balance for an address
func (api *APIServer) getBalance(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	vars := mux.Vars(r)
	address := vars["address"]
	
	balance := api.blockchain.GetBalance(address)
	
	result := map[string]interface{}{
		"address": address,
		"balance": balance,
	}
	
	json.NewEncoder(w).Encode(result)
}

// startMining starts the mining process
func (api *APIServer) startMining(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	api.miner.Start()
	
	result := map[string]interface{}{
		"success": true,
		"message": "Mining started",
	}
	
	json.NewEncoder(w).Encode(result)
}

// stopMining stops the mining process
func (api *APIServer) stopMining(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	api.miner.Stop()
	
	result := map[string]interface{}{
		"success": true,
		"message": "Mining stopped",
	}
	
	json.NewEncoder(w).Encode(result)
}

// getMiningStatus returns mining status
func (api *APIServer) getMiningStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	api.miner.mutex.RLock()
	defer api.miner.mutex.RUnlock()
	
	result := map[string]interface{}{
		"is_mining":     api.miner.isActive,
		"miner_id":      api.miner.id,
		"miner_address": api.miner.address,
		"workers":       api.miner.workers,
		"difficulty":    api.miner.difficulty,
	}
	
	json.NewEncoder(w).Encode(result)
}

// getPeers returns connected peers
func (api *APIServer) getPeers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	api.node.PeersMutex.RLock()
	defer api.node.PeersMutex.RUnlock()
	
	var peers []map[string]interface{}
	for _, peer := range api.node.Peers {
		peers = append(peers, map[string]interface{}{
			"id":        peer.ID,
			"address":   peer.Address,
			"port":      peer.Port,
			"last_seen": peer.LastSeen,
		})
	}
	
	result := map[string]interface{}{
		"peers": peers,
		"count": len(peers),
	}
	
	json.NewEncoder(w).Encode(result)
}

// getNetworkStatus returns network status
func (api *APIServer) getNetworkStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	api.node.PeersMutex.RLock()
	peerCount := len(api.node.Peers)
	api.node.PeersMutex.RUnlock()
	
	api.blockchain.mutex.RLock()
	blockCount := len(api.blockchain.Blocks)
	api.blockchain.mutex.RUnlock()
	
	result := map[string]interface{}{
		"node_id":     api.node.ID,
		"peers":       peerCount,
		"blocks":      blockCount,
		"pending_txs":// handleBlock handles block messages
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