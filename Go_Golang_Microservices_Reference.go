// GO (GOLANG) HIGH-PERFORMANCE APIs & MICROSERVICES - Comprehensive Reference - by Richard Rembert
// Go excels at building high-performance, concurrent APIs and microservices with excellent
// performance characteristics, built-in concurrency, and minimal resource footprint

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════════════════

/*
GO DEVELOPMENT SETUP:

1. Install Go:
   # Download from https://golang.org/dl/
   # Or using package managers:
   
   # macOS (Homebrew)
   brew install go
   
   # Ubuntu/Debian
   sudo apt update
   sudo apt install golang-go
   
   # Verify installation
   go version

2. Environment Setup:
   # Add to ~/.bashrc or ~/.zshrc
   export GOROOT=/usr/local/go
   export GOPATH=$HOME/go
   export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
   
   # For Go 1.11+ with modules (recommended)
   export GO111MODULE=on

3. Initialize New Project:
   mkdir my-microservice
   cd my-microservice
   go mod init github.com/username/my-microservice

4. Essential Dependencies (go.mod):
   module github.com/username/my-microservice
   
   go 1.21
   
   require (
       // Web Framework
       github.com/gin-gonic/gin v1.9.1
       github.com/gorilla/mux v1.8.0
       github.com/labstack/echo/v4 v4.11.1
       
       // Database
       github.com/lib/pq v1.10.9
       gorm.io/gorm v1.25.4
       gorm.io/driver/postgres v1.5.2
       github.com/go-redis/redis/v8 v8.11.5
       
       // Microservices
       google.golang.org/grpc v1.57.0
       google.golang.org/protobuf v1.31.0
       github.com/nats-io/nats.go v1.28.0
       
       // Monitoring & Logging
       github.com/prometheus/client_golang v1.16.0
       github.com/sirupsen/logrus v1.9.3
       go.uber.org/zap v1.25.0
       
       // Configuration
       github.com/spf13/viper v1.16.0
       
       // Authentication & Security
       github.com/golang-jwt/jwt/v5 v5.0.0
       golang.org/x/crypto v0.12.0
       
       // Testing
       github.com/stretchr/testify v1.8.4
       github.com/testcontainers/testcontainers-go v0.22.0
       
       // Utilities
       github.com/google/uuid v1.3.0
       github.com/shopspring/decimal v1.3.1
   )

5. Project Structure:
   my-microservice/
   ├── cmd/
   │   ├── api/
   │   │   └── main.go
   │   └── worker/
   │       └── main.go
   ├── internal/
   │   ├── config/
   │   ├── handlers/
   │   ├── middleware/
   │   ├── models/
   │   ├── repository/
   │   ├── service/
   │   └── utils/
   ├── pkg/
   │   ├── logger/
   │   ├── database/
   │   └── auth/
   ├── api/
   │   └── proto/
   ├── deployments/
   │   ├── docker/
   │   └── k8s/
   ├── scripts/
   ├── tests/
   ├── docs/
   ├── go.mod
   ├── go.sum
   ├── Dockerfile
   └── README.md

6. Build and Run:
   # Development
   go run cmd/api/main.go
   
   # Build binary
   go build -o bin/api cmd/api/main.go
   
   # Cross-compile
   GOOS=linux GOARCH=amd64 go build -o bin/api-linux cmd/api/main.go
   
   # Install dependencies
   go mod tidy
   go mod vendor
*/

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/spf13/viper"
)

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. CONFIGURATION MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

// Config holds all configuration for the application
type Config struct {
	Server   ServerConfig   `mapstructure:"server"`
	Database DatabaseConfig `mapstructure:"database"`
	Redis    RedisConfig    `mapstructure:"redis"`
	JWT      JWTConfig      `mapstructure:"jwt"`
	Logging  LoggingConfig  `mapstructure:"logging"`
	Metrics  MetricsConfig  `mapstructure:"metrics"`
}

type ServerConfig struct {
	Host         string        `mapstructure:"host"`
	Port         int           `mapstructure:"port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
	IdleTimeout  time.Duration `mapstructure:"idle_timeout"`
	GracefulStop time.Duration `mapstructure:"graceful_stop"`
}

type DatabaseConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	User         string `mapstructure:"user"`
	Password     string `mapstructure:"password"`
	Name         string `mapstructure:"name"`
	SSLMode      string `mapstructure:"ssl_mode"`
	MaxOpenConns int    `mapstructure:"max_open_conns"`
	MaxIdleConns int    `mapstructure:"max_idle_conns"`
	MaxLifetime  time.Duration `mapstructure:"max_lifetime"`
}

type RedisConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Password string `mapstructure:"password"`
	DB       int    `mapstructure:"db"`
	PoolSize int    `mapstructure:"pool_size"`
}

type JWTConfig struct {
	SecretKey      string        `mapstructure:"secret_key"`
	ExpirationTime time.Duration `mapstructure:"expiration_time"`
	Issuer         string        `mapstructure:"issuer"`
}

type LoggingConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
	Output string `mapstructure:"output"`
}

type MetricsConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	Path    string `mapstructure:"path"`
	Port    int    `mapstructure:"port"`
}

// LoadConfig loads configuration from files and environment variables
func LoadConfig() (*Config, error) {
	// Set default values
	viper.SetDefault("server.host", "0.0.0.0")
	viper.SetDefault("server.port", 8080)
	viper.SetDefault("server.read_timeout", "10s")
	viper.SetDefault("server.write_timeout", "10s")
	viper.SetDefault("server.idle_timeout", "60s")
	viper.SetDefault("server.graceful_stop", "30s")
	
	viper.SetDefault("database.host", "localhost")
	viper.SetDefault("database.port", 5432)
	viper.SetDefault("database.ssl_mode", "disable")
	viper.SetDefault("database.max_open_conns", 25)
	viper.SetDefault("database.max_idle_conns", 25)
	viper.SetDefault("database.max_lifetime", "5m")
	
	viper.SetDefault("redis.host", "localhost")
	viper.SetDefault("redis.port", 6379)
	viper.SetDefault("redis.db", 0)
	viper.SetDefault("redis.pool_size", 10)
	
	viper.SetDefault("jwt.expiration_time", "24h")
	viper.SetDefault("jwt.issuer", "my-microservice")
	
	viper.SetDefault("logging.level", "info")
	viper.SetDefault("logging.format", "json")
	viper.SetDefault("logging.output", "stdout")
	
	viper.SetDefault("metrics.enabled", true)
	viper.SetDefault("metrics.path", "/metrics")
	viper.SetDefault("metrics.port", 9090)

	// Environment variable binding
	viper.AutomaticEnv()
	viper.SetEnvPrefix("APP")
	
	// Configuration file
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("./configs")
	viper.AddConfigPath("/etc/myservice")

	// Read config file (optional)
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, err
		}
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, err
	}

	return &config, nil
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. LOGGING AND MONITORING
// ═══════════════════════════════════════════════════════════════════════════════

package logger

import (
	"context"
	"os"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Logger wraps zap logger with additional functionality
type Logger struct {
	*zap.Logger
	sugar *zap.SugaredLogger
}

// NewLogger creates a new logger instance
func NewLogger(level, format, output string) (*Logger, error) {
	var config zap.Config
	
	switch format {
	case "json":
		config = zap.NewProductionConfig()
	default:
		config = zap.NewDevelopmentConfig()
	}
	
	// Set log level
	switch level {
	case "debug":
		config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	case "info":
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	case "warn":
		config.Level = zap.NewAtomicLevelAt(zap.WarnLevel)
	case "error":
		config.Level = zap.NewAtomicLevelAt(zap.ErrorLevel)
	default:
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
	}
	
	// Set output
	if output != "stdout" {
		config.OutputPaths = []string{output}
	}
	
	// Add caller information
	config.EncoderConfig.CallerKey = "caller"
	config.DisableCaller = false
	
	// Add timestamp
	config.EncoderConfig.TimeKey = "timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	
	zapLogger, err := config.Build()
	if err != nil {
		return nil, err
	}
	
	return &Logger{
		Logger: zapLogger,
		sugar:  zapLogger.Sugar(),
	}, nil
}

// WithContext adds context fields to logger
func (l *Logger) WithContext(ctx context.Context) *Logger {
	fields := extractFieldsFromContext(ctx)
	newLogger := l.Logger.With(fields...)
	return &Logger{
		Logger: newLogger,
		sugar:  newLogger.Sugar(),
	}
}

// WithFields adds custom fields to logger
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	zapFields := make([]zap.Field, 0, len(fields))
	for k, v := range fields {
		zapFields = append(zapFields, zap.Any(k, v))
	}
	newLogger := l.Logger.With(zapFields...)
	return &Logger{
		Logger: newLogger,
		sugar:  newLogger.Sugar(),
	}
}

// Structured logging methods
func (l *Logger) InfoCtx(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithContext(ctx).Info(msg, fields...)
}

func (l *Logger) ErrorCtx(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithContext(ctx).Error(msg, fields...)
}

func (l *Logger) WarnCtx(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithContext(ctx).Warn(msg, fields...)
}

func (l *Logger) DebugCtx(ctx context.Context, msg string, fields ...zap.Field) {
	l.WithContext(ctx).Debug(msg, fields...)
}

// Sugar methods for easier usage
func (l *Logger) Infof(template string, args ...interface{}) {
	l.sugar.Infof(template, args...)
}

func (l *Logger) Errorf(template string, args ...interface{}) {
	l.sugar.Errorf(template, args...)
}

func (l *Logger) Warnf(template string, args ...interface{}) {
	l.sugar.Warnf(template, args...)
}

func (l *Logger) Debugf(template string, args ...interface{}) {
	l.sugar.Debugf(template, args...)
}

// extractFieldsFromContext extracts logging fields from context
func extractFieldsFromContext(ctx context.Context) []zap.Field {
	var fields []zap.Field
	
	if requestID := ctx.Value("request_id"); requestID != nil {
		fields = append(fields, zap.String("request_id", requestID.(string)))
	}
	
	if userID := ctx.Value("user_id"); userID != nil {
		fields = append(fields, zap.String("user_id", userID.(string)))
	}
	
	if traceID := ctx.Value("trace_id"); traceID != nil {
		fields = append(fields, zap.String("trace_id", traceID.(string)))
	}
	
	return fields
}

// Metrics package for Prometheus integration
package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// HTTP metrics
	HTTPRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status_code"},
	)

	HTTPRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	// Database metrics
	DatabaseConnectionsActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "database_connections_active",
			Help: "Number of active database connections",
		},
	)

	DatabaseQueryDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "database_query_duration_seconds",
			Help:    "Database query duration in seconds",
			Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		},
		[]string{"operation", "table"},
	)

	// Application metrics
	ActiveUsers = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "active_users_total",
			Help: "Number of currently active users",
		},
	)

	BusinessOperationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "business_operations_total",
			Help: "Total number of business operations",
		},
		[]string{"operation", "status"},
	)
)

// RecordHTTPRequest records HTTP request metrics
func RecordHTTPRequest(method, endpoint, statusCode string, duration time.Duration) {
	HTTPRequestsTotal.WithLabelValues(method, endpoint, statusCode).Inc()
	HTTPRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

// RecordDatabaseQuery records database query metrics
func RecordDatabaseQuery(operation, table string, duration time.Duration) {
	DatabaseQueryDuration.WithLabelValues(operation, table).Observe(duration.Seconds())
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. DATABASE LAYER AND MODELS
// ═══════════════════════════════════════════════════════════════════════════════

package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	gormLogger "gorm.io/gorm/logger"
	"github.com/go-redis/redis/v8"
	_ "github.com/lib/pq"
)

// Database holds database connections
type Database struct {
	DB    *gorm.DB
	Redis *redis.Client
}

// NewDatabase creates a new database instance
func NewDatabase(dbConfig DatabaseConfig, redisConfig RedisConfig) (*Database, error) {
	// PostgreSQL connection
	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		dbConfig.Host, dbConfig.Port, dbConfig.User, dbConfig.Password, dbConfig.Name, dbConfig.SSLMode)
	
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: gormLogger.Default.LogMode(gormLogger.Info),
		NowFunc: func() time.Time {
			return time.Now().UTC()
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure connection pool
	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to get underlying sql.DB: %w", err)
	}

	sqlDB.SetMaxOpenConns(dbConfig.MaxOpenConns)
	sqlDB.SetMaxIdleConns(dbConfig.MaxIdleConns)
	sqlDB.SetConnMaxLifetime(dbConfig.MaxLifetime)

	// Redis connection
	rdb := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", redisConfig.Host, redisConfig.Port),
		Password: redisConfig.Password,
		DB:       redisConfig.DB,
		PoolSize: redisConfig.PoolSize,
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &Database{
		DB:    db,
		Redis: rdb,
	}, nil
}

// Close closes database connections
func (d *Database) Close() error {
	sqlDB, err := d.DB.DB()
	if err != nil {
		return err
	}
	
	if err := sqlDB.Close(); err != nil {
		return err
	}
	
	return d.Redis.Close()
}

// Health check
func (d *Database) Health(ctx context.Context) error {
	// Check PostgreSQL
	sqlDB, err := d.DB.DB()
	if err != nil {
		return err
	}
	
	if err := sqlDB.PingContext(ctx); err != nil {
		return fmt.Errorf("postgres health check failed: %w", err)
	}
	
	// Check Redis
	if err := d.Redis.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("redis health check failed: %w", err)
	}
	
	return nil
}

// Models
package models

import (
	"time"
	"github.com/google/uuid"
	"gorm.io/gorm"
	"golang.org/x/crypto/bcrypt"
)

// BaseModel contains common fields for all models
type BaseModel struct {
	ID        uuid.UUID      `json:"id" gorm:"type:uuid;primary_key;default:gen_random_uuid()"`
	CreatedAt time.Time      `json:"created_at" gorm:"not null"`
	UpdatedAt time.Time      `json:"updated_at" gorm:"not null"`
	DeletedAt gorm.DeletedAt `json:"-" gorm:"index"`
}

// User represents a user in the system
type User struct {
	BaseModel
	Email     string    `json:"email" gorm:"uniqueIndex;not null"`
	Username  string    `json:"username" gorm:"uniqueIndex;not null"`
	Password  string    `json:"-" gorm:"not null"`
	FirstName string    `json:"first_name" gorm:"not null"`
	LastName  string    `json:"last_name" gorm:"not null"`
	Role      UserRole  `json:"role" gorm:"not null;default:'user'"`
	Status    UserStatus `json:"status" gorm:"not null;default:'active'"`
	LastLoginAt *time.Time `json:"last_login_at"`
	
	// Relationships
	Posts    []Post    `json:"posts,omitempty" gorm:"foreignKey:UserID"`
	Comments []Comment `json:"comments,omitempty" gorm:"foreignKey:UserID"`
}

type UserRole string

const (
	UserRoleUser      UserRole = "user"
	UserRoleModerator UserRole = "moderator"
	UserRoleAdmin     UserRole = "admin"
)

type UserStatus string

const (
	UserStatusActive    UserStatus = "active"
	UserStatusInactive  UserStatus = "inactive"
	UserStatusSuspended UserStatus = "suspended"
)

// SetPassword hashes and sets the user password
func (u *User) SetPassword(password string) error {
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return err
	}
	u.Password = string(hashedPassword)
	return nil
}

// CheckPassword verifies the provided password
func (u *User) CheckPassword(password string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(u.Password), []byte(password))
	return err == nil
}

// FullName returns the user's full name
func (u *User) FullName() string {
	return u.FirstName + " " + u.LastName
}

// IsAdmin checks if user has admin role
func (u *User) IsAdmin() bool {
	return u.Role == UserRoleAdmin
}

// CanModerate checks if user can moderate content
func (u *User) CanModerate() bool {
	return u.Role == UserRoleAdmin || u.Role == UserRoleModerator
}

// Post represents a blog post or article
type Post struct {
	BaseModel
	Title       string     `json:"title" gorm:"not null"`
	Slug        string     `json:"slug" gorm:"uniqueIndex;not null"`
	Content     string     `json:"content" gorm:"type:text;not null"`
	Excerpt     string     `json:"excerpt" gorm:"type:text"`
	Status      PostStatus `json:"status" gorm:"not null;default:'draft'"`
	Featured    bool       `json:"featured" gorm:"default:false"`
	ViewCount   int64      `json:"view_count" gorm:"default:0"`
	PublishedAt *time.Time `json:"published_at"`
	
	// Foreign keys
	UserID     uuid.UUID `json:"user_id" gorm:"not null"`
	CategoryID uuid.UUID `json:"category_id" gorm:"not null"`
	
	// Relationships
	User     User      `json:"user" gorm:"foreignKey:UserID"`
	Category Category  `json:"category" gorm:"foreignKey:CategoryID"`
	Comments []Comment `json:"comments,omitempty" gorm:"foreignKey:PostID"`
	Tags     []Tag     `json:"tags,omitempty" gorm:"many2many:post_tags;"`
}

type PostStatus string

const (
	PostStatusDraft     PostStatus = "draft"
	PostStatusPublished PostStatus = "published"
	PostStatusArchived  PostStatus = "archived"
)

// IsPublished checks if post is published
func (p *Post) IsPublished() bool {
	return p.Status == PostStatusPublished && p.PublishedAt != nil
}

// ReadingTime calculates estimated reading time in minutes
func (p *Post) ReadingTime() int {
	words := len([]rune(p.Content)) / 5 // Rough estimate: 5 chars per word
	minutes := words / 200             // 200 words per minute
	if minutes < 1 {
		return 1
	}
	return minutes
}

// Category represents a post category
type Category struct {
	BaseModel
	Name        string `json:"name" gorm:"not null"`
	Slug        string `json:"slug" gorm:"uniqueIndex;not null"`
	Description string `json:"description" gorm:"type:text"`
	Color       string `json:"color" gorm:"default:'#3B82F6'"`
	PostCount   int64  `json:"post_count" gorm:"default:0"`
	
	// Relationships
	Posts []Post `json:"posts,omitempty" gorm:"foreignKey:CategoryID"`
}

// Comment represents a comment on a post
type Comment struct {
	BaseModel
	Content  string        `json:"content" gorm:"type:text;not null"`
	Status   CommentStatus `json:"status" gorm:"not null;default:'pending'"`
	
	// Foreign keys
	PostID   uuid.UUID  `json:"post_id" gorm:"not null"`
	UserID   uuid.UUID  `json:"user_id" gorm:"not null"`
	ParentID *uuid.UUID `json:"parent_id"`
	
	// Relationships
	Post     Post      `json:"post" gorm:"foreignKey:PostID"`
	User     User      `json:"user" gorm:"foreignKey:UserID"`
	Parent   *Comment  `json:"parent,omitempty" gorm:"foreignKey:ParentID"`
	Replies  []Comment `json:"replies,omitempty" gorm:"foreignKey:ParentID"`
}

type CommentStatus string

const (
	CommentStatusPending  CommentStatus = "pending"
	CommentStatusApproved CommentStatus = "approved"
	CommentStatusRejected CommentStatus = "rejected"
)

// IsApproved checks if comment is approved
func (c *Comment) IsApproved() bool {
	return c.Status == CommentStatusApproved
}

// Tag represents a content tag
type Tag struct {
	BaseModel
	Name      string `json:"name" gorm:"uniqueIndex;not null"`
	Slug      string `json:"slug" gorm:"uniqueIndex;not null"`
	PostCount int64  `json:"post_count" gorm:"default:0"`
	
	// Relationships
	Posts []Post `json:"posts,omitempty" gorm:"many2many:post_tags;"`
}

// Migration function
func MigrateModels(db *gorm.DB) error {
	return db.AutoMigrate(
		&User{},
		&Category{},
		&Post{},
		&Comment{},
		&Tag{},
	)
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           5. REPOSITORY LAYER
// ═══════════════════════════════════════════════════════════════════════════════

package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/go-redis/redis/v8"
	"gorm.io/gorm"
)

// BaseRepository provides common repository functionality
type BaseRepository struct {
	db    *gorm.DB
	redis *redis.Client
}

// NewBaseRepository creates a new base repository
func NewBaseRepository(db *gorm.DB, redis *redis.Client) *BaseRepository {
	return &BaseRepository{
		db:    db,
		redis: redis,
	}
}

// CacheGet retrieves data from cache
func (r *BaseRepository) CacheGet(ctx context.Context, key string, dest interface{}) error {
	val, err := r.redis.Get(ctx, key).Result()
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(val), dest)
}

// CacheSet stores data in cache
func (r *BaseRepository) CacheSet(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return r.redis.Set(ctx, key, data, expiration).Err()
}

// CacheDelete removes data from cache
func (r *BaseRepository) CacheDelete(ctx context.Context, keys ...string) error {
	if len(keys) == 0 {
		return nil
	}
	return r.redis.Del(ctx, keys...).Err()
}

// UserRepository handles user data operations
type UserRepository struct {
	*BaseRepository
}

// NewUserRepository creates a new user repository
func NewUserRepository(db *gorm.DB, redis *redis.Client) *UserRepository {
	return &UserRepository{
		BaseRepository: NewBaseRepository(db, redis),
	}
}

// Create creates a new user
func (r *UserRepository) Create(ctx context.Context, user *User) error {
	if err := r.db.WithContext(ctx).Create(user).Error; err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}
	
	// Cache the user
	cacheKey := fmt.Sprintf("user:%s", user.ID.String())
	r.CacheSet(ctx, cacheKey, user, time.Hour)
	
	return nil
}

// GetByID retrieves a user by ID
func (r *UserRepository) GetByID(ctx context.Context, id uuid.UUID) (*User, error) {
	// Try cache first
	cacheKey := fmt.Sprintf("user:%s", id.String())
	var user User
	
	if err := r.CacheGet(ctx, cacheKey, &user); err == nil {
		return &user, nil
	}
	
	// Cache miss, get from database
	if err := r.db.WithContext(ctx).First(&user, id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	
	// Cache the result
	r.CacheSet(ctx, cacheKey, &user, time.Hour)
	
	return &user, nil
}

// GetByEmail retrieves a user by email
func (r *UserRepository) GetByEmail(ctx context.Context, email string) (*User, error) {
	var user User
	if err := r.db.WithContext(ctx).Where("email = ?", email).First(&user).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	return &user, nil
}

// Update updates a user
func (r *UserRepository) Update(ctx context.Context, user *User) error {
	if err := r.db.WithContext(ctx).Save(user).Error; err != nil {
		return fmt.Errorf("failed to update user: %w", err)
	}
	
	// Invalidate cache
	cacheKey := fmt.Sprintf("user:%s", user.ID.String())
	r.CacheDelete(ctx, cacheKey)
	
	return nil
}

// Delete deletes a user
func (r *UserRepository) Delete(ctx context.Context, id uuid.UUID) error {
	if err := r.db.WithContext(ctx).Delete(&User{}, id).Error; err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}
	
	// Invalidate cache
	cacheKey := fmt.Sprintf("user:%s", id.String())
	r.CacheDelete(ctx, cacheKey)
	
	return nil
}

// List retrieves users with pagination
func (r *UserRepository) List(ctx context.Context, offset, limit int) ([]User, int64, error) {
	var users []User
	var total int64
	
	// Get total count
	if err := r.db.WithContext(ctx).Model(&User{}).Count(&total).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to count users: %w", err)
	}
	
	// Get users with pagination
	if err := r.db.WithContext(ctx).
		Offset(offset).
		Limit(limit).
		Order("created_at DESC").
		Find(&users).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to list users: %w", err)
	}
	
	return users, total, nil
}

// PostRepository handles post data operations
type PostRepository struct {
	*BaseRepository
}

// NewPostRepository creates a new post repository
func NewPostRepository(db *gorm.DB, redis *redis.Client) *PostRepository {
	return &PostRepository{
		BaseRepository: NewBaseRepository(db, redis),
	}
}

// Create creates a new post
func (r *PostRepository) Create(ctx context.Context, post *Post) error {
	if err := r.db.WithContext(ctx).Create(post).Error; err != nil {
		return fmt.Errorf("failed to create post: %w", err)
	}
	
	// Invalidate related caches
	r.CacheDelete(ctx, "posts:published", "posts:featured")
	
	return nil
}

// GetByID retrieves a post by ID with relationships
func (r *PostRepository) GetByID(ctx context.Context, id uuid.UUID) (*Post, error) {
	var post Post
	if err := r.db.WithContext(ctx).
		Preload("User").
		Preload("Category").
		Preload("Tags").
		Preload("Comments", func(db *gorm.DB) *gorm.DB {
			return db.Where("status = ?", CommentStatusApproved).Order("created_at DESC")
		}).
		Preload("Comments.User").
		First(&post, id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("post not found")
		}
		return nil, fmt.Errorf("failed to get post: %w", err)
	}
	return &post, nil
}

// GetBySlug retrieves a post by slug
func (r *PostRepository) GetBySlug(ctx context.Context, slug string) (*Post, error) {
	var post Post
	if err := r.db.WithContext(ctx).
		Preload("User").
		Preload("Category").
		Preload("Tags").
		Where("slug = ?", slug).
		First(&post).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("post not found")
		}
		return nil, fmt.Errorf("failed to get post: %w", err)
	}
	return &post, nil
}

// List retrieves posts with filters and pagination
func (r *PostRepository) List(ctx context.Context, filters PostFilters) ([]Post, int64, error) {
	query := r.db.WithContext(ctx).Model(&Post{}).
		Preload("User").
		Preload("Category").
		Preload("Tags")
	
	// Apply filters
	if filters.Status != "" {
		query = query.Where("status = ?", filters.Status)
	}
	
	if filters.CategoryID != uuid.Nil {
		query = query.Where("category_id = ?", filters.CategoryID)
	}
	
	if filters.UserID != uuid.Nil {
		query = query.Where("user_id = ?", filters.UserID)
	}
	
	if filters.Featured != nil {
		query = query.Where("featured = ?", *filters.Featured)
	}
	
	if filters.Search != "" {
		query = query.Where("title ILIKE ? OR content ILIKE ?", 
			"%"+filters.Search+"%", "%"+filters.Search+"%")
	}
	
	// Get total count
	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to count posts: %w", err)
	}
	
	// Apply pagination and ordering
	var posts []Post
	if err := query.
		Offset(filters.Offset).
		Limit(filters.Limit).
		Order(filters.OrderBy + " " + filters.Order).
		Find(&posts).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to list posts: %w", err)
	}
	
	return posts, total, nil
}

// GetPublished retrieves published posts
func (r *PostRepository) GetPublished(ctx context.Context, limit int) ([]Post, error) {
	// Try cache first
	cacheKey := fmt.Sprintf("posts:published:%d", limit)
	var posts []Post
	
	if err := r.CacheGet(ctx, cacheKey, &posts); err == nil {
		return posts, nil
	}
	
	// Cache miss, get from database
	if err := r.db.WithContext(ctx).
		Preload("User").
		Preload("Category").
		Where("status = ?", PostStatusPublished).
		Order("published_at DESC").
		Limit(limit).
		Find(&posts).Error; err != nil {
		return nil, fmt.Errorf("failed to get published posts: %w", err)
	}
	
	// Cache the result
	r.CacheSet(ctx, cacheKey, posts, 15*time.Minute)
	
	return posts, nil
}

// IncrementViewCount increments the view count for a post
func (r *PostRepository) IncrementViewCount(ctx context.Context, id uuid.UUID) error {
	return r.db.WithContext(ctx).
		Model(&Post{}).
		Where("id = ?", id).
		Update("view_count", gorm.Expr("view_count + 1")).Error
}

// PostFilters defines filters for post queries
type PostFilters struct {
	Status     PostStatus
	CategoryID uuid.UUID
	UserID     uuid.UUID
	Featured   *bool
	Search     string
	Offset     int
	Limit      int
	OrderBy    string
	Order      string
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. SERVICE LAYER
// ═══════════════════════════════════════════════════════════════════════════════

package service

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

// UserService handles user business logic
type UserService struct {
	userRepo   *UserRepository
	jwtSecret  string
	jwtExpiry  time.Duration
	jwtIssuer  string
}

// NewUserService creates a new user service
func NewUserService(userRepo *UserRepository, jwtSecret string, jwtExpiry time.Duration, jwtIssuer string) *UserService {
	return &UserService{
		userRepo:  userRepo,
		jwtSecret: jwtSecret,
		jwtExpiry: jwtExpiry,
		jwtIssuer: jwtIssuer,
	}
}

// CreateUser creates a new user
func (s *UserService) CreateUser(ctx context.Context, req CreateUserRequest) (*User, error) {
	// Validate input
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}
	
	// Check if user already exists
	existing, err := s.userRepo.GetByEmail(ctx, req.Email)
	if err == nil && existing != nil {
		return nil, fmt.Errorf("user with email %s already exists", req.Email)
	}
	
	// Create user
	user := &User{
		Email:     req.Email,
		Username:  req.Username,
		FirstName: req.FirstName,
		LastName:  req.LastName,
		Role:      UserRoleUser,
		Status:    UserStatusActive,
	}
	
	if err := user.SetPassword(req.Password); err != nil {
		return nil, fmt.Errorf("failed to set password: %w", err)
	}
	
	if err := s.userRepo.Create(ctx, user); err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}
	
	return user, nil
}

// Authenticate authenticates a user and returns a JWT token
func (s *UserService) Authenticate(ctx context.Context, email, password string) (*AuthResponse, error) {
	user, err := s.userRepo.GetByEmail(ctx, email)
	if err != nil {
		return nil, fmt.Errorf("invalid credentials")
	}
	
	if !user.CheckPassword(password) {
		return nil, fmt.Errorf("invalid credentials")
	}
	
	if user.Status != UserStatusActive {
		return nil, fmt.Errorf("user account is not active")
	}
	
	// Generate JWT token
	token, err := s.generateJWT(user)
	if err != nil {
		return nil, fmt.Errorf("failed to generate token: %w", err)
	}
	
	// Update last login time
	now := time.Now()
	user.LastLoginAt = &now
	s.userRepo.Update(ctx, user)
	
	return &AuthResponse{
		Token:     token,
		User:      user,
		ExpiresAt: time.Now().Add(s.jwtExpiry),
	}, nil
}

// GetUserByID retrieves a user by ID
func (s *UserService) GetUserByID(ctx context.Context, id uuid.UUID) (*User, error) {
	return s.userRepo.GetByID(ctx, id)
}

// UpdateUser updates user information
func (s *UserService) UpdateUser(ctx context.Context, id uuid.UUID, req UpdateUserRequest) (*User, error) {
	user, err := s.userRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	
	// Update fields
	if req.FirstName != "" {
		user.FirstName = req.FirstName
	}
	if req.LastName != "" {
		user.LastName = req.LastName
	}
	if req.Username != "" {
		user.Username = req.Username
	}
	
	if err := s.userRepo.Update(ctx, user); err != nil {
		return nil, fmt.Errorf("failed to update user: %w", err)
	}
	
	return user, nil
}

// ChangePassword changes user password
func (s *UserService) ChangePassword(ctx context.Context, userID uuid.UUID, oldPassword, newPassword string) error {
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return err
	}
	
	if !user.CheckPassword(oldPassword) {
		return fmt.Errorf("invalid current password")
	}
	
	if err := user.SetPassword(newPassword); err != nil {
		return fmt.Errorf("failed to set new password: %w", err)
	}
	
	return s.userRepo.Update(ctx, user)
}

// ValidateToken validates a JWT token and returns the user
func (s *UserService) ValidateToken(tokenString string) (*User, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(s.jwtSecret), nil
	})
	
	if err != nil {
		return nil, fmt.Errorf("invalid token: %w", err)
	}
	
	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		userIDStr, ok := claims["user_id"].(string)
		if !ok {
			return nil, fmt.Errorf("invalid token claims")
		}
		
		userID, err := uuid.Parse(userIDStr)
		if err != nil {
			return nil, fmt.Errorf("invalid user ID in token")
		}
		
		user, err := s.userRepo.GetByID(context.Background(), userID)
		if err != nil {
			return nil, fmt.Errorf("user not found")
		}
		
		return user, nil
	}
	
	return nil, fmt.Errorf("invalid token")
}

// generateJWT generates a JWT token for a user
func (s *UserService) generateJWT(user *User) (string, error) {
	claims := jwt.MapClaims{
		"user_id":  user.ID.String(),
		"email":    user.Email,
		"username": user.Username,
		"role":     user.Role,
		"iss":      s.jwtIssuer,
		"exp":      time.Now().Add(s.jwtExpiry).Unix(),
		"iat":      time.Now().Unix(),
	}
	
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(s.jwtSecret))
}

// Request/Response types
type CreateUserRequest struct {
	Email     string `json:"email" validate:"required,email"`
	Username  string `json:"username" validate:"required,min=3,max=50"`
	Password  string `json:"password" validate:"required,min=8"`
	FirstName string `json:"first_name" validate:"required"`
	LastName  string `json:"last_name" validate:"required"`
}

func (r *CreateUserRequest) Validate() error {
	if r.Email == "" {
		return fmt.Errorf("email is required")
	}
	if r.Username == "" {
		return fmt.Errorf("username is required")
	}
	if len(r.Password) < 8 {
		return fmt.Errorf("password must be at least 8 characters")
	}
	if r.FirstName == "" {
		return fmt.Errorf("first name is required")
	}
	if r.LastName == "" {
		return fmt.Errorf("last name is required")
	}
	return nil
}

type UpdateUserRequest struct {
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Username  string `json:"username"`
}

type AuthResponse struct {
	Token     string    `json:"token"`
	User      *User     `json:"user"`
	ExpiresAt time.Time `json:"expires_at"`
}

// PostService handles post business logic
type PostService struct {
	postRepo     *PostRepository
	categoryRepo *CategoryRepository
	userRepo     *UserRepository
}

// NewPostService creates a new post service
func NewPostService(postRepo *PostRepository, categoryRepo *CategoryRepository, userRepo *UserRepository) *PostService {
	return &PostService{
		postRepo:     postRepo,
		categoryRepo: categoryRepo,
		userRepo:     userRepo,
	}
}

// CreatePost creates a new post
func (s *PostService) CreatePost(ctx context.Context, userID uuid.UUID, req CreatePostRequest) (*Post, error) {
	// Validate input
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}
	
	// Verify user exists
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("user not found: %w", err)
	}
	
	// Verify category exists
	category, err := s.categoryRepo.GetByID(ctx, req.CategoryID)
	if err != nil {
		return nil, fmt.Errorf("category not found: %w", err)
	}
	
	// Generate slug
	slug := generateSlug(req.Title)
	
	// Create post
	post := &Post{
		Title:      req.Title,
		Slug:       slug,
		Content:    req.Content,
		Excerpt:    req.Excerpt,
		Status:     PostStatusDraft,
		UserID:     userID,
		CategoryID: req.CategoryID,
	}
	
	if err := s.postRepo.Create(ctx, post); err != nil {
		return nil, fmt.Errorf("failed to create post: %w", err)
	}
	
	// Load relationships
	return s.postRepo.GetByID(ctx, post.ID)
}

// GetPost retrieves a post by ID
func (s *PostService) GetPost(ctx context.Context, id uuid.UUID) (*Post, error) {
	post, err := s.postRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	
	// Increment view count asynchronously
	go func() {
		s.postRepo.IncrementViewCount(context.Background(), id)
	}()
	
	return post, nil
}

// GetPostBySlug retrieves a post by slug
func (s *PostService) GetPostBySlug(ctx context.Context, slug string) (*Post, error) {
	return s.postRepo.GetBySlug(ctx, slug)
}

// UpdatePost updates a post
func (s *PostService) UpdatePost(ctx context.Context, id uuid.UUID, userID uuid.UUID, req UpdatePostRequest) (*Post, error) {
	post, err := s.postRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	
	// Check ownership or admin permission
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return nil, err
	}
	
	if post.UserID != userID && !user.CanModerate() {
		return nil, fmt.Errorf("permission denied")
	}
	
	// Update fields
	if req.Title != "" {
		post.Title = req.Title
		post.Slug = generateSlug(req.Title)
	}
	if req.Content != "" {
		post.Content = req.Content
	}
	if req.Excerpt != "" {
		post.Excerpt = req.Excerpt
	}
	if req.CategoryID != uuid.Nil {
		post.CategoryID = req.CategoryID
	}
	
	if err := s.postRepo.Update(ctx, post); err != nil {
		return nil, fmt.Errorf("failed to update post: %w", err)
	}
	
	return s.postRepo.GetByID(ctx, post.ID)
}

// PublishPost publishes a draft post
func (s *PostService) PublishPost(ctx context.Context, id uuid.UUID, userID uuid.UUID) (*Post, error) {
	post, err := s.postRepo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	
	// Check ownership or admin permission
	user, err := s.userRepo.GetByID(ctx, userID)
	if err != nil {
		return nil, err
	}
	
	if post.UserID != userID && !user.CanModerate() {
		return nil, fmt.Errorf("permission denied")
	}
	
	if post.Status != PostStatusDraft {
		return nil, fmt.Errorf("only draft posts can be published")
	}
	
	// Publish the post
	now := time.Now()
	post.Status = PostStatusPublished
	post.PublishedAt = &now
	
	if err := s.postRepo.Update(ctx, post); err != nil {
		return nil, fmt.Errorf("failed to publish post: %w", err)
	}
	
	return post, nil
}

// ListPosts retrieves posts with filters
func (s *PostService) ListPosts(ctx context.Context, filters PostFilters) (*PostListResponse, error) {
	// Set defaults
	if filters.Limit <= 0 {
		filters.Limit = 20
	}
	if filters.Limit > 100 {
		filters.Limit = 100
	}
	if filters.OrderBy == "" {
		filters.OrderBy = "created_at"
	}
	if filters.Order == "" {
		filters.Order = "DESC"
	}
	
	posts, total, err := s.postRepo.List(ctx, filters)
	if err != nil {
		return nil, err
	}
	
	return &PostListResponse{
		Posts: posts,
		Total: total,
		Page:  filters.Offset/filters.Limit + 1,
		Limit: filters.Limit,
	}, nil
}

// generateSlug generates a URL-friendly slug from a title
func generateSlug(title string) string {
	slug := strings.ToLower(title)
	slug = strings.ReplaceAll(slug, " ", "-")
	// Remove special characters (simplified)
	slug = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			return r
		}
		return -1
	}, slug)
	return slug
}

// Request/Response types for posts
type CreatePostRequest struct {
	Title      string    `json:"title" validate:"required"`
	Content    string    `json:"content" validate:"required"`
	Excerpt    string    `json:"excerpt"`
	CategoryID uuid.UUID `json:"category_id" validate:"required"`
}

func (r *CreatePostRequest) Validate() error {
	if r.Title == "" {
		return fmt.Errorf("title is required")
	}
	if r.Content == "" {
		return fmt.Errorf("content is required")
	}
	if r.CategoryID == uuid.Nil {
		return fmt.Errorf("category ID is required")
	}
	return nil
}

type UpdatePostRequest struct {
	Title      string    `json:"title"`
	Content    string    `json:"content"`
	Excerpt    string    `json:"excerpt"`
	CategoryID uuid.UUID `json:"category_id"`
}

type PostListResponse struct {
	Posts []Post `json:"posts"`
	Total int64  `json:"total"`
	Page  int    `json:"page"`
	Limit int    `json:"limit"`
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. HTTP HANDLERS AND MIDDLEWARE
// ═══════════════════════════════════════════════════════════════════════════════

package handlers

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// UserHandler handles user-related HTTP requests
type UserHandler struct {
	userService *UserService
	logger      *logger.Logger
}

// NewUserHandler creates a new user handler
func NewUserHandler(userService *UserService, logger *logger.Logger) *UserHandler {
	return &UserHandler{
		userService: userService,
		logger:      logger,
	}
}

// Register handles user registration
func (h *UserHandler) Register(c *gin.Context) {
	var req CreateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	user, err := h.userService.CreateUser(c.Request.Context(), req)
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to create user", 
			zap.Error(err), zap.String("email", req.Email))
		
		if strings.Contains(err.Error(), "already exists") {
			c.JSON(http.StatusConflict, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user"})
		return
	}
	
	h.logger.InfoCtx(c.Request.Context(), "User created successfully", 
		zap.String("user_id", user.ID.String()), zap.String("email", user.Email))
	
	c.JSON(http.StatusCreated, gin.H{
		"message": "User created successfully",
		"user":    user,
	})
}

// Login handles user authentication
func (h *UserHandler) Login(c *gin.Context) {
	var req struct {
		Email    string `json:"email" binding:"required,email"`
		Password string `json:"password" binding:"required"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	authResp, err := h.userService.Authenticate(c.Request.Context(), req.Email, req.Password)
	if err != nil {
		h.logger.WarnCtx(c.Request.Context(), "Authentication failed", 
			zap.Error(err), zap.String("email", req.Email))
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid credentials"})
		return
	}
	
	h.logger.InfoCtx(c.Request.Context(), "User authenticated successfully", 
		zap.String("user_id", authResp.User.ID.String()), zap.String("email", authResp.User.Email))
	
	c.JSON(http.StatusOK, authResp)
}

// GetProfile handles getting user profile
func (h *UserHandler) GetProfile(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	user, err := h.userService.GetUserByID(c.Request.Context(), userID.(uuid.UUID))
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to get user profile", zap.Error(err))
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"user": user})
}

// UpdateProfile handles updating user profile
func (h *UserHandler) UpdateProfile(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	var req UpdateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	user, err := h.userService.UpdateUser(c.Request.Context(), userID.(uuid.UUID), req)
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to update user profile", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update profile"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Profile updated successfully",
		"user":    user,
	})
}

// ChangePassword handles password change
func (h *UserHandler) ChangePassword(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	var req struct {
		OldPassword string `json:"old_password" binding:"required"`
		NewPassword string `json:"new_password" binding:"required,min=8"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	err := h.userService.ChangePassword(c.Request.Context(), userID.(uuid.UUID), req.OldPassword, req.NewPassword)
	if err != nil {
		if strings.Contains(err.Error(), "invalid current password") {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid current password"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to change password", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to change password"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"message": "Password changed successfully"})
}

// PostHandler handles post-related HTTP requests
type PostHandler struct {
	postService *PostService
	logger      *logger.Logger
}

// NewPostHandler creates a new post handler
func NewPostHandler(postService *PostService, logger *logger.Logger) *PostHandler {
	return &PostHandler{
		postService: postService,
		logger:      logger,
	}
}

// CreatePost handles post creation
func (h *PostHandler) CreatePost(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	var req CreatePostRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	post, err := h.postService.CreatePost(c.Request.Context(), userID.(uuid.UUID), req)
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to create post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create post"})
		return
	}
	
	h.logger.InfoCtx(c.Request.Context(), "Post created successfully", 
		zap.String("post_id", post.ID.String()), zap.String("user_id", userID.(uuid.UUID).String()))
	
	c.JSON(http.StatusCreated, gin.H{
		"message": "Post created successfully",
		"post":    post,
	})
}

// GetPost handles getting a single post
func (h *PostHandler) GetPost(c *gin.Context) {
	idParam := c.Param("id")
	postID, err := uuid.Parse(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid post ID"})
		return
	}
	
	post, err := h.postService.GetPost(c.Request.Context(), postID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to get post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"post": post})
}

// GetPostBySlug handles getting a post by slug
func (h *PostHandler) GetPostBySlug(c *gin.Context) {
	slug := c.Param("slug")
	
	post, err := h.postService.GetPostBySlug(c.Request.Context(), slug)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to get post by slug", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"post": post})
}

// ListPosts handles listing posts with filters
func (h *PostHandler) ListPosts(c *gin.Context) {
	filters := PostFilters{
		Status:  PostStatus(c.Query("status")),
		Search:  c.Query("search"),
		OrderBy: c.Query("order_by"),
		Order:   c.Query("order"),
	}
	
	// Parse pagination
	if page := c.Query("page"); page != "" {
		if p, err := strconv.Atoi(page); err == nil && p > 0 {
			filters.Offset = (p - 1) * filters.Limit
		}
	}
	
	if limit := c.Query("limit"); limit != "" {
		if l, err := strconv.Atoi(limit); err == nil && l > 0 {
			filters.Limit = l
		}
	}
	
	if filters.Limit == 0 {
		filters.Limit = 20
	}
	
	// Parse category ID
	if categoryID := c.Query("category_id"); categoryID != "" {
		if id, err := uuid.Parse(categoryID); err == nil {
			filters.CategoryID = id
		}
	}
	
	// Parse featured filter
	if featured := c.Query("featured"); featured != "" {
		if f, err := strconv.ParseBool(featured); err == nil {
			filters.Featured = &f
		}
	}
	
	response, err := h.postService.ListPosts(c.Request.Context(), filters)
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to list posts", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list posts"})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

// UpdatePost handles post updates
func (h *PostHandler) UpdatePost(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	idParam := c.Param("id")
	postID, err := uuid.Parse(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid post ID"})
		return
	}
	
	var req UpdatePostRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	post, err := h.postService.UpdatePost(c.Request.Context(), postID, userID.(uuid.UUID), req)
	if err != nil {
		if strings.Contains(err.Error(), "permission denied") {
			c.JSON(http.StatusForbidden, gin.H{"error": "Permission denied"})
			return
		}
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to update post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Post updated successfully",
		"post":    post,
	})
}

// PublishPost handles post publishing
func (h *PostHandler) PublishPost(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	idParam := c.Param("id")
	postID, err := uuid.Parse(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid post ID"})
		return
	}
	
	post, err := h.postService.PublishPost(c.Request.Context(), postID, userID.(uuid.UUID))
	if err != nil {
		if strings.Contains(err.Error(), "permission denied") {
			c.JSON(http.StatusForbidden, gin.H{"error": "Permission denied"})
			return
		}
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to publish post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to publish post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Post published successfully",
		"post":    post,
	})
}

// Middleware
package middleware

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// AuthMiddleware validates JWT tokens and sets user context
func AuthMiddleware(userService *UserService) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
			c.Abort()
			return
		}
		
		tokenParts := strings.Split(authHeader, " ")
		if len(tokenParts) != 2 || tokenParts[0] != "Bearer" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid authorization header format"})
			c.Abort()
			return
		}
		
		token := tokenParts[1]
		user, err := userService.ValidateToken(token)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}
		
		// Set user context
		c.Set("user_id", user.ID)
		c.Set("user", user)
		c.Next()
	}
}

// OptionalAuthMiddleware validates JWT tokens but doesn't require them
func OptionalAuthMiddleware(userService *UserService) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.Next()
			return
		}
		
		tokenParts := strings.Split(authHeader, " ")
		if len(tokenParts) != 2 || tokenParts[0] != "Bearer" {
			c.Next()
			return
		}
		
		token := tokenParts[1]
		user, err := userService.ValidateToken(token)
		if err != nil {
			c.Next()
			return
		}
		
		// Set user context
		c.Set("user_id", user.ID)
		c.Set("user", user)
		c.Next()
	}
}

// RequestIDMiddleware adds a unique request ID to each request
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := uuid.New().String()
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)
		c.Next()
	}
}

// LoggingMiddleware logs HTTP requests
func LoggingMiddleware(logger *logger.Logger) gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		logger.InfoCtx(param.Request.Context(), "HTTP Request",
			zap.String("method", param.Method),
			zap.String("path", param.Path),
			zap.Int("status", param.StatusCode),
			zap.Duration("latency", param.Latency),
			zap.String("client_ip", param.ClientIP),
			zap.String("user_agent", param.Request.UserAgent()),
		)
		return ""
	})
}

// MetricsMiddleware records HTTP metrics
func MetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		c.Next()
		
		duration := time.Since(start)
		method := c.Request.Method
		route := c.FullPath()
		if route == "" {
			route = "unknown"
		}
		status := strconv.Itoa(c.Writer.Status())
		
		metrics.RecordHTTPRequest(method, route, status, duration)
	}
}

// CORSMiddleware handles Cross-Origin Resource Sharing
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Request-ID")
		c.Header("Access-Control-Expose-Headers", "X-Request-ID")
		c.Header("Access-Control-Max-Age", "86400")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		
		c.Next()
	}
}

// RateLimitMiddleware implements simple rate limiting
func RateLimitMiddleware(rps int) gin.HandlerFunc {
	// This is a simplified rate limiter
	// In production, use a proper rate limiting library like golang.org/x/time/rate
	return func(c *gin.Context) {
		// Implementation would go here
		c.Next()
	}
}

// SecurityHeadersMiddleware adds security headers
func SecurityHeadersMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'")
		c.Next()
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           8. GRPC AND MICROSERVICES COMMUNICATION
// ═══════════════════════════════════════════════════════════════════════════════

// api/proto/user.proto
/*
syntax = "proto3";

package user;

option go_package = "github.com/username/my-microservice/pkg/proto/user";

service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
}

message User {
  string id = 1;
  string email = 2;
  string username = 3;
  string first_name = 4;
  string last_name = 5;
  string role = 6;
  string status = 7;
  int64 created_at = 8;
  int64 updated_at = 9;
}

message GetUserRequest {
  string id = 1;
}

message GetUserResponse {
  User user = 1;
}

message CreateUserRequest {
  string email = 1;
  string username = 2;
  string password = 3;
  string first_name = 4;
  string last_name = 5;
}

message CreateUserResponse {
  User user = 1;
}

message UpdateUserRequest {
  string id = 1;
  string first_name = 2;
  string last_name = 3;
  string username = 4;
}

message UpdateUserResponse {
  User user = 1;
}

message DeleteUserRequest {
  string id = 1;
}

message DeleteUserResponse {
  bool success = 1;
}

message ListUsersRequest {
  int32 page = 1;
  int32 limit = 2;
  string search = 3;
}

message ListUsersResponse {
  repeated User users = 1;
  int64 total = 2;
  int32 page = 3;
  int32 limit = 4;
}
*/

// Generate protobuf code:
// protoc --go_out=. --go_opt=paths=source_relative \
//        --go-grpc_out=. --go-grpc_opt=paths=source_relative \
//        api/proto/user.proto

package grpc

import (
	"context"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

// UserGRPCServer implements the gRPC user service
type UserGRPCServer struct {
	userService *UserService
	logger      *logger.Logger
	UnimplementedUserServiceServer
}

// NewUserGRPCServer creates a new gRPC user server
func NewUserGRPCServer(userService *UserService, logger *logger.Logger) *UserGRPCServer {
	return &UserGRPCServer{
		userService: userService,
		logger:      logger,
	}
}

// GetUser implements the GetUser gRPC method
func (s *UserGRPCServer) GetUser(ctx context.Context, req *GetUserRequest) (*GetUserResponse, error) {
	userID, err := uuid.Parse(req.Id)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid user ID: %v", err)
	}
	
	user, err := s.userService.GetUserByID(ctx, userID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			return nil, status.Errorf(codes.NotFound, "user not found")
		}
		s.logger.ErrorCtx(ctx, "Failed to get user via gRPC", zap.Error(err))
		return nil, status.Errorf(codes.Internal, "internal server error")
	}
	
	return &GetUserResponse{
		User: convertUserToProto(user),
	}, nil
}

// CreateUser implements the CreateUser gRPC method
func (s *UserGRPCServer) CreateUser(ctx context.Context, req *CreateUserRequest) (*CreateUserResponse, error) {
	createReq := CreateUserRequest{
		Email:     req.Email,
		Username:  req.Username,
		Password:  req.Password,
		FirstName: req.FirstName,
		LastName:  req.LastName,
	}
	
	user, err := s.userService.CreateUser(ctx, createReq)
	if err != nil {
		if strings.Contains(err.Error(), "already exists") {
			return nil, status.Errorf(codes.AlreadyExists, err.Error())
		}
		if strings.Contains(err.Error(), "validation") {
			return nil, status.Errorf(codes.InvalidArgument, err.Error())
		}
		
		s.logger.ErrorCtx(ctx, "Failed to create user via gRPC", zap.Error(err))
		return nil, status.Errorf(codes.Internal, "internal server error")
	}
	
	return &CreateUserResponse{
		User: convertUserToProto(user),
	}, nil
}

// ListUsers implements the ListUsers gRPC method
func (s *UserGRPCServer) ListUsers(ctx context.Context, req *ListUsersRequest) (*ListUsersResponse, error) {
	limit := int(req.Limit)
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	
	page := int(req.Page)
	if page <= 0 {
		page = 1
	}
	
	offset := (page - 1) * limit
	
	users, total, err := s.userService.userRepo.List(ctx, offset, limit)
	if err != nil {
		s.logger.ErrorCtx(ctx, "Failed to list users via gRPC", zap.Error(err))
		return nil, status.Errorf(codes.Internal, "internal server error")
	}
	
	protoUsers := make([]*User, len(users))
	for i, user := range users {
		protoUsers[i] = convertUserToProto(&user)
	}
	
	return &ListUsersResponse{
		Users: protoUsers,
		Total: total,
		Page:  int32(page),
		Limit: int32(limit),
	}, nil
}

// convertUserToProto converts domain user to protobuf user
func convertUserToProto(user *User) *User {
	return &User{
		Id:        user.ID.String(),
		Email:     user.Email,
		Username:  user.Username,
		FirstName: user.FirstName,
		LastName:  user.LastName,
		Role:      string(user.Role),
		Status:    string(user.Status),
		CreatedAt: user.CreatedAt.Unix(),
		UpdatedAt: user.UpdatedAt.Unix(),
	}
}

// GRPCServer holds the gRPC server configuration
type GRPCServer struct {
	server *grpc.Server
	logger *logger.Logger
}

// NewGRPCServer creates a new gRPC server
func NewGRPCServer(userService *UserService, logger *logger.Logger) *GRPCServer {
	server := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			loggingInterceptor(logger),
			metricsInterceptor(),
			recoveryInterceptor(logger),
		),
	)
	
	// Register services
	RegisterUserServiceServer(server, NewUserGRPCServer(userService, logger))
	
	// Register health check
	healthServer := health.NewServer()
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(server, healthServer)
	
	// Register reflection (for development)
	reflection.Register(server)
	
	return &GRPCServer{
		server: server,
		logger: logger,
	}
}

// Start starts the gRPC server
func (s *GRPCServer) Start(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", addr, err)
	}
	
	s.logger.Info("Starting gRPC server", zap.String("address", addr))
	return s.server.Serve(lis)
}

// Stop gracefully stops the gRPC server
func (s *GRPCServer) Stop() {
	s.logger.Info("Stopping gRPC server")
	s.server.GracefulStop()
}

// gRPC Interceptors
func loggingInterceptor(logger *logger.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()
		
		resp, err := handler(ctx, req)
		
		duration := time.Since(start)
		
		fields := []zap.Field{
			zap.String("method", info.FullMethod),
			zap.Duration("duration", duration),
		}
		
		if err != nil {
			fields = append(fields, zap.Error(err))
			logger.ErrorCtx(ctx, "gRPC request failed", fields...)
		} else {
			logger.InfoCtx(ctx, "gRPC request completed", fields...)
		}
		
		return resp, err
	}
}

func metricsInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		start := time.Now()
		
		resp, err := handler(ctx, req)
		
		duration := time.Since(start)
		status := "success"
		if err != nil {
			status = "error"
		}
		
		// Record gRPC metrics (would need to define these metrics)
		// metrics.RecordGRPCRequest(info.FullMethod, status, duration)
		
		return resp, err
	}
}

func recoveryInterceptor(logger *logger.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				logger.ErrorCtx(ctx, "gRPC panic recovered", 
					zap.String("method", info.FullMethod),
					zap.Any("panic", r))
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()
		
		return handler(ctx, req)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           9. MESSAGE QUEUES AND EVENT PROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

package messaging

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/nats-io/nats.go"
)

// Event represents a domain event
type Event struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Source      string                 `json:"source"`
	Subject     string                 `json:"subject"`
	Data        map[string]interface{} `json:"data"`
	Timestamp   time.Time              `json:"timestamp"`
	Version     string                 `json:"version"`
}

// EventBus handles event publishing and subscription
type EventBus struct {
	nc     *nats.Conn
	js     nats.JetStreamContext
	logger *logger.Logger
}

// NewEventBus creates a new event bus
func NewEventBus(natsURL string, logger *logger.Logger) (*EventBus, error) {
	nc, err := nats.Connect(natsURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %w", err)
	}
	
	js, err := nc.JetStream()
	if err != nil {
		return nil, fmt.Errorf("failed to create JetStream context: %w", err)
	}
	
	return &EventBus{
		nc:     nc,
		js:     js,
		logger: logger,
	}, nil
}

// Publish publishes an event
func (eb *EventBus) Publish(ctx context.Context, event Event) error {
	event.ID = uuid.New().String()
	event.Timestamp = time.Now().UTC()
	event.Version = "1.0"
	
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}
	
	subject := fmt.Sprintf("events.%s.%s", event.Source, event.Type)
	
	_, err = eb.js.Publish(subject, data)
	if err != nil {
		return fmt.Errorf("failed to publish event: %w", err)
	}
	
	eb.logger.InfoCtx(ctx, "Event published", 
		zap.String("event_id", event.ID),
		zap.String("event_type", event.Type),
		zap.String("subject", subject))
	
	return nil
}

// Subscribe subscribes to events with a handler
func (eb *EventBus) Subscribe(subject string, handler EventHandler) error {
	_, err := eb.js.Subscribe(subject, func(msg *nats.Msg) {
		var event Event
		if err := json.Unmarshal(msg.Data, &event); err != nil {
			eb.logger.Error("Failed to unmarshal event", zap.Error(err))
			msg.Nak()
			return
		}
		
		ctx := context.Background()
		if err := handler.Handle(ctx, event); err != nil {
			eb.logger.ErrorCtx(ctx, "Event handler failed", 
				zap.Error(err),
				zap.String("event_id", event.ID),
				zap.String("event_type", event.Type))
			msg.Nak()
			return
		}
		
		msg.Ack()
	}, nats.Durable("durable-consumer"))
	
	return err
}

// Close closes the event bus connection
func (eb *EventBus) Close() {
	eb.nc.Close()
}

// EventHandler interface for event handlers
type EventHandler interface {
	Handle(ctx context.Context, event Event) error
}

// UserEventHandler handles user-related events
type UserEventHandler struct {
	logger *logger.Logger
}

// NewUserEventHandler creates a new user event handler
func NewUserEventHandler(logger *logger.Logger) *UserEventHandler {
	return &UserEventHandler{
		logger: logger,
	}
}

// Handle handles user events
func (h *UserEventHandler) Handle(ctx context.Context, event Event) error {
	switch event.Type {
	case "user.created":
		return h.handleUserCreated(ctx, event)
	case "user.updated":
		return h.handleUserUpdated(ctx, event)
	case "user.deleted":
		return h.handleUserDeleted(ctx, event)
	default:
		h.logger.WarnCtx(ctx, "Unknown event type", zap.String("event_type", event.Type))
		return nil
	}
}

func (h *UserEventHandler) handleUserCreated(ctx context.Context, event Event) error {
	userID, ok := event.Data["user_id"].(string)
	if !ok {
		return fmt.Errorf("missing user_id in event data")
	}
	
	// Handle user creation (e.g., send welcome email, create related records)
	h.logger.InfoCtx(ctx, "Handling user created event", zap.String("user_id", userID))
	
	// Implementation would go here
	return nil
}

func (h *UserEventHandler) handleUserUpdated(ctx context.Context, event Event) error {
	userID, ok := event.Data["user_id"].(string)
	if !ok {
		return fmt.Errorf("missing user_id in event data")
	}
	
	// Handle user update (e.g., sync to other services)
	h.logger.InfoCtx(ctx, "Handling user updated event", zap.String("user_id", userID))
	
	return nil
}

func (h *UserEventHandler) handleUserDeleted(ctx context.Context, event Event) error {
	userID, ok := event.Data["user_id"].(string)
	if !ok {
		return fmt.Errorf("missing user_id in event data")
	}// GetByID retrieves a user by ID
func (r *UserRepository) GetByID(ctx context.Context, id uuid.UUID) (*User, error) {
	// Try cache first
	cacheKey := fmt.Sprintf("user:%s", id.String())
	var user User
	
	if err := r.CacheGet(ctx, cacheKey, &user); err == nil {
		return &user, nil
	}
	
	// Cache miss, get from database
	if err := r.db.WithContext(ctx).First(&user, id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	
	// Cache the result
	r.CacheSet(ctx, cacheKey, &user, time.Hour)
	
	return &user, nil
}

// GetByEmail retrieves a user by email
func (r *UserRepository) GetByEmail(ctx context.Context, email string) (*User, error) {
	var user User
	if err := r.db.WithContext(ctx).Where("email = ?", email).First(&user).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	return &user, nil
}

// Update updates a user
func (r *UserRepository) Update(ctx context.Context, user *User) error {
	if err := r.db.WithContext(ctx).Save(user).Error; err != nil {
		return fmt.Errorf("failed to update user: %w", err)
	}
	
	// Invalidate cache
	cacheKey := fmt.Sprintf("user:%s", user.ID.String())
	r.CacheDelete(ctx, cacheKey)
	
	return nil
}

// Delete deletes a user
func (r *UserRepository) Delete(ctx context.Context, id uuid.UUID) error {
	if err := r.db.WithContext(ctx).Delete(&User{}, id).Error; err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}
	
	// Invalidate cache
	cacheKey := fmt.Sprintf("user:%s", id.String())
	r.CacheDelete(ctx, cacheKey)
	
	return nil
}

// List retrieves users with pagination
func (r *UserRepository) List(ctx context.Context, offset, limit int) ([]User, int64, error) {
	var users []User
	var total int64
	
	// Get total count
	if err := r.db.WithContext(ctx).Model(&User{}).Count(&total).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to count users: %w", err)
	}
	
	// Get users with pagination
	if err := r.db.WithContext(ctx).
		Offset(offset).
		Limit(limit).
		Order("created_at DESC").
		Find(&users).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to list users: %w", err)
	}
	
	return users, total, nil
}

// PostRepository handles post data operations
type PostRepository struct {
	*BaseRepository
}

// NewPostRepository creates a new post repository
func NewPostRepository(db *gorm.DB, redis *redis.Client) *PostRepository {
	return &PostRepository{
		BaseRepository: NewBaseRepository(db, redis),
	}
}

// Create creates a new post
func (r *PostRepository) Create(ctx context.Context, post *Post) error {
	if err := r.db.WithContext(ctx).Create(post).Error; err != nil {
		return fmt.Errorf("failed to create post: %w", err)
	}
	
	// Invalidate related caches
	r.CacheDelete(ctx, "posts:published", "posts:featured")
	
	return nil
}

// GetByID retrieves a post by ID with relationships
func (r *PostRepository) GetByID(ctx context.Context, id uuid.UUID) (*Post, error) {
	var post Post
	if err := r.db.WithContext(ctx).
		Preload("User").
		Preload("Category").
		Preload("Tags").
		Preload("Comments", func(db *gorm.DB) *gorm.DB {
			return db.Where("status = ?", CommentStatusApproved).Order("created_at DESC")
		}).
		Preload("Comments.User").
		First(&post, id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("post not found")
		}
		return nil, fmt.Errorf("failed to get post: %w", err)
	}
	return &post, nil
}

// GetBySlug retrieves a post by slug
func (r *PostRepository) GetBySlug(ctx context.Context, slug string) (*Post, error) {
	var post Post
	if err := r.db.WithContext(ctx).
		Preload("User").
		Preload("Category").
		Preload("Tags").
		Where("slug = ?", slug).
		First(&post).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("post not found")
		}
		return nil, fmt.Errorf("failed to get post: %w", err)
	}
	return &post, nil
}

// List retrieves posts with filters and pagination
func (r *PostRepository) List(ctx context.Context, filters PostFilters) ([]Post, int64, error) {
	query := r.db.WithContext(ctx).Model(&Post{}).
		Preload("User").
		Preload("Category").
		Preload("Tags")
	
	// Apply filters
	if filters.Status != "" {
		query = query.Where("status = ?", filters.Status)
	}
	
	if filters.CategoryID != uuid.Nil {
		query = query.Where("category_id = ?", filters.CategoryID)
	}
	
	if filters.UserID != uuid.Nil {
		query = query.Where("user_id = ?", filters.UserID)
	}
	
	if filters.Featured != nil {
		query = query.Where("featured = ?", *filters.Featured)
	}
	
	if filters.Search != "" {
		query = query.Where("title ILIKE ? OR content ILIKE ?", 
			"%"+filters.Search+"%", "%"+filters.Search+"%")
	}
	
	// Get total count
	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to count posts: %w", err)
	}
	
	// Apply pagination and ordering
	var posts []Post
	if err := query.
		Offset(filters.Offset).
		Limit(filters.Limit).
		Order(filters.OrderBy + " " + filters.Order).
		Find(&posts).Error; err != nil {
		return nil, 0, fmt.Errorf("failed to list posts: %w", err)
	}
	
	return posts, total, nil
}

// GetPublished retrieves published posts
func (r *PostRepository) GetPublished(ctx context.Context, limit int) ([]Post, error) {
	// Try cache first
	cacheKey := fmt.Sprintf("posts:published:%d", limit)
	var posts []Post
	
	if err := r.CacheGet(ctx, cacheKey, &posts); err == nil {
		return posts, nil
	}
	
	// Cache miss, get from database
	if err := r.db.WithContext(ctx).
		Preload("User").
		Preload("Category").
		Where("status = ?", PostStatusPublished).
		Order("published_at DESC").
		Limit(limit).
		Find(&posts).Error; err != nil {
		return nil, fmt.Errorf("failed to get published posts: %w", err)
	}
	
	// Cache the result
	r.CacheSet(ctx, cacheKey, posts, 15*time.Minute)
	
	return posts, nil
}

// IncrementViewCount increments the view count for a post
func (r *PostRepository) IncrementViewCount(ctx context.Context, id uuid.UUID) error {
	return r.db.WithContext(ctx).
		Model(&Post{}).
		Where("id = ?", id).
		Update("view_count", gorm.Expr("view_count + 1")).Error
}

// PostFilters defines filters for post queries
type PostFilters struct {
	Status     PostStatus
	CategoryID uuid.UUID
	UserID     uuid.UUID
	Featured   *bool
	Search     string
	Offset     int
	Limit      int
	OrderBy    string
	Order      string
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           10. TESTING AND BENCHMARKING
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
)

// Test Suite
type APITestSuite struct {
	suite.Suite
	app        *gin.Engine
	db         *Database
	container  testcontainers.Container
	userService *UserService
	userHandler *UserHandler
}

// SetupSuite runs before all tests
func (suite *APITestSuite) SetupSuite() {
	// Setup test database using testcontainers
	ctx := context.Background()
	
	postgresContainer, err := postgres.RunContainer(ctx,
		testcontainers.WithImage("postgres:15"),
		postgres.WithDatabase("testdb"),
		postgres.WithUsername("testuser"),
		postgres.WithPassword("testpass"),
	)
	suite.Require().NoError(err)
	suite.container = postgresContainer
	
	// Get connection details
	host, err := postgresContainer.Host(ctx)
	suite.Require().NoError(err)
	
	port, err := postgresContainer.MappedPort(ctx, "5432")
	suite.Require().NoError(err)
	
	// Setup database
	dbConfig := DatabaseConfig{
		Host:         host,
		Port:         port.Int(),
		User:         "testuser",
		Password:     "testpass",
		Name:         "testdb",
		SSLMode:      "disable",
		MaxOpenConns: 10,
		MaxIdleConns: 5,
		MaxLifetime:  time.Minute,
	}
	
	redisConfig := RedisConfig{
		Host:     "localhost",
		Port:     6379,
		Password: "",
		DB:       1, // Use test database
		PoolSize: 5,
	}
	
	db, err := NewDatabase(dbConfig, redisConfig)
	suite.Require().NoError(err)
	suite.db = db
	
	// Run migrations
	err = MigrateModels(db.DB)
	suite.Require().NoError(err)
	
	// Setup services
	userRepo := NewUserRepository(db.DB, db.Redis)
	suite.userService = NewUserService(userRepo, "test-secret", time.Hour, "test-issuer")
	
	// Setup handlers
	logger, err := NewLogger("debug", "json", "stdout")
	suite.Require().NoError(err)
	
	suite.userHandler = NewUserHandler(suite.userService, logger)
	
	// Setup Gin app
	gin.SetMode(gin.TestMode)
	suite.app = gin.New()
	
	// Setup routes
	api := suite.app.Group("/api/v1")
	api.POST("/register", suite.userHandler.Register)
	api.POST("/login", suite.userHandler.Login)
	
	auth := api.Group("/")
	auth.Use(AuthMiddleware(suite.userService))
	auth.GET("/profile", suite.userHandler.GetProfile)
	auth.PUT("/profile", suite.userHandler.UpdateProfile)
}

// TearDownSuite runs after all tests
func (suite *APITestSuite) TearDownSuite() {
	if suite.db != nil {
		suite.db.Close()
	}
	if suite.container != nil {
		suite.container.Terminate(context.Background())
	}
}

// SetupTest runs before each test
func (suite *APITestSuite) SetupTest() {
	// Clean database before each test
	suite.db.DB.Exec("TRUNCATE users CASCADE")
}

// Test user registration
func (suite *APITestSuite) TestUserRegistration() {
	payload := CreateUserRequest{
		Email:     "test@example.com",
		Username:  "testuser",
		Password:  "password123",
		FirstName: "Test",
		LastName:  "User",
	}
	
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest("POST", "/api/v1/register", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	
	w := httptest.NewRecorder()
	suite.app.ServeHTTP(w, req)
	
	assert.Equal(suite.T(), http.StatusCreated, w.Code)
	
	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), "User created successfully", response["message"])
	assert.NotNil(suite.T(), response["user"])
}

// Test user registration with invalid data
func (suite *APITestSuite) TestUserRegistrationInvalidData() {
	payload := CreateUserRequest{
		Email:    "invalid-email",
		Username: "",
		Password: "123", // Too short
	}
	
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest("POST", "/api/v1/register", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	
	w := httptest.NewRecorder()
	suite.app.ServeHTTP(w, req)
	
	assert.Equal(suite.T(), http.StatusInternalServerError, w.Code)
}

// Test user login
func (suite *APITestSuite) TestUserLogin() {
	// First, create a user
	user := &User{
		Email:     "test@example.com",
		Username:  "testuser",
		FirstName: "Test",
		LastName:  "User",
		Role:      UserRoleUser,
		Status:    UserStatusActive,
	}
	user.SetPassword("password123")
	
	err := suite.userService.userRepo.Create(context.Background(), user)
	suite.Require().NoError(err)
	
	// Now test login
	payload := map[string]string{
		"email":    "test@example.com",
		"password": "password123",
	}
	
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest("POST", "/api/v1/login", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	
	w := httptest.NewRecorder()
	suite.app.ServeHTTP(w, req)
	
	assert.Equal(suite.T(), http.StatusOK, w.Code)
	
	var response AuthResponse
	err = json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), response.Token)
	assert.Equal(suite.T(), user.Email, response.User.Email)
}

// Test protected endpoint
func (suite *APITestSuite) TestProtectedEndpoint() {
	// Create and login user to get token
	user := &User{
		Email:     "test@example.com",
		Username:  "testuser",
		FirstName: "Test",
		LastName:  "User",
		Role:      UserRoleUser,
		Status:    UserStatusActive,
	}
	user.SetPassword("password123")
	
	err := suite.userService.userRepo.Create(context.Background(), user)
	suite.Require().NoError(err)
	
	authResp, err := suite.userService.Authenticate(context.Background(), "test@example.com", "password123")
	suite.Require().NoError(err)
	
	// Test protected endpoint with valid token
	req := httptest.NewRequest("GET", "/api/v1/profile", nil)
	req.Header.Set("Authorization", "Bearer "+authResp.Token)
	
	w := httptest.NewRecorder()
	suite.app.ServeHTTP(w, req)
	
	assert.Equal(suite.T(), http.StatusOK, w.Code)
}

// Test protected endpoint without token
func (suite *APITestSuite) TestProtectedEndpointWithoutToken() {
	req := httptest.NewRequest("GET", "/api/v1/profile", nil)
	
	w := httptest.NewRecorder()
	suite.app.ServeHTTP(w, req)
	
	assert.Equal(suite.T(), http.StatusUnauthorized, w.Code)
}

// Run the test suite
func TestAPITestSuite(t *testing.T) {
	suite.Run(t, new(APITestSuite))
}

// Unit Tests
func TestUserService_CreateUser(t *testing.T) {
	// Mock repository
	mockRepo := &MockUserRepository{}
	userService := NewUserService(mockRepo, "secret", time.Hour, "issuer")
	
	req := CreateUserRequest{
		Email:     "test@example.com",
		Username:  "testuser",
		Password:  "password123",
		FirstName: "Test",
		LastName:  "User",
	}
	
	// Setup mock expectations
	mockRepo.On("GetByEmail", mock.Anything, req.Email).Return(nil, fmt.Errorf("not found"))
	mockRepo.On("Create", mock.Anything, mock.AnythingOfType("*User")).Return(nil)
	
	user, err := userService.CreateUser(context.Background(), req)
	
	assert.NoError(t, err)
	assert.NotNil(t, user)
	assert.Equal(t, req.Email, user.Email)
	assert.Equal(t, req.Username, user.Username)
	
	mockRepo.AssertExpectations(t)
}

// Mock Repository
type MockUserRepository struct {
	mock.Mock
}

func (m *MockUserRepository) Create(ctx context.Context, user *User) error {
	args := m.Called(ctx, user)
	return args.Error(0)
}

func (m *MockUserRepository) GetByID(ctx context.Context, id uuid.UUID) (*User, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepository) GetByEmail(ctx context.Context, email string) (*User, error) {
	args := m.Called(ctx, email)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*User), args.Error(1)
}

func (m *MockUserRepository) Update(ctx context.Context, user *User) error {
	args := m.Called(ctx, user)
	return args.Error(0)
}

func (m *MockUserRepository) Delete(ctx context.Context, id uuid.UUID) error {
	args := m.Called(ctx, id)
	return args.Error(0)
}

func (m *MockUserRepository) List(ctx context.Context, offset, limit int) ([]User, int64, error) {
	args := m.Called(ctx, offset, limit)
	return args.Get(0).([]User), args.Get(1).(int64), args.Error(2)
}

// Benchmark Tests
func BenchmarkUserService_CreateUser(b *testing.B) {
	// Setup
	mockRepo := &MockUserRepository{}
	userService := NewUserService(mockRepo, "secret", time.Hour, "issuer")
	
	req := CreateUserRequest{
		Email:     "test@example.com",
		Username:  "testuser",
		Password:  "password123",
		FirstName: "Test",
		LastName:  "User",
	}
	
	mockRepo.On("GetByEmail", mock.Anything, mock.Anything).Return(nil, fmt.Errorf("not found"))
	mockRepo.On("Create", mock.Anything, mock.Anything).Return(nil)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		userService.CreateUser(context.Background(), req)
	}
}

func BenchmarkPasswordHashing(b *testing.B) {
	user := &User{}
	password := "password123"
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		user.SetPassword(password)
	}
}

func BenchmarkJWTGeneration(b *testing.B) {
	userService := NewUserService(nil, "secret", time.Hour, "issuer")
	user := &User{
		ID:       uuid.New(),
		Email:    "test@example.com",
		Username: "testuser",
		Role:     UserRoleUser,
	}
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		userService.generateJWT(user)
	}
}

// Load Testing Example
func TestConcurrentUserCreation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping load test in short mode")
	}
	
	// This would be a more comprehensive load test
	// using tools like vegeta or custom goroutines
	
	concurrency := 100
	requests := 1000
	
	// Setup test server
	// ... server setup code ...
	
	results := make(chan error, requests)
	
	for i := 0; i < concurrency; i++ {
		go func() {
			for j := 0; j < requests/concurrency; j++ {
				// Make HTTP request
				// results <- err
			}
		}()
	}
	
	// Collect results
	var errors []error
	for i := 0; i < requests; i++ {
		if err := <-results; err != nil {
			errors = append(errors, err)
		}
	}
	
	errorRate := float64(len(errors)) / float64(requests)
	assert.Less(t, errorRate, 0.01, "Error rate should be less than 1%")
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           11. DEPLOYMENT AND CONTAINERIZATION
// ═══════════════════════════════════════════════════════════════════════════════

// Dockerfile
/*
# Multi-stage build for smaller production image
FROM golang:1.21-alpine AS builder

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main cmd/api/main.go

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

WORKDIR /root/

# Copy binary from builder stage
COPY --from=builder /app/main .

# Copy configuration files
COPY --from=builder /app/configs ./configs

# Change ownership
RUN chown -R appuser:appgroup /root/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./main"]
*/

// docker-compose.yml
/*
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - APP_DATABASE_HOST=postgres
      - APP_DATABASE_PORT=5432
      - APP_DATABASE_USER=myapp
      - APP_DATABASE_PASSWORD=mypassword
      - APP_DATABASE_NAME=myapp
      - APP_REDIS_HOST=redis
      - APP_REDIS_PORT=6379
      - APP_JWT_SECRET_KEY=my-super-secret-key
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - app-network

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=myapp
      - POSTGRES_PASSWORD=mypassword
      - POSTGRES_DB=myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - app-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - app-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
*/

// Kubernetes Deployment
// k8s/namespace.yaml
/*
apiVersion: v1
kind: Namespace
metadata:
  name: myapp
*/

// k8s/configmap.yaml
/*
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
  namespace: myapp
data:
  config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
    database:
      host: postgres-service
      port: 5432
      name: myapp
      ssl_mode: "require"
    redis:
      host: redis-service
      port: 6379
    logging:
      level: "info"
      format: "json"
*/

// k8s/secret.yaml
/*
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
  namespace: myapp
type: Opaque
data:
  database-user: bXlhcHA=        # base64 encoded "myapp"
  database-password: bXlwYXNzd29yZA==  # base64 encoded "mypassword"
  jwt-secret: bXktc3VwZXItc2VjcmV0LWtleQ==  # base64 encoded secret
*/

// k8s/deployment.yaml
/*
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-api
  namespace: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp-api
  template:
    metadata:
      labels:
        app: myapp-api
    spec:
      containers:
      - name: api
        image: myapp:latest
        ports:
        - containerPort: 8080
        env:
        - name: APP_DATABASE_USER
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-user
        - name: APP_DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-password
        - name: APP_JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: jwt-secret
        volumeMounts:
        - name: config-volume
          mountPath: /app/configs
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: config-volume
        configMap:
          name: myapp-config
*/

// k8s/service.yaml
/*
apiVersion: v1
kind: Service
metadata:
  name: myapp-api-service
  namespace: myapp
spec:
  selector:
    app: myapp-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
*/

// k8s/ingress.yaml
/*
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  namespace: myapp
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.myapp.com
    secretName: myapp-tls
  rules:
  - host: api.myapp.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: myapp-api-service
            port:
              number: 80
*/

// Health Check Handler
package handlers

func (h *HealthHandler) HealthCheck(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()
	
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC(),
		"version":   os.Getenv("APP_VERSION"),
		"checks":    make(map[string]interface{}),
	}
	
	// Database health check
	if err := h.db.Health(ctx); err != nil {
		health["checks"].(map[string]interface{})["database"] = map[string]interface{}{
			"status": "unhealthy",
			"error":  err.Error(),
		}
		health["status"] = "unhealthy"
	} else {
		health["checks"].(map[string]interface{})["database"] = map[string]interface{}{
			"status": "healthy",
		}
	}
	
	// Redis health check
	if err := h.db.Redis.Ping(ctx).Err(); err != nil {
		health["checks"].(map[string]interface{})["redis"] = map[string]interface{}{
			"status": "unhealthy",
			"error":  err.Error(),
		}
		health["status"] = "unhealthy"
	} else {
		health["checks"].(map[string]interface{})["redis"] = map[string]interface{}{
			"status": "healthy",
		}
	}
	
	// External service checks would go here
	
	status := http.StatusOK
	if health["status"] == "unhealthy" {
		status = http.StatusServiceUnavailable
	}
	
	c.JSON(status, health)
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           12. MAIN APPLICATION ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════

// cmd/api/main.go
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	// Load configuration
	config, err := LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize logger
	logger, err := NewLogger(config.Logging.Level, config.Logging.Format, config.Logging.Output)
	if err != nil {
		log.Fatalf("Failed to initialize logger: %v", err)
	}
	defer logger.Sync()

	logger.Info("Starting application",
		zap.String("environment", os.Getenv("GO_ENV")),
		zap.String("version", os.Getenv("APP_VERSION")))

	// Initialize database
	db, err := NewDatabase(config.Database, config.Redis)
	if err != nil {
		logger.Fatal("Failed to initialize database", zap.Error(err))
	}
	defer db.Close()

	// Run migrations
	if err := MigrateModels(db.DB); err != nil {
		logger.Fatal("Failed to run database migrations", zap.Error(err))
	}

	// Initialize repositories
	userRepo := NewUserRepository(db.DB, db.Redis)
	postRepo := NewPostRepository(db.DB, db.Redis)
	categoryRepo := NewCategoryRepository(db.DB, db.Redis)

	// Initialize services
	userService := NewUserService(userRepo, config.JWT.SecretKey, config.JWT.ExpirationTime, config.JWT.Issuer)
	postService := NewPostService(postRepo, categoryRepo, userRepo)

	// Initialize handlers
	userHandler := NewUserHandler(userService, logger)
	postHandler := NewPostHandler(postService, logger)
	healthHandler := NewHealthHandler(db, logger)

	// Initialize event bus
	eventBus, err := NewEventBus("nats://localhost:4222", logger)
	if err != nil {
		logger.Warn("Failed to initialize event bus", zap.Error(err))
	}
	defer eventBus.Close()

	// Initialize worker pool
	workerPool := NewWorkerPool(5, 100, logger)
	workerPool.Start()
	defer workerPool.Stop()

	// Setup HTTP server
	if os.Getenv("GO_ENV") == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	
	// Global middleware
	router.Use(RequestIDMiddleware())
	router.Use(LoggingMiddleware(logger))
	router.Use(MetricsMiddleware())
	router.Use(CORSMiddleware())
	router.Use(SecurityHeadersMiddleware())		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to change password"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"message": "Password changed successfully"})
}

// PostHandler handles post-related HTTP requests
type PostHandler struct {
	postService *PostService
	logger      *logger.Logger
}

// NewPostHandler creates a new post handler
func NewPostHandler(postService *PostService, logger *logger.Logger) *PostHandler {
	return &PostHandler{
		postService: postService,
		logger:      logger,
	}
}

// CreatePost handles post creation
func (h *PostHandler) CreatePost(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	var req CreatePostRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	post, err := h.postService.CreatePost(c.Request.Context(), userID.(uuid.UUID), req)
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to create post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create post"})
		return
	}
	
	h.logger.InfoCtx(c.Request.Context(), "Post created successfully", 
		zap.String("post_id", post.ID.String()), zap.String("user_id", userID.(uuid.UUID).String()))
	
	c.JSON(http.StatusCreated, gin.H{
		"message": "Post created successfully",
		"post":    post,
	})
}

// GetPost handles getting a single post
func (h *PostHandler) GetPost(c *gin.Context) {
	idParam := c.Param("id")
	postID, err := uuid.Parse(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid post ID"})
		return
	}
	
	post, err := h.postService.GetPost(c.Request.Context(), postID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to get post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"post": post})
}

// GetPostBySlug handles getting a post by slug
func (h *PostHandler) GetPostBySlug(c *gin.Context) {
	slug := c.Param("slug")
	
	post, err := h.postService.GetPostBySlug(c.Request.Context(), slug)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to get post by slug", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{"post": post})
}

// ListPosts handles listing posts with filters
func (h *PostHandler) ListPosts(c *gin.Context) {
	filters := PostFilters{
		Status:  PostStatus(c.Query("status")),
		Search:  c.Query("search"),
		OrderBy: c.Query("order_by"),
		Order:   c.Query("order"),
	}
	
	// Parse pagination
	if page := c.Query("page"); page != "" {
		if p, err := strconv.Atoi(page); err == nil && p > 0 {
			filters.Offset = (p - 1) * filters.Limit
		}
	}
	
	if limit := c.Query("limit"); limit != "" {
		if l, err := strconv.Atoi(limit); err == nil && l > 0 {
			filters.Limit = l
		}
	}
	
	if filters.Limit == 0 {
		filters.Limit = 20
	}
	
	// Parse category ID
	if categoryID := c.Query("category_id"); categoryID != "" {
		if id, err := uuid.Parse(categoryID); err == nil {
			filters.CategoryID = id
		}
	}
	
	// Parse featured filter
	if featured := c.Query("featured"); featured != "" {
		if f, err := strconv.ParseBool(featured); err == nil {
			filters.Featured = &f
		}
	}
	
	response, err := h.postService.ListPosts(c.Request.Context(), filters)
	if err != nil {
		h.logger.ErrorCtx(c.Request.Context(), "Failed to list posts", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list posts"})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

// UpdatePost handles post updates
func (h *PostHandler) UpdatePost(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	idParam := c.Param("id")
	postID, err := uuid.Parse(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid post ID"})
		return
	}
	
	var req UpdatePostRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	post, err := h.postService.UpdatePost(c.Request.Context(), postID, userID.(uuid.UUID), req)
	if err != nil {
		if strings.Contains(err.Error(), "permission denied") {
			c.JSON(http.StatusForbidden, gin.H{"error": "Permission denied"})
			return
		}
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to update post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Post updated successfully",
		"post":    post,
	})
}

// PublishPost handles post publishing
func (h *PostHandler) PublishPost(c *gin.Context) {
	userID, exists := c.Get("user_id")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}
	
	idParam := c.Param("id")
	postID, err := uuid.Parse(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid post ID"})
		return
	}
	
	post, err := h.postService.PublishPost(c.Request.Context(), postID, userID.(uuid.UUID))
	if err != nil {
		if strings.Contains(err.Error(), "permission denied") {
			c.JSON(http.StatusForbidden, gin.H{"error": "Permission denied"})
			return
		}
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{"error": "Post not found"})
			return
		}
		
		h.logger.ErrorCtx(c.Request.Context(), "Failed to publish post", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to publish post"})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Post published successfully",
		"post":    post,
	})
}

// Middleware
package middleware

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// AuthMiddleware validates JWT tokens and sets user context
func AuthMiddleware(userService *UserService) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization header required"})
			c.Abort()
			return
		}
		
		tokenParts := strings.Split(authHeader, " ")
		if len(tokenParts) != 2 || tokenParts[0] != "Bearer" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid authorization header format"})
			c.Abort()
			return
		}
		
		token := tokenParts[1]
		user, err := userService.ValidateToken(token)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}
		
		// Set user context
		c.Set("user_id", user.ID)
		c.Set("user", user)
		c.Next()
	}
}

// OptionalAuthMiddleware validates JWT tokens but doesn't require them
func OptionalAuthMiddleware(userService *UserService) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.Next()
			return
		}
		
		tokenParts := strings.Split(authHeader, " ")
		if len(tokenParts) != 2 || tokenParts[0] != "Bearer" {
			c.Next()
			return
		}
		
		token := tokenParts[1]
		user, err := userService.ValidateToken(token)
		if err != nil {
			c.Next()
			return
		}
		
		// Set user context
		c.Set("user_id", user.ID)
		c.Set("user", user)
		c.Next()
	}
}

// RequestIDMiddleware adds a unique request ID to each request
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := uuid.New().String()
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)
		c.Next()
	}
}

// LoggingMiddleware logs HTTP requests
func LoggingMiddleware(logger *logger.Logger) gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		logger.InfoCtx(param.Request.Context(), "HTTP Request",
			zap.String("method", param.Method),
			zap.String("path", param.Path),
			zap.Int("status", param.StatusCode),
			zap.Duration("latency", param.Latency),
			zap.String("client_ip", param.ClientIP),
			zap.String("user_agent", param.Request.UserAgent()),
		)
		return ""
	})
}

// MetricsMiddleware records HTTP metrics
func MetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		c.Next()
		
		duration := time.Since(start)
		method := c.Request.Method
		route := c.FullPath()
		if route == "" {
			route = "unknown"
		}
		status := strconv.Itoa(c.Writer.Status())
		
		metrics.RecordHTTPRequest(method, route, status, duration)
	}
}

// CORSMiddleware handles Cross-Origin Resource Sharing
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Request-ID")
		c.Header("Access-Control-Expose-Headers", "X-Request-ID")
		c.Header("Access-Control-Max-Age", "86400")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		
		c.Next()
	}
}

// RateLimitMiddleware implements simple rate limiting
func RateLimitMiddleware(rps int) gin.HandlerFunc {
	// This is a simplified rate limiter
	// In production, use a proper rate limiting library like golang.org/x/time/rate
	return func(c *gin.Context) {
		// Implementation would go here
		c.Next()
	}
}

// SecurityHeadersMiddleware adds security headers
func SecurityHeadersMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'")
		c.Next()
	}
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           13. PERFORMANCE OPTIMIZATION AND BEST PRACTICES
// ═══════════════════════════════════════════════════════════════════════════════

/*
GO PERFORMANCE OPTIMIZATION BEST PRACTICES:

1. MEMORY MANAGEMENT:
   - Use object pooling for frequently allocated objects
   - Minimize garbage collection pressure
   - Use appropriate data structures
   - Avoid memory leaks in goroutines

2. CONCURRENCY PATTERNS:
   - Use worker pools for bounded parallelism
   - Implement proper context cancellation
   - Use channels for communication, not shared memory
   - Avoid goroutine leaks

3. DATABASE OPTIMIZATION:
   - Use connection pooling effectively
   - Implement proper indexing
   - Use prepared statements
   - Batch operations when possible
   - Implement read replicas for scaling

4. CACHING STRATEGIES:
   - Cache frequently accessed data
   - Use Redis for distributed caching
   - Implement cache invalidation strategies
   - Use local caching for hot data

5. HTTP/API OPTIMIZATION:
   - Use HTTP/2 where possible
   - Implement proper compression
   - Use CDNs for static content
   - Implement rate limiting
   - Use connection keep-alive

6. PROFILING AND MONITORING:
   - Use pprof for CPU and memory profiling
   - Implement comprehensive metrics
   - Use distributed tracing
   - Monitor goroutine counts and GC stats

EXAMPLE IMPLEMENTATIONS:
*/

package performance

import (
	"context"
	"runtime"
	"sync"
	"time"
)

// Object Pool for reducing allocations
type UserPool struct {
	pool sync.Pool
}

func NewUserPool() *UserPool {
	return &UserPool{
		pool: sync.Pool{
			New: func() interface{} {
				return &User{}
			},
		},
	}
}

func (p *UserPool) Get() *User {
	return p.pool.Get().(*User)
}

func (p *UserPool) Put(user *User) {
	// Reset user fields before putting back
	*user = User{}
	p.pool.Put(user)
}

// Connection Pool Manager
type ConnectionManager struct {
	maxConns    int
	activeConns int
	idleConns   chan *Connection
	mu          sync.Mutex
}

type Connection struct {
	ID        string
	CreatedAt time.Time
	LastUsed  time.Time
}

func NewConnectionManager(maxConns int) *ConnectionManager {
	return &ConnectionManager{
		maxConns:  maxConns,
		idleConns: make(chan *Connection, maxConns),
	}
}

func (cm *ConnectionManager) GetConnection(ctx context.Context) (*Connection, error) {
	select {
	case conn := <-cm.idleConns:
		conn.LastUsed = time.Now()
		return conn, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		cm.mu.Lock()
		if cm.activeConns < cm.maxConns {
			cm.activeConns++
			cm.mu.Unlock()
			return &Connection{
				ID:        fmt.Sprintf("conn-%d", time.Now().UnixNano()),
				CreatedAt: time.Now(),
				LastUsed:  time.Now(),
			}, nil
		}
		cm.mu.Unlock()
		
		// Wait for available connection
		select {
		case conn := <-cm.idleConns:
			conn.LastUsed = time.Now()
			return conn, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (cm *ConnectionManager) ReleaseConnection(conn *Connection) {
	select {
	case cm.idleConns <- conn:
	default:
		// Pool is full, close connection
		cm.mu.Lock()
		cm.activeConns--
		cm.mu.Unlock()
	}
}

// Circuit Breaker Pattern
type CircuitBreaker struct {
	name          string
	maxFailures   int
	resetTimeout  time.Duration
	state         CircuitState
	failures      int
	lastFailTime  time.Time
	mu            sync.RWMutex
}

type CircuitState int

const (
	StateClosed CircuitState = iota
	StateOpen
	StateHalfOpen
)

func NewCircuitBreaker(name string, maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		name:         name,
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        StateClosed,
	}
}

func (cb *CircuitBreaker) Call(fn func() error) error {
	if !cb.canExecute() {
		return fmt.Errorf("circuit breaker %s is open", cb.name)
	}
	
	err := fn()
	cb.recordResult(err)
	return err
}

func (cb *CircuitBreaker) canExecute() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		return time.Since(cb.lastFailTime) >= cb.resetTimeout
	case StateHalfOpen:
		return true
	default:
		return false
	}
}

func (cb *CircuitBreaker) recordResult(err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	
	if err != nil {
		cb.failures++
		cb.lastFailTime = time.Now()
		
		if cb.failures >= cb.maxFailures {
			cb.state = StateOpen
		}
	} else {
		cb.failures = 0
		cb.state = StateClosed
	}
}

// Rate Limiter using Token Bucket
type TokenBucket struct {
	capacity     int
	tokens       int
	refillRate   int
	lastRefill   time.Time
	mu           sync.Mutex
}

func NewTokenBucket(capacity, refillRate int) *TokenBucket {
	return &TokenBucket{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill)
	
	// Refill tokens
	tokensToAdd := int(elapsed.Seconds()) * tb.refillRate
	tb.tokens = min(tb.capacity, tb.tokens+tokensToAdd)
	tb.lastRefill = now
	
	if tb.tokens > 0 {
		tb.tokens--
		return true
	}
	
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Memory-efficient batch processor
type BatchProcessor struct {
	batchSize   int
	maxWait     time.Duration
	processor   func([]interface{}) error
	items       []interface{}
	timer       *time.Timer
	mu          sync.Mutex
	done        chan struct{}
}

func NewBatchProcessor(batchSize int, maxWait time.Duration, processor func([]interface{}) error) *BatchProcessor {
	bp := &BatchProcessor{
		batchSize: batchSize,
		maxWait:   maxWait,
		processor: processor,
		items:     make([]interface{}, 0, batchSize),
		done:      make(chan struct{}),
	}
	
	go bp.run()
	return bp
}

func (bp *BatchProcessor) Add(item interface{}) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	
	bp.items = append(bp.items, item)
	
	if len(bp.items) == 1 {
		bp.timer = time.AfterFunc(bp.maxWait, bp.flush)
	}
	
	if len(bp.items) >= bp.batchSize {
		bp.timer.Stop()
		bp.processBatch()
	}
}

func (bp *BatchProcessor) flush() {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	bp.processBatch()
}

func (bp *BatchProcessor) processBatch() {
	if len(bp.items) == 0 {
		return
	}
	
	batch := make([]interface{}, len(bp.items))
	copy(batch, bp.items)
	bp.items = bp.items[:0]
	
	go func() {
		if err := bp.processor(batch); err != nil {
			// Handle error (log, retry, etc.)
		}
	}()
}

func (bp *BatchProcessor) run() {
	<-bp.done
}

func (bp *BatchProcessor) Close() {
	close(bp.done)
	bp.mu.Lock()
	if bp.timer != nil {
		bp.timer.Stop()
	}
	bp.processBatch()
	bp.mu.Unlock()
}

// Performance monitoring
type PerformanceMonitor struct {
	stats map[string]*OperationStats
	mu    sync.RWMutex
}

type OperationStats struct {
	Count        int64
	TotalTime    time.Duration
	MinTime      time.Duration
	MaxTime      time.Duration
	ErrorCount   int64
	LastExecuted time.Time
}

func NewPerformanceMonitor() *PerformanceMonitor {
	return &PerformanceMonitor{
		stats: make(map[string]*OperationStats),
	}
}

func (pm *PerformanceMonitor) Record(operation string, duration time.Duration, err error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	stats, exists := pm.stats[operation]
	if !exists {
		stats = &OperationStats{
			MinTime: duration,
			MaxTime: duration,
		}
		pm.stats[operation] = stats
	}
	
	stats.Count++
	stats.TotalTime += duration
	stats.LastExecuted = time.Now()
	
	if duration < stats.MinTime {
		stats.MinTime = duration
	}
	if duration > stats.MaxTime {
		stats.MaxTime = duration
	}
	
	if err != nil {
		stats.ErrorCount++
	}
}

func (pm *PerformanceMonitor) GetStats(operation string) *OperationStats {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	
	stats, exists := pm.stats[operation]
	if !exists {
		return nil
	}
	
	// Return a copy to avoid race conditions
	return &OperationStats{
		Count:        stats.Count,
		TotalTime:    stats.TotalTime,
		MinTime:      stats.MinTime,
		MaxTime:      stats.MaxTime,
		ErrorCount:   stats.ErrorCount,
		LastExecuted: stats.LastExecuted,
	}
}

func (pm *PerformanceMonitor) GetAverageTime(operation string) time.Duration {
	stats := pm.GetStats(operation)
	if stats == nil || stats.Count == 0 {
		return 0
	}
	return time.Duration(int64(stats.TotalTime) / stats.Count)
}

// Memory usage monitoring
func GetMemoryStats() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return map[string]interface{}{
		"alloc_mb":      bToMb(m.Alloc),
		"total_alloc_mb": bToMb(m.TotalAlloc),
		"sys_mb":        bToMb(m.Sys),
		"num_gc":        m.NumGC,
		"gc_cpu_fraction": m.GCCPUFraction,
		"goroutines":    runtime.NumGoroutine(),
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

/*
═══════════════════════════════════════════════════════════════════════════════
                           14. SECURITY IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════

SECURITY BEST PRACTICES FOR GO MICROSERVICES:

1. AUTHENTICATION & AUTHORIZATION:
   - Use strong JWT tokens with proper expiration
   - Implement role-based access control (RBAC)
   - Use secure password hashing (bcrypt)
   - Implement proper session management

2. INPUT VALIDATION:
   - Validate all user inputs
   - Use parameterized queries to prevent SQL injection
   - Sanitize data before processing
   - Implement rate limiting

3. ENCRYPTION & TLS:
   - Use TLS for all communications
   - Encrypt sensitive data at rest
   - Use strong encryption algorithms
   - Manage certificates properly

4. MONITORING & LOGGING:
   - Log security events
   - Monitor for suspicious activities
   - Implement alerting for security incidents
   - Regular security audits

5. DEPENDENCY MANAGEMENT:
   - Keep dependencies updated
   - Use dependency scanning tools
   - Minimize attack surface
   - Use trusted sources only
*/

package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
)

// Input validation utilities
type Validator struct {
	emailRegex    *regexp.Regexp
	usernameRegex *regexp.Regexp
}

func NewValidator() *Validator {
	return &Validator{
		emailRegex:    regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}),
		usernameRegex: regexp.MustCompile(`^[a-zA-Z0-9_]{3,30}),
	}
}

func (v *Validator) ValidateEmail(email string) error {
	if len(email) > 254 {
		return fmt.Errorf("email too long")
	}
	if !v.emailRegex.MatchString(email) {
		return fmt.Errorf("invalid email format")
	}
	return nil
}

func (v *Validator) ValidateUsername(username string) error {
	if len(username) < 3 || len(username) > 30 {
		return fmt.Errorf("username must be between 3 and 30 characters")
	}
	if !v.usernameRegex.MatchString(username) {
		return fmt.Errorf("username can only contain letters, numbers, and underscores")
	}
	return nil
}

func (v *Validator) ValidatePassword(password string) error {
	if len(password) < 8 {
		return fmt.Errorf("password must be at least 8 characters")
	}
	if len(password) > 128 {
		return fmt.Errorf("password too long")
	}
	
	var hasUpper, hasLower, hasDigit, hasSpecial bool
	for _, char := range password {
		switch {
		case 'A' <= char && char <= 'Z':
			hasUpper = true
		case 'a' <= char && char <= 'z':
			hasLower = true
		case '0' <= char && char <= '9':
			hasDigit = true
		default:
			hasSpecial = true
		}
	}
	
	if !hasUpper {
		return fmt.Errorf("password must contain at least one uppercase letter")
	}
	if !hasLower {
		return fmt.Errorf("password must contain at least one lowercase letter")
	}
	if !hasDigit {
		return fmt.Errorf("password must contain at least one digit")
	}
	if !hasSpecial {
		return fmt.Errorf("password must contain at least one special character")
	}
	
	return nil
}

// Sanitize input to prevent XSS
func (v *Validator) SanitizeString(input string) string {
	// Remove potentially dangerous characters
	input = strings.ReplaceAll(input, "<script", "")
	input = strings.ReplaceAll(input, "</script>", "")
	input = strings.ReplaceAll(input, "javascript:", "")
	input = strings.ReplaceAll(input, "on", "") // Remove event handlers
	
	return strings.TrimSpace(input)
}

// Encryption utilities
type EncryptionService struct {
	key []byte
}

func NewEncryptionService(secretKey string) *EncryptionService {
	hash := sha256.Sum256([]byte(secretKey))
	return &EncryptionService{
		key: hash[:],
	}
}

func (e *EncryptionService) Encrypt(plaintext string) (string, error) {
	block, err := aes.NewCipher(e.key)
	if err != nil {
		return "", err
	}
	
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	
	nonce := make([]byte, gcm.NonceSize())
	if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}
	
	ciphertext := gcm.Seal(nonce, nonce, []byte(plaintext), nil)
	return base64.StdEncoding.EncodeToString(ciphertext), nil
}

func (e *EncryptionService) Decrypt(ciphertext string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(ciphertext)
	if err != nil {
		return "", err
	}
	
	block, err := aes.NewCipher(e.key)
	if err != nil {
		return "", err
	}
	
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	
	nonceSize := gcm.NonceSize()
	if len(data) < nonceSize {
		return "", fmt.Errorf("ciphertext too short")
	}
	
	nonce, ciphertext := data[:nonceSize], data[nonceSize:]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return "", err
	}
	
	return string(plaintext), nil
}

// Security audit logger
type SecurityAuditLogger struct {
	logger *logger.Logger
}

func NewSecurityAuditLogger(logger *logger.Logger) *SecurityAuditLogger {
	return &SecurityAuditLogger{
		logger: logger,
	}
}

func (sal *SecurityAuditLogger) LogLoginAttempt(ctx context.Context, email string, success bool, clientIP string) {
	sal.logger.InfoCtx(ctx, "Login attempt",
		zap.String("event_type", "login_attempt"),
		zap.String("email", email),
		zap.Bool("success", success),
		zap.String("client_ip", clientIP),
		zap.Time("timestamp", time.Now()))
}

func (sal *SecurityAuditLogger) LogPasswordChange(ctx context.Context, userID string, clientIP string) {
	sal.logger.InfoCtx(ctx, "Password changed",
		zap.String("event_type", "password_change"),
		zap.String("user_id", userID),
		zap.String("client_ip", clientIP),
		zap.Time("timestamp", time.Now()))
}

func (sal *SecurityAuditLogger) LogSuspiciousActivity(ctx context.Context, activity string, userID string, clientIP string, details map[string]interface{}) {
	fields := []zap.Field{
		zap.String("event_type", "suspicious_activity"),
		zap.String("activity", activity),
		zap.String("user_id", userID),
		zap.String("client_ip", clientIP),
		zap.Time("timestamp", time.Now()),
	}
	
	for k, v := range details {
		fields = append(fields, zap.Any(k, v))
	}
	
	sal.logger.WarnCtx(ctx, "Suspicious activity detected", fields...)
}

/*
═══════════════════════════════════════════════════════════════════════════════
                           15. CONCLUSION AND ADDITIONAL RESOURCES
═══════════════════════════════════════════════════════════════════════════════

This comprehensive Go reference covers:

✅ Project setup and configuration management
✅ Structured logging and monitoring with Prometheus
✅ Database layer with GORM and Redis caching
✅ Repository pattern implementation
✅ Service layer with business logic
✅ HTTP handlers and middleware
✅ gRPC microservices communication
✅ Message queues and event processing
✅ Comprehensive testing strategies
✅ Containerization and Kubernetes deployment
✅ Performance optimization techniques
✅ Security best practices
✅ Production-ready application structure

ADDITIONAL GO RESOURCES:

1. DOCUMENTATION:
   - Official Go Documentation: https://golang.org/doc/
   - Effective Go: https://golang.org/doc/effective_go.html
   - Go Code Review Comments: https://github.com/golang/go/wiki/CodeReviewComments

2. PERFORMANCE:
   - Go Performance Tools: https://golang.org/doc/diagnostics.html
   - pprof Tutorial: https://blog.golang.org/pprof
   - Memory Profiling: https://golang.org/pkg/runtime/pprof/

3. TESTING:
   - Testing Package: https://golang.org/pkg/testing/
   - Testify Framework: https://github.com/stretchr/testify
   - Gomega Matcher Library: https://onsi.github.io/gomega/

4. MICROSERVICES:
   - gRPC Go Tutorial: https://grpc.io/docs/languages/go/
   - Go Kit Microservices: https://gokit.io/
   - Micro Framework: https://micro.mu/

5. DEPLOYMENT:
   - Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/
   - Kubernetes Go Client: https://github.com/kubernetes/client-go
   - Helm Charts: https://helm.sh/docs/

PERFORMANCE BENCHMARKS:
- Go can handle 100k+ concurrent connections with proper design
- Memory usage typically 10-20MB for basic microservices
- Request latency often sub-millisecond for cached operations
- Excellent horizontal scaling characteristics

Go excels at building high-performance, concurrent systems with minimal resource
usage, making it ideal for microservices architecture, APIs, and distributed systems.
This reference provides production-ready patterns for building scalable applications.
*/