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
		c.JSON(http.StatusInternalServerError, gin.H{"error": "