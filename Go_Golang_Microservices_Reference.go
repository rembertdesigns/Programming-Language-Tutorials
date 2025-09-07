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