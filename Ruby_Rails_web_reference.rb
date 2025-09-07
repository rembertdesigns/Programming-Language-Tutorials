# RUBY ON RAILS WEB DEVELOPMENT - Comprehensive Reference - by Richard Rembert
# Ruby on Rails enables rapid prototyping and development of full-stack web applications
# with convention over configuration, built-in security, and powerful abstractions

# ═══════════════════════════════════════════════════════════════════════════════
#                           1. SETUP AND ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

=begin
RUBY ON RAILS DEVELOPMENT SETUP:

1. Install Ruby (using rbenv recommended):
   # Install rbenv
   curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash
   
   # Install Ruby
   rbenv install 3.2.0
   rbenv global 3.2.0
   
   # Verify installation
   ruby -v

2. Install Rails:
   gem install rails -v 7.1.0
   rails -v

3. Install dependencies:
   # Database (PostgreSQL recommended)
   brew install postgresql
   
   # Node.js (for asset pipeline)
   brew install node
   
   # Redis (for caching and background jobs)
   brew install redis

4. Create new Rails application:
   rails new my_web_app --database=postgresql --css=tailwind --javascript=stimulus
   cd my_web_app
   
   # Or for API-only application
   rails new my_api --api --database=postgresql

5. Essential Gems (Gemfile):
   # Core gems
   gem 'rails', '~> 7.1.0'
   gem 'pg', '~> 1.1'
   gem 'puma', '~> 5.0'
   gem 'sass-rails', '>= 6'
   gem 'webpacker', '~> 5.0'
   gem 'turbo-rails'
   gem 'stimulus-rails'
   gem 'jbuilder', '~> 2.7'
   
   # Authentication and Authorization
   gem 'devise'
   gem 'omniauth'
   gem 'omniauth-rails_csrf_protection'
   gem 'pundit'
   
   # Database and Models
   gem 'pg_search'
   gem 'kaminari'
   gem 'paranoia'
   gem 'paper_trail'
   
   # API and JSON
   gem 'fast_jsonapi'
   gem 'rack-cors'
   gem 'jwt'
   
   # Background Jobs
   gem 'sidekiq'
   gem 'sidekiq-web'
   
   # File Upload
   gem 'image_processing', '~> 1.2'
   gem 'aws-sdk-s3'
   
   # Development and Testing
   group :development, :test do
     gem 'rspec-rails'
     gem 'factory_bot_rails'
     gem 'faker'
     gem 'pry-rails'
     gem 'byebug'
   end
   
   group :development do
     gem 'web-console', '>= 4.1.0'
     gem 'listen', '~> 3.3'
     gem 'spring'
     gem 'annotate'
     gem 'bullet'
   end

6. Database setup:
   rails db:create
   rails db:migrate
   rails db:seed

7. Start server:
   rails server
   # Visit http://localhost:3000
=end

# ═══════════════════════════════════════════════════════════════════════════════
#                           2. APPLICATION STRUCTURE AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# config/application.rb
module MyWebApp
    class Application < Rails::Application
      config.load_defaults 7.1
      
      # Time zone
      config.time_zone = 'UTC'
      
      # API configuration
      config.api_only = false
      
      # CORS configuration
      config.middleware.insert_before 0, Rack::Cors do
        allow do
          origins 'localhost:3000', 'localhost:3001'
          resource '*',
            headers: :any,
            methods: [:get, :post, :put, :patch, :delete, :options, :head],
            credentials: true
        end
      end
      
      # Auto-load lib directory
      config.autoload_paths << Rails.root.join('lib')
      
      # Session store
      config.session_store :cookie_store, key: '_my_web_app_session'
      
      # Active Job adapter
      config.active_job.queue_adapter = :sidekiq
      
      # Asset host for CDN
      # config.asset_host = 'https://cdn.example.com'
    end
  end
  
  # config/database.yml
  default: &default
    adapter: postgresql
    encoding: unicode
    pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>
    host: <%= ENV.fetch("DB_HOST") { "localhost" } %>
    username: <%= ENV.fetch("DB_USERNAME") { "postgres" } %>
    password: <%= ENV.fetch("DB_PASSWORD") { "" } %>
  
  development:
    <<: *default
    database: my_web_app_development
  
  test:
    <<: *default
    database: my_web_app_test
  
  production:
    <<: *default
    database: my_web_app_production
    url: <%= ENV['DATABASE_URL'] %>
  
  # config/routes.rb
  Rails.application.routes.draw do
    # Root route
    root 'home#index'
    
    # Authentication routes (Devise)
    devise_for :users, controllers: {
      registrations: 'users/registrations',
      sessions: 'users/sessions',
      passwords: 'users/passwords'
    }
    
    # API namespace
    namespace :api do
      namespace :v1 do
        resources :users, only: [:show, :update]
        resources :posts do
          resources :comments, except: [:show]
        end
        resources :categories, only: [:index, :show]
        
        # Authentication endpoints
        post 'auth/login', to: 'authentication#login'
        post 'auth/logout', to: 'authentication#logout'
        post 'auth/refresh', to: 'authentication#refresh'
      end
    end
    
    # Web routes
    resources :posts do
      member do
        patch :publish
        patch :unpublish
        post :toggle_featured
      end
      
      collection do
        get :published
        get :drafts
      end
      
      resources :comments, except: [:index, :show]
    end
    
    resources :categories, except: [:destroy]
    resources :users, only: [:index, :show, :edit, :update]
    
    # Admin routes
    namespace :admin do
      resources :users do
        member do
          patch :activate
          patch :deactivate
        end
      end
      resources :posts
      resources :categories
      root 'dashboard#index'
    end
    
    # Static pages
    get 'about', to: 'pages#about'
    get 'contact', to: 'pages#contact'
    post 'contact', to: 'pages#create_contact'
    get 'privacy', to: 'pages#privacy'
    get 'terms', to: 'pages#terms'
    
    # Sidekiq Web UI (for development)
    if Rails.env.development?
      require 'sidekiq/web'
      mount Sidekiq::Web => '/sidekiq'
    end
  end
  
  # ═══════════════════════════════════════════════════════════════════════════════
  #                           3. MODELS AND DATABASE
  # ═══════════════════════════════════════════════════════════════════════════════
  
  # app/models/application_record.rb
  class ApplicationRecord < ActiveRecord::Base
    primary_abstract_class
    
    # Global scopes
    scope :recent, -> { order(created_at: :desc) }
    scope :oldest, -> { order(created_at: :asc) }
    
    # Pagination
    paginates_per 20
  end
  
  # app/models/user.rb
  class User < ApplicationRecord
    # Devise modules
    devise :database_authenticatable, :registerable,
           :recoverable, :rememberable, :validatable,
           :confirmable, :lockable, :trackable
    
    # Associations
    has_many :posts, dependent: :destroy
    has_many :comments, dependent: :destroy
    has_many :authored_posts, -> { published }, class_name: 'Post', foreign_key: 'user_id'
    
    # File attachments
    has_one_attached :avatar
    
    # Validations
    validates :first_name, :last_name, presence: true
    validates :username, presence: true, uniqueness: { case_sensitive: false }
    validates :email, presence: true, uniqueness: { case_sensitive: false }
    
    # Callbacks
    before_save :normalize_username
    after_create :send_welcome_email
    
    # Enums
    enum role: { user: 0, moderator: 1, admin: 2 }
    enum status: { inactive: 0, active: 1, suspended: 2 }
    
    # Scopes
    scope :active, -> { where(status: :active) }
    scope :with_avatar, -> { joins(:avatar_attachment) }
    scope :search, ->(query) { where("first_name ILIKE ? OR last_name ILIKE ? OR email ILIKE ?", "%#{query}%", "%#{query}%", "%#{query}%") }
    
    # Instance methods
    def full_name
      "#{first_name} #{last_name}".strip
    end
    
    def display_name
      username.presence || full_name
    end
    
    def admin?
      role == 'admin'
    end
    
    def can_moderate?
      admin? || moderator?
    end
    
    def avatar_url(size: :medium)
      return unless avatar.attached?
      
      case size
      when :small
        Rails.application.routes.url_helpers.rails_representation_url(avatar.variant(resize_to_limit: [50, 50]))
      when :medium
        Rails.application.routes.url_helpers.rails_representation_url(avatar.variant(resize_to_limit: [150, 150]))
      when :large
        Rails.application.routes.url_helpers.rails_representation_url(avatar.variant(resize_to_limit: [300, 300]))
      else
        Rails.application.routes.url_helpers.rails_blob_url(avatar)
      end
    end
    
    private
    
    def normalize_username
      self.username = username.downcase.strip if username.present?
    end
    
    def send_welcome_email
      UserMailer.welcome(self).deliver_later
    end
  end
  
  # app/models/post.rb
  class Post < ApplicationRecord
    include PgSearch::Model
    
    # Associations
    belongs_to :user
    belongs_to :category
    has_many :comments, dependent: :destroy
    has_many :post_tags, dependent: :destroy
    has_many :tags, through: :post_tags
    
    # File attachments
    has_one_attached :featured_image
    has_rich_text :content
    
    # Validations
    validates :title, presence: true, length: { maximum: 255 }
    validates :content, presence: true
    validates :slug, presence: true, uniqueness: true
    validates :excerpt, length: { maximum: 500 }
    
    # Callbacks
    before_validation :generate_slug
    before_save :generate_excerpt
    after_create :notify_subscribers
    
    # Enums
    enum status: { draft: 0, published: 1, archived: 2 }
    enum visibility: { public: 0, private: 1, protected: 2 }
    
    # Scopes
    scope :published, -> { where(status: :published) }
    scope :drafts, -> { where(status: :draft) }
    scope :featured, -> { where(featured: true) }
    scope :by_category, ->(category) { joins(:category).where(categories: { slug: category }) }
    scope :recent, -> { order(published_at: :desc) }
    scope :popular, -> { order(views_count: :desc) }
    
    # Search configuration
    pg_search_scope :search_full_text,
      against: [:title, :excerpt],
      associated_against: {
        user: [:first_name, :last_name],
        category: [:name]
      },
      using: {
        tsearch: { prefix: true, dictionary: "english" }
      }
    
    # Class methods
    def self.trending(limit: 10)
      published
        .where('published_at > ?', 7.days.ago)
        .order(views_count: :desc)
        .limit(limit)
    end
    
    def self.by_tag(tag_name)
      joins(:tags).where(tags: { name: tag_name })
    end
    
    # Instance methods
    def published?
      status == 'published' && published_at.present?
    end
    
    def can_be_published?
      draft? && title.present? && content.present?
    end
    
    def publish!
      update!(status: :published, published_at: Time.current) if can_be_published?
    end
    
    def unpublish!
      update!(status: :draft, published_at: nil)
    end
    
    def reading_time
      word_count = content.to_plain_text.split.size
      (word_count / 200.0).ceil # Assuming 200 words per minute
    end
    
    def previous_post
      self.class.published
          .where('published_at < ?', published_at)
          .order(published_at: :desc)
          .first
    end
    
    def next_post
      self.class.published
          .where('published_at > ?', published_at)
          .order(published_at: :asc)
          .first
    end
    
    def increment_views!
      increment!(:views_count)
    end
    
    def featured_image_url(size: :medium)
      return unless featured_image.attached?
      
      case size
      when :thumbnail
        Rails.application.routes.url_helpers.rails_representation_url(featured_image.variant(resize_to_limit: [300, 200]))
      when :medium
        Rails.application.routes.url_helpers.rails_representation_url(featured_image.variant(resize_to_limit: [600, 400]))
      when :large
        Rails.application.routes.url_helpers.rails_representation_url(featured_image.variant(resize_to_limit: [1200, 800]))
      else
        Rails.application.routes.url_helpers.rails_blob_url(featured_image)
      end
    end
    
    private
    
    def generate_slug
      return if slug.present? && !title_changed?
      
      base_slug = title.to_s.parameterize
      unique_slug = base_slug
      counter = 1
      
      while self.class.exists?(slug: unique_slug)
        unique_slug = "#{base_slug}-#{counter}"
        counter += 1
      end
      
      self.slug = unique_slug
    end
    
    def generate_excerpt
      return if excerpt.present?
      
      plain_content = content.to_plain_text
      self.excerpt = plain_content.truncate(200)
    end
    
    def notify_subscribers
      NotifySubscribersJob.perform_later(self) if published?
    end
  end
  
  # app/models/category.rb
  class Category < ApplicationRecord
    # Associations
    has_many :posts, dependent: :destroy
    has_one_attached :image
    
    # Validations
    validates :name, presence: true, uniqueness: { case_sensitive: false }
    validates :slug, presence: true, uniqueness: { case_sensitive: false }
    validates :description, length: { maximum: 500 }
    
    # Callbacks
    before_validation :generate_slug
    
    # Scopes
    scope :active, -> { where(active: true) }
    scope :with_posts, -> { joins(:posts).distinct }
    scope :ordered, -> { order(:name) }
    
    # Instance methods
    def posts_count
      posts.published.count
    end
    
    def latest_post
      posts.published.recent.first
    end
    
    private
    
    def generate_slug
      self.slug = name.to_s.parameterize if name.present? && (slug.blank? || name_changed?)
    end
  end
  
  # app/models/comment.rb
  class Comment < ApplicationRecord
    # Associations
    belongs_to :post
    belongs_to :user
    belongs_to :parent, class_name: 'Comment', optional: true
    has_many :replies, class_name: 'Comment', foreign_key: 'parent_id', dependent: :destroy
    
    # Validations
    validates :content, presence: true, length: { minimum: 3, maximum: 1000 }
    
    # Callbacks
    after_create :notify_post_author
    
    # Enums
    enum status: { pending: 0, approved: 1, rejected: 2 }
    
    # Scopes
    scope :approved, -> { where(status: :approved) }
    scope :top_level, -> { where(parent_id: nil) }
    scope :recent, -> { order(created_at: :desc) }
    
    # Instance methods
    def reply?
      parent_id.present?
    end
    
    def top_level?
      parent_id.nil?
    end
    
    private
    
    def notify_post_author
      CommentNotificationJob.perform_later(self) unless user == post.user
    end
  end
  
  # app/models/tag.rb
  class Tag < ApplicationRecord
    # Associations
    has_many :post_tags, dependent: :destroy
    has_many :posts, through: :post_tags
    
    # Validations
    validates :name, presence: true, uniqueness: { case_sensitive: false }
    validates :slug, presence: true, uniqueness: { case_sensitive: false }
    
    # Callbacks
    before_validation :generate_slug
    
    # Scopes
    scope :popular, -> { joins(:posts).group('tags.id').order('COUNT(posts.id) DESC') }
    scope :ordered, -> { order(:name) }
    
    # Instance methods
    def posts_count
      posts.published.count
    end
    
    private
    
    def generate_slug
      self.slug = name.to_s.parameterize if name.present? && (slug.blank? || name_changed?)
    end
  end
  
  # Database Migrations
  =begin
  # Create Users table
  rails generate devise User first_name:string last_name:string username:string role:integer status:integer
  
  # Add additional fields to users
  class AddFieldsToUsers < ActiveRecord::Migration[7.1]
    def change
      add_column :users, :first_name, :string, null: false
      add_column :users, :last_name, :string, null: false
      add_column :users, :username, :string, null: false
      add_column :users, :role, :integer, default: 0
      add_column :users, :status, :integer, default: 1
      add_column :users, :bio, :text
      add_column :users, :website, :string
      add_column :users, :location, :string
      
      add_index :users, :username, unique: true
      add_index :users, :role
      add_index :users, :status
    end
  end
  
  # Create Categories table
  class CreateCategories < ActiveRecord::Migration[7.1]
    def change
      create_table :categories do |t|
        t.string :name, null: false
        t.string :slug, null: false
        t.text :description
        t.string :color, default: '#3B82F6'
        t.boolean :active, default: true
        t.integer :posts_count, default: 0
        
        t.timestamps
      end
      
      add_index :categories, :slug, unique: true
      add_index :categories, :active
    end
  end
  
  # Create Posts table
  class CreatePosts < ActiveRecord::Migration[7.1]
    def change
      create_table :posts do |t|
        t.string :title, null: false
        t.string :slug, null: false
        t.text :excerpt
        t.integer :status, default: 0
        t.integer :visibility, default: 0
        t.boolean :featured, default: false
        t.integer :views_count, default: 0
        t.datetime :published_at
        t.references :user, null: false, foreign_key: true
        t.references :category, null: false, foreign_key: true
        
        t.timestamps
      end
      
      add_index :posts, :slug, unique: true
      add_index :posts, :status
      add_index :posts, :featured
      add_index :posts, :published_at
      add_index :posts, [:user_id, :status]
    end
  end
  
  # Create Comments table
  class CreateComments < ActiveRecord::Migration[7.1]
    def change
      create_table :comments do |t|
        t.text :content, null: false
        t.integer :status, default: 0
        t.references :post, null: false, foreign_key: true
        t.references :user, null: false, foreign_key: true
        t.references :parent, foreign_key: { to_table: :comments }, null: true
        
        t.timestamps
      end
      
      add_index :comments, :status
      add_index :comments, [:post_id, :status]
    end
  end
  
  # Create Tags and PostTags tables
  class CreateTags < ActiveRecord::Migration[7.1]
    def change
      create_table :tags do |t|
        t.string :name, null: false
        t.string :slug, null: false
        t.text :description
        t.integer :posts_count, default: 0
        
        t.timestamps
      end
      
      add_index :tags, :slug, unique: true
    end
  end
  
  class CreatePostTags < ActiveRecord::Migration[7.1]
    def change
      create_table :post_tags do |t|
        t.references :post, null: false, foreign_key: true
        t.references :tag, null: false, foreign_key: true
        
        t.timestamps
      end
      
      add_index :post_tags, [:post_id, :tag_id], unique: true
    end
  end
  =end
  
  # ═══════════════════════════════════════════════════════════════════════════════
  #                           4. CONTROLLERS AND ROUTING
  # ═══════════════════════════════════════════════════════════════════════════════
  
  # app/controllers/application_controller.rb
  class ApplicationController < ActionController::Base
    protect_from_forgery with: :exception
    
    # Authentication
    before_action :authenticate_user!, except: [:index, :show]
    before_action :configure_permitted_parameters, if: :devise_controller?
    
    # Authorization
    include Pundit::Authorization
    after_action :verify_authorized, except: [:index, :show], unless: :skip_authorization?
    after_action :verify_policy_scoped, only: :index
    
    # Error handling
    rescue_from Pundit::NotAuthorizedError, with: :user_not_authorized
    rescue_from ActiveRecord::RecordNotFound, with: :not_found
    
    protected
    
    def configure_permitted_parameters
      devise_parameter_sanitizer.permit(:sign_up, keys: [:first_name, :last_name, :username])
      devise_parameter_sanitizer.permit(:account_update, keys: [:first_name, :last_name, :username, :bio, :website, :location, :avatar])
    end
    
    def skip_authorization?
      devise_controller? || params[:controller] =~ /(^(rails_)?admin)|(^pages$)/
    end
    
    private
    
    def user_not_authorized
      flash[:alert] = "You are not authorized to perform this action."
      redirect_back(fallback_location: root_path)
    end
    
    def not_found
      render file: "#{Rails.root}/public/404", layout: false, status: :not_found
    end
    
    def set_current_user
      Current.user = current_user
    end
  end
  
  # app/controllers/home_controller.rb
  class HomeController < ApplicationController
    def index
      @featured_posts = Post.published.featured.recent.limit(3)
      @recent_posts = Post.published.recent.limit(6)
      @categories = Category.active.with_posts.ordered.limit(8)
      @popular_tags = Tag.popular.limit(10)
    end
  end
  
  # app/controllers/posts_controller.rb
  class PostsController < ApplicationController
    before_action :set_post, only: [:show, :edit, :update, :destroy, :publish, :unpublish, :toggle_featured]
    before_action :authenticate_user!, except: [:index, :show]
    
    def index
      @posts = policy_scope(Post).includes(:user, :category, :tags)
      @posts = @posts.published if params[:status] != 'all'
      @posts = @posts.by_category(params[:category]) if params[:category].present?
      @posts = @posts.by_tag(params[:tag]) if params[:tag].present?
      @posts = @posts.search_full_text(params[:search]) if params[:search].present?
      @posts = @posts.page(params[:page]).per(12)
      
      @categories = Category.active.ordered
      @tags = Tag.popular.limit(20)
    end
    
    def show
      authorize @post
      @post.increment_views!
      @comments = @post.comments.approved.top_level.includes(:user, :replies).recent
      @comment = Comment.new
      @related_posts = Post.published
                           .where.not(id: @post.id)
                           .where(category: @post.category)
                           .recent
                           .limit(3)
    end
    
    def new
      @post = current_user.posts.build
      authorize @post
    end
    
    def create
      @post = current_user.posts.build(post_params)
      authorize @post
      
      if @post.save
        redirect_to @post, notice: 'Post was successfully created.'
      else
        render :new, status: :unprocessable_entity
      end
    end
    
    def edit
      authorize @post
    end
    
    def update
      authorize @post
      
      if @post.update(post_params)
        redirect_to @post, notice: 'Post was successfully updated.'
      else
        render :edit, status: :unprocessable_entity
      end
    end
    
    def destroy
      authorize @post
      @post.destroy
      redirect_to posts_path, notice: 'Post was successfully deleted.'
    end
    
    def publish
      authorize @post
      
      if @post.publish!
        redirect_to @post, notice: 'Post was successfully published.'
      else
        redirect_to @post, alert: 'Unable to publish post.'
      end
    end
    
    def unpublish
      authorize @post
      @post.unpublish!
      redirect_to @post, notice: 'Post was unpublished.'
    end
    
    def toggle_featured
      authorize @post
      @post.update!(featured: !@post.featured?)
      redirect_to @post, notice: "Post #{@post.featured? ? 'featured' : 'unfeatured'} successfully."
    end
    
    def published
      @posts = policy_scope(Post).published.includes(:user, :category).recent.page(params[:page])
    end
    
    def drafts
      @posts = policy_scope(Post).drafts.includes(:user, :category).recent.page(params[:page])
    end
    
    private
    
    def set_post
      @post = Post.friendly.find(params[:id])
    end
    
    def post_params
      params.require(:post).permit(:title, :content, :excerpt, :category_id, :featured_image, :visibility, tag_ids: [])
    end
  end
  
  # app/controllers/comments_controller.rb
  class CommentsController < ApplicationController
    before_action :set_post
    before_action :set_comment, only: [:show, :edit, :update, :destroy]
    before_action :authenticate_user!
    
    def create
      @comment = @post.comments.build(comment_params)
      @comment.user = current_user
      authorize @comment
      
      if @comment.save
        redirect_to @post, notice: 'Comment was successfully created.'
      else
        redirect_to @post, alert: 'Unable to create comment.'
      end
    end
    
    def edit
      authorize @comment
    end
    
    def update
      authorize @comment
      
      if @comment.update(comment_params)
        redirect_to @post, notice: 'Comment was successfully updated.'
      else
        render :edit, status: :unprocessable_entity
      end
    end
    
    def destroy
      authorize @comment
      @comment.destroy
      redirect_to @post, notice: 'Comment was successfully deleted.'
    end
    
    private
    
    def set_post
      @post = Post.friendly.find(params[:post_id])
    end
    
    def set_comment
      @comment = @post.comments.find(params[:id])
    end
    
    def comment_params
      params.require(:comment).permit(:content, :parent_id)
    end
  end