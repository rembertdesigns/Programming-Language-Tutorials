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


# ═══════════════════════════════════════════════════════════════════════════════
#                           5. API CONTROLLERS
# ═══════════════════════════════════════════════════════════════════════════════

# app/controllers/api/v1/base_controller.rb
class Api::V1::BaseController < ActionController::API
    include Pundit::Authorization
    
    before_action :authenticate_api_user!
    
    # Error handling
    rescue_from ActiveRecord::RecordNotFound, with: :not_found
    rescue_from Pundit::NotAuthorizedError, with: :unauthorized
    rescue_from ActionController::ParameterMissing, with: :bad_request
    
    protected
    
    def authenticate_api_user!
      token = request.headers['Authorization']&.split(' ')&.last
      return render_unauthorized unless token
      
      begin
        decoded_token = JWT.decode(token, Rails.application.secrets.secret_key_base, true, algorithm: 'HS256')
        user_id = decoded_token[0]['user_id']
        @current_user = User.find(user_id)
      rescue JWT::DecodeError, ActiveRecord::RecordNotFound
        render_unauthorized
      end
    end
    
    def current_user
      @current_user
    end
    
    private
    
    def render_error(message, status = :unprocessable_entity)
      render json: { error: message }, status: status
    end
    
    def render_unauthorized
      render json: { error: 'Unauthorized' }, status: :unauthorized
    end
    
    def not_found
      render json: { error: 'Record not found' }, status: :not_found
    end
    
    def unauthorized
      render json: { error: 'Access denied' }, status: :forbidden
    end
    
    def bad_request
      render json: { error: 'Bad request' }, status: :bad_request
    end
  end
  
  # app/controllers/api/v1/authentication_controller.rb
  class Api::V1::AuthenticationController < Api::V1::BaseController
    skip_before_action :authenticate_api_user!, only: [:login, :refresh]
    
    def login
      user = User.find_by(email: params[:email])
      
      if user&.valid_password?(params[:password])
        token = generate_jwt_token(user)
        refresh_token = generate_refresh_token(user)
        
        render json: {
          user: UserSerializer.new(user).serializable_hash,
          token: token,
          refresh_token: refresh_token,
          expires_at: 24.hours.from_now
        }
      else
        render json: { error: 'Invalid credentials' }, status: :unauthorized
      end
    end
    
    def logout
      # In a real app, you'd blacklist the token
      render json: { message: 'Logged out successfully' }
    end
    
    def refresh
      refresh_token = params[:refresh_token]
      return render_unauthorized unless refresh_token
      
      begin
        decoded_token = JWT.decode(refresh_token, Rails.application.secrets.secret_key_base, true, algorithm: 'HS256')
        user = User.find(decoded_token[0]['user_id'])
        new_token = generate_jwt_token(user)
        
        render json: {
          token: new_token,
          expires_at: 24.hours.from_now
        }
      rescue JWT::DecodeError, ActiveRecord::RecordNotFound
        render_unauthorized
      end
    end
    
    private
    
    def generate_jwt_token(user)
      payload = {
        user_id: user.id,
        exp: 24.hours.from_now.to_i
      }
      JWT.encode(payload, Rails.application.secrets.secret_key_base, 'HS256')
    end
    
    def generate_refresh_token(user)
      payload = {
        user_id: user.id,
        exp: 7.days.from_now.to_i,
        type: 'refresh'
      }
      JWT.encode(payload, Rails.application.secrets.secret_key_base, 'HS256')
    end
  end
  
  # app/controllers/api/v1/posts_controller.rb
  class Api::V1::PostsController < Api::V1::BaseController
    before_action :set_post, only: [:show, :update, :destroy]
    skip_before_action :authenticate_api_user!, only: [:index, :show]
    
    def index
      @posts = Post.published.includes(:user, :category, :tags)
      @posts = @posts.by_category(params[:category]) if params[:category].present?
      @posts = @posts.search_full_text(params[:search]) if params[:search].present?
      @posts = @posts.page(params[:page]).per(params[:per_page] || 20)
      
      render json: {
        posts: PostSerializer.new(@posts).serializable_hash,
        meta: pagination_meta(@posts)
      }
    end
    
    def show
      render json: PostSerializer.new(@post).serializable_hash
    end
    
    def create
      @post = current_user.posts.build(post_params)
      authorize @post
      
      if @post.save
        render json: PostSerializer.new(@post).serializable_hash, status: :created
      else
        render json: { errors: @post.errors }, status: :unprocessable_entity
      end
    end
    
    def update
      authorize @post
      
      if @post.update(post_params)
        render json: PostSerializer.new(@post).serializable_hash
      else
        render json: { errors: @post.errors }, status: :unprocessable_entity
      end
    end
    
    def destroy
      authorize @post
      @post.destroy
      head :no_content
    end
    
    private
    
    def set_post
      @post = Post.friendly.find(params[:id])
    end
    
    def post_params
      params.require(:post).permit(:title, :content, :excerpt, :category_id, :status, :visibility, tag_ids: [])
    end
    
    def pagination_meta(collection)
      {
        current_page: collection.current_page,
        next_page: collection.next_page,
        prev_page: collection.prev_page,
        total_pages: collection.total_pages,
        total_count: collection.total_count
      }
    end
  end
  
  # app/controllers/api/v1/comments_controller.rb
  class Api::V1::CommentsController < Api::V1::BaseController
    before_action :set_post
    before_action :set_comment, only: [:update, :destroy]
    
    def index
      @comments = @post.comments.approved.includes(:user, :replies)
      render json: CommentSerializer.new(@comments).serializable_hash
    end
    
    def create
      @comment = @post.comments.build(comment_params)
      @comment.user = current_user
      authorize @comment
      
      if @comment.save
        render json: CommentSerializer.new(@comment).serializable_hash, status: :created
      else
        render json: { errors: @comment.errors }, status: :unprocessable_entity
      end
    end
    
    def update
      authorize @comment
      
      if @comment.update(comment_params)
        render json: CommentSerializer.new(@comment).serializable_hash
      else
        render json: { errors: @comment.errors }, status: :unprocessable_entity
      end
    end
    
    def destroy
      authorize @comment
      @comment.destroy
      head :no_content
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
  
  # ═══════════════════════════════════════════════════════════════════════════════
  #                           6. SERIALIZERS AND JSON API
  # ═══════════════════════════════════════════════════════════════════════════════
  
  # app/serializers/application_serializer.rb
  class ApplicationSerializer
    include FastJsonapi::ObjectSerializer
  end
  
  # app/serializers/user_serializer.rb
  class UserSerializer < ApplicationSerializer
    attributes :id, :email, :first_name, :last_name, :username, :bio, :website, :location, :created_at
    
    attribute :full_name do |user|
      user.full_name
    end
    
    attribute :avatar_url do |user|
      user.avatar_url if user.avatar.attached?
    end
    
    attribute :posts_count do |user|
      user.posts.published.count
    end
  end
  
  # app/serializers/post_serializer.rb
  class PostSerializer < ApplicationSerializer
    attributes :id, :title, :slug, :excerpt, :content, :status, :featured, :views_count, :published_at, :created_at, :updated_at
    
    belongs_to :user
    belongs_to :category
    has_many :tags
    has_many :comments
    
    attribute :reading_time do |post|
      post.reading_time
    end
    
    attribute :featured_image_url do |post|
      post.featured_image_url if post.featured_image.attached?
    end
    
    attribute :author do |post|
      {
        id: post.user.id,
        name: post.user.full_name,
        username: post.user.username,
        avatar_url: post.user.avatar_url
      }
    end
  end
  
  # app/serializers/category_serializer.rb
  class CategorySerializer < ApplicationSerializer
    attributes :id, :name, :slug, :description, :color, :posts_count
    
    has_many :posts
  end
  
  # app/serializers/comment_serializer.rb
  class CommentSerializer < ApplicationSerializer
    attributes :id, :content, :status, :created_at, :updated_at
    
    belongs_to :user
    belongs_to :post
    belongs_to :parent, optional: true
    has_many :replies
    
    attribute :author do |comment|
      {
        id: comment.user.id,
        name: comment.user.full_name,
        username: comment.user.username,
        avatar_url: comment.user.avatar_url
      }
    end
  end
  
  # ═══════════════════════════════════════════════════════════════════════════════
  #                           7. VIEWS AND FRONTEND
  # ═══════════════════════════════════════════════════════════════════════════════
  
  # app/views/layouts/application.html.erb
  =begin
  <!DOCTYPE html>
  <html>
    <head>
      <title>My Web App</title>
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <%= csrf_meta_tags %>
      <%= csp_meta_tag %>
      
      <%= stylesheet_link_tag "application", "data-turbo-track": "reload" %>
      <%= javascript_pack_tag "application", "data-turbo-track": "reload" %>
      
      <!-- SEO Meta Tags -->
      <meta name="description" content="<%= content_for?(:description) ? yield(:description) : 'My awesome web application' %>">
      <meta name="keywords" content="<%= content_for?(:keywords) ? yield(:keywords) : 'rails, web, app' %>">
      
      <!-- Open Graph -->
      <meta property="og:title" content="<%= content_for?(:title) ? yield(:title) : 'My Web App' %>">
      <meta property="og:description" content="<%= content_for?(:description) ? yield(:description) : 'My awesome web application' %>">
      <meta property="og:image" content="<%= content_for?(:og_image) ? yield(:og_image) : asset_url('og-image.png') %>">
      <meta property="og:url" content="<%= request.original_url %>">
      
      <!-- Favicons -->
      <%= favicon_link_tag asset_path('favicon.ico') %>
    </head>
  
    <body class="<%= yield(:body_class) %>">
      <!-- Navigation -->
      <nav class="bg-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4">
          <div class="flex justify-between h-16">
            <div class="flex items-center">
              <%= link_to root_path, class: "flex-shrink-0 flex items-center" do %>
                <span class="font-bold text-xl text-gray-800">My Web App</span>
              <% end %>
              
              <div class="hidden md:ml-6 md:flex md:space-x-8">
                <%= link_to "Home", root_path, class: "text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium" %>
                <%= link_to "Posts", posts_path, class: "text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium" %>
                <%= link_to "Categories", categories_path, class: "text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium" %>
                <%= link_to "About", about_path, class: "text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium" %>
              </div>
            </div>
            
            <div class="flex items-center space-x-4">
              <% if user_signed_in? %>
                <div class="relative" data-controller="dropdown">
                  <button data-dropdown-target="trigger" class="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    <% if current_user.avatar.attached? %>
                      <%= image_tag current_user.avatar_url(:small), class: "h-8 w-8 rounded-full" %>
                    <% else %>
                      <div class="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center">
                        <span class="text-sm font-medium text-gray-700"><%= current_user.first_name.first %></span>
                      </div>
                    <% end %>
                  </button>
                  
                  <div data-dropdown-target="menu" class="hidden absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50">
                    <%= link_to "Profile", user_path(current_user), class: "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" %>
                    <%= link_to "My Posts", posts_path(user: current_user), class: "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" %>
                    <%= link_to "Settings", edit_user_registration_path, class: "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" %>
                    <div class="border-t border-gray-100"></div>
                    <%= link_to "Sign out", destroy_user_session_path, method: :delete, class: "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100" %>
                  </div>
                </div>
              <% else %>
                <%= link_to "Sign in", new_user_session_path, class: "text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium" %>
                <%= link_to "Sign up", new_user_registration_path, class: "bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md text-sm font-medium" %>
              <% end %>
            </div>
          </div>
        </div>
      </nav>
  
      <!-- Flash Messages -->
      <% if notice %>
        <div class="bg-green-50 border border-green-200 text-green-800 px-4 py-3" role="alert">
          <%= notice %>
        </div>
      <% end %>
      
      <% if alert %>
        <div class="bg-red-50 border border-red-200 text-red-800 px-4 py-3" role="alert">
          <%= alert %>
        </div>
      <% end %>
  
      <!-- Main Content -->
      <main class="min-h-screen bg-gray-50">
        <%= yield %>
      </main>
  
      <!-- Footer -->
      <footer class="bg-gray-800 text-white">
        <div class="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
          <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 class="text-lg font-semibold mb-4">My Web App</h3>
              <p class="text-gray-300">Building amazing web applications with Ruby on Rails.</p>
            </div>
            <div>
              <h3 class="text-lg font-semibold mb-4">Quick Links</h3>
              <ul class="space-y-2">
                <li><%= link_to "About", about_path, class: "text-gray-300 hover:text-white" %></li>
                <li><%= link_to "Contact", contact_path, class: "text-gray-300 hover:text-white" %></li>
                <li><%= link_to "Privacy", privacy_path, class: "text-gray-300 hover:text-white" %></li>
                <li><%= link_to "Terms", terms_path, class: "text-gray-300 hover:text-white" %></li>
              </ul>
            </div>
            <div>
              <h3 class="text-lg font-semibold mb-4">Connect</h3>
              <div class="flex space-x-4">
                <a href="#" class="text-gray-300 hover:text-white">Twitter</a>
                <a href="#" class="text-gray-300 hover:text-white">GitHub</a>
                <a href="#" class="text-gray-300 hover:text-white">LinkedIn</a>
              </div>
            </div>
          </div>
          <div class="mt-8 pt-8 border-t border-gray-700 text-center">
            <p class="text-gray-300">&copy; <%= Date.current.year %> My Web App. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </body>
  </html>
  =end
  
  # app/views/home/index.html.erb
  =begin
  <% content_for :title, "Welcome to My Web App" %>
  <% content_for :description, "Discover amazing content and connect with our community" %>
  
  <!-- Hero Section -->
  <section class="bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
    <div class="max-w-7xl mx-auto px-4 py-24">
      <div class="text-center">
        <h1 class="text-4xl md:text-6xl font-bold mb-6">Welcome to My Web App</h1>
        <p class="text-xl md:text-2xl mb-8 text-indigo-100">Discover amazing content and connect with our community</p>
        <div class="space-x-4">
          <%= link_to "Get Started", posts_path, class: "bg-white text-indigo-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition duration-300" %>
          <%= link_to "Learn More", about_path, class: "border border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-indigo-600 transition duration-300" %>
        </div>
      </div>
    </div>
  </section>
  
  <!-- Featured Posts -->
  <% if @featured_posts.any? %>
  <section class="py-16">
    <div class="max-w-7xl mx-auto px-4">
      <h2 class="text-3xl font-bold text-gray-900 mb-8">Featured Posts</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
        <% @featured_posts.each do |post| %>
          <article class="bg-white rounded-lg shadow-md overflow-hidden">
            <% if post.featured_image.attached? %>
              <%= link_to post_path(post) do %>
                <%= image_tag post.featured_image_url(:medium), class: "w-full h-48 object-cover" %>
              <% end %>
            <% end %>
            <div class="p-6">
              <div class="flex items-center mb-2">
                <span class="bg-<%= post.category.color %> text-white px-2 py-1 rounded text-sm"><%= post.category.name %></span>
                <span class="text-gray-500 text-sm ml-2"><%= time_ago_in_words(post.published_at) %> ago</span>
              </div>
              <h3 class="text-xl font-semibold mb-2">
                <%= link_to post.title, post_path(post), class: "text-gray-900 hover:text-indigo-600" %>
              </h3>
              <p class="text-gray-600 mb-4"><%= post.excerpt %></p>
              <div class="flex items-center">
                <% if post.user.avatar.attached? %>
                  <%= image_tag post.user.avatar_url(:small), class: "w-8 h-8 rounded-full mr-3" %>
                <% end %>
                <span class="text-sm text-gray-700"><%= post.user.full_name %></span>
              </div>
            </div>
          </article>
        <% end %>
      </div>
    </div>
  </section>
  <% end %>
  
  <!-- Recent Posts -->
  <section class="py-16 bg-gray-50">
    <div class="max-w-7xl mx-auto px-4">
      <div class="flex justify-between items-center mb-8">
        <h2 class="text-3xl font-bold text-gray-900">Recent Posts</h2>
        <%= link_to "View All Posts", posts_path, class: "text-indigo-600 hover:text-indigo-800 font-medium" %>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        <% @recent_posts.each do |post| %>
          <article class="bg-white rounded-lg shadow-md overflow-hidden">
            <% if post.featured_image.attached? %>
              <%= link_to post_path(post) do %>
                <%= image_tag post.featured_image_url(:thumbnail), class: "w-full h-32 object-cover" %>
              <% end %>
            <% end %>
            <div class="p-4">
              <div class="flex items-center mb-2">
                <span class="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs"><%= post.category.name %></span>
                <span class="text-gray-500 text-xs ml-2"><%= post.reading_time %> min read</span>
              </div>
              <h3 class="text-lg font-semibold mb-2">
                <%= link_to post.title, post_path(post), class: "text-gray-900 hover:text-indigo-600" %>
              </h3>
              <p class="text-gray-600 text-sm"><%= truncate(post.excerpt, length: 100) %></p>
            </div>
          </article>
        <% end %>
      </div>
    </div>
  </section>
  
  <!-- Categories -->
  <section class="py-16">
    <div class="max-w-7xl mx-auto px-4">
      <h2 class="text-3xl font-bold text-gray-900 mb-8">Explore Categories</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <% @categories.each do |category| %>
          <%= link_to category_path(category), class: "block" do %>
            <div class="bg-white p-6 rounded-lg shadow-md text-center hover:shadow-lg transition duration-300">
              <h3 class="font-semibold text-gray-900 mb-2"><%= category.name %></h3>
              <p class="text-gray-600 text-sm"><%= pluralize(category.posts_count, 'post') %></p>
            </div>
          <% end %>
        <% end %>
      </div>
    </div>
  </section>
  =end
  
  # ═══════════════════════════════════════════════════════════════════════════════
  #                           8. AUTHORIZATION WITH PUNDIT
  # ═══════════════════════════════════════════════════════════════════════════════
  
  # app/policies/application_policy.rb
  class ApplicationPolicy
    attr_reader :user, :record
  
    def initialize(user, record)
      @user = user
      @record = record
    end
  
    def index?
      true
    end
  
    def show?
      true
    end
  
    def create?
      user.present?
    end
  
    def new?
      create?
    end
  
    def update?
      user.present? && (user == record.user || user.admin?)
    end
  
    def edit?
      update?
    end
  
    def destroy?
      user.present? && (user == record.user || user.admin?)
    end
  
    class Scope
      def initialize(user, scope)
        @user = user
        @scope = scope
      end
  
      def resolve
        raise NotImplementedError, "You must define #resolve in #{self.class}"
      end
  
      private
  
      attr_reader :user, :scope
    end
  end
  
  # app/policies/post_policy.rb
  class PostPolicy < ApplicationPolicy
    def index?
      true
    end
  
    def show?
      record.published? || user == record.user || user&.can_moderate?
    end
  
    def create?
      user.present?
    end
  
    def update?
      user == record.user || user&.can_moderate?
    end
  
    def destroy?
      user == record.user || user&.admin?
    end
  
    def publish?
      user == record.user || user&.can_moderate?
    end
  
    def unpublish?
      publish?
    end
  
    def toggle_featured?
      user&.can_moderate?
    end
  
    class Scope < Scope
      def resolve
        if user&.admin?
          scope.all
        elsif user&.moderator?
          scope.where("status = ? OR user_id = ?", Post.statuses[:published], user.id)
        elsif user.present?
          scope.where("status = ? OR user_id = ?", Post.statuses[:published], user.id)
        else
          scope.published
        end
      end
    end
  end
  
  # app/policies/comment_policy.rb
  class CommentPolicy < ApplicationPolicy
    def create?
      user.present?
    end
  
    def update?
      user == record.user || user&.can_moderate?
    end
  
    def destroy?
      user == record.user || user&.can_moderate?
    end
  
    def approve?
      user&.can_moderate?
    end
  
    def reject?
      user&.can_moderate?
    end
  end
  
  # app/policies/user_policy.rb
  class UserPolicy < ApplicationPolicy
    def index?
      user&.can_moderate?
    end
  
    def show?
      true
    end
  
    def update?
      user == record || user&.admin?
    end
  
    def destroy?
      user&.admin? && user != record
    end
  
    def activate?
      user&.admin?
    end
  
    def deactivate?
      user&.admin? && user != record
    end
  end


  # ═══════════════════════════════════════════════════════════════════════════════
#                           9. BACKGROUND JOBS AND MAILERS
# ═══════════════════════════════════════════════════════════════════════════════

# app/jobs/application_job.rb
class ApplicationJob < ActiveJob::Base
    # Automatically retry jobs that encountered a deadlock
    retry_on ActiveRecord::Deadlocked
  
    # Most jobs are safe to ignore if the underlying records are no longer available
    discard_on ActiveJob::DeserializationError
  end
  
  # app/jobs/notify_subscribers_job.rb
  class NotifySubscribersJob < ApplicationJob
    queue_as :default
  
    def perform(post)
      # Get all subscribers (this would be a real model in your app)
      subscribers = User.where(subscribed: true)
      
      subscribers.find_each do |subscriber|
        PostMailer.new_post_notification(subscriber, post).deliver_now
      end
    end
  end
  
  # app/jobs/comment_notification_job.rb
  class CommentNotificationJob < ApplicationJob
    queue_as :default
  
    def perform(comment)
      return unless comment.post.user.email_notifications?
      
      CommentMailer.new_comment_notification(comment).deliver_now
    end
  end
  
  # app/jobs/cleanup_old_posts_job.rb
  class CleanupOldPostsJob < ApplicationJob
    queue_as :low_priority
  
    def perform
      # Archive posts older than 2 years
      Post.where('created_at < ?', 2.years.ago)
          .where(status: :published)
          .update_all(status: :archived)
    end
  end
  
  # app/mailers/application_mailer.rb
  class ApplicationMailer < ActionMailer::Base
    default from: 'noreply@mywebapp.com'
    layout 'mailer'
    
    private
    
    def mail_with_name(to_user, subject)
      mail(
        to: "#{to_user.full_name} <#{to_user.email}>",
        subject: subject
      )
    end
  end
  
  # app/mailers/user_mailer.rb
  class UserMailer < ApplicationMailer
    def welcome(user)
      @user = user
      @login_url = new_user_session_url
      
      mail_with_name(@user, 'Welcome to My Web App!')
    end
    
    def password_reset(user)
      @user = user
      @reset_url = edit_user_password_url(@user, reset_password_token: @user.reset_password_token)
      
      mail_with_name(@user, 'Password Reset Instructions')
    end
  end
  
  # app/mailers/post_mailer.rb
  class PostMailer < ApplicationMailer
    def new_post_notification(subscriber, post)
      @subscriber = subscriber
      @post = post
      @post_url = post_url(@post)
      @unsubscribe_url = unsubscribe_url(token: @subscriber.unsubscribe_token)
      
      mail_with_name(@subscriber, "New post: #{@post.title}")
    end
  end
  
  # app/mailers/comment_mailer.rb
  class CommentMailer < ApplicationMailer
    def new_comment_notification(comment)
      @comment = comment
      @post = comment.post
      @author = @post.user
      @post_url = post_url(@post, anchor: "comment-#{@comment.id}")
      
      mail_with_name(@author, "New comment on your post: #{@post.title}")
    end
  end
  
# ═══════════════════════════════════════════════════════════════════════════════
#                           10. TESTING WITH RSPEC
# ═══════════════════════════════════════════════════════════════════════════════

# spec/rails_helper.rb
=begin
require 'spec_helper'
ENV['RAILS_ENV'] ||= 'test'
require_relative '../config/environment'
abort("The Rails environment is running in production mode!") if Rails.env.production?
require 'rspec/rails'

# Add additional requires below this line. Rails is not loaded until this point!
require 'factory_bot_rails'
require 'faker'

# Checks for pending migrations and applies them before tests are run.
begin
  ActiveRecord::Migration.maintain_test_schema!
rescue ActiveRecord::PendingMigrationError => e
  abort e.to_s.strip
end

RSpec.configure do |config|
  # Remove this line if you're not using ActiveRecord or ActiveRecord fixtures
  config.fixture_path = "#{::Rails.root}/spec/fixtures"
  
  # If you're not using ActiveRecord, or you'd prefer not to run each of your
  # examples within a transaction, remove the following line or assign false
  # instead of true
  config.use_transactional_fixtures = true
  
  # You can uncomment this line to turn off ActiveRecord support entirely.
  # config.use_active_record = false
  
  # RSpec Rails can automatically mix in different behaviours to your tests
  # based on their file location, for example enabling you to call `get` and
  # `post` in specs under `spec/controllers`.
  config.infer_spec_type_from_file_location!
  
  # Filter lines from Rails gems in backtraces.
  config.filter_rails_from_backtrace!
  
  # Include FactoryBot methods
  config.include FactoryBot::Syntax::Methods
  
  # Include Devise test helpers
  config.include Devise::Test::ControllerHelpers, type: :controller
  config.include Devise::Test::IntegrationHelpers, type: :request
  
  # Database cleaner
  config.before(:suite) do
    DatabaseCleaner.strategy = :transaction
    DatabaseCleaner.clean_with(:truncation)
  end
  
  config.around(:each) do |example|
    DatabaseCleaner.cleaning do
      example.run
    end
  end
end
=end

# spec/factories/users.rb
FactoryBot.define do
    factory :user do
      first_name { Faker::Name.first_name }
      last_name { Faker::Name.last_name }
      username { Faker::Internet.unique.username(specifier: 5..15) }
      email { Faker::Internet.unique.email }
      password { 'password123' }
      password_confirmation { 'password123' }
      bio { Faker::Lorem.paragraph }
      website { Faker::Internet.url }
      location { Faker::Address.city }
      confirmed_at { Time.current }
      
      trait :admin do
        role { :admin }
      end
      
      trait :moderator do
        role { :moderator }
      end
      
      trait :with_avatar do
        after(:build) do |user|
          user.avatar.attach(
            io: File.open(Rails.root.join('spec', 'fixtures', 'files', 'avatar.jpg')),
            filename: 'avatar.jpg',
            content_type: 'image/jpeg'
          )
        end
      end
    end
  end
  
  # spec/factories/categories.rb
  FactoryBot.define do
    factory :category do
      name { Faker::Book.genre }
      slug { name.parameterize }
      description { Faker::Lorem.paragraph }
      color { %w[#3B82F6 #EF4444 #10B981 #F59E0B #8B5CF6].sample }
      active { true }
    end
  end
  
  # spec/factories/posts.rb
  FactoryBot.define do
    factory :post do
      title { Faker::Lorem.sentence(word_count: 4) }
      slug { title.parameterize }
      excerpt { Faker::Lorem.paragraph }
      content { Faker::Lorem.paragraphs(number: 5).join("\n\n") }
      status { :published }
      visibility { :public }
      published_at { Time.current }
      association :user
      association :category
      
      trait :draft do
        status { :draft }
        published_at { nil }
      end
      
      trait :featured do
        featured { true }
      end
      
      trait :with_image do
        after(:build) do |post|
          post.featured_image.attach(
            io: File.open(Rails.root.join('spec', 'fixtures', 'files', 'featured_image.jpg')),
            filename: 'featured_image.jpg',
            content_type: 'image/jpeg'
          )
        end
      end
    end
  end
  
  # spec/factories/comments.rb
  FactoryBot.define do
    factory :comment do
      content { Faker::Lorem.paragraph }
      status { :approved }
      association :post
      association :user
      
      trait :pending do
        status { :pending }
      end
      
      trait :reply do
        association :parent, factory: :comment
      end
    end
  end
  
  # spec/models/user_spec.rb
  require 'rails_helper'
  
  RSpec.describe User, type: :model do
    describe 'validations' do
      subject { build(:user) }
      
      it { should validate_presence_of(:first_name) }
      it { should validate_presence_of(:last_name) }
      it { should validate_presence_of(:username) }
      it { should validate_presence_of(:email) }
      it { should validate_uniqueness_of(:username).case_insensitive }
      it { should validate_uniqueness_of(:email).case_insensitive }
    end
    
    describe 'associations' do
      it { should have_many(:posts).dependent(:destroy) }
      it { should have_many(:comments).dependent(:destroy) }
      it { should have_one_attached(:avatar) }
    end
    
    describe 'enums' do
      it { should define_enum_for(:role).with_values(user: 0, moderator: 1, admin: 2) }
      it { should define_enum_for(:status).with_values(inactive: 0, active: 1, suspended: 2) }
    end
    
    describe 'scopes' do
      let!(:active_user) { create(:user, status: :active) }
      let!(:inactive_user) { create(:user, status: :inactive) }
      
      it 'returns only active users' do
        expect(User.active).to include(active_user)
        expect(User.active).not_to include(inactive_user)
      end
    end
    
    describe 'instance methods' do
      let(:user) { create(:user, first_name: 'John', last_name: 'Doe') }
      
      describe '#full_name' do
        it 'returns the full name' do
          expect(user.full_name).to eq('John Doe')
        end
      end
      
      describe '#admin?' do
        it 'returns true for admin users' do
          admin = create(:user, :admin)
          expect(admin.admin?).to be true
        end
        
        it 'returns false for non-admin users' do
          expect(user.admin?).to be false
        end
      end
      
      describe '#can_moderate?' do
        it 'returns true for admin users' do
          admin = create(:user, :admin)
          expect(admin.can_moderate?).to be true
        end
        
        it 'returns true for moderator users' do
          moderator = create(:user, :moderator)
          expect(moderator.can_moderate?).to be true
        end
        
        it 'returns false for regular users' do
          expect(user.can_moderate?).to be false
        end
      end
    end
    
    describe 'callbacks' do
      it 'normalizes username before save' do
        user = build(:user, username: 'TestUser123')
        user.save
        expect(user.username).to eq('testuser123')
      end
      
      it 'sends welcome email after create' do
        expect {
          create(:user)
        }.to have_enqueued_job(ActionMailer::MailDeliveryJob)
      end
    end
  end
  
  # spec/models/post_spec.rb
  require 'rails_helper'
  
  RSpec.describe Post, type: :model do
    describe 'validations' do
      subject { build(:post) }
      
      it { should validate_presence_of(:title) }
      it { should validate_presence_of(:content) }
      it { should validate_presence_of(:slug) }
      it { should validate_uniqueness_of(:slug) }
      it { should validate_length_of(:title).is_at_most(255) }
      it { should validate_length_of(:excerpt).is_at_most(500) }
    end
    
    describe 'associations' do
      it { should belong_to(:user) }
      it { should belong_to(:category) }
      it { should have_many(:comments).dependent(:destroy) }
      it { should have_many(:post_tags).dependent(:destroy) }
      it { should have_many(:tags).through(:post_tags) }
    end
    
    describe 'enums' do
      it { should define_enum_for(:status).with_values(draft: 0, published: 1, archived: 2) }
      it { should define_enum_for(:visibility).with_values(public: 0, private: 1, protected: 2) }
    end
    
    describe 'scopes' do
      let!(:published_post) { create(:post, status: :published) }
      let!(:draft_post) { create(:post, :draft) }
      let!(:featured_post) { create(:post, featured: true) }
      
      it 'returns only published posts' do
        expect(Post.published).to include(published_post)
        expect(Post.published).not_to include(draft_post)
      end
      
      it 'returns only draft posts' do
        expect(Post.drafts).to include(draft_post)
        expect(Post.drafts).not_to include(published_post)
      end
      
      it 'returns only featured posts' do
        expect(Post.featured).to include(featured_post)
      end
    end
    
    describe 'instance methods' do
      let(:post) { create(:post) }
      
      describe '#published?' do
        it 'returns true for published posts' do
          expect(post.published?).to be true
        end
        
        it 'returns false for draft posts' do
          draft_post = create(:post, :draft)
          expect(draft_post.published?).to be false
        end
      end
      
      describe '#reading_time' do
        it 'calculates reading time based on content' do
          # Create post with 400 words (should be 2 minutes at 200 words/minute)
          content = (1..400).map { 'word' }.join(' ')
          post = create(:post, content: content)
          expect(post.reading_time).to eq(2)
        end
      end
      
      describe '#publish!' do
        let(:draft_post) { create(:post, :draft) }
        
        it 'publishes a draft post' do
          expect {
            draft_post.publish!
          }.to change(draft_post, :status).from('draft').to('published')
            .and change(draft_post, :published_at).from(nil)
        end
      end
      
      describe '#unpublish!' do
        it 'unpublishes a published post' do
          expect {
            post.unpublish!
          }.to change(post, :status).from('published').to('draft')
            .and change(post, :published_at).to(nil)
        end
      end
    end
    
    describe 'callbacks' do
      it 'generates slug before validation' do
        post = build(:post, title: 'Test Post Title', slug: nil)
        post.valid?
        expect(post.slug).to eq('test-post-title')
      end
      
      it 'generates unique slug if title already exists' do
        create(:post, title: 'Test Title', slug: 'test-title')
        post = build(:post, title: 'Test Title')
        post.valid?
        expect(post.slug).to eq('test-title-1')
      end
      
      it 'generates excerpt if not provided' do
        content = 'This is a very long content that should be truncated' * 10
        post = build(:post, content: content, excerpt: nil)
        post.save
        expect(post.excerpt).to be_present
        expect(post.excerpt.length).to be <= 200
      end
    end
  end
  
  # spec/controllers/posts_controller_spec.rb
  require 'rails_helper'
  
  RSpec.describe PostsController, type: :controller do
    let(:user) { create(:user) }
    let(:admin) { create(:user, :admin) }
    let(:category) { create(:category) }
    let(:post_instance) { create(:post, user: user, category: category) }
    let(:draft_post) { create(:post, :draft, user: user, category: category) }
    
    describe 'GET #index' do
      before do
        create_list(:post, 3, category: category)
        create_list(:post, 2, :draft, category: category)
      end
      
      it 'returns published posts only' do
        get :index
        expect(assigns(:posts).map(&:status).uniq).to eq(['published'])
      end
      
      it 'filters by category when specified' do
        other_category = create(:category)
        other_post = create(:post, category: other_category)
        
        get :index, params: { category: category.slug }
        expect(assigns(:posts)).not_to include(other_post)
      end
      
      it 'searches posts when search term provided' do
        searchable_post = create(:post, title: 'Unique Searchable Title')
        
        get :index, params: { search: 'Unique Searchable' }
        expect(assigns(:posts)).to include(searchable_post)
      end
    end
    
    describe 'GET #show' do
      context 'when post is published' do
        it 'shows the post to anyone' do
          get :show, params: { id: post_instance.slug }
          expect(response).to have_http_status(:success)
          expect(assigns(:post)).to eq(post_instance)
        end
        
        it 'increments views count' do
          expect {
            get :show, params: { id: post_instance.slug }
          }.to change { post_instance.reload.views_count }.by(1)
        end
      end
      
      context 'when post is draft' do
        it 'shows the post to the author' do
          sign_in user
          get :show, params: { id: draft_post.slug }
          expect(response).to have_http_status(:success)
        end
        
        it 'denies access to other users' do
          other_user = create(:user)
          sign_in other_user
          expect {
            get :show, params: { id: draft_post.slug }
          }.to raise_error(Pundit::NotAuthorizedError)
        end
      end
    end
    
    describe 'POST #create' do
      let(:valid_attributes) do
        {
          title: 'New Post',
          content: 'Post content',
          category_id: category.id
        }
      end
      
      context 'when user is signed in' do
        before { sign_in user }
        
        it 'creates a new post' do
          expect {
            post :create, params: { post: valid_attributes }
          }.to change(Post, :count).by(1)
        end
        
        it 'assigns the post to current user' do
          post :create, params: { post: valid_attributes }
          expect(assigns(:post).user).to eq(user)
        end
        
        it 'redirects to the post on success' do
          post :create, params: { post: valid_attributes }
          expect(response).to redirect_to(assigns(:post))
        end
      end
      
      context 'when user is not signed in' do
        it 'redirects to sign in' do
          post :create, params: { post: valid_attributes }
          expect(response).to redirect_to(new_user_session_path)
        end
      end
    end
    
    describe 'PATCH #update' do
      context 'when user owns the post' do
        before { sign_in user }
        
        it 'updates the post' do
          patch :update, params: { 
            id: post_instance.slug, 
            post: { title: 'Updated Title' } 
          }
          expect(post_instance.reload.title).to eq('Updated Title')
        end
        
        it 'redirects to the post on success' do
          patch :update, params: { 
            id: post_instance.slug, 
            post: { title: 'Updated Title' } 
          }
          expect(response).to redirect_to(post_instance)
        end
      end
      
      context 'when user does not own the post' do
        let(:other_user) { create(:user) }
        
        before { sign_in other_user }
        
        it 'denies access' do
          expect {
            patch :update, params: { 
              id: post_instance.slug, 
              post: { title: 'Updated Title' } 
            }
          }.to raise_error(Pundit::NotAuthorizedError)
        end
      end
    end
    
    describe 'DELETE #destroy' do
      context 'when user owns the post' do
        before { sign_in user }
        
        it 'deletes the post' do
          post_to_delete = create(:post, user: user)
          expect {
            delete :destroy, params: { id: post_to_delete.slug }
          }.to change(Post, :count).by(-1)
        end
        
        it 'redirects to posts index' do
          delete :destroy, params: { id: post_instance.slug }
          expect(response).to redirect_to(posts_path)
        end
      end
      
      context 'when admin user' do
        before { sign_in admin }
        
        it 'allows deletion of any post' do
          expect {
            delete :destroy, params: { id: post_instance.slug }
          }.to change(Post, :count).by(-1)
        end
      end
    end
    
    describe 'PATCH #publish' do
      before { sign_in user }
      
      it 'publishes a draft post' do
        patch :publish, params: { id: draft_post.slug }
        expect(draft_post.reload.status).to eq('published')
      end
      
      it 'redirects with success notice' do
        patch :publish, params: { id: draft_post.slug }
        expect(response).to redirect_to(draft_post)
        expect(flash[:notice]).to eq('Post was successfully published.')
      end
    end
  end
  
  # spec/requests/api/v1/posts_spec.rb
  require 'rails_helper'
  
  RSpec.describe 'Api::V1::Posts', type: :request do
    let(:user) { create(:user) }
    let(:admin) { create(:user, :admin) }
    let(:category) { create(:category) }
    let!(:published_posts) { create_list(:post, 3, category: category) }
    let!(:draft_posts) { create_list(:post, 2, :draft, category: category) }
    
    describe 'GET /api/v1/posts' do
      it 'returns published posts' do
        get '/api/v1/posts'
        
        expect(response).to have_http_status(:success)
        json_response = JSON.parse(response.body)
        expect(json_response['posts']['data'].length).to eq(3)
      end
      
      it 'includes pagination metadata' do
        get '/api/v1/posts'
        
        json_response = JSON.parse(response.body)
        expect(json_response['meta']).to include(
          'current_page',
          'total_pages',
          'total_count'
        )
      end
      
      it 'filters by category' do
        other_category = create(:category)
        create(:post, category: other_category)
        
        get "/api/v1/posts?category=#{category.slug}"
        
        json_response = JSON.parse(response.body)
        expect(json_response['posts']['data'].length).to eq(3)
      end
      
      it 'searches posts' do
        searchable_post = create(:post, title: 'Unique Search Term')
        
        get '/api/v1/posts?search=Unique Search'
        
        json_response = JSON.parse(response.body)
        expect(json_response['posts']['data'].length).to eq(1)
      end
    end
    
    describe 'GET /api/v1/posts/:id' do
      let(:post_instance) { published_posts.first }
      
      it 'returns the post' do
        get "/api/v1/posts/#{post_instance.slug}"
        
        expect(response).to have_http_status(:success)
        json_response = JSON.parse(response.body)
        expect(json_response['data']['attributes']['title']).to eq(post_instance.title)
      end
      
      it 'returns 404 for non-existent post' do
        get '/api/v1/posts/non-existent'
        expect(response).to have_http_status(:not_found)
      end
    end
    
    describe 'POST /api/v1/posts' do
      let(:valid_attributes) do
        {
          title: 'New API Post',
          content: 'Post content via API',
          category_id: category.id
        }
      end
      
      context 'with valid authentication' do
        before { authenticate_api_user(user) }
        
        it 'creates a new post' do
          expect {
            post '/api/v1/posts', params: { post: valid_attributes }
          }.to change(Post, :count).by(1)
          
          expect(response).to have_http_status(:created)
        end
        
        it 'returns the created post' do
          post '/api/v1/posts', params: { post: valid_attributes }
          
          json_response = JSON.parse(response.body)
          expect(json_response['data']['attributes']['title']).to eq('New API Post')
        end
      end
      
      context 'without authentication' do
        it 'returns unauthorized' do
          post '/api/v1/posts', params: { post: valid_attributes }
          expect(response).to have_http_status(:unauthorized)
        end
      end
      
      context 'with invalid attributes' do
        before { authenticate_api_user(user) }
        
        it 'returns validation errors' do
          post '/api/v1/posts', params: { post: { title: '' } }
          
          expect(response).to have_http_status(:unprocessable_entity)
          json_response = JSON.parse(response.body)
          expect(json_response['errors']).to be_present
        end
      end
    end
    
    describe 'PUT /api/v1/posts/:id' do
      let(:post_instance) { create(:post, user: user) }
      
      before { authenticate_api_user(user) }
      
      it 'updates the post' do
        put "/api/v1/posts/#{post_instance.slug}", 
            params: { post: { title: 'Updated Title' } }
        
        expect(response).to have_http_status(:success)
        expect(post_instance.reload.title).to eq('Updated Title')
      end
      
      it 'returns the updated post' do
        put "/api/v1/posts/#{post_instance.slug}", 
            params: { post: { title: 'Updated Title' } }
        
        json_response = JSON.parse(response.body)
        expect(json_response['data']['attributes']['title']).to eq('Updated Title')
      end
    end
    
    describe 'DELETE /api/v1/posts/:id' do
      let(:post_instance) { create(:post, user: user) }
      
      before { authenticate_api_user(user) }
      
      it 'deletes the post' do
        delete "/api/v1/posts/#{post_instance.slug}"
        
        expect(response).to have_http_status(:no_content)
        expect(Post.exists?(post_instance.id)).to be false
      end
    end
  end
  
  # Helper method for API authentication in tests
  def authenticate_api_user(user)
    token = JWT.encode(
      { user_id: user.id, exp: 24.hours.from_now.to_i },
      Rails.application.secrets.secret_key_base,
      'HS256'
    )
    request.headers['Authorization'] = "Bearer #{token}"
  end


# ═══════════════════════════════════════════════════════════════════════════════
#                           11. PERFORMANCE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# config/application.rb - Performance configurations
module MyWebApp
    class Application < Rails::Application
      # ... existing configuration ...
      
      # Asset pipeline optimizations
      config.assets.compile = false
      config.assets.digest = true
      config.assets.compress = true
      
      # Gzip compression
      config.middleware.use Rack::Deflater
      
      # Cache store configuration
      config.cache_store = :redis_cache_store, {
        url: ENV['REDIS_URL'],
        namespace: 'myapp_cache'
      }
      
      # Session store with Redis
      config.session_store :redis_store,
        servers: [ENV['REDIS_URL']],
        expire_after: 1.week,
        key: '_myapp_session'
    end
  end
  
  # app/models/concerns/cacheable.rb
  module Cacheable
    extend ActiveSupport::Concern
    
    class_methods do
      def cached_find(id, expires_in: 1.hour)
        Rails.cache.fetch("#{self.name.downcase}/#{id}", expires_in: expires_in) do
          find(id)
        end
      end
      
      def cached_count(expires_in: 10.minutes)
        Rails.cache.fetch("#{self.name.downcase}/count", expires_in: expires_in) do
          count
        end
      end
    end
    
    def cache_key_with_version
      "#{super}/#{updated_at.to_i}"
    end
    
    def expire_cache
      Rails.cache.delete("#{self.class.name.downcase}/#{id}")
      Rails.cache.delete("#{self.class.name.downcase}/count")
    end
  end
  
  # Include in models that need caching
  class Post < ApplicationRecord
    include Cacheable
    
    after_update :expire_cache
    after_destroy :expire_cache
    
    # ... rest of model ...
  end
  
  # app/controllers/concerns/caching.rb
  module Caching
    extend ActiveSupport::Concern
    
    private
    
    def cache_page(key, expires_in: 1.hour, &block)
      Rails.cache.fetch(key, expires_in: expires_in) do
        yield
      end
    end
    
    def set_cache_headers(max_age: 1.hour)
      response.cache_control[:max_age] = max_age.to_i
      response.cache_control[:public] = true
    end
  end
  
  # Database optimization examples
  class OptimizedPostsQuery
    def self.homepage_posts
      Post.includes(:user, :category, :tags)
          .published
          .featured
          .select(:id, :title, :slug, :excerpt, :published_at, :user_id, :category_id)
          .recent
          .limit(3)
    end
    
    def self.posts_with_stats
      Post.joins(:user, :category)
          .select('posts.*, users.first_name, users.last_name, categories.name as category_name')
          .where(status: :published)
          .order(published_at: :desc)
    end
    
    def self.popular_posts(limit: 10)
      Rails.cache.fetch("popular_posts/#{limit}", expires_in: 1.hour) do
        Post.published
            .select(:id, :title, :slug, :views_count)
            .order(views_count: :desc)
            .limit(limit)
      end
    end
  end
  
  # Background job for cache warming
  class CacheWarmupJob < ApplicationJob
    queue_as :low_priority
    
    def perform
      # Warm up popular posts cache
      OptimizedPostsQuery.popular_posts
      
      # Warm up categories cache
      Category.active.includes(:posts).order(:name)
      
      # Warm up homepage data
      OptimizedPostsQuery.homepage_posts
    end
  end
  
  # Database indexes for performance
  =begin
  # Add these to your migrations for better query performance
  
  class AddIndexesForPerformance < ActiveRecord::Migration[7.1]
    def change
      # Posts indexes
      add_index :posts, [:status, :published_at]
      add_index :posts, [:category_id, :status]
      add_index :posts, [:user_id, :status]
      add_index :posts, [:featured, :status]
      add_index :posts, :views_count
      
      # Comments indexes
      add_index :comments, [:post_id, :status, :created_at]
      add_index :comments, [:parent_id]
      
      # Users indexes
      add_index :users, [:status, :created_at]
      add_index :users, :username
      
      # Full-text search indexes (PostgreSQL)
      add_index :posts, :title, using: :gin, opclass: :gin_trgm_ops
      add_index :posts, :excerpt, using: :gin, opclass: :gin_trgm_ops
    end
  end
  =end
  
  # ═══════════════════════════════════════════════════════════════════════════════
  #                           12. DEPLOYMENT AND PRODUCTION
  # ═══════════════════════════════════════════════════════════════════════════════
  
  # config/environments/production.rb
  Rails.application.configure do
    # Settings specified here will take precedence over those in config/application.rb.
    
    # Code is not reloaded between requests.
    config.cache_classes = true
    
    # Eager load code on boot.
    config.eager_load = true
    
    # Full error reports are disabled and caching is turned on.
    config.consider_all_requests_local = false
    config.action_controller.perform_caching = true
    
    # Ensures that a master key has been made available
    config.require_master_key = true
    
    # Disable serving static files from the `/public` folder by default since
    # Apache or NGINX already handles this.
    config.public_file_server.enabled = ENV['RAILS_SERVE_STATIC_FILES'].present?
    
    # Compress CSS using a preprocessor.
    config.assets.css_compressor = :sass
    
    # Do not fallback to assets pipeline if a precompiled asset is missed.
    config.assets.compile = false
    
    # Enable serving of images, stylesheets, and JavaScripts from an asset server.
    config.asset_host = ENV['ASSET_HOST'] if ENV['ASSET_HOST'].present?
    
    # Specifies the header that your server uses for sending files.
    config.action_dispatch.x_sendfile_header = 'X-Accel-Redirect' # for NGINX
    
    # Store uploaded files on the local file system (see config/storage.yml for options).
    config.active_storage.variant_processor = :mini_magick
    
    # Mount Action Cable outside main process or domain.
    # config.action_cable.mount_path = nil
    # config.action_cable.url = 'wss://example.com/cable'
    # config.action_cable.allowed_request_origins = [ 'http://example.com', /http:\/\/example.*/ ]
    
    # Force all access to the app over SSL
    config.force_ssl = true
    
    # Include generic and useful information about system operation
    config.log_level = :info
    
    # Prepend all log lines with the following tags.
    config.log_tags = [ :request_id ]# RUBY ON RAILS WEB DEVELOPMENT - Comprehensive Reference - by Richard Rembert
  # Ruby on Rails enables rapid prototyping and development of full-stack web applications
  # with convention over configuration, built-in security, and powerful abstractions