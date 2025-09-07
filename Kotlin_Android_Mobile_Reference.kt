// KOTLIN ANDROID MOBILE DEVELOPMENT - Comprehensive Reference - by Richard Rembert
// Kotlin is the preferred language for Android development, offering modern syntax,
// null safety, coroutines, and seamless Java interoperability for building robust mobile apps

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND PROJECT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/*
KOTLIN ANDROID DEVELOPMENT SETUP:

1. Android Studio Installation:
   - Download from: https://developer.android.com/studio
   - Include Android SDK, emulator, and build tools
   - Enable Kotlin plugin (included by default)

2. Project Creation:
   - Choose "Empty Activity" or "Basic Activity"
   - Select Kotlin as language
   - Choose minimum SDK (API 24+ recommended)
   - Enable Jetpack Compose for modern UI

3. Essential Dependencies (app/build.gradle.kts):
   
   android {
       compileSdk 34
       
       defaultConfig {
           applicationId "com.example.myapp"
           minSdk 24
           targetSdk 34
           versionCode 1
           versionName "1.0"
           
           testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
           vectorDrawables.useSupportLibrary = true
       }
       
       buildFeatures {
           compose = true
           viewBinding = true
           dataBinding = true
       }
       
       compileOptions {
           sourceCompatibility = JavaVersion.VERSION_17
           targetCompatibility = JavaVersion.VERSION_17
       }
       
       kotlinOptions {
           jvmTarget = "17"
       }
       
       composeOptions {
           kotlinCompilerExtensionVersion = "1.5.4"
       }
   }
   
   dependencies {
       // Core Android
       implementation("androidx.core:core-ktx:1.12.0")
       implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
       implementation("androidx.activity:activity-compose:1.8.1")
       
       // Jetpack Compose
       implementation(platform("androidx.compose:compose-bom:2023.10.01"))
       implementation("androidx.compose.ui:ui")
       implementation("androidx.compose.ui:ui-graphics")
       implementation("androidx.compose.ui:ui-tooling-preview")
       implementation("androidx.compose.material3:material3")
       
       // Navigation
       implementation("androidx.navigation:navigation-compose:2.7.5")
       implementation("androidx.navigation:navigation-fragment-ktx:2.7.5")
       implementation("androidx.navigation:navigation-ui-ktx:2.7.5")
       
       // Architecture Components
       implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
       implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
       implementation("androidx.room:room-runtime:2.6.0")
       implementation("androidx.room:room-ktx:2.6.0")
       kapt("androidx.room:room-compiler:2.6.0")
       
       // Dependency Injection
       implementation("com.google.dagger:hilt-android:2.48")
       implementation("androidx.hilt:hilt-navigation-compose:1.1.0")
       kapt("com.google.dagger:hilt-compiler:2.48")
       
       // Networking
       implementation("com.squareup.retrofit2:retrofit:2.9.0")
       implementation("com.squareup.retrofit2:converter-gson:2.9.0")
       implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
       
       // Image Loading
       implementation("io.coil-kt:coil-compose:2.5.0")
       
       // Coroutines
       implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
       
       // Testing
       testImplementation("junit:junit:4.13.2")
       testImplementation("org.mockito:mockito-core:5.6.0")
       testImplementation("androidx.arch.core:core-testing:2.2.0")
       testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
       
       androidTestImplementation("androidx.test.ext:junit:1.1.5")
       androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
       androidTestImplementation("androidx.compose.ui:ui-test-junit4")
       
       debugImplementation("androidx.compose.ui:ui-tooling")
       debugImplementation("androidx.compose.ui:ui-test-manifest")
   }

4. Project Structure:
   app/
   ├── src/
   │   ├── main/
   │   │   ├── java/com/example/myapp/
   │   │   │   ├── data/
   │   │   │   │   ├── local/
   │   │   │   │   ├── remote/
   │   │   │   │   ├── repository/
   │   │   │   │   └── model/
   │   │   │   ├── domain/
   │   │   │   │   ├── repository/
   │   │   │   │   ├── usecase/
   │   │   │   │   └── model/
   │   │   │   ├── presentation/
   │   │   │   │   ├── ui/
   │   │   │   │   ├── viewmodel/
   │   │   │   │   └── navigation/
   │   │   │   ├── di/
   │   │   │   └── utils/
   │   │   ├── res/
   │   │   └── AndroidManifest.xml
   │   ├── test/
   │   └── androidTest/
   └── build.gradle.kts
*/

package com.example.myapp

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. APPLICATION CLASS AND DEPENDENCY INJECTION
// ═══════════════════════════════════════════════════════════════════════════════

@HiltAndroidApp
class MyApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize any required libraries
        initializeLibraries()
        
        // Setup crash reporting (Firebase Crashlytics, Bugsnag, etc.)
        setupCrashReporting()
        
        // Setup analytics
        setupAnalytics()
    }
    
    private fun initializeLibraries() {
        // Initialize third-party libraries if needed
        // Example: Timber for logging, LeakCanary for memory leak detection
    }
    
    private fun setupCrashReporting() {
        // Setup crash reporting service
    }
    
    private fun setupAnalytics() {
        // Setup analytics service
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. DEPENDENCY INJECTION WITH HILT
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.di

import android.content.Context
import androidx.room.Room
import com.example.myapp.data.local.AppDatabase
import com.example.myapp.data.local.UserDao
import com.example.myapp.data.remote.ApiService
import com.example.myapp.data.repository.UserRepositoryImpl
import com.example.myapp.domain.repository.UserRepository
import com.example.myapp.utils.Constants
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    
    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app_database"
        )
        .fallbackToDestructiveMigration()
        .build()
    }
    
    @Provides
    fun provideUserDao(database: AppDatabase): UserDao = database.userDao()
}

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {
    
    @Provides
    @Singleton
    fun provideGson(): Gson {
        return GsonBuilder()
            .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
            .create()
    }
    
    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        val loggingInterceptor = HttpLoggingInterceptor().apply {
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BODY
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        }
        
        return OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .addInterceptor { chain ->
                val request = chain.request().newBuilder()
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Accept", "application/json")
                    .build()
                chain.proceed(request)
            }
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }
    
    @Provides
    @Singleton
    fun provideRetrofit(gson: Gson, okHttpClient: OkHttpClient): Retrofit {
        return Retrofit.Builder()
            .baseUrl(Constants.BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }
    
    @Provides
    @Singleton
    fun provideApiService(retrofit: Retrofit): ApiService = retrofit.create(ApiService::class.java)
}

@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {
    
    @Provides
    @Singleton
    fun provideUserRepository(
        apiService: ApiService,
        userDao: UserDao
    ): UserRepository {
        return UserRepositoryImpl(apiService, userDao)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. DATA MODELS AND ENTITIES
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.google.gson.annotations.SerializedName
import java.util.Date

// Domain Models (Clean Architecture)
package com.example.myapp.domain.model

data class User(
    val id: String,
    val email: String,
    val username: String,
    val firstName: String,
    val lastName: String,
    val avatarUrl: String? = null,
    val isActive: Boolean = true,
    val createdAt: Date,
    val updatedAt: Date
) {
    val fullName: String
        get() = "$firstName $lastName"
    
    val initials: String
        get() = "${firstName.firstOrNull()?.uppercase()}${lastName.firstOrNull()?.uppercase()}"
}

data class Post(
    val id: String,
    val title: String,
    val content: String,
    val excerpt: String,
    val imageUrl: String? = null,
    val authorId: String,
    val author: User? = null,
    val categoryId: String,
    val category: Category? = null,
    val tags: List<String> = emptyList(),
    val isPublished: Boolean = false,
    val viewCount: Int = 0,
    val likeCount: Int = 0,
    val commentCount: Int = 0,
    val publishedAt: Date? = null,
    val createdAt: Date,
    val updatedAt: Date
) {
    val readingTimeMinutes: Int
        get() = (content.split(" ").size / 200).coerceAtLeast(1)
    
    val isPublic: Boolean
        get() = isPublished && publishedAt != null
}

data class Category(
    val id: String,
    val name: String,
    val slug: String,
    val description: String,
    val color: String,
    val postCount: Int = 0
)

data class Comment(
    val id: String,
    val content: String,
    val postId: String,
    val authorId: String,
    val author: User? = null,
    val parentId: String? = null,
    val replies: List<Comment> = emptyList(),
    val createdAt: Date,
    val updatedAt: Date
)

// Data Transfer Objects (DTOs)
package com.example.myapp.data.model

data class UserDto(
    @SerializedName("id") val id: String,
    @SerializedName("email") val email: String,
    @SerializedName("username") val username: String,
    @SerializedName("first_name") val firstName: String,
    @SerializedName("last_name") val lastName: String,
    @SerializedName("avatar_url") val avatarUrl: String?,
    @SerializedName("is_active") val isActive: Boolean,
    @SerializedName("created_at") val createdAt: String,
    @SerializedName("updated_at") val updatedAt: String
)

data class PostDto(
    @SerializedName("id") val id: String,
    @SerializedName("title") val title: String,
    @SerializedName("content") val content: String,
    @SerializedName("excerpt") val excerpt: String,
    @SerializedName("image_url") val imageUrl: String?,
    @SerializedName("author_id") val authorId: String,
    @SerializedName("author") val author: UserDto?,
    @SerializedName("category_id") val categoryId: String,
    @SerializedName("category") val category: CategoryDto?,
    @SerializedName("tags") val tags: List<String>,
    @SerializedName("is_published") val isPublished: Boolean,
    @SerializedName("view_count") val viewCount: Int,
    @SerializedName("like_count") val likeCount: Int,
    @SerializedName("comment_count") val commentCount: Int,
    @SerializedName("published_at") val publishedAt: String?,
    @SerializedName("created_at") val createdAt: String,
    @SerializedName("updated_at") val updatedAt: String
)

data class CategoryDto(
    @SerializedName("id") val id: String,
    @SerializedName("name") val name: String,
    @SerializedName("slug") val slug: String,
    @SerializedName("description") val description: String,
    @SerializedName("color") val color: String,
    @SerializedName("post_count") val postCount: Int
)

// Room Database Entities
package com.example.myapp.data.local.entity

@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey val id: String,
    val email: String,
    val username: String,
    val firstName: String,
    val lastName: String,
    val avatarUrl: String?,
    val isActive: Boolean,
    val createdAt: Long,
    val updatedAt: Long
)

@Entity(tableName = "posts")
data class PostEntity(
    @PrimaryKey val id: String,
    val title: String,
    val content: String,
    val excerpt: String,
    val imageUrl: String?,
    val authorId: String,
    val categoryId: String,
    val tags: String, // JSON string
    val isPublished: Boolean,
    val viewCount: Int,
    val likeCount: Int,
    val commentCount: Int,
    val publishedAt: Long?,
    val createdAt: Long,
    val updatedAt: Long
)

@Entity(tableName = "categories")
data class CategoryEntity(
    @PrimaryKey val id: String,
    val name: String,
    val slug: String,
    val description: String,
    val color: String,
    val postCount: Int
)

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. DATA MAPPERS
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.data.mapper

import com.example.myapp.data.local.entity.UserEntity
import com.example.myapp.data.model.UserDto
import com.example.myapp.domain.model.User
import java.text.SimpleDateFormat
import java.util.*

object UserMapper {
    
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.getDefault())
    
    fun dtoToDomain(dto: UserDto): User {
        return User(
            id = dto.id,
            email = dto.email,
            username = dto.username,
            firstName = dto.firstName,
            lastName = dto.lastName,
            avatarUrl = dto.avatarUrl,
            isActive = dto.isActive,
            createdAt = dateFormat.parse(dto.createdAt) ?: Date(),
            updatedAt = dateFormat.parse(dto.updatedAt) ?: Date()
        )
    }
    
    fun dtoToEntity(dto: UserDto): UserEntity {
        return UserEntity(
            id = dto.id,
            email = dto.email,
            username = dto.username,
            firstName = dto.firstName,
            lastName = dto.lastName,
            avatarUrl = dto.avatarUrl,
            isActive = dto.isActive,
            createdAt = dateFormat.parse(dto.createdAt)?.time ?: 0L,
            updatedAt = dateFormat.parse(dto.updatedAt)?.time ?: 0L
        )
    }
    
    fun entityToDomain(entity: UserEntity): User {
        return User(
            id = entity.id,
            email = entity.email,
            username = entity.username,
            firstName = entity.firstName,
            lastName = entity.lastName,
            avatarUrl = entity.avatarUrl,
            isActive = entity.isActive,
            createdAt = Date(entity.createdAt),
            updatedAt = Date(entity.updatedAt)
        )
    }
    
    fun domainToEntity(user: User): UserEntity {
        return UserEntity(
            id = user.id,
            email = user.email,
            username = user.username,
            firstName = user.firstName,
            lastName = user.lastName,
            avatarUrl = user.avatarUrl,
            isActive = user.isActive,
            createdAt = user.createdAt.time,
            updatedAt = user.updatedAt.time
        )
    }
}

object PostMapper {
    
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.getDefault())
    private val gson = Gson()
    
    fun dtoToDomain(dto: PostDto): Post {
        return Post(
            id = dto.id,
            title = dto.title,
            content = dto.content,
            excerpt = dto.excerpt,
            imageUrl = dto.imageUrl,
            authorId = dto.authorId,
            author = dto.author?.let { UserMapper.dtoToDomain(it) },
            categoryId = dto.categoryId,
            category = dto.category?.let { CategoryMapper.dtoToDomain(it) },
            tags = dto.tags,
            isPublished = dto.isPublished,
            viewCount = dto.viewCount,
            likeCount = dto.likeCount,
            commentCount = dto.commentCount,
            publishedAt = dto.publishedAt?.let { dateFormat.parse(it) },
            createdAt = dateFormat.parse(dto.createdAt) ?: Date(),
            updatedAt = dateFormat.parse(dto.updatedAt) ?: Date()
        )
    }
    
    fun entityToDomain(entity: PostEntity): Post {
        val tags = try {
            gson.fromJson(entity.tags, Array<String>::class.java).toList()
        } catch (e: Exception) {
            emptyList<String>()
        }
        
        return Post(
            id = entity.id,
            title = entity.title,
            content = entity.content,
            excerpt = entity.excerpt,
            imageUrl = entity.imageUrl,
            authorId = entity.authorId,
            categoryId = entity.categoryId,
            tags = tags,
            isPublished = entity.isPublished,
            viewCount = entity.viewCount,
            likeCount = entity.likeCount,
            commentCount = entity.commentCount,
            publishedAt = entity.publishedAt?.let { Date(it) },
            createdAt = Date(entity.createdAt),
            updatedAt = Date(entity.updatedAt)
        )
    }
    
    fun domainToEntity(post: Post): PostEntity {
        return PostEntity(
            id = post.id,
            title = post.title,
            content = post.content,
            excerpt = post.excerpt,
            imageUrl = post.imageUrl,
            authorId = post.authorId,
            categoryId = post.categoryId,
            tags = gson.toJson(post.tags),
            isPublished = post.isPublished,
            viewCount = post.viewCount,
            likeCount = post.likeCount,
            commentCount = post.commentCount,
            publishedAt = post.publishedAt?.time,
            createdAt = post.createdAt.time,
            updatedAt = post.updatedAt.time
        )
    }
}

object CategoryMapper {
    
    fun dtoToDomain(dto: CategoryDto): Category {
        return Category(
            id = dto.id,
            name = dto.name,
            slug = dto.slug,
            description = dto.description,
            color = dto.color,
            postCount = dto.postCount
        )
    }
    
    fun entityToDomain(entity: CategoryEntity): Category {
        return Category(
            id = entity.id,
            name = entity.name,
            slug = entity.slug,
            description = entity.description,
            color = entity.color,
            postCount = entity.postCount
        )
    }
    
    fun domainToEntity(category: Category): CategoryEntity {
        return CategoryEntity(
            id = category.id,
            name = category.name,
            slug = category.slug,
            description = category.description,
            color = category.color,
            postCount = category.postCount
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. DATABASE LAYER (ROOM)
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.data.local

import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.example.myapp.data.local.entity.CategoryEntity
import com.example.myapp.data.local.entity.PostEntity
import com.example.myapp.data.local.entity.UserEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface UserDao {
    
    @Query("SELECT * FROM users WHERE id = :id")
    suspend fun getUserById(id: String): UserEntity?
    
    @Query("SELECT * FROM users WHERE email = :email")
    suspend fun getUserByEmail(email: String): UserEntity?
    
    @Query("SELECT * FROM users ORDER BY firstName, lastName")
    fun getAllUsers(): Flow<List<UserEntity>>
    
    @Query("SELECT * FROM users WHERE firstName LIKE '%' || :query || '%' OR lastName LIKE '%' || :query || '%' OR username LIKE '%' || :query || '%'")
    fun searchUsers(query: String): Flow<List<UserEntity>>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: UserEntity)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUsers(users: List<UserEntity>)
    
    @Update
    suspend fun updateUser(user: UserEntity)
    
    @Delete
    suspend fun deleteUser(user: UserEntity)
    
    @Query("DELETE FROM users WHERE id = :id")
    suspend fun deleteUserById(id: String)
    
    @Query("DELETE FROM users")
    suspend fun deleteAllUsers()
}

@Dao
interface PostDao {
    
    @Query("SELECT * FROM posts WHERE id = :id")
    suspend fun getPostById(id: String): PostEntity?
    
    @Query("SELECT * FROM posts WHERE isPublished = 1 ORDER BY publishedAt DESC")
    fun getPublishedPosts(): Flow<List<PostEntity>>
    
    @Query("SELECT * FROM posts WHERE authorId = :authorId ORDER BY createdAt DESC")
    fun getPostsByAuthor(authorId: String): Flow<List<PostEntity>>
    
    @Query("SELECT * FROM posts WHERE categoryId = :categoryId AND isPublished = 1 ORDER BY publishedAt DESC")
    fun getPostsByCategory(categoryId: String): Flow<List<PostEntity>>
    
    @Query("SELECT * FROM posts WHERE title LIKE '%' || :query || '%' OR content LIKE '%' || :query || '%'")
    fun searchPosts(query: String): Flow<List<PostEntity>>
    
    @Query("SELECT * FROM posts ORDER BY viewCount DESC LIMIT :limit")
    fun getPopularPosts(limit: Int): Flow<List<PostEntity>>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPost(post: PostEntity)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPosts(posts: List<PostEntity>)
    
    @Update
    suspend fun updatePost(post: PostEntity)
    
    @Delete
    suspend fun deletePost(post: PostEntity)
    
    @Query("DELETE FROM posts WHERE id = :id")
    suspend fun deletePostById(id: String)
    
    @Query("UPDATE posts SET viewCount = viewCount + 1 WHERE id = :id")
    suspend fun incrementViewCount(id: String)
    
    @Query("UPDATE posts SET likeCount = :likeCount WHERE id = :id")
    suspend fun updateLikeCount(id: String, likeCount: Int)
}

@Dao
interface CategoryDao {
    
    @Query("SELECT * FROM categories WHERE id = :id")
    suspend fun getCategoryById(id: String): CategoryEntity?
    
    @Query("SELECT * FROM categories ORDER BY name")
    fun getAllCategories(): Flow<List<CategoryEntity>>
    
    @Query("SELECT * FROM categories WHERE name LIKE '%' || :query || '%'")
    fun searchCategories(query: String): Flow<List<CategoryEntity>>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCategory(category: CategoryEntity)
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCategories(categories: List<CategoryEntity>)
    
    @Update
    suspend fun updateCategory(category: CategoryEntity)
    
    @Delete
    suspend fun deleteCategory(category: CategoryEntity)
}

@Database(
    entities = [UserEntity::class, PostEntity::class, CategoryEntity::class],
    version = 1,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    
    abstract fun userDao(): UserDao
    abstract fun postDao(): PostDao
    abstract fun categoryDao(): CategoryDao
    
    companion object {
        const val DATABASE_NAME = "app_database"
        
        // Example migration (when needed)
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE users ADD COLUMN bio TEXT")
            }
        }
    }
}

class Converters {
    
    @TypeConverter
    fun fromStringList(value: List<String>): String {
        return Gson().toJson(value)
    }
    
    @TypeConverter
    fun toStringList(value: String): List<String> {
        return try {
            Gson().fromJson(value, Array<String>::class.java).toList()
        } catch (e: Exception) {
            emptyList()
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           7. NETWORK LAYER (RETROFIT)
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.data.remote

import com.example.myapp.data.model.*
import retrofit2.Response
import retrofit2.http.*

interface ApiService {
    
    // Authentication
    @POST("auth/login")
    suspend fun login(@Body request: LoginRequest): Response<AuthResponse>
    
    @POST("auth/register")
    suspend fun register(@Body request: RegisterRequest): Response<AuthResponse>
    
    @POST("auth/refresh")
    suspend fun refreshToken(@Body request: RefreshTokenRequest): Response<AuthResponse>
    
    @POST("auth/logout")
    suspend fun logout(): Response<Unit>
    
    // Users
    @GET("users/{id}")
    suspend fun getUserById(@Path("id") id: String): Response<ApiResponse<UserDto>>
    
    @GET("users")
    suspend fun getUsers(
        @Query("page") page: Int = 1,
        @Query("limit") limit: Int = 20,
        @Query("search") search: String? = null
    ): Response<PaginatedResponse<UserDto>>
    
    @PUT("users/{id}")
    suspend fun updateUser(
        @Path("id") id: String,
        @Body request: UpdateUserRequest
    ): Response<ApiResponse<UserDto>>
    
    @DELETE("users/{id}")
    suspend fun deleteUser(@Path("id") id: String): Response<Unit>
    
    // Posts
    @GET("posts")
    suspend fun getPosts(
        @Query("page") page: Int = 1,
        @Query("limit") limit: Int = 20,
        @Query("category") category: String? = null,
        @Query("author") author: String? = null,
        @Query("search") search: String? = null,
        @Query("published") published: Boolean? = null
    ): Response<PaginatedResponse<PostDto>>
    
    @GET("posts/{id}")
    suspend fun getPostById(@Path("id") id: String): Response<ApiResponse<PostDto>>
    
    @POST("posts")
    suspend fun createPost(@Body request: CreatePostRequest): Response<ApiResponse<PostDto>>
    
    @PUT("posts/{id}")
    suspend fun updatePost(
        @Path("id") id: String,
        @Body request: UpdatePostRequest
    ): Response<ApiResponse<PostDto>>
    
    @DELETE("posts/{id}")
    suspend fun deletePost(@Path("id") id: String): Response<Unit>
    
    @POST("posts/{id}/like")
    suspend fun likePost(@Path("id") id: String): Response<Unit>
    
    @DELETE("posts/{id}/like")
    suspend fun unlikePost(@Path("id") id: String): Response<Unit>
    
    @POST("posts/{id}/view")
    suspend fun incrementPostView(@Path("id") id: String): Response<Unit>
    
    // Categories
    @GET("categories")
    suspend fun getCategories(): Response<ApiResponse<List<CategoryDto>>>
    
    @GET("categories/{id}")
    suspend fun getCategoryById(@Path("id") id: String): Response<ApiResponse<CategoryDto>>
    
    // Comments
    @GET("posts/{postId}/comments")
    suspend fun getComments(
        @Path("postId") postId: String,
        @Query("page") page: Int = 1,
        @Query("limit") limit: Int = 20
    ): Response<PaginatedResponse<CommentDto>>
    
    @POST("posts/{postId}/comments")
    suspend fun createComment(
        @Path("postId") postId: String,
        @Body request: CreateCommentRequest
    ): Response<ApiResponse<CommentDto>>
    
    @PUT("comments/{id}")
    suspend fun updateComment(
        @Path("id") id: String,
        @Body request: UpdateCommentRequest
    ): Response<ApiResponse<CommentDto>>
    
    @DELETE("comments/{id}")
    suspend fun deleteComment(@Path("id") id: String): Response<Unit>
}

// API Request/Response Models
data class LoginRequest(
    val email: String,
    val password: String
)

data class RegisterRequest(
    val email: String,
    val username: String,
    val firstName: String,
    val lastName: String,
    val password: String
)

data class RefreshTokenRequest(
    val refreshToken: String
)

data class AuthResponse(
    val accessToken: String,
    val refreshToken: String,
    val expiresIn: Long,
    val user: UserDto
)

data class UpdateUserRequest(
    val firstName: String?,
    val lastName: String?,
    val username: String?,
    val bio: String?
)

data class CreatePostRequest(
    val title: String,
    val content: String,
    val excerpt: String?,
    val categoryId: String,
    val tags: List<String>,
    val isPublished: Boolean = false
)

data class UpdatePostRequest(
    val title: String?,
    val content: String?,
    val excerpt: String?,
    val categoryId: String?,
    val tags: List<String>?,
    val isPublished: Boolean?
)

data class CreateCommentRequest(
    val content: String,
    val parentId: String? = null
)

data class UpdateCommentRequest(
    val content: String
)

// Generic API Response Wrappers
data class ApiResponse<T>(
    val success: Boolean,
    val data: T?,
    val message: String?,
    val errors: List<String>?
)

data class PaginatedResponse<T>(
    val success: Boolean,
    val data: List<T>,
    val pagination: PaginationInfo,
    val message: String?,
    val errors: List<String>?
)

data class PaginationInfo(
    val currentPage: Int,
    val totalPages: Int,
    val totalItems: Int,
    val itemsPerPage: Int,
    val hasNext: Boolean,
    val hasPrevious: Boolean
)

// ═══════════════════════════════════════════════════════════════════════════════
//                           8. REPOSITORY LAYER
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.domain.repository

import com.example.myapp.domain.model.*
import com.example.myapp.utils.Resource
import kotlinx.coroutines.flow.Flow

interface UserRepository {
    suspend fun login(email: String, password: String): Resource<User>
    suspend fun register(
        email: String,
        username: String,
        firstName: String,
        lastName: String,
        password: String
    ): Resource<User>
    suspend fun refreshToken(): Resource<String>
    suspend fun logout(): Resource<Unit>
    suspend fun getCurrentUser(): Flow<User?>
    suspend fun getUserById(id: String): Resource<User>
    suspend fun updateUser(user: User): Resource<User>
    suspend fun searchUsers(query: String): Flow<List<User>>
    suspend fun clearUserData()
}

interface PostRepository {
    suspend fun getPosts(
        page: Int = 1,
        limit: Int = 20,
        categoryId: String? = null,
        authorId: String? = null,
        search: String? = null
    ): Resource<List<Post>>
    
    suspend fun getPostById(id: String): Resource<Post>
    suspend fun createPost(post: CreatePostRequest): Resource<Post>
    suspend fun updatePost(id: String, post: UpdatePostRequest): Resource<Post>
    suspend fun deletePost(id: String): Resource<Unit>
    suspend fun likePost(id: String): Resource<Unit>
    suspend fun unlikePost(id: String): Resource<Unit>
    suspend fun incrementPostView(id: String): Resource<Unit>
    
    // Local caching
    fun getLocalPosts(): Flow<List<Post>>
    fun getPostsByCategory(categoryId: String): Flow<List<Post>>
    fun getPostsByAuthor(authorId: String): Flow<List<Post>>
    fun searchLocalPosts(query: String): Flow<List<Post>>
    suspend fun cachePost(post: Post)
    suspend fun cachePosts(posts: List<Post>)
}

interface CategoryRepository {
    suspend fun getCategories(): Resource<List<Category>>
    suspend fun getCategoryById(id: String): Resource<Category>
    fun getLocalCategories(): Flow<List<Category>>
    suspend fun cacheCategories(categories: List<Category>)
}

// Repository Implementation
package com.example.myapp.data.repository

import com.example.myapp.data.local.UserDao
import com.example.myapp.data.mapper.UserMapper
import com.example.myapp.data.remote.ApiService
import com.example.myapp.domain.model.User
import com.example.myapp.domain.repository.UserRepository
import com.example.myapp.utils.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UserRepositoryImpl @Inject constructor(
    private val apiService: ApiService,
    private val userDao: UserDao,
    private val tokenManager: TokenManager
) : UserRepository {
    
    override suspend fun login(email: String, password: String): Resource<User> {
        return try {
            val response = apiService.login(LoginRequest(email, password))
            if (response.isSuccessful && response.body() != null) {
                val authResponse = response.body()!!
                
                // Save tokens
                tokenManager.saveTokens(
                    accessToken = authResponse.accessToken,
                    refreshToken = authResponse.refreshToken
                )
                
                // Cache user
                val user = UserMapper.dtoToDomain(authResponse.user)
                userDao.insertUser(UserMapper.domainToEntity(user))
                
                Resource.Success(user)
            } else {
                Resource.Error("Login failed: ${response.message()}")
            }
        } catch (e: Exception) {
            Resource.Error("Network error: ${e.message}")
        }
    }
    
    override suspend fun register(
        email: String,
        username: String,
        firstName: String,
        lastName: String,
        password: String
    ): Resource<User> {
        return try {
            val request = RegisterRequest(email, username, firstName, lastName, password)
            val response = apiService.register(request)
            
            if (response.isSuccessful && response.body() != null) {
                val authResponse = response.body()!!
                
                // Save tokens
                tokenManager.saveTokens(
                    accessToken = authResponse.accessToken,
                    refreshToken = authResponse.refreshToken
                )
                
                // Cache user
                val user = UserMapper.dtoToDomain(authResponse.user)
                userDao.insertUser(UserMapper.domainToEntity(user))
                
                Resource.Success(user)
            } else {
                Resource.Error("Registration failed: ${response.message()}")
            }
        } catch (e: Exception) {
            Resource.Error("Network error: ${e.message}")
        }
    }
    
    override suspend fun refreshToken(): Resource<String> {
        return try {
            val refreshToken = tokenManager.getRefreshToken()
            if (refreshToken.isNullOrEmpty()) {
                return Resource.Error("No refresh token available")
            }
            
            val response = apiService.refreshToken(RefreshTokenRequest(refreshToken))
            if (response.isSuccessful && response.body() != null) {
                val authResponse = response.body()!!
                tokenManager.saveTokens(
                    accessToken = authResponse.accessToken,
                    refreshToken = authResponse.refreshToken
                )
                Resource.Success(authResponse.accessToken)
            } else {
                Resource.Error("Token refresh failed")
            }
        } catch (e: Exception) {
            Resource.Error("Token refresh error: ${e.message}")
        }
    }
    
    override suspend fun logout(): Resource<Unit> {
        return try {
            // Call API to invalidate token on server
            apiService.logout()
            
            // Clear local data
            clearUserData()
            
            Resource.Success(Unit)
        } catch (e: Exception) {
            // Even if API call fails, clear local data
            clearUserData()
            Resource.Success(Unit)
        }
    }
    
    override suspend fun getCurrentUser(): Flow<User?> {
        return userDao.getAllUsers().map { entities ->
            entities.firstOrNull()?.let { UserMapper.entityToDomain(it) }
        }
    }
    
    override suspend fun getUserById(id: String): Resource<User> {
        return try {
            // Try local first
            val localUser = userDao.getUserById(id)
            if (localUser != null) {
                return Resource.Success(UserMapper.entityToDomain(localUser))
            }
            
            // Fetch from API
            val response = apiService.getUserById(id)
            if (response.isSuccessful && response.body()?.data != null) {
                val user = UserMapper.dtoToDomain(response.body()!!.data!!)
                userDao.insertUser(UserMapper.domainToEntity(user))
                Resource.Success(user)
            } else {
                Resource.Error("User not found")
            }
        } catch (e: Exception) {
            Resource.Error("Error fetching user: ${e.message}")
        }
    }
    
    override suspend fun updateUser(user: User): Resource<User> {
        return try {
            val request = UpdateUserRequest(
                firstName = user.firstName,
                lastName = user.lastName,
                username = user.username,
                bio = null // Add bio field if needed
            )
            
            val response = apiService.updateUser(user.id, request)
            if (response.isSuccessful && response.body()?.data != null) {
                val updatedUser = UserMapper.dtoToDomain(response.body()!!.data!!)
                userDao.updateUser(UserMapper.domainToEntity(updatedUser))
                Resource.Success(updatedUser)
            } else {
                Resource.Error("Update failed: ${response.message()}")
            }
        } catch (e: Exception) {
            Resource.Error("Update error: ${e.message}")
        }
    }
    
    override suspend fun searchUsers(query: String): Flow<List<User>> {
        return userDao.searchUsers(query).map { entities ->
            entities.map { UserMapper.entityToDomain(it) }
        }
    }
    
    override suspend fun clearUserData() {
        tokenManager.clearTokens()
        userDao.deleteAllUsers()
    }
}

// Similar implementation for PostRepositoryImpl and CategoryRepositoryImpl...

// ═══════════════════════════════════════════════════════════════════════════════
//                           9. USE CASES (DOMAIN LAYER)
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.domain.usecase

import com.example.myapp.domain.model.User
import com.example.myapp.domain.repository.UserRepository
import com.example.myapp.utils.Resource
import javax.inject.Inject

class LoginUseCase @Inject constructor(
    private val userRepository: UserRepository
) {
    suspend operator fun invoke(email: String, password: String): Resource<User> {
        // Add business logic validation here if needed
        if (email.isBlank()) {
            return Resource.Error("Email cannot be empty")
        }
        
        if (password.isBlank()) {
            return Resource.Error("Password cannot be empty")
        }
        
        if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            return Resource.Error("Invalid email format")
        }
        
        return userRepository.login(email, password)
    }
}

class RegisterUseCase @Inject constructor(
    private val userRepository: UserRepository
) {
    suspend operator fun invoke(
        email: String,
        username: String,
        firstName: String,
        lastName: String,
        password: String
    ): Resource<User> {
        // Validation
        when {
            email.isBlank() -> return Resource.Error("Email is required")
            username.isBlank() -> return Resource.Error("Username is required")
            firstName.isBlank() -> return Resource.Error("First name is required")
            lastName.isBlank() -> return Resource.Error("Last name is required")
            password.length < 8 -> return Resource.Error("Password must be at least 8 characters")
            !android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches() -> {
                return Resource.Error("Invalid email format")
            }
            !isValidUsername(username) -> {
                return Resource.Error("Username can only contain letters, numbers, and underscores")
            }
        }
        
        return userRepository.register(email, username, firstName, lastName, password)
    }
    
    private fun isValidUsername(username: String): Boolean {
        return username.matches(Regex("^[a-zA-Z0-9_]{3,20}$"))
    }
}

class GetCurrentUserUseCase @Inject constructor(
    private val userRepository: UserRepository
) {
    suspend operator fun invoke() = userRepository.getCurrentUser()
}

class LogoutUseCase @Inject constructor(
    private val userRepository: UserRepository
) {
    suspend operator fun invoke() = userRepository.logout()
}

class GetPostsUseCase @Inject constructor(
    private val postRepository: PostRepository
) {
    suspend operator fun invoke(
        page: Int = 1,
        limit: Int = 20,
        categoryId: String? = null,
        authorId: String? = null,
        search: String? = null
    ) = postRepository.getPosts(page, limit, categoryId, authorId, search)
}

class GetPostByIdUseCase @Inject constructor(
    private val postRepository: PostRepository
) {
    suspend operator fun invoke(id: String): Resource<Post> {
        if (id.isBlank()) {
            return Resource.Error("Post ID cannot be empty")
        }
        
        return postRepository.getPostById(id)
    }
}

class CreatePostUseCase @Inject constructor(
    private val postRepository: PostRepository
) {
    suspend operator fun invoke(
        title: String,
        content: String,
        excerpt: String?,
        categoryId: String,
        tags: List<String>,
        isPublished: Boolean = false
    ): Resource<Post> {
        when {
            title.isBlank() -> return Resource.Error("Title is required")
            content.isBlank() -> return Resource.Error("Content is required")
            categoryId.isBlank() -> return Resource.Error("Category is required")
        }
        
        val request = CreatePostRequest(
            title = title.trim(),
            content = content.trim(),
            excerpt = excerpt?.trim(),
            categoryId = categoryId,
            tags = tags.map { it.trim() }.filter { it.isNotBlank() },
            isPublished = isPublished
        )
        
        return postRepository.createPost(request)
    }
}

class LikePostUseCase @Inject constructor(
    private val postRepository: PostRepository
) {
    suspend operator fun invoke(postId: String, isLiked: Boolean): Resource<Unit> {
        return if (isLiked) {
            postRepository.unlikePost(postId)
        } else {
            postRepository.likePost(postId)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           10. VIEWMODELS AND STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.myapp.domain.model.User
import com.example.myapp.domain.usecase.*
import com.example.myapp.utils.Resource
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class AuthViewModel @Inject constructor(
    private val loginUseCase: LoginUseCase,
    private val registerUseCase: RegisterUseCase,
    private val getCurrentUserUseCase: GetCurrentUserUseCase,
    private val logoutUseCase: LogoutUseCase
) : ViewModel() {
    
    private val _authState = MutableStateFlow(AuthState())
    val authState: StateFlow<AuthState> = _authState.asStateFlow()
    
    private val _isLoggedIn = MutableStateFlow(false)
    val isLoggedIn: StateFlow<Boolean> = _isLoggedIn.asStateFlow()
    
    init {
        checkAuthStatus()
    }
    
    fun login(email: String, password: String) {
        viewModelScope.launch {
            _authState.value = _authState.value.copy(isLoading = true, error = null)
            
            when (val result = loginUseCase(email, password)) {
                is Resource.Success -> {
                    _authState.value = _authState.value.copy(
                        isLoading = false,
                        user = result.data,
                        error = null
                    )
                    _isLoggedIn.value = true
                }
                is Resource.Error -> {
                    _authState.value = _authState.value.copy(
                        isLoading = false,
                        error = result.message
                    )
                }
                is Resource.Loading -> {
                    _authState.value = _authState.value.copy(isLoading = true)
                }
            }
        }
    }
    
    fun register(
        email: String,
        username: String,
        firstName: String,
        lastName: String,
        password: String
    ) {
        viewModelScope.launch {
            _authState.value = _authState.value.copy(isLoading = true, error = null)
            
            when (val result = registerUseCase(email, username, firstName, lastName, password)) {
                is Resource.Success -> {
                    _authState.value = _authState.value.copy(
                        isLoading = false,
                        user = result.data,
                        error = null
                    )
                    _isLoggedIn.value = true
                }
                is Resource.Error -> {
                    _authState.value = _authState.value.copy(
                        isLoading = false,
                        error = result.message
                    )
                }
                is Resource.Loading -> {
                    _authState.value = _authState.value.copy(isLoading = true)
                }
            }
        }
    }
    
    fun logout() {
        viewModelScope.launch {
            logoutUseCase()
            _authState.value = AuthState()
            _isLoggedIn.value = false
        }
    }
    
    fun clearError() {
        _authState.value = _authState.value.copy(error = null)
    }
    
    private fun checkAuthStatus() {
        viewModelScope.launch {
            getCurrentUserUseCase().collect { user ->
                _isLoggedIn.value = user != null
                _authState.value = _authState.value.copy(user = user)
            }
        }
    }
}

data class AuthState(
    val isLoading: Boolean = false,
    val user: User? = null,
    val error: String? = null
)

@HiltViewModel
class PostsViewModel @Inject constructor(
    private val getPostsUseCase: GetPostsUseCase,
    private val getPostByIdUseCase: GetPostByIdUseCase,
    private val likePostUseCase: LikePostUseCase,
    private val postRepository: PostRepository
) : ViewModel() {
    
    private val _postsState = MutableStateFlow(PostsState())
    val postsState: StateFlow<PostsState> = _postsState.asStateFlow()
    
    private val _selectedPost = MutableStateFlow<Post?>(null)
    val selectedPost: StateFlow<Post?> = _selectedPost.asStateFlow()
    
    private var currentPage = 1
    private var isLastPage = false
    
    init {
        loadPosts()
    }
    
    fun loadPosts(
        refresh: Boolean = false,
        categoryId: String? = null,
        search: String? = null
    ) {
        if (refresh) {
            currentPage = 1
            isLastPage = false
            _postsState.value = _postsState.value.copy(posts = emptyList())
        }
        
        if (_postsState.value.isLoading || isLastPage) return
        
        viewModelScope.launch {
            _postsState.value = _postsState.value.copy(isLoading = true, error = null)
            
            when (val result = getPostsUseCase(
                page = currentPage,
                categoryId = categoryId,
                search = search
            )) {
                is Resource.Success -> {
                    val newPosts = result.data ?: emptyList()
                    val currentPosts = if (refresh) emptyList() else _postsState.value.posts
                    
                    _postsState.value = _postsState.value.copy(
                        isLoading = false,
                        posts = currentPosts + newPosts,
                        error = null
                    )
                    
                    if (newPosts.isEmpty() || newPosts.size < 20) {
                        isLastPage = true
                    } else {
                        currentPage++
                    }
                }
                is Resource.Error -> {
                    _postsState.value = _postsState.value.copy(
                        isLoading = false,
                        error = result.message
                    )
                }
                is Resource.Loading -> {
                    _postsState.value = _postsState.value.copy(isLoading = true)
                }
            }
        }
    }
    
    fun getPostById(id: String) {
        viewModelScope.launch {
            when (val result = getPostByIdUseCase(id)) {
                is Resource.Success -> {
                    _selectedPost.value = result.data
                }
                is Resource.Error -> {
                    _postsState.value = _postsState.value.copy(error = result.message)
                }
                is Resource.Loading -> {
                    // Handle loading if needed
                }
            }
        }
    }
    
    fun toggleLike(post: Post) {
        viewModelScope.launch {
            when (likePostUseCase(post.id, post.isLiked)) {
                is Resource.Success -> {
                    // Update local state
                    val updatedPosts = _postsState.value.posts.map { p ->
                        if (p.id == post.id) {
                            p.copy(
                                isLiked = !p.isLiked,
                                likeCount = if (p.isLiked) p.likeCount - 1 else p.likeCount + 1
                            )
                        } else p
                    }
                    _postsState.value = _postsState.value.copy(posts = updatedPosts)
                    
                    // Update selected post if it's the same
                    if (_selectedPost.value?.id == post.id) {
                        _selectedPost.value = _selectedPost.value?.copy(
                            isLiked = !post.isLiked,
                            likeCount = if (post.isLiked) post.likeCount - 1 else post.likeCount + 1
                        )
                    }
                }
                is Resource.Error -> {
                    _postsState.value = _postsState.value.copy(error = result.message)
                }
                is Resource.Loading -> {
                    // Handle loading if needed
                }
            }
        }
    }
    
    fun clearError() {
        _postsState.value = _postsState.value.copy(error = null)
    }
    
    fun searchPosts(query: String) {
        loadPosts(refresh = true, search = query)
    }
}

data class PostsState(
    val isLoading: Boolean = false,
    val posts: List<Post> = emptyList(),
    val error: String? = null
)

// Add extension property to Post model for like state
val Post.isLiked: Boolean
    get() = false // This would be determined by checking if current user liked the post

// ═══════════════════════════════════════════════════════════════════════════════
//                           11. JETPACK COMPOSE UI
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.presentation.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LoginScreen(
    onNavigateToRegister: () -> Unit,
    onLoginSuccess: () -> Unit,
    viewModel: AuthViewModel = hiltViewModel()
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    
    val authState by viewModel.authState.collectAsStateWithLifecycle()
    val isLoggedIn by viewModel.isLoggedIn.collectAsStateWithLifecycle()
    
    // Navigate on successful login
    LaunchedEffect(isLoggedIn) {
        if (isLoggedIn) {
            onLoginSuccess()
        }
    }
    
    // Show error snackbar
    authState.error?.let { error ->
        LaunchedEffect(error) {
            // Show snackbar with error
            viewModel.clearError()
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Logo or App Title
        Text(
            text = "Welcome Back",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 32.dp)
        )
        
        // Email Field
        OutlinedTextField(
            value = email,
            onValueChange = { email = it },
            label = { Text("Email") },
            leadingIcon = {
                Icon(Icons.Default.Email, contentDescription = null)
            },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        )
        
        // Password Field
        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Password") },
            leadingIcon = {
                Icon(Icons.Default.Lock, contentDescription = null)
            },
            trailingIcon = {
                IconButton(onClick = { passwordVisible = !passwordVisible }) {
                    Icon(
                        if (passwordVisible) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                        contentDescription = if (passwordVisible) "Hide password" else "Show password"
                    )
                }
            },
            visualTransformation = if (passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp)
        )
        
        // Login Button
        Button(
            onClick = { viewModel.login(email, password) },
            enabled = !authState.isLoading && email.isNotBlank() && password.isNotBlank(),
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            if (authState.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    color = MaterialTheme.colorScheme.onPrimary
                )
            } else {
                Text("Login")
            }
        }
        
        // Register Link
        TextButton(
            onClick = onNavigateToRegister,
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text("Don't have an account? Register")
        }
        
        // Error Display
        authState.error?.let { error ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.Error,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onErrorContainer,
                        modifier = Modifier.padding(end = 8.dp)
                    )
                    Text(
                        text = error,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PostsScreen(
    onPostClick: (String) -> Unit,
    viewModel: PostsViewModel = hiltViewModel()
) {
    val postsState by viewModel.postsState.collectAsStateWithLifecycle()
    var searchQuery by remember { mutableStateOf("") }
    
    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Search Bar
        OutlinedTextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            label = { Text("Search posts...") },
            leadingIcon = {
                Icon(Icons.Default.Search, contentDescription = null)
            },
            trailingIcon = {
                if (searchQuery.isNotEmpty()) {
                    IconButton(onClick = { 
                        searchQuery = ""
                        viewModel.loadPosts(refresh = true)
                    }) {
                        Icon(Icons.Default.Clear, contentDescription = "Clear search")
                    }
                }
            },
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        )
        
        // Posts List
        when {
            postsState.isLoading && postsState.posts.isEmpty() -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }
            postsState.posts.isEmpty() && !postsState.isLoading -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No posts found",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            else -> {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(
                        items = postsState.posts,
                        key = { it.id }
                    ) { post ->
                        PostCard(
                            post = post,
                            onPostClick = { onPostClick(post.id) },
                            onLikeClick = { viewModel.toggleLike(post) }
                        )
                    }
                    
                    // Loading indicator for pagination
                    if (postsState.isLoading) {
                        item {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                CircularProgressIndicator()
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Search functionality
    LaunchedEffect(searchQuery) {
        if (searchQuery.length >= 3) {
            viewModel.searchPosts(searchQuery)
        } else if (searchQuery.isEmpty()) {
            viewModel.loadPosts(refresh = true)
        }
    }
    
    // Error handling
    postsState.error?.let { error ->
        LaunchedEffect(error) {
            // Show snackbar with error
            viewModel.clearError()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PostCard(
    post: Post,
    onPostClick: () -> Unit,
    onLikeClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        onClick = onPostClick,
        modifier = modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Post Title
            Text(
                text = post.title,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                maxLines = 2,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            // Post Excerpt
            Text(
                text = post.excerpt,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 3,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            // Author and Category Info
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    post.author?.let { author ->
                        Text(
                            text = "by ${author.fullName}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    
                    post.category?.let { category ->
                        Text(
                            text = category.name,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                
                // Reading time
                Text(
                    text = "${post.readingTimeMinutes} min read",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            // Action Row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Engagement Stats
                Row(
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            Icons.Default.Visibility,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = "${post.viewCount}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            Icons.Default.Comment,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = "${post.commentCount}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                
                // Like Button
                IconButton(onClick = onLikeClick) {
                    Icon(
                        if (post.isLiked) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                        contentDescription = if (post.isLiked) "Unlike" else "Like",
                        tint = if (post.isLiked) Color.Red else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PostDetailScreen(
    postId: String,
    onBackClick: () -> Unit,
    viewModel: PostsViewModel = hiltViewModel()
) {
    val selectedPost by viewModel.selectedPost.collectAsStateWithLifecycle()
    
    LaunchedEffect(postId) {
        viewModel.getPostById(postId)
    }
    
    selectedPost?.let { post ->
        Column(
            modifier = Modifier.fillMaxSize()
        ) {
            // Top App Bar
            TopAppBar(
                title = { Text("Post Details") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { /* Share */ }) {
                        Icon(Icons.Default.Share, contentDescription = "Share")
                    }
                }
            )
            
            // Post Content
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    // Post Title
                    Text(
                        text = post.title,
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold
                    )
                }
                
                item {
                    // Author and Meta Info
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            post.author?.let { author ->
                                Text(
                                    text = author.fullName,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Medium,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                            
                            Text(
                                text = "Published ${formatDate(post.publishedAt)}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        
                        Text(
                            text = "${post.readingTimeMinutes} min read",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                
                item {
                    Divider()
                }
                
                item {
                    // Post Content
                    Text(
                        text = post.content,
                        style = MaterialTheme.typography.bodyLarge,
                        lineHeight = MaterialTheme.typography.bodyLarge.lineHeight * 1.5
                    )
                }
                
                item {
                    // Tags
                    if (post.tags.isNotEmpty()) {
                        LazyRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(post.tags) { tag ->
                                AssistChip(
                                    onClick = { /* Navigate to tag */ },
                                    label = { Text(tag) }
                                )
                            }
                        }
                    }
                }
                
                item {
                    // Action Buttons
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        // Like Button
                        Button(
                            onClick = { viewModel.toggleLike(post) },
                            colors = if (post.isLiked) {
                                ButtonDefaults.buttonColors(
                                    containerColor = Color.Red,
                                    contentColor = Color.White
                                )
                            } else {
                                ButtonDefaults.buttonColors()
                            }
                        ) {
                            Icon(
                                if (post.isLiked) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                                contentDescription = null,
                                modifier = Modifier.padding(end = 8.dp)
                            )
                            Text("${post.likeCount}")
                        }
                        
                        // Comment Button
                        OutlinedButton(
                            onClick = { /* Navigate to comments */ }
                        ) {
                            Icon(
                                Icons.Default.Comment,
                                contentDescription = null,
                                modifier = Modifier.padding(end = 8.dp)
                            )
                            Text("${post.commentCount}")
                        }
                        
                        // Share Button
                        OutlinedButton(
                            onClick = { /* Share post */ }
                        ) {
                            Icon(
                                Icons.Default.Share,
                                contentDescription = null,
                                modifier = Modifier.padding(end = 8.dp)
                            )
                            Text("Share")
                        }
                    }
                }
            }
        }
    } ?: run {
        // Loading state
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            CircularProgressIndicator()
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           12. NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.presentation.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.myapp.presentation.ui.screens.*

@Composable
fun AppNavigation(
    navController: NavHostController = rememberNavController(),
    startDestination: String = "splash"
) {
    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        // Splash Screen
        composable("splash") {
            SplashScreen(
                onNavigateToAuth = {
                    navController.navigate("auth") {
                        popUpTo("splash") { inclusive = true }
                    }
                },
                onNavigateToMain = {
                    navController.navigate("main") {
                        popUpTo("splash") { inclusive = true }
                    }
                }
            )
        }
        
        // Authentication Flow
        composable("auth") {
            AuthScreen(
                onLoginSuccess = {
                    navController.navigate("main") {
                        popUpTo("auth") { inclusive = true }
                    }
                }
            )
        }
        
        // Main App Flow
        composable("main") {
            MainScreen(
                onLogout = {
                    navController.navigate("auth") {
                        popUpTo("main") { inclusive = true }
                    }
                }
            )
        }
        
        // Post Detail
        composable("post/{postId}") { backStackEntry ->
            val postId = backStackEntry.arguments?.getString("postId") ?: ""
            PostDetailScreen(
                postId = postId,
                onBackClick = { navController.popBackStack() }
            )
        }
        
        // Profile
        composable("profile") {
            ProfileScreen(
                onBackClick = { navController.popBackStack() }
            )
        }
        
        // Settings
        composable("settings") {
            SettingsScreen(
                onBackClick = { navController.popBackStack() }
            )
        }
    }
}

@Composable
fun MainScreen(
    onLogout: () -> Unit
) {
    val navController = rememberNavController()
    
    MainBottomNavigation(
        navController = navController,
        onLogout = onLogout
    )
}

@Composable
fun MainBottomNavigation(
    navController: NavHostController,
    onLogout: () -> Unit
) {
    Scaffold(
        bottomBar = {
            NavigationBar {
                val items = listOf(
                    BottomNavItem("posts", "Posts", Icons.Default.Article),
                    BottomNavItem("explore", "Explore", Icons.Default.Explore),
                    BottomNavItem("favorites", "Favorites", Icons.Default.Favorite),
                    BottomNavItem("profile", "Profile", Icons.Default.Person)
                )
                
                items.forEach { item ->
                    NavigationBarItem(
                        selected = false, // Implement current route logic
                        onClick = {
                            navController.navigate(item.route) {
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label) }
                    )
                }
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = "posts",
            modifier = Modifier.padding(paddingValues)
        ) {
            composable("posts") {
                PostsScreen(
                    onPostClick = { postId ->
                        navController.navigate("post/$postId")
                    }
                )
            }
            
            composable("explore") {
                ExploreScreen()
            }
            
            composable("favorites") {
                FavoritesScreen()
            }
            
            composable("profile") {
                ProfileScreen(
                    onLogout = onLogout,
                    onSettingsClick = {
                        navController.navigate("settings")
                    }
                )
            }
        }
    }
}

data class BottomNavItem(
    val route: String,
    val label: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector
)

// ═══════════════════════════════════════════════════════════════════════════════
//                           13. UTILS AND HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp.utils

sealed class Resource<T>(
    val data: T? = null,
    val message: String? = null
) {
    class Success<T>(data: T) : Resource<T>(data)
    class Error<T>(message: String, data: T? = null) : Resource<T>(data, message)
    class Loading<T>(data: T? = null) : Resource<T>(data)
}

object Constants {
    const val BASE_URL = "https://api.example.com/v1/"
    const val DATABASE_NAME = "app_database"
    
    // Shared Preferences
    const val PREFS_NAME = "app_prefs"
    const val KEY_ACCESS_TOKEN = "access_token"
    const val KEY_REFRESH_TOKEN = "refresh_token"
    const val KEY_USER_ID = "user_id"
    
    // Request Codes
    const val REQUEST_CODE_CAMERA = 1001
    const val REQUEST_CODE_GALLERY = 1002
    
    // Notification Channels
    const val NOTIFICATION_CHANNEL_GENERAL = "general"
    const val NOTIFICATION_CHANNEL_POSTS = "posts"
}

package com.example.myapp.utils

import android.content.Context
import android.content.SharedPreferences
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class TokenManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val prefs: SharedPreferences = context.getSharedPreferences(
        Constants.PREFS_NAME,
        Context.MODE_PRIVATE
    )
    
    fun saveTokens(accessToken: String, refreshToken: String) {
        prefs.edit()
            .putString(Constants.KEY_ACCESS_TOKEN, accessToken)
            .putString(Constants.KEY_REFRESH_TOKEN, refreshToken)
            .apply()
    }
    
    fun getAccessToken(): String? {
        return prefs.getString(Constants.KEY_ACCESS_TOKEN, null)
    }
    
    fun getRefreshToken(): String? {
        return prefs.getString(Constants.KEY_REFRESH_TOKEN, null)
    }
    
    fun clearTokens() {
        prefs.edit()
            .remove(Constants.KEY_ACCESS_TOKEN)
            .remove(Constants.KEY_REFRESH_TOKEN)
            .remove(Constants.KEY_USER_ID)
            .apply()
    }
    
    fun isLoggedIn(): Boolean {
        return getAccessToken() != null
    }
}

package com.example.myapp.utils

import java.text.SimpleDateFormat
import java.util.*

object DateUtils {
    private val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
    private val timeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())
    private val dateTimeFormat = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
    
    fun formatDate(date: Date?): String {
        return if (date != null) {
            dateFormat.format(date)
        } else {
            ""
        }
    }
    
    fun formatTime(date: Date?): String {
        return if (date != null) {
            timeFormat.format(date)
        } else {
            ""
        }
    }
    
    fun formatDateTime(date: Date?): String {
        return if (date != null) {
            dateTimeFormat.format(date)
        } else {
            ""
        }
    }
    
    fun getTimeAgo(date: Date?): String {
        if (date == null) return ""
        
        val now = Date()
        val diff = now.time - date.time
        
        val seconds = diff / 1000
        val minutes = seconds / 60
        val hours = minutes / 60
        val days = hours / 24
        
        return when {
            seconds < 60 -> "Just now"
            minutes < 60 -> "${minutes}m ago"
            hours < 24 -> "${hours}h ago"
            days < 7 -> "${days}d ago"
            else -> formatDate(date)
        }
    }
}

package com.example.myapp.utils

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast

object AppUtils {
    
    fun showToast(context: Context, message: String, length: Int = Toast.LENGTH_SHORT) {
        Toast.makeText(context, message, length).show()
    }
    
    fun openUrl(context: Context, url: String) {
        try {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            context.startActivity(intent)
        } catch (e: Exception) {
            showToast(context, "Cannot open URL")
        }
    }
    
    fun shareText(context: Context, text: String, title: String = "Share") {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, text)
        }
        context.startActivity(Intent.createChooser(intent, title))
    }
    
    fun openEmail(context: Context, email: String, subject: String = "", body: String = "") {
        try {
            val intent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("mailto:$email")
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body)
            }
            context.startActivity(intent)
        } catch (e: Exception) {
            showToast(context, "No email app found")
        }
    }
}

package com.example.myapp.utils

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onStart

inline fun <ResultType, RequestType> networkBoundResource(
    crossinline query: () -> Flow<ResultType>,
    crossinline fetch: suspend () -> RequestType,
    crossinline saveFetchResult: suspend (RequestType) -> Unit,
    crossinline shouldFetch: (ResultType) -> Boolean = { true }
) = flow<Resource<ResultType>> {
    val data = query().first()
    val flow = if (shouldFetch(data)) {
        emit(Resource.Loading(data))
        try {
            saveFetchResult(fetch())
            query().map { Resource.Success(it) }
        } catch (throwable: Throwable) {
            query().map { Resource.Error(throwable.message ?: "Unknown error", it) }
        }
    } else {
        query().map { Resource.Success(it) }
    }
    emitAll(flow)
}.catch { emit(Resource.Error(it.message ?: "Unknown error")) }

// Extension functions
fun String.isValidEmail(): Boolean {
    return android.util.Patterns.EMAIL_ADDRESS.matcher(this).matches()
}

fun String.capitalizeWords(): String {
    return split(" ").joinToString(" ") { word ->
        word.lowercase().replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           14. TESTING
// ═══════════════════════════════════════════════════════════════════════════════

package com.example.myapp

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import androidx.test.core.app.ApplicationProvider
import com.example.myapp.data.local.AppDatabase
import com.example.myapp.data.local.UserDao
import com.example.myapp.data.local.entity.UserEntity
import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*

@RunWith(AndroidJUnit4::class)
class UserDaoTest {
    
    @get:Rule
    val instantExecutorRule = InstantTaskExecutorRule()
    
    private lateinit var database: AppDatabase
    private lateinit var userDao: UserDao
    
    @Before
    fun setup() {
        database = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            AppDatabase::class.java
        ).allowMainThreadQueries().build()
        
        userDao = database.userDao()
    }
    
    @After
    fun teardown() {
        database.close()
    }
    
    @Test
    fun insertAndGetUser() = runTest {
        // Given
        val user = UserEntity(
            id = "1",
            email = "test@example.com",
            username = "testuser",
            firstName = "Test",
            lastName = "User",
            avatarUrl = null,
            isActive = true,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis()
        )
        
        // When
        userDao.insertUser(user)
        val retrievedUser = userDao.getUserById("1")
        
        // Then
        assertNotNull(retrievedUser)
        assertEquals(user.email, retrievedUser?.email)
        assertEquals(user.username, retrievedUser?.username)
    }
    
    @Test
    fun getAllUsers() = runTest {
        // Given
        val users = listOf(
            UserEntity("1", "user1@test.com", "user1", "User", "One", null, true, 0L, 0L),
            UserEntity("2", "user2@test.com", "user2", "User", "Two", null, true, 0L, 0L)
        )
        
        // When
        userDao.insertUsers(users)
        val allUsers = userDao.getAllUsers().first()
        
        // Then
        assertEquals(2, allUsers.size)
        assertTrue(allUsers.any { it.username == "user1" })
        assertTrue(allUsers.any { it.username == "user2" })
    }
    
    @Test
    fun searchUsers() = runTest {
        // Given
        val users = listOf(
            UserEntity("1", "john@test.com", "john_doe", "John", "Doe", null, true, 0L, 0L),
            UserEntity("2", "jane@test.com", "jane_smith", "Jane", "Smith", null, true, 0L, 0L),
            UserEntity("3", "bob@test.com", "bob_johnson", "Bob", "Johnson", null, true, 0L, 0L)
        )
        
        // When
        userDao.insertUsers(users)
        val searchResults = userDao.searchUsers("john").first()
        
        // Then
        assertEquals(2, searchResults.size) // John Doe and Bob Johnson
        assertTrue(searchResults.any { it.firstName == "John" })
        assertTrue(searchResults.any { it.lastName == "Johnson" })
    }
}

// Unit Tests for ViewModels
package com.example.myapp.presentation.viewmodel

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import com.example.myapp.domain.model.User
import com.example.myapp.domain.usecase.*
import com.example.myapp.utils.Resource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.whenever
import java.util.*
import org.junit.Assert.*

@OptIn(ExperimentalCoroutinesApi::class)
class AuthViewModelTest {
    
    @get:Rule
    val instantExecutorRule = InstantTaskExecutorRule()
    
    private val testDispatcher = UnconfinedTestDispatcher()
    
    @Mock
    private lateinit var loginUseCase: LoginUseCase
    
    @Mock
    private lateinit var registerUseCase: RegisterUseCase
    
    @Mock
    private lateinit var getCurrentUserUseCase: GetCurrentUserUseCase
    
    @Mock
    private lateinit var logoutUseCase: LogoutUseCase
    
    private lateinit var viewModel: AuthViewModel
    
    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        Dispatchers.setMain(testDispatcher)
        
        whenever(getCurrentUserUseCase()).thenReturn(flowOf(null))
        
        viewModel = AuthViewModel(
            loginUseCase,
            registerUseCase,
            getCurrentUserUseCase,
            logoutUseCase
        )
    }
    
    @After
    fun teardown() {
        Dispatchers.resetMain()
    }
    
    @Test
    fun `login success updates state correctly    @GET("posts/{id}")
    suspend fun getPostById(@Path("id") id: String): Response<ApiResponse<PostDto>>
    
    @POST("posts")
    suspend fun createPost(@Body request: CreatePostRequest): Response<ApiResponse<PostDto>>
    
    @PUT("posts/{id}")
    suspend fun updatePost(
        @Path("id") id: String,
        @Body request: UpdatePostRequest
    ): Response<ApiResponse<PostDto>>
    
    @DELETE("posts/{id}")
    suspend fun deletePost(@Path("id") id: String): Response<Unit>
    
    @POST("posts/{id}/like")
    suspend fun likePost(@Path("id") id: String): Response<Unit>
    
    @DELETE("posts/{id}/like")
    suspend fun unlikePost(@Path("id") id: String): Response<Unit>
    
    @POST("posts/{id}/view")
    suspend fun incrementPostView(@Path("id") id: String): Response<Unit>
    
    // Categories
    @GET("categories")
    suspend fun getCategories(): Response<ApiResponse<List<CategoryDto>>>
    
    @GET("categories/{id}")
    suspend fun getCategoryById(@Path("id") id: String): Response<ApiResponse<CategoryDto>>
    
    // Comments
    @GET("posts/{postId}/comments")
    suspend fun getComments(
        @Path("postId") postId: String,
        @Query("page") page: Int = 1,
        @Query("limit") limit: Int = 20
    ): Response<PaginatedResponse<CommentDto>>
    
    @POST("posts/{postId}/comments")
    suspend fun createComment(
        @Path("postId") postId: String,
        @Body request: CreateCommentRequest
    ): Response<ApiResponse<CommentDto>>
    
    @PUT("comments/{id}")
    suspend fun updateComment(
        @Path("id") id: String,
        @Body request: UpdateCommentRequest
    ): Response<ApiResponse<CommentDto>>
    
    @DELETE("comments/{id}")
    suspend fun deleteComment(@Path("id") id: String): Response<Unit>
}

// API Request/Response Models
data class LoginRequest(
    val email: String,
    val password: String
)

data class RegisterRequest(
    val email: String,
    val username: String,
    val firstName: String,
    val lastName: String,
    val password: String
)

data class RefreshTokenRequest(
    val refreshToken: String
)

data class AuthResponse(
    val accessToken: String,
    val refreshToken: String,
    val expiresIn: Long,
    val user: UserDto
)

data class UpdateUserRequest(
    val firstName: String?,
    val lastName: String?,
    val username: String?,
    val bio: String?
)

data class CreatePostRequest(
    val title: String,
    val content: String,
    val excerpt: String?,
    val categoryId: String,
    val tags: List<String>,
    val isPublished: Boolean = false
)

data class UpdatePostRequest(
    val title: String?,
    val content: String?,
    val excerpt: String?,
    val categoryId: String?,
    val tags: List<String>?,
    val isPublished: Boolean?
)

data class CreateCommentRequest(
    val content: String,
    val parentId: String? = null
)

data class UpdateCommentRequest(
    val content: String
)

// Generic API Response Wrappers
data class ApiResponse<T>(
    val success: Boolean,
    val data: T?,
    val message: String?,
    val errors: List<String>?
)

data class PaginatedResponse<T>(
    val success: Boolean,
    val data: List<T>,
    val pagination: PaginationInfo,
    val message: String?,
    val errors: List<String>?
)

data class PaginationInfo(
    val currentPage: Int,
    val totalPages: Int,
    val totalItems: Int,
    val itemsPerPage: Int,
    val hasNext: Boolean,
    val hasPrevious: Boolean
)