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