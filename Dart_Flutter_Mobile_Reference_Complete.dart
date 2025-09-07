// DART FLUTTER MOBILE DEVELOPMENT - Comprehensive Reference - by Richard Rembert
// Dart with Flutter enables building high-performance, cross-platform mobile applications
// for iOS and Android with a single codebase using modern reactive programming patterns

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND PROJECT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/*
FLUTTER DEVELOPMENT SETUP:

1. Flutter Installation:
   - Download Flutter SDK from flutter.dev
   - Set PATH environment variable
   - Run `flutter doctor` to verify installation
   - Install Android Studio / Xcode for platform SDKs

2. Project Creation:
   flutter create my_app
   cd my_app
   flutter run

3. Essential Dependencies (pubspec.yaml):
   dependencies:
     flutter:
       sdk: flutter
     http: ^1.1.0              # HTTP client
     provider: ^6.1.1          # State management
     shared_preferences: ^2.2.2 # Local storage
     sqflite: ^2.3.0           # SQLite database
     path_provider: ^2.1.1     # File system paths
     image_picker: ^1.0.4      # Camera/gallery access
     permission_handler: ^11.0.1 # Runtime permissions
     firebase_core: ^2.24.2    # Firebase integration
     firebase_auth: ^4.15.3    # Authentication
     cloud_firestore: ^4.13.6  # Firestore database
     cached_network_image: ^3.3.0 # Image caching
     flutter_secure_storage: ^9.0.0 # Secure storage
     connectivity_plus: ^5.0.1 # Network connectivity
     
   dev_dependencies:
     flutter_test:
       sdk: flutter
     flutter_lints: ^3.0.1     # Linting rules
     mockito: ^5.4.2           # Mocking for tests
     integration_test:
       sdk: flutter

4. Project Structure:
   my_app/
   ├── lib/
   │   ├── main.dart
   │   ├── app/
   │   │   ├── app.dart
   │   │   ├── routes.dart
   │   │   └── theme.dart
   │   ├── core/
   │   │   ├── constants/
   │   │   ├── errors/
   │   │   ├── network/
   │   │   ├── utils/
   │   │   └── services/
   │   ├── data/
   │   │   ├── datasources/
   │   │   ├── models/
   │   │   └── repositories/
   │   ├── domain/
   │   │   ├── entities/
   │   │   ├── repositories/
   │   │   └── usecases/
   │   ├── presentation/
   │   │   ├── pages/
   │   │   ├── widgets/
   │   │   ├── providers/
   │   │   └── bloc/
   │   └── features/
   │       ├── authentication/
   │       ├── posts/
   │       ├── profile/
   │       └── settings/
   ├── test/
   ├── integration_test/
   ├── android/
   ├── ios/
   ├── assets/
   │   ├── images/
   │   ├── fonts/
   │   └── translations/
   └── pubspec.yaml

5. Configuration Files:
   - pubspec.yaml (Dependencies and assets)
   - analysis_options.yaml (Linting configuration)
   - flutter_launcher_icons.yaml (App icons)
   - android/app/build.gradle (Android configuration)
   - ios/Runner/Info.plist (iOS configuration)
*/

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'dart:convert';
import 'dart:async';
import 'dart:io';

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. APP ENTRY POINT AND CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize services
  await _initializeServices();
  
  // Set system UI overlay style
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.white,
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );
  
  // Set preferred orientations
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  
  runApp(MyApp());
}

Future<void> _initializeServices() async {
  // Initialize local storage
  await SharedPreferences.getInstance();
  
  // Initialize database
  await DatabaseService.instance.database;
  
  // Initialize Firebase (if using)
  // await Firebase.initializeApp();
  
  // Initialize analytics
  AnalyticsService.instance.initialize();
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppStateProvider()),
        ChangeNotifierProvider(create: (_) => AuthProvider()),
        ChangeNotifierProvider(create: (_) => ThemeProvider()),
        ChangeNotifierProvider(create: (_) => ConnectivityProvider()),
      ],
      child: Consumer<ThemeProvider>(
        builder: (context, themeProvider, child) {
          return MaterialApp(
            title: 'My Flutter App',
            debugShowCheckedModeBanner: false,
            theme: AppTheme.lightTheme,
            darkTheme: AppTheme.darkTheme,
            themeMode: themeProvider.themeMode,
            initialRoute: AppRoutes.splash,
            onGenerateRoute: AppRoutes.generateRoute,
            navigatorKey: NavigationService.navigatorKey,
            builder: (context, child) {
              return MediaQuery(
                data: MediaQuery.of(context).copyWith(
                  textScaleFactor: 1.0, // Prevent text scaling
                ),
                child: child!,
              );
            },
          );
        },
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. APP STATE AND PROVIDERS
// ═══════════════════════════════════════════════════════════════════════════════

class AppStateProvider extends ChangeNotifier {
  bool _isLoading = false;
  String? _errorMessage;
  bool _isOnline = true;
  
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  bool get isOnline => _isOnline;
  
  void setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }
  
  void setError(String? error) {
    _errorMessage = error;
    notifyListeners();
  }
  
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }
  
  void setOnlineStatus(bool online) {
    _isOnline = online;
    notifyListeners();
  }
}

class ThemeProvider extends ChangeNotifier {
  ThemeMode _themeMode = ThemeMode.system;
  
  ThemeMode get themeMode => _themeMode;
  
  bool get isDarkMode {
    if (_themeMode == ThemeMode.system) {
      return WidgetsBinding.instance.platformDispatcher.platformBrightness == Brightness.dark;
    }
    return _themeMode == ThemeMode.dark;
  }
  
  void setThemeMode(ThemeMode mode) {
    _themeMode = mode;
    _saveThemeMode(mode);
    notifyListeners();
  }
  
  void toggleTheme() {
    _themeMode = _themeMode == ThemeMode.light ? ThemeMode.dark : ThemeMode.light;
    _saveThemeMode(_themeMode);
    notifyListeners();
  }
  
  Future<void> _saveThemeMode(ThemeMode mode) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('theme_mode', mode.toString());
  }
  
  Future<void> loadThemeMode() async {
    final prefs = await SharedPreferences.getInstance();
    final themeModeString = prefs.getString('theme_mode');
    if (themeModeString != null) {
      _themeMode = ThemeMode.values.firstWhere(
        (mode) => mode.toString() == themeModeString,
        orElse: () => ThemeMode.system,
      );
      notifyListeners();
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. MODELS AND DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// Base Entity
abstract class Entity {
  final String id;
  final DateTime createdAt;
  final DateTime updatedAt;
  
  Entity({
    required this.id,
    required this.createdAt,
    required this.updatedAt,
  });
}

// User Model
class User extends Entity {
  final String email;
  final String username;
  final String firstName;
  final String lastName;
  final String? avatarUrl;
  final String? bio;
  final bool isActive;
  
  User({
    required String id,
    required this.email,
    required this.username,
    required this.firstName,
    required this.lastName,
    this.avatarUrl,
    this.bio,
    required this.isActive,
    required DateTime createdAt,
    required DateTime updatedAt,
  }) : super(id: id, createdAt: createdAt, updatedAt: updatedAt);
  
  String get fullName => '$firstName $lastName';
  
  String get initials => '${firstName[0]}${lastName[0]}'.toUpperCase();
  
  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      email: json['email'],
      username: json['username'],
      firstName: json['first_name'],
      lastName: json['last_name'],
      avatarUrl: json['avatar_url'],
      bio: json['bio'],
      isActive: json['is_active'] ?? true,
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'username': username,
      'first_name': firstName,
      'last_name': lastName,
      'avatar_url': avatarUrl,
      'bio': bio,
      'is_active': isActive,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }
  
  User copyWith({
    String? email,
    String? username,
    String? firstName,
    String? lastName,
    String? avatarUrl,
    String? bio,
    bool? isActive,
  }) {
    return User(
      id: id,
      email: email ?? this.email,
      username: username ?? this.username,
      firstName: firstName ?? this.firstName,
      lastName: lastName ?? this.lastName,
      avatarUrl: avatarUrl ?? this.avatarUrl,
      bio: bio ?? this.bio,
      isActive: isActive ?? this.isActive,
      createdAt: createdAt,
      updatedAt: DateTime.now(),
    );
  }
}

// Post Model
class Post extends Entity {
  final String title;
  final String content;
  final String excerpt;
  final String? imageUrl;
  final String authorId;
  final User? author;
  final String categoryId;
  final Category? category;
  final List<String> tags;
  final bool isPublished;
  final int viewCount;
  final int likeCount;
  final int commentCount;
  final bool isLiked;
  final DateTime? publishedAt;
  
  Post({
    required String id,
    required this.title,
    required this.content,
    required this.excerpt,
    this.imageUrl,
    required this.authorId,
    this.author,
    required this.categoryId,
    this.category,
    required this.tags,
    required this.isPublished,
    required this.viewCount,
    required this.likeCount,
    required this.commentCount,
    required this.isLiked,
    this.publishedAt,
    required DateTime createdAt,
    required DateTime updatedAt,
  }) : super(id: id, createdAt: createdAt, updatedAt: updatedAt);
  
  int get readingTimeMinutes {
    const wordsPerMinute = 200;
    final wordCount = content.split(' ').length;
    return (wordCount / wordsPerMinute).ceil().clamp(1, 999);
  }
  
  bool get isPublic => isPublished && publishedAt != null;
  
  String get formattedPublishDate {
    if (publishedAt == null) return '';
    return DateUtils.formatDate(publishedAt!);
  }
  
  factory Post.fromJson(Map<String, dynamic> json) {
    return Post(
      id: json['id'],
      title: json['title'],
      content: json['content'],
      excerpt: json['excerpt'],
      imageUrl: json['image_url'],
      authorId: json['author_id'],
      author: json['author'] != null ? User.fromJson(json['author']) : null,
      categoryId: json['category_id'],
      category: json['category'] != null ? Category.fromJson(json['category']) : null,
      tags: List<String>.from(json['tags'] ?? []),
      isPublished: json['is_published'] ?? false,
      viewCount: json['view_count'] ?? 0,
      likeCount: json['like_count'] ?? 0,
      commentCount: json['comment_count'] ?? 0,
      isLiked: json['is_liked'] ?? false,
      publishedAt: json['published_at'] != null ? DateTime.parse(json['published_at']) : null,
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'content': content,
      'excerpt': excerpt,
      'image_url': imageUrl,
      'author_id': authorId,
      'category_id': categoryId,
      'tags': tags,
      'is_published': isPublished,
      'view_count': viewCount,
      'like_count': likeCount,
      'comment_count': commentCount,
      'is_liked': isLiked,
      'published_at': publishedAt?.toIso8601String(),
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }
}

// Category Model
class Category extends Entity {
  final String name;
  final String slug;
  final String description;
  final String color;
  final int postCount;
  
  Category({
    required String id,
    required this.name,
    required this.slug,
    required this.description,
    required this.color,
    required this.postCount,
    required DateTime createdAt,
    required DateTime updatedAt,
  }) : super(id: id, createdAt: createdAt, updatedAt: updatedAt);
  
  Color get colorValue => ColorUtils.fromHex(color);
  
  factory Category.fromJson(Map<String, dynamic> json) {
    return Category(
      id: json['id'],
      name: json['name'],
      slug: json['slug'],
      description: json['description'],
      color: json['color'],
      postCount: json['post_count'] ?? 0,
      createdAt: DateTime.parse(json['created_at']),
      updatedAt: DateTime.parse(json['updated_at']),
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'slug': slug,
      'description': description,
      'color': color,
      'post_count': postCount,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }
}

// API Response Models
class ApiResponse<T> {
  final bool success;
  final T? data;
  final String? message;
  final List<String>? errors;
  
  ApiResponse({
    required this.success,
    this.data,
    this.message,
    this.errors,
  });
  
  factory ApiResponse.fromJson(Map<String, dynamic> json, T Function(dynamic)? fromJson) {
    return ApiResponse<T>(
      success: json['success'] ?? false,
      data: json['data'] != null && fromJson != null ? fromJson(json['data']) : json['data'],
      message: json['message'],
      errors: json['errors'] != null ? List<String>.from(json['errors']) : null,
    );
  }
}

class PaginatedResponse<T> {
  final bool success;
  final List<T> data;
  final PaginationInfo pagination;
  final String? message;
  final List<String>? errors;
  
  PaginatedResponse({
    required this.success,
    required this.data,
    required this.pagination,
    this.message,
    this.errors,
  });
  
  factory PaginatedResponse.fromJson(
    Map<String, dynamic> json,
    T Function(Map<String, dynamic>) fromJson,
  ) {
    return PaginatedResponse<T>(
      success: json['success'] ?? false,
      data: json['data'] != null 
        ? (json['data'] as List).map((item) => fromJson(item)).toList()
        : [],
      pagination: PaginationInfo.fromJson(json['pagination'] ?? {}),
      message: json['message'],
      errors: json['errors'] != null ? List<String>.from(json['errors']) : null,
    );
  }
}

class PaginationInfo {
  final int currentPage;
  final int totalPages;
  final int totalItems;
  final int itemsPerPage;
  final bool hasNext;
  final bool hasPrevious;
  
  PaginationInfo({
    required this.currentPage,
    required this.totalPages,
    required this.totalItems,
    required this.itemsPerPage,
    required this.hasNext,
    required this.hasPrevious,
  });
  
  factory PaginationInfo.fromJson(Map<String, dynamic> json) {
    return PaginationInfo(
      currentPage: json['current_page'] ?? 1,
      totalPages: json['total_pages'] ?? 1,
      totalItems: json['total_items'] ?? 0,
      itemsPerPage: json['items_per_page'] ?? 20,
      hasNext: json['has_next'] ?? false,
      hasPrevious: json['has_previous'] ?? false,
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. NETWORKING LAYER
// ═══════════════════════════════════════════════════════════════════════════════

class ApiService {
  static const String baseUrl = 'https://api.example.com/v1';
  static const Duration timeout = Duration(seconds: 30);
  
  final http.Client _client;
  final SecureStorageService _storage;
  
  ApiService() : _client = http.Client(), _storage = SecureStorageService.instance;
  
  // Generic GET request
  Future<T> get<T>(
    String endpoint, {
    Map<String, String>? queryParams,
    T Function(Map<String, dynamic>)? fromJson,
  }) async {
    try {
      final uri = _buildUri(endpoint, queryParams);
      final headers = await _buildHeaders();
      
      final response = await _client.get(uri, headers: headers).timeout(timeout);
      
      return _handleResponse<T>(response, fromJson);
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  // Generic POST request
  Future<T> post<T>(
    String endpoint, {
    Map<String, dynamic>? body,
    T Function(Map<String, dynamic>)? fromJson,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final headers = await _buildHeaders();
      
      final response = await _client.post(
        uri,
        headers: headers,
        body: body != null ? json.encode(body) : null,
      ).timeout(timeout);
      
      return _handleResponse<T>(response, fromJson);
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  // Generic PUT request
  Future<T> put<T>(
    String endpoint, {
    Map<String, dynamic>? body,
    T Function(Map<String, dynamic>)? fromJson,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final headers = await _buildHeaders();
      
      final response = await _client.put(
        uri,
        headers: headers,
        body: body != null ? json.encode(body) : null,
      ).timeout(timeout);
      
      return _handleResponse<T>(response, fromJson);
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  // Generic DELETE request
  Future<T> delete<T>(
    String endpoint, {
    T Function(Map<String, dynamic>)? fromJson,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final headers = await _buildHeaders();
      
      final response = await _client.delete(uri, headers: headers).timeout(timeout);
      
      return _handleResponse<T>(response, fromJson);
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  // File upload
  Future<T> uploadFile<T>(
    String endpoint,
    File file, {
    Map<String, String>? fields,
    T Function(Map<String, dynamic>)? fromJson,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final headers = await _buildHeaders(includeContentType: false);
      
      final request = http.MultipartRequest('POST', uri);
      request.headers.addAll(headers);
      
      if (fields != null) {
        request.fields.addAll(fields);
      }
      
      request.files.add(
        await http.MultipartFile.fromPath('file', file.path),
      );
      
      final streamedResponse = await request.send().timeout(timeout);
      final response = await http.Response.fromStream(streamedResponse);
      
      return _handleResponse<T>(response, fromJson);
    } catch (e) {
      throw _handleError(e);
    }
  }
  
  Uri _buildUri(String endpoint, [Map<String, String>? queryParams]) {
    final uri = Uri.parse('$baseUrl$endpoint');
    if (queryParams != null && queryParams.isNotEmpty) {
      return uri.replace(queryParameters: queryParams);
    }
    return uri;
  }
  
  Future<Map<String, String>> _buildHeaders({bool includeContentType = true}) async {
    final headers = <String, String>{};
    
    if (includeContentType) {
      headers['Content-Type'] = 'application/json';
    }
    
    final token = await _storage.getAccessToken();
    if (token != null) {
      headers['Authorization'] = 'Bearer $token';
    }
    
    return headers;
  }
  
  T _handleResponse<T>(http.Response response, T Function(Map<String, dynamic>)? fromJson) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) {
        return true as T;
      }
      
      final jsonData = json.decode(response.body);
      
      if (fromJson != null) {
        return fromJson(jsonData);
      }
      
      return jsonData as T;
    } else {
      throw ApiException(
        statusCode: response.statusCode,
        message: _parseErrorMessage(response.body),
      );
    }
  }
  
  String _parseErrorMessage(String responseBody) {
    try {
      final jsonData = json.decode(responseBody);
      return jsonData['message'] ?? 'Unknown error occurred';
    } catch (e) {
      return 'Unknown error occurred';
    }
  }
  
  AppException _handleError(dynamic error) {
    if (error is SocketException) {
      return NetworkException('No internet connection');
    } else if (error is TimeoutException) {
      return NetworkException('Request timeout');
    } else if (error is ApiException) {
      return error;
    } else {
      return AppException('Unknown error occurred: ${error.toString()}');
    }
  }
  
  void dispose() {
    _client.close();
  }
}

// API Endpoints
class ApiEndpoints {
  // Authentication
  static const String login = '/auth/login';
  static const String register = '/auth/register';
  static const String refreshToken = '/auth/refresh';
  static const String logout = '/auth/logout';
  static const String currentUser = '/auth/me';
  
  // Posts
  static const String posts = '/posts';
  static String post(String id) => '/posts/$id';
  static String likePost(String id) => '/posts/$id/like';
  static String unlikePost(String id) => '/posts/$id/unlike';
  
  // Categories
  static const String categories = '/categories';
  static String category(String id) => '/categories/$id';
  
  // Users
  static const String users = '/users';
  static String user(String id) => '/users/$id';
}

// Custom Exceptions
abstract class AppException implements Exception {
  final String message;
  AppException(this.message);
  
  @override
  String toString() => message;
}

class NetworkException extends AppException {
  NetworkException(String message) : super(message);
}

class ApiException extends AppException {
  final int statusCode;
  
  ApiException({required this.statusCode, required String message}) : super(message);
  
  bool get isUnauthorized => statusCode == 401;
  bool get isForbidden => statusCode == 403;
  bool get isNotFound => statusCode == 404;
  bool get isServerError => statusCode >= 500;
}

class ValidationException extends AppException {
  final Map<String, List<String>> errors;
  
  ValidationException({required String message, required this.errors}) : super(message);
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. LOCAL DATABASE (SQLite)
// ═══════════════════════════════════════════════════════════════════════════════

class DatabaseService {
  static final DatabaseService instance = DatabaseService._internal();
  DatabaseService._internal();
  
  static Database? _database;
  
  Future<Database> get database async {
    _database ??= await _initDatabase();
    return _database!;
  }
  
  Future<Database> _initDatabase() async {
    final databasesPath = await getDatabasesPath();
    final path = join(databasesPath, 'app_database.db');
    
    return await openDatabase(
      path,
      version: 1,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }
  
  Future<void> _onCreate(Database db, int version) async {
    // Users table
    await db.execute('''
      CREATE TABLE users (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        username TEXT NOT NULL UNIQUE,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        avatar_url TEXT,
        bio TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    ''');
    
    // Posts table
    await db.execute('''
      CREATE TABLE posts (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        excerpt TEXT NOT NULL,
        image_url TEXT,
        author_id TEXT NOT NULL,
        category_id TEXT NOT NULL,
        tags TEXT,
        is_published INTEGER NOT NULL DEFAULT 0,
        view_count INTEGER NOT NULL DEFAULT 0,
        like_count INTEGER NOT NULL DEFAULT 0,
        comment_count INTEGER NOT NULL DEFAULT 0,
        is_liked INTEGER NOT NULL DEFAULT 0,
        published_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (author_id) REFERENCES users (id),
        FOREIGN KEY (category_id) REFERENCES categories (id)
      )
    ''');
    
    // Categories table
    await db.execute('''
      CREATE TABLE categories (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        slug TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL,
        color TEXT NOT NULL,
        post_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    ''');
    
    // Create indexes
    await db.execute('CREATE INDEX idx_posts_author ON posts(author_id)');
    await db.execute('CREATE INDEX idx_posts_category ON posts(category_id)');
    await db.execute('CREATE INDEX idx_posts_published ON posts(is_published, published_at)');
  }
  
  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    // Handle database schema upgrades
    if (oldVersion < 2) {
      // Add new columns or tables for version 2
    }
  }
  
  // User operations
  Future<void> insertUser(User user) async {
    final db = await database;
    await db.insert(
      'users',
      user.toJson(),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }
  
  Future<User?> getUser(String id) async {
    final db = await database;
    final maps = await db.query(
      'users',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    
    if (maps.isNotEmpty) {
      return User.fromJson(maps.first);
    }
    return null;
  }
  
  Future<List<User>> getAllUsers() async {
    final db = await database;
    final maps = await db.query('users', orderBy: 'created_at DESC');
    return maps.map((map) => User.fromJson(map)).toList();
  }
  
  Future<void> deleteUser(String id) async {
    final db = await database;
    await db.delete('users', where: 'id = ?', whereArgs: [id]);
  }
  
  // Post operations
  Future<void> insertPost(Post post) async {
    final db = await database;
    final postData = post.toJson();
    postData['tags'] = post.tags.join(','); // Convert list to string
    await db.insert(
      'posts',
      postData,
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }
  
  Future<void> insertPosts(List<Post> posts) async {
    final db = await database;
    final batch = db.batch();
    
    for (final post in posts) {
      final postData = post.toJson();
      postData['tags'] = post.tags.join(',');
      batch.insert('posts', postData, conflictAlgorithm: ConflictAlgorithm.replace);
    }
    
    await batch.commit();
  }
  
  Future<Post?> getPost(String id) async {
    final db = await database;
    final maps = await db.query(
      'posts',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    
    if (maps.isNotEmpty) {
      final postData = Map<String, dynamic>.from(maps.first);
      postData['tags'] = (postData['tags'] as String).split(',').where((tag) => tag.isNotEmpty).toList();
      return Post.fromJson(postData);
    }
    return null;
  }
  
  Future<List<Post>> getPosts({
    int limit = 20,
    int offset = 0,
    String? categoryId,
    String? searchQuery,
  }) async {
    final db = await database;
    String whereClause = 'is_published = 1';
    List<dynamic> whereArgs = [];
    
    if (categoryId != null) {
      whereClause += ' AND category_id = ?';
      whereArgs.add(categoryId);
    }
    
    if (searchQuery != null && searchQuery.isNotEmpty) {
      whereClause += ' AND (title LIKE ? OR content LIKE ?)';
      whereArgs.addAll(['%$searchQuery%', '%$searchQuery%']);
    }
    
    final maps = await db.query(
      'posts',
      where: whereClause,
      whereArgs: whereArgs,
      orderBy: 'published_at DESC, created_at DESC',
      limit: limit,
      offset: offset,
    );
    
    return maps.map((map) {
      final postData = Map<String, dynamic>.from(map);
      postData['tags'] = (postData['tags'] as String).split(',').where((tag) => tag.isNotEmpty).toList();
      return Post.fromJson(postData);
    }).toList();
  }
  
  Future<List<Post>> searchPosts(String query) async {
    final db = await database;
    final maps = await db.query(
      'posts',
      where: 'is_published = 1 AND (title LIKE ? OR content LIKE ?)',
      whereArgs: ['%$query%', '%$query%'],
      orderBy: 'published_at DESC',
    );
    
    return maps.map((map) {
      final postData = Map<String, dynamic>.from(map);
      postData['tags'] = (postData['tags'] as String).split(',').where((tag) => tag.isNotEmpty).toList();
      return Post.fromJson(postData);
    }).toList();
  }
  
  Future<void> clearAllData() async {
    final db = await database;
    await db.delete('posts');
    await db.delete('users');
    await db.delete('categories');
  }
  
  Future<void> close() async {
    final db = await database;
    await db.close();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. SECURE STORAGE SERVICE
// ═══════════════════════════════════════════════════════════════════════════════

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class SecureStorageService {
  static final SecureStorageService instance = SecureStorageService._internal();
  SecureStorageService._internal();
  
  final FlutterSecureStorage _storage = const FlutterSecureStorage(
    aOptions: AndroidOptions(
      encryptedSharedPreferences: true,
    ),
    iOptions: IOSOptions(
      accessibility: IOSAccessibility.first_unlock_this_device,
    ),
  );
  
  // Token management
  Future<void> setAccessToken(String token) async {
    await _storage.write(key: 'access_token', value: token);
  }
  
  Future<String?> getAccessToken() async {
    return await _storage.read(key: 'access_token');
  }
  
  Future<void> setRefreshToken(String token) async {
    await _storage.write(key: 'refresh_token', value: token);
  }
  
  Future<String?> getRefreshToken() async {
    return await _storage.read(key: 'refresh_token');
  }
  
  Future<void> clearTokens() async {
    await _storage.delete(key: 'access_token');
    await _storage.delete(key: 'refresh_token');
  }
  
  // User data
  Future<void> setUserData(String userData) async {
    await _storage.write(key: 'user_data', value: userData);
  }
  
  Future<String?> getUserData() async {
    return await _storage.read(key: 'user_data');
  }
  
  Future<void> clearUserData() async {
    await _storage.delete(key: 'user_data');
  }
  
  // Clear all data
  Future<void> clearAll() async {
    await _storage.deleteAll();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           8. AUTHENTICATION PROVIDER
// ═══════════════════════════════════════════════════════════════════════════════

class AuthProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService();
  final SecureStorageService _storage = SecureStorageService.instance;
  
  User? _currentUser;
  bool _isAuthenticated = false;
  bool _isLoading = false;
  String? _errorMessage;
  
  User? get currentUser => _currentUser;
  bool get isAuthenticated => _isAuthenticated;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  
  AuthProvider() {
    _checkAuthenticationStatus();
  }
  
  Future<void> _checkAuthenticationStatus() async {
    final token = await _storage.getAccessToken();
    if (token != null) {
      await getCurrentUser();
    }
  }
  
  Future<bool> login(String email, String password) async {
    try {
      _setLoading(true);
      _clearError();
      
      final response = await _apiService.post<ApiResponse<Map<String, dynamic>>>(
        ApiEndpoints.login,
        body: {'email': email, 'password': password},
        fromJson: (json) => ApiResponse.fromJson(json, (data) => data as Map<String, dynamic>),
      );
      
      if (response.success && response.data != null) {
        await _handleSuccessfulAuth(response.data!);
        return true;
      } else {
        _setError(response.message ?? 'Login failed');
        return false;
      }
    } catch (e) {
      _setError(_getErrorMessage(e));
      return false;
    } finally {
      _setLoading(false);
    }
  }
  
  Future<bool> register({
    required String email,
    required String username,
    required String firstName,
    required String lastName,
    required String password,
  }) async {
    try {
      _setLoading(true);
      _clearError();
      
      final response = await _apiService.post<ApiResponse<Map<String, dynamic>>>(
        ApiEndpoints.register,
        body: {
          'email': email,
          'username': username,
          'first_name': firstName,
          'last_name': lastName,
          'password': password,
        },
        fromJson: (json) => ApiResponse.fromJson(json, (data) => data as Map<String, dynamic>),
      );
      
      if (response.success && response.data != null) {
        await _handleSuccessfulAuth(response.data!);
        return true;
      } else {
        _setError(response.message ?? 'Registration failed');
        return false;
      }
    } catch (e) {
      _setError(_getErrorMessage(e));
      return false;
    } finally {
      _setLoading(false);
    }
  }
  
  Future<void> logout() async {
    try {
      await _apiService.post(ApiEndpoints.logout);
    } catch (e) {
      // Continue with local logout even if API call fails
    }
    
    await _handleLogout();
  }
  
  Future<void> getCurrentUser() async {
    try {
      final response = await _apiService.get<ApiResponse<Map<String, dynamic>>>(
        ApiEndpoints.currentUser,
        fromJson: (json) => ApiResponse.fromJson(json, (data) => data as Map<String, dynamic>),
      );
      
      if (response.success && response.data != null) {
        _currentUser = User.fromJson(response.data!);
        _isAuthenticated = true;
        await DatabaseService.instance.insertUser(_currentUser!);
        notifyListeners();
      } else {
        await _handleLogout();
      }
    } catch (e) {
      await _handleLogout();
    }
  }
  
  Future<void> refreshToken() async {
    try {
      final refreshToken = await _storage.getRefreshToken();
      if (refreshToken == null) {
        await _handleLogout();
        return;
      }
      
      final response = await _apiService.post<ApiResponse<Map<String, dynamic>>>(
        ApiEndpoints.refreshToken,
        body: {'refresh_token': refreshToken},
        fromJson: (json) => ApiResponse.fromJson(json, (data) => data as Map<String, dynamic>),
      );
      
      if (response.success && response.data != null) {
        await _handleSuccessfulAuth(response.data!);
      } else {
        await _handleLogout();
      }
    } catch (e) {
      await _handleLogout();
    }
  }
  
  Future<void> _handleSuccessfulAuth(Map<String, dynamic> authData) async {
    final accessToken = authData['access_token'] as String;
    final refreshToken = authData['refresh_token'] as String;
    final userData = authData['user'] as Map<String, dynamic>;
    
    await _storage.setAccessToken(accessToken);
    await _storage.setRefreshToken(refreshToken);
    
    _currentUser = User.fromJson(userData);
    _isAuthenticated = true;
    
    await DatabaseService.instance.insertUser(_currentUser!);
    await _storage.setUserData(json.encode(_currentUser!.toJson()));
    
    notifyListeners();
  }
  
  Future<void> _handleLogout() async {
    await _storage.clearAll();
    await DatabaseService.instance.clearAllData();
    
    _currentUser = null;
    _isAuthenticated = false;
    
    notifyListeners();
  }
  
  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }
  
  void _setError(String error) {
    _errorMessage = error;
    notifyListeners();
  }
  
  void _clearError() {
    _errorMessage = null;
    notifyListeners();
  }
  
  String _getErrorMessage(dynamic error) {
    if (error is NetworkException) {
      return error.message;
    } else if (error is ApiException) {
      return error.message;
    } else {
      return 'An unexpected error occurred';
    }
  }
  
  void clearError() {
    _clearError();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           9. POSTS PROVIDER
// ═══════════════════════════════════════════════════════════════════════════════

class PostsProvider extends ChangeNotifier {
  final ApiService _apiService = ApiService();
  final DatabaseService _dbService = DatabaseService.instance;
  
  List<Post> _posts = [];
  Post? _selectedPost;
  bool _isLoading = false;
  bool _isLoadingMore = false;
  String? _errorMessage;
  String _searchQuery = '';
  Category? _selectedCategory;
  int _currentPage = 1;
  bool _hasMorePages = true;
  
  List<Post> get posts => _posts;
  Post? get selectedPost => _selectedPost;
  bool get isLoading => _isLoading;
  bool get isLoadingMore => _isLoadingMore;
  String? get errorMessage => _errorMessage;
  String get searchQuery => _searchQuery;
  Category? get selectedCategory => _selectedCategory;
  bool get hasMorePages => _hasMorePages;
  
  Future<void> loadPosts({bool refresh = false}) async {
    if (refresh) {
      _currentPage = 1;
      _posts.clear();
      _hasMorePages = true;
    }
    
    if (!_hasMorePages || _isLoading) return;
    
    try {
      if (refresh) {
        _setLoading(true);
      } else {
        _setLoadingMore(true);
      }
      
      _clearError();
      
      final queryParams = <String, String>{
        'page': _currentPage.toString(),
        'limit': '20',
      };
      
      if (_selectedCategory != null) {
        queryParams['category_id'] = _selectedCategory!.id;
      }
      
      if (_searchQuery.isNotEmpty) {
        queryParams['search'] = _searchQuery;
      }
      
      final response = await _apiService.get<PaginatedResponse<Post>>(
        ApiEndpoints.posts,
        queryParams: queryParams,
        fromJson: (json) => PaginatedResponse.fromJson(json, (data) => Post.fromJson(data)),
      );
      
      if (response.success) {
        if (refresh) {
          _posts = response.data;
        } else {
          _posts.addAll(response.data);
        }
        
        _hasMorePages = response.pagination.hasNext;
        _currentPage++;
        
        // Cache posts locally
        await _dbService.insertPosts(response.data);
      } else {
        _setError(response.message ?? 'Failed to load posts');
      }
    } catch (e) {
      _setError(_getErrorMessage(e));
      
      // Load from local database if network fails
      if (refresh) {
        await _loadPostsFromLocal();
      }
    } finally {
      _setLoading(false);
      _setLoadingMore(false);
    }
  }
  
  Future<void> _loadPostsFromLocal() async {
    try {
      final localPosts = await _dbService.getPosts(
        categoryId: _selectedCategory?.id,
        searchQuery: _searchQuery.isNotEmpty ? _searchQuery : null,
      );
      _posts = localPosts;
      notifyListeners();
    } catch (e) {
      // Handle local storage error
    }
  }
  
  Future<void> loadPost(String postId) async {
    try {
      _setLoading(true);
      _clearError();
      
      final response = await _apiService.get<ApiResponse<Post>>(
        ApiEndpoints.post(postId),
        fromJson: (json) => ApiResponse.fromJson(json, (data) => Post.fromJson(data)),
      );
      
      if (response.success && response.data != null) {
        _selectedPost = response.data;
        await _dbService.insertPost(_selectedPost!);
      } else {
        _setError(response.message ?? 'Failed to load post');
        
        // Try loading from local database
        _selectedPost = await _dbService.getPost(postId);
      }
    } catch (e) {
      _setError(_getErrorMessage(e));
      _selectedPost = await _dbService.getPost(postId);
    } finally {
      _setLoading(false);
    }
  }
  
  Future<void> likePost(String postId) async {
    try {
      final post = _posts.firstWhere((p) => p.id == postId);
      final isCurrentlyLiked = post.isLiked;
      
      // Optimistic update
      _updatePostLikeStatus(postId, !isCurrentlyLiked);
      
      final endpoint = isCurrentlyLiked 
          ? ApiEndpoints.unlikePost(postId)
          : ApiEndpoints.likePost(postId);
      
      await _apiService.post(endpoint);
    } catch (e) {
      // Revert optimistic update on error
      final post = _posts.firstWhere((p) => p.id == postId);
      _updatePostLikeStatus(postId, !post.isLiked);
      _setError(_getErrorMessage(e));
    }
  }
  
  void _updatePostLikeStatus(String postId, bool isLiked) {
    final index = _posts.indexWhere((p) => p.id == postId);
    if (index != -1) {
      final post = _posts[index];
      _posts[index] = Post(
        id: post.id,
        title: post.title,
        content: post.content,
        excerpt: post.excerpt,
        imageUrl: post.imageUrl,
        authorId: post.authorId,
        author: post.author,
        categoryId: post.categoryId,
        category: post.category,
        tags: post.tags,
        isPublished: post.isPublished,
        viewCount: post.viewCount,
        likeCount: isLiked ? post.likeCount + 1 : post.likeCount - 1,
        commentCount: post.commentCount,
        isLiked: isLiked,
        publishedAt: post.publishedAt,
        createdAt: post.createdAt,
        updatedAt: post.updatedAt,
      );
      notifyListeners();
    }
  }
  
  void setSearchQuery(String query) {
    _searchQuery = query;
    loadPosts(refresh: true);
  }
  
  void setSelectedCategory(Category? category) {
    _selectedCategory = category;
    loadPosts(refresh: true);
  }
  
  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }
  
  void _setLoadingMore(bool loading) {
    _isLoadingMore = loading;
    notifyListeners();
  }
  
  void _setError(String error) {
    _errorMessage = error;
    notifyListeners();
  }
  
  void _clearError() {
    _errorMessage = null;
    notifyListeners();
  }
  
  String _getErrorMessage(dynamic error) {
    if (error is NetworkException) {
      return error.message;
    } else if (error is ApiException) {
      return error.message;
    } else {
      return 'An unexpected error occurred';
    }
  }
  
  void clearError() {
    _clearError();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           10. CONNECTIVITY PROVIDER
// ═══════════════════════════════════════════════════════════════════════════════

import 'package:connectivity_plus/connectivity_plus.dart';

class ConnectivityProvider extends ChangeNotifier {
  final Connectivity _connectivity = Connectivity();
  ConnectivityResult _connectionStatus = ConnectivityResult.none;
  late StreamSubscription<ConnectivityResult> _connectivitySubscription;
  
  ConnectivityResult get connectionStatus => _connectionStatus;
  bool get isConnected => _connectionStatus != ConnectivityResult.none;
  bool get isWifi => _connectionStatus == ConnectivityResult.wifi;
  bool get isMobile => _connectionStatus == ConnectivityResult.mobile;
  
  ConnectivityProvider() {
    _initConnectivity();
    _connectivitySubscription = _connectivity.onConnectivityChanged.listen(_updateConnectionStatus);
  }
  
  Future<void> _initConnectivity() async {
    try {
      final result = await _connectivity.checkConnectivity();
      _updateConnectionStatus(result);
    } catch (e) {
      _connectionStatus = ConnectivityResult.none;
      notifyListeners();
    }
  }
  
  void _updateConnectionStatus(ConnectivityResult result) {
    _connectionStatus = result;
    notifyListeners();
  }
  
  @override
  void dispose() {
    _connectivitySubscription.cancel();
    super.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           11. ROUTING AND NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

class AppRoutes {
  static const String splash = '/';
  static const String login = '/login';
  static const String register = '/register';
  static const String home = '/home';
  static const String posts = '/posts';
  static const String postDetail = '/post-detail';
  static const String profile = '/profile';
  static const String settings = '/settings';
  static const String createPost = '/create-post';
  
  static Route<dynamic> generateRoute(RouteSettings settings) {
    switch (settings.name) {
      case splash:
        return MaterialPageRoute(builder: (_) => SplashScreen());
      case login:
        return MaterialPageRoute(builder: (_) => LoginScreen());
      case register:
        return MaterialPageRoute(builder: (_) => RegisterScreen());
      case home:
        return MaterialPageRoute(builder: (_) => HomeScreen());
      case posts:
        return MaterialPageRoute(builder: (_) => PostsScreen());
      case postDetail:
        final args = settings.arguments as Map<String, dynamic>?;
        final postId = args?['postId'] as String?;
        if (postId != null) {
          return MaterialPageRoute(builder: (_) => PostDetailScreen(postId: postId));
        }
        return _errorRoute();
      case profile:
        return MaterialPageRoute(builder: (_) => ProfileScreen());
      case settings:
        return MaterialPageRoute(builder: (_) => SettingsScreen());
      case createPost:
        return MaterialPageRoute(builder: (_) => CreatePostScreen());
      default:
        return _errorRoute();
    }
  }
  
  static Route<dynamic> _errorRoute() {
    return MaterialPageRoute(
      builder: (_) => Scaffold(
        appBar: AppBar(title: const Text('Error')),
        body: const Center(
          child: Text('Page not found'),
        ),
      ),
    );
  }
}

class NavigationService {
  static final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();
  
  static NavigatorState? get navigator => navigatorKey.currentState;
  
  static Future<dynamic> navigateTo(String routeName, {Object? arguments}) {
    return navigator!.pushNamed(routeName, arguments: arguments);
  }
  
  static Future<dynamic> navigateAndReplace(String routeName, {Object? arguments}) {
    return navigator!.pushReplacementNamed(routeName, arguments: arguments);
  }
  
  static Future<dynamic> navigateAndClearStack(String routeName, {Object? arguments}) {
    return navigator!.pushNamedAndRemoveUntil(
      routeName,
      (route) => false,
      arguments: arguments,
    );
  }
  
  static void goBack([dynamic result]) {
    navigator!.pop(result);
  }
  
  static bool canGoBack() {
    return navigator!.canPop();
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
//                           12. THEME AND STYLING
// ═══════════════════════════════════════════════════════════════════════════════

class AppTheme {
  static const Color primaryColor = Color(0xFF2196F3);
  static const Color primaryColorDark = Color(0xFF1976D2);
  static const Color accentColor = Color(0xFFFF4081);
  static const Color backgroundColor = Color(0xFFF5F5F5);
  static const Color surfaceColor = Colors.white;
  static const Color errorColor = Color(0xFFE53E3E);
  static const Color successColor = Color(0xFF38A169);
  static const Color warningColor = Color(0xFFD69E2E);
  
  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryColor,
        brightness: Brightness.light,
      ),
      appBarTheme: const AppBarTheme(
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w600,
          color: Colors.black87,
        ),
        iconTheme: IconThemeData(color: Colors.black87),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
        filled: true,
        fillColor: Colors.grey.shade50,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
      cardTheme: CardTheme(
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
    );
  }
  
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryColor,
        brightness: Brightness.dark,
      ),
      appBarTheme: const AppBarTheme(
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        iconTheme: IconThemeData(color: Colors.white),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey.shade600),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey.shade600),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
        filled: true,
        fillColor: Colors.grey.shade800,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
      cardTheme: CardTheme(
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
    );
  }
}