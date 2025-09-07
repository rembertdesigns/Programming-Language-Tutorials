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


// ═══════════════════════════════════════════════════════════════════════════════
//                           13. SCREENS AND UI COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

// Splash Screen
class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _initializeApp();
  }
  
  Future<void> _initializeApp() async {
    // Simulate initialization time
    await Future.delayed(const Duration(seconds: 2));
    
    final authProvider = Provider.of<AuthProvider>(context, listen: false);
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    
    // Load theme preference
    await themeProvider.loadThemeMode();
    
    // Navigate to appropriate screen
    if (authProvider.isAuthenticated) {
      NavigationService.navigateAndReplace(AppRoutes.home);
    } else {
      NavigationService.navigateAndReplace(AppRoutes.login);
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [AppTheme.primaryColor, AppTheme.primaryColorDark],
          ),
        ),
        child: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.article,
                size: 80,
                color: Colors.white,
              ),
              SizedBox(height: 24),
              Text(
                'My Flutter App',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              SizedBox(height: 16),
              CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Login Screen
class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _obscurePassword = true;
  
  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }
  
  Future<void> _login() async {
    if (_formKey.currentState!.validate()) {
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      
      final success = await authProvider.login(
        _emailController.text.trim(),
        _passwordController.text,
      );
      
      if (success) {
        NavigationService.navigateAndClearStack(AppRoutes.home);
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Icon(
                Icons.article,
                size: 80,
                color: AppTheme.primaryColor,
              ),
              const SizedBox(height: 32),
              const Text(
                'Welcome Back',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              const Text(
                'Sign in to your account',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),
              Form(
                key: _formKey,
                child: Column(
                  children: [
                    TextFormField(
                      controller: _emailController,
                      keyboardType: TextInputType.emailAddress,
                      decoration: const InputDecoration(
                        labelText: 'Email',
                        hintText: 'Enter your email',
                        prefixIcon: Icon(Icons.email_outlined),
                      ),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Please enter your email';
                        }
                        if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}).hasMatch(value)) {
                          return 'Please enter a valid email';
                        }
                        return null;
                      },
                    ),
                    const SizedBox(height: 16),
                    TextFormField(
                      controller: _passwordController,
                      obscureText: _obscurePassword,
                      decoration: InputDecoration(
                        labelText: 'Password',
                        hintText: 'Enter your password',
                        prefixIcon: const Icon(Icons.lock_outlined),
                        suffixIcon: IconButton(
                          icon: Icon(_obscurePassword ? Icons.visibility : Icons.visibility_off),
                          onPressed: () {
                            setState(() {
                              _obscurePassword = !_obscurePassword;
                            });
                          },
                        ),
                      ),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Please enter your password';
                        }
                        return null;
                      },
                    ),
                    const SizedBox(height: 24),
                    Consumer<AuthProvider>(
                      builder: (context, authProvider, child) {
                        return SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: authProvider.isLoading ? null : _login,
                            child: authProvider.isLoading
                                ? const SizedBox(
                                    height: 20,
                                    width: 20,
                                    child: CircularProgressIndicator(strokeWidth: 2),
                                  )
                                : const Text('Sign In'),
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              Consumer<AuthProvider>(
                builder: (context, authProvider, child) {
                  if (authProvider.errorMessage != null) {
                    return Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: AppTheme.errorColor.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: AppTheme.errorColor.withOpacity(0.3)),
                      ),
                      child: Row(
                        children: [
                          Icon(Icons.error_outline, color: AppTheme.errorColor),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              authProvider.errorMessage!,
                              style: TextStyle(color: AppTheme.errorColor),
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.close),
                            onPressed: authProvider.clearError,
                            iconSize: 20,
                          ),
                        ],
                      ),
                    );
                  }
                  return const SizedBox.shrink();
                },
              ),
              const SizedBox(height: 24),
              TextButton(
                onPressed: () {
                  NavigationService.navigateTo(AppRoutes.register);
                },
                child: const Text("Don't have an account? Sign up"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Home Screen with Bottom Navigation
class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;
  
  final List<Widget> _pages = [
    PostsScreen(),
    ExploreScreen(),
    FavoritesScreen(),
    ProfileScreen(),
  ];
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.article_outlined),
            activeIcon: Icon(Icons.article),
            label: 'Posts',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.explore_outlined),
            activeIcon: Icon(Icons.explore),
            label: 'Explore',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.favorite_outline),
            activeIcon: Icon(Icons.favorite),
            label: 'Favorites',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline),
            activeIcon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}

// Posts Screen
class PostsScreen extends StatefulWidget {
  @override
  _PostsScreenState createState() => _PostsScreenState();
}

class _PostsScreenState extends State<PostsScreen> {
  final _scrollController = ScrollController();
  final _searchController = TextEditingController();
  Timer? _debounce;
  
  @override
  void initState() {
    super.initState();
    _scrollController.addListener(_onScroll);
    
    // Load posts when screen initializes
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final postsProvider = Provider.of<PostsProvider>(context, listen: false);
      if (postsProvider.posts.isEmpty) {
        postsProvider.loadPosts(refresh: true);
      }
    });
  }
  
  @override
  void dispose() {
    _scrollController.removeListener(_onScroll);
    _scrollController.dispose();
    _searchController.dispose();
    _debounce?.cancel();
    super.dispose();
  }
  
  void _onScroll() {
    if (_scrollController.position.pixels >= _scrollController.position.maxScrollExtent - 200) {
      final postsProvider = Provider.of<PostsProvider>(context, listen: false);
      if (postsProvider.hasMorePages && !postsProvider.isLoadingMore) {
        postsProvider.loadPosts();
      }
    }
  }
  
  void _onSearchChanged(String query) {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 500), () {
      final postsProvider = Provider.of<PostsProvider>(context, listen: false);
      postsProvider.setSearchQuery(query);
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Posts'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () {
              NavigationService.navigateTo(AppRoutes.createPost);
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // Search Bar
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: TextField(
              controller: _searchController,
              decoration: const InputDecoration(
                hintText: 'Search posts...',
                prefixIcon: Icon(Icons.search),
              ),
              onChanged: _onSearchChanged,
            ),
          ),
          
          // Posts List
          Expanded(
            child: Consumer<PostsProvider>(
              builder: (context, postsProvider, child) {
                if (postsProvider.isLoading && postsProvider.posts.isEmpty) {
                  return const Center(child: CircularProgressIndicator());
                }
                
                if (postsProvider.posts.isEmpty && !postsProvider.isLoading) {
                  return const EmptyStateWidget(
                    icon: Icons.article_outlined,
                    title: 'No Posts Found',
                    subtitle: 'Start by creating your first post',
                  );
                }
                
                return RefreshIndicator(
                  onRefresh: () => postsProvider.loadPosts(refresh: true),
                  child: ListView.builder(
                    controller: _scrollController,
                    itemCount: postsProvider.posts.length + (postsProvider.isLoadingMore ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == postsProvider.posts.length) {
                        return const Center(
                          child: Padding(
                            padding: EdgeInsets.all(16.0),
                            child: CircularProgressIndicator(),
                          ),
                        );
                      }
                      
                      final post = postsProvider.posts[index];
                      return PostCard(
                        post: post,
                        onTap: () {
                          NavigationService.navigateTo(
                            AppRoutes.postDetail,
                            arguments: {'postId': post.id},
                          );
                        },
                        onLike: () {
                          postsProvider.likePost(post.id);
                        },
                      );
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

// Post Card Widget
class PostCard extends StatelessWidget {
  final Post post;
  final VoidCallback onTap;
  final VoidCallback onLike;
  
  const PostCard({
    Key? key,
    required this.post,
    required this.onTap,
    required this.onLike,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Author Info
              Row(
                children: [
                  CircleAvatar(
                    radius: 16,
                    backgroundImage: post.author?.avatarUrl != null
                        ? NetworkImage(post.author!.avatarUrl!)
                        : null,
                    child: post.author?.avatarUrl == null
                        ? Text(post.author?.initials ?? '??')
                        : null,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          post.author?.fullName ?? 'Unknown Author',
                          style: const TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                          ),
                        ),
                        Text(
                          post.formattedPublishDate,
                          style: TextStyle(
                            color: Colors.grey[600],
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (post.category != null)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: post.category!.colorValue.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        post.category!.name,
                        style: TextStyle(
                          color: post.category!.colorValue,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 12),
              
              // Post Content
              Text(
                post.title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 8),
              Text(
                post.excerpt,
                style: TextStyle(
                  color: Colors.grey[700],
                  fontSize: 14,
                ),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
              
              // Post Image
              if (post.imageUrl != null) ...[
                const SizedBox(height: 12),
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: CachedNetworkImageWidget(
                    imageUrl: post.imageUrl!,
                    height: 200,
                    width: double.infinity,
                    fit: BoxFit.cover,
                  ),
                ),
              ],
              
              // Tags
              if (post.tags.isNotEmpty) ...[
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 4,
                  children: post.tags.take(3).map((tag) {
                    return Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.grey[200],
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '#$tag',
                        style: TextStyle(
                          color: Colors.grey[700],
                          fontSize: 12,
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
              
              const SizedBox(height: 12),
              
              // Action Buttons
              Row(
                children: [
                  InkWell(
                    onTap: onLike,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          post.isLiked ? Icons.favorite : Icons.favorite_border,
                          color: post.isLiked ? Colors.red : Colors.grey[600],
                          size: 20,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          post.likeCount.toString(),
                          style: TextStyle(
                            color: Colors.grey[600],
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 16),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.comment_outlined,
                        color: Colors.grey[600],
                        size: 20,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        post.commentCount.toString(),
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(width: 16),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.visibility_outlined,
                        color: Colors.grey[600],
                        size: 20,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        post.viewCount.toString(),
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                  const Spacer(),
                  Text(
                    '${post.readingTimeMinutes} min read',
                    style: TextStyle(
                      color: Colors.grey[600],
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Post Detail Screen
class PostDetailScreen extends StatefulWidget {
  final String postId;
  
  const PostDetailScreen({Key? key, required this.postId}) : super(key: key);
  
  @override
  _PostDetailScreenState createState() => _PostDetailScreenState();
}

class _PostDetailScreenState extends State<PostDetailScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final postsProvider = Provider.of<PostsProvider>(context, listen: false);
      postsProvider.loadPost(widget.postId);
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Post Detail'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () {
              // Implement share functionality
            },
          ),
        ],
      ),
      body: Consumer<PostsProvider>(
        builder: (context, postsProvider, child) {
          if (postsProvider.isLoading && postsProvider.selectedPost == null) {
            return const Center(child: CircularProgressIndicator());
          }
          
          final post = postsProvider.selectedPost;
          if (post == null) {
            return const EmptyStateWidget(
              icon: Icons.error_outline,
              title: 'Post Not Found',
              subtitle: 'The requested post could not be loaded',
            );
          }
          
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Post Header
                Text(
                  post.title,
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 16),
                
                // Author Info
                Row(
                  children: [
                    CircleAvatar(
                      radius: 20,
                      backgroundImage: post.author?.avatarUrl != null
                          ? NetworkImage(post.author!.avatarUrl!)
                          : null,
                      child: post.author?.avatarUrl == null
                          ? Text(post.author?.initials ?? '??')
                          : null,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            post.author?.fullName ?? 'Unknown Author',
                            style: const TextStyle(
                              fontWeight: FontWeight.w600,
                              fontSize: 16,
                            ),
                          ),
                          Text(
                            '${post.formattedPublishDate} • ${post.readingTimeMinutes} min read',
                            style: TextStyle(
                              color: Colors.grey[600],
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 24),
                
                // Post Image
                if (post.imageUrl != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: CachedNetworkImageWidget(
                      imageUrl: post.imageUrl!,
                      height: 250,
                      width: double.infinity,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(height: 24),
                ],
                
                // Post Content
                Text(
                  post.content,
                  style: const TextStyle(
                    fontSize: 16,
                    height: 1.6,
                  ),
                ),
                const SizedBox(height: 24),
                
                // Tags
                if (post.tags.isNotEmpty) ...[
                  const Text(
                    'Tags',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: post.tags.map((tag) {
                      return Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: AppTheme.primaryColor.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Text(
                          '#$tag',
                          style: const TextStyle(
                            color: AppTheme.primaryColor,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                  const SizedBox(height: 24),
                ],
                
                // Action Buttons
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () {
                          postsProvider.likePost(post.id);
                        },
                        icon: Icon(
                          post.isLiked ? Icons.favorite : Icons.favorite_border,
                          color: post.isLiked ? Colors.red : null,
                        ),
                        label: Text('${post.likeCount} Likes'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: post.isLiked 
                              ? Colors.red.withOpacity(0.1)
                              : null,
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () {
                          // Navigate to comments
                        },
                        icon: const Icon(Icons.comment_outlined),
                        label: Text('${post.commentCount} Comments'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           14. UTILITY WIDGETS AND SERVICES
// ═══════════════════════════════════════════════════════════════════════════════

// Cached Network Image Widget
import 'package:cached_network_image/cached_network_image.dart';

class CachedNetworkImageWidget extends StatelessWidget {
  final String imageUrl;
  final double? width;
  final double? height;
  final BoxFit fit;
  final Widget? placeholder;
  final Widget? errorWidget;
  
  const CachedNetworkImageWidget({
    Key? key,
    required this.imageUrl,
    this.width,
    this.height,
    this.fit = BoxFit.cover,
    this.placeholder,
    this.errorWidget,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return CachedNetworkImage(
      imageUrl: imageUrl,
      width: width,
      height: height,
      fit: fit,
      placeholder: (context, url) => placeholder ?? Container(
        color: Colors.grey[200],
        child: const Center(
          child: CircularProgressIndicator(),
        ),
      ),
      errorWidget: (context, url, error) => errorWidget ?? Container(
        color: Colors.grey[200],
        child: const Icon(
          Icons.error_outline,
          color: Colors.grey,
        ),
      ),
    );
  }
}

// Empty State Widget
class EmptyStateWidget extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Widget? action;
  
  const EmptyStateWidget({
    Key? key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.action,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 80,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 16),
            Text(
              title,
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              subtitle,
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
            if (action != null) ...[
              const SizedBox(height: 24),
              action!,
            ],
          ],
        ),
      ),
    );
  }
}