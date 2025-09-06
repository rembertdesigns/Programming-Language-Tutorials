// DART FLUTTER WEB - Comprehensive Development Reference - by Richard Rembert
// Dart with Flutter Web enables building high-performance, cross-platform web applications
// with native-like performance and beautiful UIs from a single codebase

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND ENVIRONMENT
// ═══════════════════════════════════════════════════════════════════════════════

/*
DART FLUTTER WEB DEVELOPMENT SETUP:

1. Install Flutter SDK:
   - Download from https://flutter.dev/docs/get-started/install
   - Add Flutter to PATH
   - Run: flutter doctor

2. Enable web support:
   flutter config --enable-web

3. Create new Flutter web project:
   flutter create --platforms web my_web_app
   cd my_web_app
   flutter run -d chrome

4. Essential dependencies (pubspec.yaml):
   dependencies:
     flutter:
       sdk: flutter
     
     # State management
     provider: ^6.1.1
     bloc: ^8.1.2
     flutter_bloc: ^8.1.3
     riverpod: ^2.4.9
     
     # HTTP and networking
     http: ^1.1.0
     dio: ^5.3.2
     socket_io_client: ^2.0.3
     
     # UI components
     material_design_icons_flutter: ^7.0.7296
     flutter_svg: ^2.0.9
     cached_network_image: ^3.3.0
     animations: ^2.0.11
     
     # Routing
     go_router: ^12.1.3
     auto_route: ^7.8.4
     
     # Utilities
     shared_preferences: ^2.2.2
     intl: ^0.19.0
     uuid: ^4.2.1
     json_annotation: ^4.8.1
     
     # Firebase integration
     firebase_core: ^2.24.2
     firebase_auth: ^4.15.3
     cloud_firestore: ^4.13.6
     
   dev_dependencies:
     flutter_test:
       sdk: flutter
     build_runner: ^2.4.7
     json_serializable: ^6.7.1
     flutter_lints: ^3.0.1
     test: ^1.24.9

5. Web-specific configuration (web/index.html):
   <!DOCTYPE html>
   <html>
   <head>
     <meta charset="UTF-8">
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     <title>My Flutter Web App</title>
     <link rel="manifest" href="manifest.json">
   </head>
   <body>
     <script src="flutter.js" defer></script>
   </body>
   </html>
*/

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:html' as html;

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. MAIN APPLICATION STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

void main() {
  runApp(MyWebApp());
}

class MyWebApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppState()),
        ChangeNotifierProvider(create: (_) => AuthService()),
        ChangeNotifierProvider(create: (_) => ThemeService()),
        ChangeNotifierProvider(create: (_) => NavigationService()),
      ],
      child: Consumer<ThemeService>(
        builder: (context, themeService, child) {
          return MaterialApp.router(
            title: 'Flutter Web App',
            theme: themeService.lightTheme,
            darkTheme: themeService.darkTheme,
            themeMode: themeService.themeMode,
            routerConfig: AppRouter.router,
            debugShowCheckedModeBanner: false,
          );
        },
      ),
    );
  }
}

// Application state management
class AppState with ChangeNotifier {
  bool _isLoading = false;
  String? _error;
  Map<String, dynamic> _data = {};

  bool get isLoading => _isLoading;
  String? get error => _error;
  Map<String, dynamic> get data => _data;

  void setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void setError(String? error) {
    _error = error;
    notifyListeners();
  }

  void setData(Map<String, dynamic> data) {
    _data = data;
    notifyListeners();
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. ROUTING AND NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

class AppRouter {
  static final GoRouter router = GoRouter(
    initialLocation: '/',
    routes: [
      GoRoute(
        path: '/',
        name: 'home',
        builder: (context, state) => const HomePage(),
      ),
      GoRoute(
        path: '/dashboard',
        name: 'dashboard',
        builder: (context, state) => const DashboardPage(),
        routes: [
          GoRoute(
            path: '/analytics',
            name: 'analytics',
            builder: (context, state) => const AnalyticsPage(),
          ),
          GoRoute(
            path: '/settings',
            name: 'settings',
            builder: (context, state) => const SettingsPage(),
          ),
        ],
      ),
      GoRoute(
        path: '/profile/:userId',
        name: 'profile',
        builder: (context, state) {
          final userId = state.pathParameters['userId']!;
          return ProfilePage(userId: userId);
        },
      ),
      GoRoute(
        path: '/auth',
        name: 'auth',
        builder: (context, state) => const AuthPage(),
      ),
      GoRoute(
        path: '/products',
        name: 'products',
        builder: (context, state) => const ProductsPage(),
        routes: [
          GoRoute(
            path: '/:productId',
            name: 'product-detail',
            builder: (context, state) {
              final productId = state.pathParameters['productId']!;
              return ProductDetailPage(productId: productId);
            },
          ),
        ],
      ),
    ],
    errorBuilder: (context, state) => ErrorPage(error: state.error.toString()),
    redirect: (context, state) {
      final authService = context.read<AuthService>();
      final isLoggedIn = authService.isAuthenticated;
      final isLoggingIn = state.location == '/auth';

      // Redirect to auth if not logged in and trying to access protected routes
      if (!isLoggedIn && !isLoggingIn && state.location.startsWith('/dashboard')) {
        return '/auth';
      }

      // Redirect to dashboard if logged in and on auth page
      if (isLoggedIn && isLoggingIn) {
        return '/dashboard';
      }

      return null; // No redirect
    },
  );
}

class NavigationService with ChangeNotifier {
  int _selectedIndex = 0;
  List<String> _history = ['/'];

  int get selectedIndex => _selectedIndex;
  List<String> get history => _history;

  void setSelectedIndex(int index) {
    _selectedIndex = index;
    notifyListeners();
  }

  void addToHistory(String route) {
    _history.add(route);
    if (_history.length > 10) {
      _history.removeAt(0);
    }
    notifyListeners();
  }

  void goBack() {
    if (_history.length > 1) {
      _history.removeLast();
      final previousRoute = _history.last;
      GoRouter.of(context as BuildContext).go(previousRoute);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. STATE MANAGEMENT AND SERVICES
// ═══════════════════════════════════════════════════════════════════════════════

// Authentication service
class AuthService with ChangeNotifier {
  bool _isAuthenticated = false;
  User? _user;
  String? _token;

  bool get isAuthenticated => _isAuthenticated;
  User? get user => _user;
  String? get token => _token;

  Future<bool> login(String email, String password) async {
    try {
      final response = await http.post(
        Uri.parse('https://api.example.com/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'password': password,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _token = data['token'];
        _user = User.fromJson(data['user']);
        _isAuthenticated = true;
        
        // Store token in local storage
        html.window.localStorage['auth_token'] = _token!;
        
        notifyListeners();
        return true;
      }
      return false;
    } catch (e) {
      print('Login error: $e');
      return false;
    }
  }

  Future<bool> register(String email, String password, String name) async {
    try {
      final response = await http.post(
        Uri.parse('https://api.example.com/auth/register'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'password': password,
          'name': name,
        }),
      );

      if (response.statusCode == 201) {
        // Auto-login after registration
        return await login(email, password);
      }
      return false;
    } catch (e) {
      print('Registration error: $e');
      return false;
    }
  }

  Future<void> logout() async {
    _isAuthenticated = false;
    _user = null;
    _token = null;
    
    // Clear local storage
    html.window.localStorage.remove('auth_token');
    
    notifyListeners();
  }

  Future<void> loadSavedToken() async {
    final savedToken = html.window.localStorage['auth_token'];
    if (savedToken != null) {
      _token = savedToken;
      // Verify token and load user data
      await _loadUserFromToken();
    }
  }

  Future<void> _loadUserFromToken() async {
    try {
      final response = await http.get(
        Uri.parse('https://api.example.com/auth/me'),
        headers: {
          'Authorization': 'Bearer $_token',
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _user = User.fromJson(data);
        _isAuthenticated = true;
        notifyListeners();
      } else {
        // Token is invalid
        await logout();
      }
    } catch (e) {
      print('Token verification error: $e');
      await logout();
    }
  }
}

// User model
class User {
  final String id;
  final String email;
  final String name;
  final String? avatar;
  final DateTime createdAt;

  User({
    required this.id,
    required this.email,
    required this.name,
    this.avatar,
    required this.createdAt,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      email: json['email'],
      name: json['name'],
      avatar: json['avatar'],
      createdAt: DateTime.parse(json['created_at']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'name': name,
      'avatar': avatar,
      'created_at': createdAt.toIso8601String(),
    };
  }
}

// Theme service
class ThemeService with ChangeNotifier {
  ThemeMode _themeMode = ThemeMode.system;
  bool _isDarkMode = false;

  ThemeMode get themeMode => _themeMode;
  bool get isDarkMode => _isDarkMode;

  ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: Colors.blue,
      brightness: Brightness.light,
    ),
    appBarTheme: const AppBarTheme(
      elevation: 0,
      centerTitle: true,
    ),
    cardTheme: CardTheme(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
      ),
      filled: true,
    ),
  );

  ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: Colors.blue,
      brightness: Brightness.dark,
    ),
    appBarTheme: const AppBarTheme(
      elevation: 0,
      centerTitle: true,
    ),
    cardTheme: CardTheme(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
      ),
      filled: true,
    ),
  );

  void setThemeMode(ThemeMode mode) {
    _themeMode = mode;
    _updateDarkMode();
    notifyListeners();
  }

  void toggleTheme() {
    _isDarkMode = !_isDarkMode;
    _themeMode = _isDarkMode ? ThemeMode.dark : ThemeMode.light;
    notifyListeners();
  }

  void _updateDarkMode() {
    switch (_themeMode) {
      case ThemeMode.dark:
        _isDarkMode = true;
        break;
      case ThemeMode.light:
        _isDarkMode = false;
        break;
      case ThemeMode.system:
        _isDarkMode = WidgetsBinding.instance.window.platformBrightness == Brightness.dark;
        break;
    }
  }
}

// HTTP service for API calls
class ApiService {
  static const String baseUrl = 'https://api.example.com';
  final http.Client _client = http.Client();

  Future<Map<String, dynamic>> get(String endpoint, {Map<String, String>? headers}) async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl$endpoint'),
        headers: {
          'Content-Type': 'application/json',
          ...?headers,
        },
      );
      return _handleResponse(response);
    } catch (e) {
      throw ApiException('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> post(String endpoint, Map<String, dynamic> data, {Map<String, String>? headers}) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl$endpoint'),
        headers: {
          'Content-Type': 'application/json',
          ...?headers,
        },
        body: json.encode(data),
      );
      return _handleResponse(response);
    } catch (e) {
      throw ApiException('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> put(String endpoint, Map<String, dynamic> data, {Map<String, String>? headers}) async {
    try {
      final response = await _client.put(
        Uri.parse('$baseUrl$endpoint'),
        headers: {
          'Content-Type': 'application/json',
          ...?headers,
        },
        body: json.encode(data),
      );
      return _handleResponse(response);
    } catch (e) {
      throw ApiException('Network error: $e');
    }
  }

  Future<void> delete(String endpoint, {Map<String, String>? headers}) async {
    try {
      final response = await _client.delete(
        Uri.parse('$baseUrl$endpoint'),
        headers: {
          'Content-Type': 'application/json',
          ...?headers,
        },
      );
      _handleResponse(response);
    } catch (e) {
      throw ApiException('Network error: $e');
    }
  }

  Map<String, dynamic> _handleResponse(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (response.body.isEmpty) return {};
      return json.decode(response.body);
    } else {
      throw ApiException('HTTP ${response.statusCode}: ${response.body}');
    }
  }

  void dispose() {
    _client.close();
  }
}

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  
  @override
  String toString() => 'ApiException: $message';
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. UI COMPONENTS AND LAYOUTS
// ═══════════════════════════════════════════════════════════════════════════════

// Responsive layout builder
class ResponsiveBuilder extends StatelessWidget {
  final Widget mobile;
  final Widget? tablet;
  final Widget desktop;

  const ResponsiveBuilder({
    Key? key,
    required this.mobile,
    this.tablet,
    required this.desktop,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < 600) {
          return mobile;
        } else if (constraints.maxWidth < 1200) {
          return tablet ?? desktop;
        } else {
          return desktop;
        }
      },
    );
  }
}

// Custom app bar
class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final List<Widget>? actions;
  final bool showBackButton;

  const CustomAppBar({
    Key? key,
    required this.title,
    this.actions,
    this.showBackButton = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      title: Text(title),
      actions: [
        ...?actions,
        Consumer<ThemeService>(
          builder: (context, themeService, child) {
            return IconButton(
              icon: Icon(
                themeService.isDarkMode 
                  ? Icons.light_mode 
                  : Icons.dark_mode,
              ),
              onPressed: themeService.toggleTheme,
            );
          },
        ),
        Consumer<AuthService>(
          builder: (context, authService, child) {
            if (authService.isAuthenticated) {
              return PopupMenuButton<String>(
                icon: CircleAvatar(
                  backgroundImage: authService.user?.avatar != null 
                    ? NetworkImage(authService.user!.avatar!) 
                    : null,
                  child: authService.user?.avatar == null 
                    ? Text(authService.user?.name.substring(0, 1).toUpperCase() ?? 'U')
                    : null,
                ),
                onSelected: (value) async {
                  switch (value) {
                    case 'profile':
                      context.go('/profile/${authService.user?.id}');
                      break;
                    case 'settings':
                      context.go('/dashboard/settings');
                      break;
                    case 'logout':
                      await authService.logout();
                      context.go('/');
                      break;
                  }
                },
                itemBuilder: (context) => [
                  const PopupMenuItem(
                    value: 'profile',
                    child: ListTile(
                      leading: Icon(Icons.person),
                      title: Text('Profile'),
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'settings',
                    child: ListTile(
                      leading: Icon(Icons.settings),
                      title: Text('Settings'),
                    ),
                  ),
                  const PopupMenuDivider(),
                  const PopupMenuItem(
                    value: 'logout',
                    child: ListTile(
                      leading: Icon(Icons.logout),
                      title: Text('Logout'),
                    ),
                  ),
                ],
              );
            }
            return TextButton(
              onPressed: () => context.go('/auth'),
              child: const Text('Login'),
            );
          },
        ),
      ],
      automaticallyImplyLeading: showBackButton,
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}

// Navigation drawer for desktop
class AppDrawer extends StatelessWidget {
  const AppDrawer({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: Column(
        children: [
          const DrawerHeader(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [Colors.blue, Colors.blueAccent],
              ),
            ),
            child: Row(
              children: [
                Icon(Icons.web, color: Colors.white, size: 32),
                SizedBox(width: 16),
                Text(
                  'Flutter Web',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView(
              children: [
                ListTile(
                  leading: const Icon(Icons.home),
                  title: const Text('Home'),
                  onTap: () {
                    context.go('/');
                    Navigator.pop(context);
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.dashboard),
                  title: const Text('Dashboard'),
                  onTap: () {
                    context.go('/dashboard');
                    Navigator.pop(context);
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.analytics),
                  title: const Text('Analytics'),
                  onTap: () {
                    context.go('/dashboard/analytics');
                    Navigator.pop(context);
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.inventory),
                  title: const Text('Products'),
                  onTap: () {
                    context.go('/products');
                    Navigator.pop(context);
                  },
                ),
                const Divider(),
                ListTile(
                  leading: const Icon(Icons.settings),
                  title: const Text('Settings'),
                  onTap: () {
                    context.go('/dashboard/settings');
                    Navigator.pop(context);
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Loading widget
class LoadingWidget extends StatelessWidget {
  final String? message;

  const LoadingWidget({Key? key, this.message}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(),
          if (message != null) ...[
            const SizedBox(height: 16),
            Text(message!),
          ],
        ],
      ),
    );
  }
}

// Error widget
class ErrorWidget extends StatelessWidget {
  final String message;
  final VoidCallback? onRetry;

  const ErrorWidget({
    Key? key,
    required this.message,
    this.onRetry,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 64, color: Colors.red),
          const SizedBox(height: 16),
          Text(
            message,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          if (onRetry != null) ...[
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: onRetry,
              child: const Text('Retry'),
            ),
          ],
        ],
      ),
    );
  }
}

// Custom card component
class CustomCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final VoidCallback? onTap;
  final Color? backgroundColor;

  const CustomCard({
    Key? key,
    required this.child,
    this.padding,
    this.onTap,
    this.backgroundColor,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      color: backgroundColor,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: padding ?? const EdgeInsets.all(16),
          child: child,
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. PAGE IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Home page
class HomePage extends StatelessWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Flutter Web App'),
      drawer: MediaQuery.of(context).size.width < 1200 ? const AppDrawer() : null,
      body: ResponsiveBuilder(
        mobile: const _MobileHomeLayout(),
        desktop: const _DesktopHomeLayout(),
      ),
    );
  }
}

class _MobileHomeLayout extends StatelessWidget {
  const _MobileHomeLayout();

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _HeroSection(),
          const SizedBox(height: 32),
          _FeaturesSection(),
          const SizedBox(height: 32),
          _CTASection(),
        ],
      ),
    );
  }
}

class _DesktopHomeLayout extends StatelessWidget {
  const _DesktopHomeLayout();

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        if (MediaQuery.of(context).size.width >= 1200) const AppDrawer(),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: Column(
              children: [
                _HeroSection(),
                const SizedBox(height: 64),
                _FeaturesSection(),
                const SizedBox(height: 64),
                _CTASection(),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _HeroSection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Theme.of(context).colorScheme.primary.withOpacity(0.1),
            Theme.of(context).colorScheme.secondary.withOpacity(0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Text(
            'Welcome to Flutter Web',
            style: Theme.of(context).textTheme.headlineLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          Text(
            'Build beautiful, fast, and responsive web applications with Flutter',
            style: Theme.of(context).textTheme.titleMedium,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          Wrap(
            spacing: 16,
            runSpacing: 16,
            alignment: WrapAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () => context.go('/dashboard'),
                child: const Text('Get Started'),
              ),
              OutlinedButton(
                onPressed: () => context.go('/products'),
                child: const Text('View Products'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _FeaturesSection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final features = [
      {
        'icon': Icons.speed,
        'title': 'High Performance',
        'description': 'Compiled to native web technologies for optimal performance',
      },
      {
        'icon': Icons.devices,
        'title': 'Cross Platform',
        'description': 'One codebase for web, mobile, and desktop applications',
      },
      {
        'icon': Icons.palette,
        'title': 'Beautiful UI',
        'description': 'Rich set of customizable widgets and animations',
      },
      {
        'icon': Icons.code,
        'title': 'Developer Experience',
        'description': 'Hot reload, excellent tooling, and strong type safety',
      },
    ];

    return Column(
      children: [
        Text(
          'Why Choose Flutter Web?',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 32),
        LayoutBuilder(
          builder: (context, constraints) {
            final crossAxisCount = constraints.maxWidth > 800 ? 2 : 1;
            return GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: crossAxisCount,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                childAspectRatio: 1.2,
              ),
              itemCount: features.length,
              itemBuilder: (context, index) {
                final feature = features[index];
                return CustomCard(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        feature['icon'] as IconData,
                        size: 48,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        feature['title'] as String,
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        feature['description'] as String,
                        style: Theme.of(context).textTheme.bodyMedium,
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                );
              },
            );
          },
        ),
      ],
    );
  }
}

class _CTASection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.primaryContainer,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Text(
            'Ready to Get Started?',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          Text(
            'Join thousands of developers building amazing web applications with Flutter',
            style: Theme.of(context).textTheme.bodyLarge,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: () => context.go('/auth'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
            ),
            child: const Text('Start Building Today'),
          ),
        ],
      ),
    );
  }
}

// Dashboard page
class DashboardPage extends StatefulWidget {
  const DashboardPage({Key? key}) : super(key: key);

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final List<Map<String, dynamic>> _stats = [
    {'title': 'Total Users', 'value': '12,345', 'icon': Icons.people, 'change': '+12%'},
    {'title': 'Revenue', 'value': '\$98,765', 'icon': Icons.attach_money, 'change': '+8%'},
    {'title': 'Orders', 'value': '1,234', 'icon': Icons.shopping_cart, 'change': '+24%'},
    {'title': 'Growth', 'value': '23%', 'icon': Icons.trending_up, 'change': '+4%'},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Dashboard'),
      drawer: MediaQuery.of(context).size.width < 1200 ? const AppDrawer() : null,
      body: ResponsiveBuilder(
        mobile: _buildMobileLayout(),
        desktop: _buildDesktopLayout(),
      ),
    );
  }

  Widget _buildMobileLayout() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildStatsGrid(),
          const SizedBox(height: 24),
          _buildRecentActivity(),
          const SizedBox(height: 24),
          _buildQuickActions(),
        ],
      ),
    );
  }

  Widget _buildDesktopLayout() {
    return Row(
      children: [
        if (MediaQuery.of(context).size.width >= 1200) const AppDrawer(),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildStatsGrid(),
                const SizedBox(height: 32),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      flex: 2,
                      child: _buildRecentActivity(),
                    ),
                    const SizedBox(width: 32),
                    Expanded(
                      child: _buildQuickActions(),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStatsGrid() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final crossAxisCount = constraints.maxWidth > 800 ? 4 : 2;
        return GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
            childAspectRatio: 1.2,
          ),
          itemCount: _stats.length,
          itemBuilder: (context, index) {
            final stat = _stats[index];
            return CustomCard(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    stat['icon'] as IconData,
                    size: 32,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    stat['value'] as String,
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    stat['title'] as String,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    stat['change'] as String,
                    style: TextStyle(
                      color: Colors.green,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildRecentActivity() {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Recent Activity',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: 5,
            separatorBuilder: (context, index) => const Divider(),
            itemBuilder: (context, index) {
              return ListTile(
                leading: CircleAvatar(
                  backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                  child: Icon(
                    Icons.person,
                    color: Theme.of(context).colorScheme.onPrimaryContainer,
                  ),
                ),
                title: Text('User ${index + 1} completed an action'),
                subtitle: Text('${index + 1} minutes ago'),
                trailing: const Icon(Icons.chevron_right),
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildQuickActions() {
    final actions = [
      {'title': 'Add Product', 'icon': Icons.add_shopping_cart, 'route': '/products'},
      {'title': 'View Analytics', 'icon': Icons.analytics, 'route': '/dashboard/analytics'},
      {'title': 'Settings', 'icon': Icons.settings, 'route': '/dashboard/settings'},
      {'title': 'Export Data', 'icon': Icons.download, 'route': null},
    ];

    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Quick Actions',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ...actions.map((action) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () {
                    if (action['route'] != null) {
                      context.go(action['route'] as String);
                    }
                  },
                  icon: Icon(action['icon'] as IconData),
                  label: Text(action['title'] as String),
                ),
              ),
            );
          }).toList(),
        ],
      ),
    );
  }
}

// Authentication page
class AuthPage extends StatefulWidget {
  const AuthPage({Key? key}) : super(key: key);

  @override
  State<AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _nameController = TextEditingController();
  bool _isLogin = true;
  bool _isLoading = false;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _submitForm() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    final authService = context.read<AuthService>();
    bool success;

    if (_isLogin) {
      success = await authService.login(
        _emailController.text.trim(),
        _passwordController.text,
      );
    } else {
      success = await authService.register(
        _emailController.text.trim(),
        _passwordController.text,
        _nameController.text.trim(),
      );
    }

    setState(() => _isLoading = false);

    if (success) {
      context.go('/dashboard');
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_isLogin ? 'Login failed' : 'Registration failed'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Authentication'),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(32),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 400),
            child: Form(
              key: _formKey,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.web,
                    size: 64,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(height: 32),
                  Text(
                    _isLogin ? 'Welcome Back' : 'Create Account',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 32),
                  if (!_isLogin) ...[
                    TextFormField(
                      controller: _nameController,
                      decoration: const InputDecoration(
                        labelText: 'Full Name',
                        prefixIcon: Icon(Icons.person),
                      ),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Please enter your name';
                        }
                        return null;
                      },
                    ),
                    const SizedBox(height: 16),
                  ],
                  TextFormField(
                    controller: _emailController,
                    decoration: const InputDecoration(
                      labelText: 'Email',
                      prefixIcon: Icon(Icons.email),
                    ),
                    keyboardType: TextInputType.emailAddress,
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your email';
                      }
                      if (!value.contains('@')) {
                        return 'Please enter a valid email';
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    controller: _passwordController,
                    decoration: const InputDecoration(
                      labelText: 'Password',
                      prefixIcon: Icon(Icons.lock),
                    ),
                    obscureText: true,
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your password';
                      }
                      if (value.length < 6) {
                        return 'Password must be at least 6 characters';
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 32),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _isLoading ? null : _submitForm,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      child: _isLoading
                          ? const CircularProgressIndicator()
                          : Text(_isLogin ? 'Login' : 'Register'),
                    ),
                  ),
                  const SizedBox(height: 16),
                  TextButton(
                    onPressed: () {
                      setState(() => _isLogin = !_isLogin);
                    },
                    child: Text(
                      _isLogin
                          ? 'Don\'t have an account? Register'
                          : 'Already have an account? Login',
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// Products page
class ProductsPage extends StatefulWidget {
  const ProductsPage({Key? key}) : super(key: key);

  @override
  State<ProductsPage> createState() => _ProductsPageState();
}

class _ProductsPageState extends State<ProductsPage> {
  final List<Product> _products = [
    Product(
      id: '1',
      name: 'Flutter Web Course',
      description: 'Learn to build amazing web apps with Flutter',
      price: 99.99,
      imageUrl: 'https://picsum.photos/300/200?random=1',
      category: 'Education',
    ),
    Product(
      id: '2',
      name: 'Dart Programming Guide',
      description: 'Complete guide to Dart programming language',
      price: 49.99,
      imageUrl: 'https://picsum.photos/300/200?random=2',
      category: 'Books',
    ),
    Product(
      id: '3',
      name: 'UI/UX Design Kit',
      description: 'Beautiful design components for Flutter',
      price: 79.99,
      imageUrl: 'https://picsum.photos/300/200?random=3',
      category: 'Design',
    ),
    Product(
      id: '4',
      name: 'State Management Pack',
      description: 'Advanced state management solutions',
      price: 129.99,
      imageUrl: 'https://picsum.photos/300/200?random=4',
      category: 'Tools',
    ),
  ];

  String _searchQuery = '';
  String _selectedCategory = 'All';

  List<Product> get filteredProducts {
    return _products.where((product) {
      final matchesSearch = product.name.toLowerCase().contains(_searchQuery.toLowerCase()) ||
          product.description.toLowerCase().contains(_searchQuery.toLowerCase());
      final matchesCategory = _selectedCategory == 'All' || product.category == _selectedCategory;
      return matchesSearch && matchesCategory;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Products'),
      drawer: MediaQuery.of(context).size.width < 1200 ? const AppDrawer() : null,
      body: ResponsiveBuilder(
        mobile: _buildMobileLayout(),
        desktop: _buildDesktopLayout(),
      ),
    );
  }

  Widget _buildMobileLayout() {
    return Column(
      children: [
        _buildSearchAndFilter(),
        Expanded(child: _buildProductGrid()),
      ],
    );
  }

  Widget _buildDesktopLayout() {
    return Row(
      children: [
        if (MediaQuery.of(context).size.width >= 1200) const AppDrawer(),
        Expanded(
          child: Column(
            children: [
              _buildSearchAndFilter(),
              Expanded(child: _buildProductGrid()),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildSearchAndFilter() {
    final categories = ['All', 'Education', 'Books', 'Design', 'Tools'];

    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          TextField(
            decoration: const InputDecoration(
              hintText: 'Search products...',
              prefixIcon: Icon(Icons.search),
            ),
            onChanged: (value) {
              setState(() => _searchQuery = value);
            },
          ),
          const SizedBox(height: 16),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: categories.map((category) {
                final isSelected = _selectedCategory == category;
                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: FilterChip(
                    label: Text(category),
                    selected: isSelected,
                    onSelected: (selected) {
                      setState(() => _selectedCategory = category);
                    },
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProductGrid() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final crossAxisCount = constraints.maxWidth > 1200 
            ? 4 
            : constraints.maxWidth > 800 
                ? 3 
                : constraints.maxWidth > 600 
                    ? 2 
                    : 1;

        return GridView.builder(
          padding: const EdgeInsets.all(16),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
            childAspectRatio: 0.75,
          ),
          itemCount: filteredProducts.length,
          itemBuilder: (context, index) {
            final product = filteredProducts[index];
            return _buildProductCard(product);
          },
        );
      },
    );
  }

  Widget _buildProductCard(Product product) {
    return CustomCard(
      onTap: () => context.go('/products/${product.id}'),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                image: DecorationImage(
                  image: NetworkImage(product.imageUrl),
                  fit: BoxFit.cover,
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            product.name,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.bold,
            ),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 4),
          Text(
            product.description,
            style: Theme.of(context).textTheme.bodySmall,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '\${product.price.toStringAsFixed(2)}',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
              Chip(
                label: Text(
                  product.category,
                  style: const TextStyle(fontSize: 12),
                ),
                materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// Product model
class Product {
  final String id;
  final String name;
  final String description;
  final double price;
  final String imageUrl;
  final String category;

  Product({
    required this.id,
    required this.name,
    required this.description,
    required this.price,
    required this.imageUrl,
    required this.category,
  });

  factory Product.fromJson(Map<String, dynamic> json) {
    return Product(
      id: json['id'],
      name: json['name'],
      description: json['description'],
      price: json['price'].toDouble(),
      imageUrl: json['image_url'],
      category: json['category'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'description': description,
      'price': price,
      'image_url': imageUrl,
      'category': category,
    };
  }
}

// Product detail page
class ProductDetailPage extends StatelessWidget {
  final String productId;

  const ProductDetailPage({Key? key, required this.productId}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // In a real app, you would fetch the product by ID
    final product = Product(
      id: productId,
      name: 'Sample Product',
      description: 'This is a sample product description with more details.',
      price: 99.99,
      imageUrl: 'https://picsum.photos/600/400?random=1',
      category: 'Sample',
    );

    return Scaffold(
      appBar: CustomAppBar(
        title: product.name,
        showBackButton: true,
      ),
      body: ResponsiveBuilder(
        mobile: _buildMobileLayout(context, product),
        desktop: _buildDesktopLayout(context, product),
      ),
    );
  }

  Widget _buildMobileLayout(BuildContext context, Product product) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildProductImage(product),
          const SizedBox(height: 24),
          _buildProductInfo(context, product),
          const SizedBox(height: 24),
          _buildActionButtons(context),
        ],
      ),
    );
  }

  Widget _buildDesktopLayout(BuildContext context, Product product) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(32),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 2,
            child: _buildProductImage(product),
          ),
          const SizedBox(width: 32),
          Expanded(
            flex: 3,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildProductInfo(context, product),
                const SizedBox(height: 32),
                _buildActionButtons(context),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProductImage(Product product) {
    return Container(
      width: double.infinity,
      height: 300,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        image: DecorationImage(
          image: NetworkImage(product.imageUrl),
          fit: BoxFit.cover,
        ),
      ),
    );
  }

  Widget _buildProductInfo(BuildContext context, Product product) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                product.name,
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Chip(label: Text(product.category)),
          ],
        ),
        const SizedBox(height: 16),
        Text(
          '\${product.price.toStringAsFixed(2)}',
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.bold,
            color: Theme.of(context).colorScheme.primary,
          ),
        ),
        const SizedBox(height: 16),
        Text(
          'Description',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          product.description,
          style: Theme.of(context).textTheme.bodyLarge,
        ),
        const SizedBox(height: 24),
        Text(
          'Features',
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('• High quality materials'),
            Text('• Fast delivery'),
            Text('• 30-day money back guarantee'),
            Text('• 24/7 customer support'),
          ],
        ),
      ],
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Added to cart!')),
              );
            },
            icon: const Icon(Icons.shopping_cart),
            label: const Text('Add to Cart'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Added to wishlist!')),
              );
            },
            icon: const Icon(Icons.favorite_border),
            label: const Text('Add to Wishlist'),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
          ),
        ),
      ],
    );
  }
}

// Analytics page
class AnalyticsPage extends StatelessWidget {
  const AnalyticsPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Analytics'),
      drawer: MediaQuery.of(context).size.width < 1200 ? const AppDrawer() : null,
      body: ResponsiveBuilder(
        mobile: _buildMobileLayout(),
        desktop: _buildDesktopLayout(),
      ),
    );
  }

  Widget _buildMobileLayout() {
    return const SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        children: [
          _AnalyticsChart(),
          SizedBox(height: 24),
          _AnalyticsMetrics(),
        ],
      ),
    );
  }

  Widget _buildDesktopLayout() {
    return Row(
      children: [
        if (MediaQuery.of(context).size.width >= 1200) const AppDrawer(),
        const Expanded(
          child: SingleChildScrollView(
            padding: EdgeInsets.all(32),
            child: Column(
              children: [
                _AnalyticsChart(),
                SizedBox(height: 32),
                _AnalyticsMetrics(),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _AnalyticsChart extends StatelessWidget {
  const _AnalyticsChart();

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Revenue Trend',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          Container(
            height: 300,
            width: double.infinity,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: Theme.of(context).colorScheme.surfaceVariant,
            ),
            child: const Center(
              child: Text('Chart Placeholder - Use charts_flutter or fl_chart'),
            ),
          ),
        ],
      ),
    );
  }
}

class _AnalyticsMetrics extends StatelessWidget {
  const _AnalyticsMetrics();

  @override
  Widget build(BuildContext context) {
    final metrics = [
      {'title': 'Page Views', 'value': '45,123', 'change': '+12%', 'color': Colors.blue},
      {'title': 'Bounce Rate', 'value': '23%', 'change': '-5%', 'color': Colors.green},
      {'title': 'Session Duration', 'value': '4:32', 'change': '+8%', 'color': Colors.orange},
      {'title': 'Conversion Rate', 'value': '3.4%', 'change': '+15%', 'color': Colors.purple},
    ];

    return LayoutBuilder(
      builder: (context, constraints) {
        final crossAxisCount = constraints.maxWidth > 800 ? 4 : 2;
        return GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
            childAspectRatio: 1.2,
          ),
          itemCount: metrics.length,
          itemBuilder: (context, index) {
            final metric = metrics[index];
            return CustomCard(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: (metric['color'] as Color).withOpacity(0.1),
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Icon(
                      Icons.analytics,
                      color: metric['color'] as Color,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    metric['value'] as String,
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    metric['title'] as String,
                    style: Theme.of(context).textTheme.bodyMedium,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    metric['change'] as String,
                    style: TextStyle(
                      color: Colors.green,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}

// Settings page
class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  bool _notificationsEnabled = true;
  bool _emailNotifications = false;
  bool _pushNotifications = true;
  double _fontSize = 16.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Settings'),
      drawer: MediaQuery.of(context).size.width < 1200 ? const AppDrawer() : null,
      body: ResponsiveBuilder(
        mobile: _buildMobileLayout(),
        desktop: _buildDesktopLayout(),
      ),
    );
  }

  Widget _buildMobileLayout() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _buildGeneralSettings(),
          const SizedBox(height: 24),
          _buildNotificationSettings(),
          const SizedBox(height: 24),
          _buildAppearanceSettings(),
          const SizedBox(height: 24),
          _buildAccountSettings(),
        ],
      ),
    );
  }

  Widget _buildDesktopLayout() {
    return Row(
      children: [
        if (MediaQuery.of(context).size.width >= 1200) const AppDrawer(),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 800),
              child: Column(
                children: [
                  _buildGeneralSettings(),
                  const SizedBox(height: 32),
                  _buildNotificationSettings(),
                  const SizedBox(height: 32),
                  _buildAppearanceSettings(),
                  const SizedBox(height: 32),
                  _buildAccountSettings(),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildGeneralSettings() {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'General',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          SwitchListTile(
            title: const Text('Enable Notifications'),
            subtitle: const Text('Receive app notifications'),
            value: _notificationsEnabled,
            onChanged: (value) {
              setState(() => _notificationsEnabled = value);
            },
          ),
          const Divider(),
          ListTile(
            title: const Text('Language'),
            subtitle: const Text('English (US)'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Show language selection dialog
            },
          ),
          const Divider(),
          ListTile(
            title: const Text('Time Zone'),
            subtitle: const Text('UTC-5 (Eastern Time)'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Show timezone selection dialog
            },
          ),
        ],
      ),
    );
  }

  Widget _buildNotificationSettings() {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Notifications',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          SwitchListTile(
            title: const Text('Email Notifications'),
            subtitle: const Text('Receive notifications via email'),
            value: _emailNotifications,
            onChanged: (value) {
              setState(() => _emailNotifications = value);
            },
          ),
          const Divider(),
          SwitchListTile(
            title: const Text('Push Notifications'),
            subtitle: const Text('Receive push notifications'),
            value: _pushNotifications,
            onChanged: (value) {
              setState(() => _pushNotifications = value);
            },
          ),
          const Divider(),
          ListTile(
            title: const Text('Notification Schedule'),
            subtitle: const Text('9:00 AM - 6:00 PM'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Show time picker dialog
            },
          ),
        ],
      ),
    );
  }

  Widget _buildAppearanceSettings() {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Appearance',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          Consumer<ThemeService>(
            builder: (context, themeService, child) {
              return Column(
                children: [
                  RadioListTile<ThemeMode>(
                    title: const Text('Light Mode'),
                    value: ThemeMode.light,
                    groupValue: themeService.themeMode,
                    onChanged: (value) => themeService.setThemeMode(value!),
                  ),
                  RadioListTile<ThemeMode>(
                    title: const Text('Dark Mode'),
                    value: ThemeMode.dark,
                    groupValue: themeService.themeMode,
                    onChanged: (value) => themeService.setThemeMode(value!),
                  ),
                  RadioListTile<ThemeMode>(
                    title: const Text('System Default'),
                    value: ThemeMode.system,
                    groupValue: themeService.themeMode,
                    onChanged: (value) => themeService.setThemeMode(value!),
                  ),
                ],
              );
            },
          ),
          const Divider(),
          ListTile(
            title: const Text('Font Size'),
            subtitle: Text('${_fontSize.round()}px'),
          ),
          Slider(
            value: _fontSize,
            min: 12.0,
            max: 24.0,
            divisions: 12,
            onChanged: (value) {
              setState(() => _fontSize = value);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildAccountSettings() {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Account',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ListTile(
            leading: const Icon(Icons.person),
            title: const Text('Edit Profile'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Navigate to profile edit
            },
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.security),
            title: const Text('Change Password'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Navigate to change password
            },
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.privacy_tip),
            title: const Text('Privacy Settings'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Navigate to privacy settings
            },
          ),
          const Divider(),
          ListTile(
            leading: Icon(Icons.logout, color: Colors.red),
            title: Text(
              'Sign Out',
              style: TextStyle(color: Colors.red),
            ),
            onTap: () async {
              final confirmed = await showDialog<bool>(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('Sign Out'),
                  content: const Text('Are you sure you want to sign out?'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context, false),
                      child: const Text('Cancel'),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(context, true),
                      child: const Text('Sign Out'),
                    ),
                  ],
                ),
              );

              if (confirmed == true) {
                await context.read<AuthService>().logout();
                context.go('/');
              }
            },
          ),
        ],
      ),
    );
  }
}

// Profile page
class ProfilePage extends StatelessWidget {
  final String userId;

  const ProfilePage({Key? key, required this.userId}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Profile'),
      body: ResponsiveBuilder(
        mobile: _buildMobileLayout(),
        desktop: _buildDesktopLayout(),
      ),
    );
  }

  Widget _buildMobileLayout() {
    return const SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        children: [
          _ProfileHeader(),
          SizedBox(height: 24),
          _ProfileStats(),
          SizedBox(height: 24),
          _ProfileActions(),
        ],
      ),
    );
  }

  Widget _buildDesktopLayout() {
    return const SingleChildScrollView(
      padding: EdgeInsets.all(32),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 1,
            child: _ProfileHeader(),
          ),
          SizedBox(width: 32),
          Expanded(
            flex: 2,
            child: Column(
              children: [
                _ProfileStats(),
                SizedBox(height: 32),
                _ProfileActions(),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ProfileHeader extends StatelessWidget {
  const _ProfileHeader();

  @override
  Widget build(BuildContext context) {
    return Consumer<AuthService>(
      builder: (context, authService, child) {
        final user = authService.user;
        
        return CustomCard(
          child: Column(
            children: [
              CircleAvatar(
                radius: 50,
                backgroundImage: user?.avatar != null 
                  ? NetworkImage(user!.avatar!) 
                  : null,
                child: user?.avatar == null 
                  ? Text(
                      user?.name.substring(0, 2).toUpperCase() ?? 'U',
                      style: const TextStyle(fontSize: 32),
                    )
                  : null,
              ),
              const SizedBox(height: 16),
              Text(
                user?.name ?? 'Unknown User',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                user?.email ?? 'unknown@example.com',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              Text(
                'Member since ${user?.createdAt.year ?? 2023}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        );
      },
    );
  }
}

class _ProfileStats extends StatelessWidget {
  const _ProfileStats();

  @override
  Widget build(BuildContext context) {
    final stats = [
      {'title': 'Products Purchased', 'value': '12'},
      {'title': 'Total Spent', 'value': '\$1,234'},
      {'title': 'Reviews Written', 'value': '8'},
      {'title': 'Loyalty Points', 'value': '2,450'},
    ];

    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Statistics',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ...stats.map((stat) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(stat['title']!),
                  Text(
                    stat['value']!,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }
}

class _ProfileActions extends StatelessWidget {
  const _ProfileActions();

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Quick Actions',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          ListTile(
            leading: const Icon(Icons.edit),
            title: const Text('Edit Profile'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Navigate to edit profile
            },
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.history),
            title: const Text('Order History'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Navigate to order history
            },
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.favorite),
            title: const Text('Wishlist'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Navigate to wishlist
            },
          ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.support_agent),
            title: const Text('Contact Support'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              // Open support chat/email
            },
          ),
        ],
      ),
    );
  }
}

// Error page
class ErrorPage extends StatelessWidget {
  final String error;

  const ErrorPage({Key? key, required this.error}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: 'Error'),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.error_outline,
              size: 64,
              color: Colors.red,
            ),
            const SizedBox(height: 16),
            Text(
              'Oops! Something went wrong',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              error,
              style: Theme.of(context).textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: () => context.go('/'),
              child: const Text('Go Home'),
            ),
          ],
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. ADVANCED FEATURES AND UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

// Form validation utilities
class FormValidators {
  static String? email(String? value) {
    if (value == null || value.isEmpty) {
      return 'Email is required';
    }
    final emailRegex = RegExp(r'^[^@]+@[^@]+\.[^@]+');
    if (!emailRegex.hasMatch(value)) {
      return 'Please enter a valid email';
    }
    return null;
  }

  static String? password(String? value) {
    if (value == null || value.isEmpty) {
      return 'Password is required';
    }
    if (value.length < 8) {
      return 'Password must be at least 8 characters';
    }
    if (!RegExp(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)').hasMatch(value)) {
      return 'Password must contain uppercase, lowercase, and number';
    }
    return null;
  }

  static String? required(String? value, [String? fieldName]) {
    if (value == null || value.isEmpty) {
      return '${fieldName ?? 'This field'} is required';
    }
    return null;
  }

  static String? minLength(String? value, int minLength, [String? fieldName]) {
    if (value == null || value.length < minLength) {
      return '${fieldName ?? 'This field'} must be at least $minLength characters';
    }
    return null;
  }

  static String? maxLength(String? value, int maxLength, [String? fieldName]) {
    if (value != null && value.length > maxLength) {
      return '${fieldName ?? 'This field'} must not exceed $maxLength characters';
    }
    return null;
  }
}

// Responsive breakpoints
class Breakpoints {
  static const double mobile = 600;
  static const double tablet = 1200;
  static const double desktop = 1200;

  static bool isMobile(BuildContext context) {
    return MediaQuery.of(context).size.width < mobile;
  }

  static bool isTablet(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    return width >= mobile && width < desktop;
  }

  static bool isDesktop(BuildContext context) {
    return MediaQuery.of(context).size.width >= desktop;
  }
}

// Custom dialogs
class AppDialogs {
  static Future<bool?> showConfirmDialog(
    BuildContext context, {
    required String title,
    required String message,
    String confirmText = 'Confirm',
    String cancelText = 'Cancel',
  }) {
    return showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: Text(cancelText),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text(confirmText),
          ),
        ],
      ),
    );
  }

  static void showErrorDialog(
    BuildContext context, {
    required String title,
    required String message,
  }) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.error, color: Colors.red),
            const SizedBox(width: 8),
            Text(title),
          ],
        ),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  static void showLoadingDialog(BuildContext context, String message) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        content: Row(
          children: [
            const CircularProgressIndicator(),
            const SizedBox(width: 16),
            Text(message),
          ],
        ),
      ),
    );
  }
}

// Animation utilities
class AppAnimations {
  static const Duration defaultDuration = Duration(milliseconds: 300);
  static const Curve defaultCurve = Curves.easeInOut;

  static Widget fadeIn({
    required Widget child,
    Duration duration = defaultDuration,
    Curve curve = defaultCurve,
  }) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: duration,
      curve: curve,
      builder: (context, value, child) {
        return Opacity(
          opacity: value,
          child: child,
        );
      },
      child: child,
    );
  }

  static Widget slideUp({
    required Widget child,
    Duration duration = defaultDuration,
    Curve curve = defaultCurve,
  }) {
    return TweenAnimationBuilder<Offset>(
      tween: Tween(begin: const Offset(0, 1), end: Offset.zero),
      duration: duration,
      curve: curve,
      builder: (context, value, child) {
        return SlideTransition(
          position: AlwaysStoppedAnimation(value),
          child: child,
        );
      },
      child: child,
    );
  }

  static Widget scaleIn({
    required Widget child,
    Duration duration = defaultDuration,
    Curve curve = defaultCurve,
  }) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: duration,
      curve: curve,
      builder: (context, value, child) {
        return Transform.scale(
          scale: value,
          child: child,
        );
      },
      child: child,
    );
  }
}

// Performance optimization utilities
class PerformanceUtils {
  static void preloadImage(String imageUrl) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final imageProvider = NetworkImage(imageUrl);
      imageProvider.resolve(const ImageConfiguration());
    });
  }

  static void debounce(VoidCallback callback, Duration duration) {
    Timer? timer;
    timer?.cancel();
    timer = Timer(duration, callback);
  }
}

// URL utilities for web
class UrlUtils {
  static void openInNewTab(String url) {
    html.window.open(url, '_blank');
  }

  static void copyToClipboard(String text) {
    Clipboard.setData(ClipboardData(text: text));
  }

  static String getCurrentUrl() {
    return html.window.location.href;
  }

  static void setPageTitle(String title) {
    html.document.title = title;
  }
}

/*
═══════════════════════════════════════════════════════════════════════════════
                           8. TESTING AND DEPLOYMENT
═══════════════════════════════════════════════════════════════════════════════

TESTING:

1. Unit Tests:
   test/unit/models_test.dart
   test/unit/services_test.dart
   test/unit/utils_test.dart

2. Widget Tests:
   test/widget/pages_test.dart
   test/widget/components_test.dart

3. Integration Tests:
   integration_test/app_test.dart

4. Test Commands:
   flutter test                    # Run all tests
   flutter test --coverage       # Generate coverage report
   flutter drive --target=integration_test/app_test.dart

DEPLOYMENT:

1. Build for Production:
   flutter build web --release

2. Configure web/index.html for production:
   - Set proper meta tags
   - Add analytics scripts
   - Configure CSP headers

3. Deploy to Firebase Hosting:
   firebase init hosting
   firebase deploy

4. Deploy to GitHub Pages:
   flutter build web --base-href "/your-repo-name/"
   
5. Deploy to Netlify:
   - Build command: flutter build web
   - Publish directory: build/web

6. Performance Optimization:
   - Enable web renderers: flutter build web --web-renderer canvaskit
   - Tree shaking: flutter build web --tree-shake-icons
   - Split defer loading: flutter build web --split-debug-info

WEB-SPECIFIC CONSIDERATIONS:

1. SEO Optimization:
   - Use proper meta tags
   - Implement server-side rendering if needed
   - Create sitemap.xml
   - Use semantic HTML

2. Performance:
   - Lazy load images and components
   - Implement code splitting
   - Use web workers for heavy computations
   - Cache static assets

3. Browser Compatibility:
   - Test on multiple browsers
   - Use web-safe fonts
   - Provide fallbacks for modern features

4. Accessibility:
   - Use semantic widgets
   - Provide proper ARIA labels
   - Ensure keyboard navigation
   - Test with screen readers

5. PWA Features:
   - Configure manifest.json
   - Implement service worker
   - Add offline capabilities
   - Enable installation prompts

EXAMPLE TEST FILES:

// test/widget/home_page_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:your_app/main.dart';

void main() {
  testWidgets('HomePage displays correctly', (WidgetTester tester) async {
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider(create: (_) => AppState()),
          ChangeNotifierProvider(create: (_) => ThemeService()),
        ],
        child: MaterialApp(home: HomePage()),
      ),
    );

    expect(find.text('Welcome to Flutter Web'), findsOneWidget);
    expect(find.byType(ElevatedButton), findsWidgets);
  });
}

// test/unit/auth_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:your_app/services/auth_service.dart';

void main() {
  group('AuthService', () {
    late AuthService authService;

    setUp(() {
      authService = AuthService();
    });

    test('initial state is not authenticated', () {
      expect(authService.isAuthenticated, false);
      expect(authService.user, null);
    });

    test('login updates authentication state', () async {
      // Mock API response
      final result = await authService.login('test@example.com', 'password');
      
      expect(result, true);
      expect(authService.isAuthenticated, true);
    });
  });
}

═══════════════════════════════════════════════════════════════════════════════
                           9. BEST PRACTICES AND TIPS
═══════════════════════════════════════════════════════════════════════════════

PROJECT STRUCTURE:
lib/
├── main.dart
├── app/
│   ├── app.dart
│   └── router.dart
├── core/
│   ├── constants/
│   ├── utils/
│   └── exceptions/
├── features/
│   ├── auth/
│   │   ├── models/
│   │   ├── services/
│   │   ├── pages/
│   │   └── widgets/
│   └── dashboard/
├── shared/
│   ├── widgets/
│   ├── services/
│   └── models/
└── l10n/

PERFORMANCE TIPS:
1. Use const constructors wherever possible
2. Implement lazy loading for large lists
3. Use AutomaticKeepAliveClientMixin for tabs
4. Optimize images (WebP format, proper sizing)
5. Use RepaintBoundary for expensive widgets
6. Implement proper state management
7. Avoid rebuilding entire widget trees

ACCESSIBILITY:
1. Use Semantics widgets for screen readers
2. Provide sufficient color contrast
3. Ensure touch targets are at least 44px
4. Support keyboard navigation
5. Use semantic HTML tags in web builds

RESPONSIVE DESIGN:
1. Use LayoutBuilder for responsive layouts
2. Implement breakpoint-based design
3. Test on multiple screen sizes
4. Use flexible and expanded widgets properly
5. Consider touch vs mouse interactions

STATE MANAGEMENT COMPARISON:
- Provider: Simple, good for small-medium apps
- Bloc: Complex but powerful, good for large apps
- Riverpod: Modern alternative to Provider
- GetX: All-in-one solution with dependency injection

SECURITY CONSIDERATIONS:
1. Validate all user inputs
2. Use HTTPS for all API calls
3. Implement proper authentication
4. Store sensitive data securely
5. Sanitize data before displaying
6. Implement rate limiting
7. Use CSP headers

DEBUGGING TOOLS:
1. Flutter Inspector for widget trees
2. Performance overlay: flutter run --profile
3. Network traffic monitoring
4. Memory usage profiling
5. Browser developer tools for web-specific issues

This comprehensive reference covers the essential aspects of Flutter Web development,
from basic setup to advanced features, testing, and deployment. Use it as a guide
for building robust, scalable web applications with Dart and Flutter.
*/