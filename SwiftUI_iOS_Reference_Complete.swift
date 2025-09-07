// SWIFT IOS MOBILE DEVELOPMENT - Comprehensive Reference - by Richard Rembert
// Swift with SwiftUI enables building modern, declarative iOS applications with powerful
// frameworks like Combine, Core Data, and native iOS integrations

// ═══════════════════════════════════════════════════════════════════════════════
//                           1. SETUP AND PROJECT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/*
SWIFT IOS DEVELOPMENT SETUP:

1. Xcode Installation:
   - Download from Mac App Store or Apple Developer Portal
   - Minimum Xcode 15.0+ for iOS 17+ development
   - Include iOS Simulator and Command Line Tools

2. Project Creation:
   - Choose "iOS App" template
   - Select "SwiftUI" for Interface
   - Choose "Swift" as Language
   - Select minimum deployment target (iOS 15+ recommended)

3. Essential Frameworks and Dependencies:
   - SwiftUI (UI Framework)
   - Combine (Reactive Programming)
   - Core Data (Local Database)
   - URLSession (Networking)
   - Foundation (Core Utilities)
   - UserNotifications (Push Notifications)
   - Security (Keychain Access)

4. Package Manager Dependencies (Package.swift or SPM):
   - Alamofire (Networking - alternative to URLSession)
   - Kingfisher (Image Loading and Caching)
   - SwiftGen (Code Generation for Resources)
   - Realm (Alternative Database)
   - Firebase SDK (Analytics, Auth, etc.)

5. Project Structure:
   MyApp/
   ├── App/
   │   ├── MyAppApp.swift
   │   ├── ContentView.swift
   │   └── Info.plist
   ├── Core/
   │   ├── Network/
   │   ├── Database/
   │   ├── Authentication/
   │   └── Utils/
   ├── Features/
   │   ├── Authentication/
   │   ├── Posts/
   │   ├── Profile/
   │   └── Settings/
   ├── Models/
   │   ├── Domain/
   │   ├── Network/
   │   └── CoreData/
   ├── Views/
   │   ├── Components/
   │   ├── Screens/
   │   └── Modifiers/
   ├── ViewModels/
   ├── Services/
   ├── Resources/
   │   ├── Assets.xcassets
   │   ├── Localizable.strings
   │   └── Colors.xcassets
   └── Tests/
       ├── UnitTests/
       ├── UITests/
       └── IntegrationTests/

6. Configuration Files:
   - Info.plist (App configuration)
   - Entitlements.plist (App capabilities)
   - Configuration.plist (Environment settings)
*/

import SwiftUI
import Combine
import Foundation

// ═══════════════════════════════════════════════════════════════════════════════
//                           2. APP ENTRY POINT AND CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

@main
struct MyAppApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var authenticationService = AuthenticationService()
    @StateObject private var networkMonitor = NetworkMonitor()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environmentObject(authenticationService)
                .environmentObject(networkMonitor)
                .onAppear {
                    setupApp()
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
                    handleAppDidEnterBackground()
                }
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
                    handleAppWillEnterForeground()
                }
        }
    }
    
    private func setupApp() {
        configureAppearance()
        setupAnalytics()
        requestNotificationPermissions()
    }
    
    private func configureAppearance() {
        // Configure global UI appearance
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor.systemBackground
        appearance.titleTextAttributes = [.foregroundColor: UIColor.label]
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
    
    private func setupAnalytics() {
        // Initialize analytics services
        AnalyticsManager.shared.initialize()
    }
    
    private func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            DispatchQueue.main.async {
                appState.notificationsEnabled = granted
            }
        }
    }
    
    private func handleAppDidEnterBackground() {
        // Save app state, pause timers, etc.
        appState.saveState()
    }
    
    private func handleAppWillEnterForeground() {
        // Refresh data, resume timers, etc.
        appState.refreshData()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           3. APP STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

class AppState: ObservableObject {
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var notificationsEnabled = false
    @Published var selectedTab: Tab = .posts
    @Published var deepLinkURL: URL?
    
    enum Tab: String, CaseIterable {
        case posts = "Posts"
        case explore = "Explore"
        case favorites = "Favorites"
        case profile = "Profile"
        
        var iconName: String {
            switch self {
            case .posts: return "doc.text"
            case .explore: return "magnifyingglass"
            case .favorites: return "heart"
            case .profile: return "person"
            }
        }
    }
    
    func showError(_ message: String) {
        errorMessage = message
    }
    
    func clearError() {
        errorMessage = nil
    }
    
    func saveState() {
        UserDefaults.standard.set(selectedTab.rawValue, forKey: "selectedTab")
    }
    
    func loadState() {
        if let tabString = UserDefaults.standard.string(forKey: "selectedTab"),
           let tab = Tab(rawValue: tabString) {
            selectedTab = tab
        }
    }
    
    func refreshData() {
        // Trigger data refresh across the app
        NotificationCenter.default.post(name: .refreshData, object: nil)
    }
}

extension Notification.Name {
    static let refreshData = Notification.Name("refreshData")
    static let userDidLogin = Notification.Name("userDidLogin")
    static let userDidLogout = Notification.Name("userDidLogout")
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           4. MODELS AND DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// MARK: - Domain Models

struct User: Identifiable, Codable, Equatable {
    let id: String
    let email: String
    let username: String
    let firstName: String
    let lastName: String
    let avatarURL: URL?
    let bio: String?
    let isActive: Bool
    let createdAt: Date
    let updatedAt: Date
    
    var fullName: String {
        "\(firstName) \(lastName)"
    }
    
    var initials: String {
        let first = firstName.prefix(1).uppercased()
        let last = lastName.prefix(1).uppercased()
        return "\(first)\(last)"
    }
}

struct Post: Identifiable, Codable, Equatable {
    let id: String
    let title: String
    let content: String
    let excerpt: String
    let imageURL: URL?
    let authorID: String
    let author: User?
    let categoryID: String
    let category: Category?
    let tags: [String]
    let isPublished: Bool
    let viewCount: Int
    let likeCount: Int
    let commentCount: Int
    let isLiked: Bool
    let publishedAt: Date?
    let createdAt: Date
    let updatedAt: Date
    
    var readingTimeMinutes: Int {
        let wordsPerMinute = 200
        let wordCount = content.components(separatedBy: .whitespacesAndNewlines).count
        return max(1, wordCount / wordsPerMinute)
    }
    
    var isPublic: Bool {
        isPublished && publishedAt != nil
    }
    
    var formattedPublishDate: String {
        guard let publishedAt = publishedAt else { return "" }
        return DateFormatter.mediumDate.string(from: publishedAt)
    }
}

struct Category: Identifiable, Codable, Equatable {
    let id: String
    let name: String
    let slug: String
    let description: String
    let color: String
    let postCount: Int
    
    var colorValue: Color {
        Color(hex: color) ?? .blue
    }
}

struct Comment: Identifiable, Codable, Equatable {
    let id: String
    let content: String
    let postID: String
    let authorID: String
    let author: User?
    let parentID: String?
    let replies: [Comment]
    let createdAt: Date
    let updatedAt: Date
    
    var isReply: Bool {
        parentID != nil
    }
    
    var timeAgo: String {
        createdAt.timeAgoDisplay()
    }
}

// MARK: - Network DTOs

struct UserDTO: Codable {
    let id: String
    let email: String
    let username: String
    let firstName: String
    let lastName: String
    let avatarURL: String?
    let bio: String?
    let isActive: Bool
    let createdAt: String
    let updatedAt: String
    
    private enum CodingKeys: String, CodingKey {
        case id, email, username, bio
        case firstName = "first_name"
        case lastName = "last_name"
        case avatarURL = "avatar_url"
        case isActive = "is_active"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

struct PostDTO: Codable {
    let id: String
    let title: String
    let content: String
    let excerpt: String
    let imageURL: String?
    let authorID: String
    let author: UserDTO?
    let categoryID: String
    let category: CategoryDTO?
    let tags: [String]
    let isPublished: Bool
    let viewCount: Int
    let likeCount: Int
    let commentCount: Int
    let isLiked: Bool
    let publishedAt: String?
    let createdAt: String
    let updatedAt: String
    
    private enum CodingKeys: String, CodingKey {
        case id, title, content, excerpt, tags
        case imageURL = "image_url"
        case authorID = "author_id"
        case author, category
        case categoryID = "category_id"
        case isPublished = "is_published"
        case viewCount = "view_count"
        case likeCount = "like_count"
        case commentCount = "comment_count"
        case isLiked = "is_liked"
        case publishedAt = "published_at"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

struct CategoryDTO: Codable {
    let id: String
    let name: String
    let slug: String
    let description: String
    let color: String
    let postCount: Int
    
    private enum CodingKeys: String, CodingKey {
        case id, name, slug, description, color
        case postCount = "post_count"
    }
}

// MARK: - API Response Models

struct APIResponse<T: Codable>: Codable {
    let success: Bool
    let data: T?
    let message: String?
    let errors: [String]?
}

struct PaginatedResponse<T: Codable>: Codable {
    let success: Bool
    let data: [T]
    let pagination: PaginationInfo
    let message: String?
    let errors: [String]?
}

struct PaginationInfo: Codable {
    let currentPage: Int
    let totalPages: Int
    let totalItems: Int
    let itemsPerPage: Int
    let hasNext: Bool
    let hasPrevious: Bool
    
    private enum CodingKeys: String, CodingKey {
        case currentPage = "current_page"
        case totalPages = "total_pages"
        case totalItems = "total_items"
        case itemsPerPage = "items_per_page"
        case hasNext = "has_next"
        case hasPrevious = "has_previous"
    }
}

// MARK: - Authentication Models

struct LoginRequest: Codable {
    let email: String
    let password: String
}

struct RegisterRequest: Codable {
    let email: String
    let username: String
    let firstName: String
    let lastName: String
    let password: String
    
    private enum CodingKeys: String, CodingKey {
        case email, username, password
        case firstName = "first_name"
        case lastName = "last_name"
    }
}

struct AuthResponse: Codable {
    let accessToken: String
    let refreshToken: String
    let expiresIn: Int
    let user: UserDTO
    
    private enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case expiresIn = "expires_in"
        case user
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           5. DATA MAPPERS
// ═══════════════════════════════════════════════════════════════════════════════

enum DataMapper {
    
    // MARK: - User Mapping
    
    static func mapUser(from dto: UserDTO) -> User {
        User(
            id: dto.id,
            email: dto.email,
            username: dto.username,
            firstName: dto.firstName,
            lastName: dto.lastName,
            avatarURL: dto.avatarURL.flatMap(URL.init),
            bio: dto.bio,
            isActive: dto.isActive,
            createdAt: DateFormatter.iso8601.date(from: dto.createdAt) ?? Date(),
            updatedAt: DateFormatter.iso8601.date(from: dto.updatedAt) ?? Date()
        )
    }
    
    // MARK: - Post Mapping
    
    static func mapPost(from dto: PostDTO) -> Post {
        Post(
            id: dto.id,
            title: dto.title,
            content: dto.content,
            excerpt: dto.excerpt,
            imageURL: dto.imageURL.flatMap(URL.init),
            authorID: dto.authorID,
            author: dto.author.map(mapUser),
            categoryID: dto.categoryID,
            category: dto.category.map(mapCategory),
            tags: dto.tags,
            isPublished: dto.isPublished,
            viewCount: dto.viewCount,
            likeCount: dto.likeCount,
            commentCount: dto.commentCount,
            isLiked: dto.isLiked,
            publishedAt: dto.publishedAt.flatMap { DateFormatter.iso8601.date(from: $0) },
            createdAt: DateFormatter.iso8601.date(from: dto.createdAt) ?? Date(),
            updatedAt: DateFormatter.iso8601.date(from: dto.updatedAt) ?? Date()
        )
    }
    
    // MARK: - Category Mapping
    
    static func mapCategory(from dto: CategoryDTO) -> Category {
        Category(
            id: dto.id,
            name: dto.name,
            slug: dto.slug,
            description: dto.description,
            color: dto.color,
            postCount: dto.postCount
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           6. NETWORKING LAYER
// ═══════════════════════════════════════════════════════════════════════════════

// MARK: - Network Manager

class NetworkManager: ObservableObject {
    static let shared = NetworkManager()
    
    private let baseURL = "https://api.example.com/v1"
    private var cancellables = Set<AnyCancellable>()
    
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()
    
    private init() {}
    
    // MARK: - Generic Request Method
    
    func request<T: Codable>(
        endpoint: APIEndpoint,
        responseType: T.Type,
        completion: @escaping (Result<T, NetworkError>) -> Void
    ) {
        guard let url = endpoint.url(baseURL: baseURL) else {
            completion(.failure(.invalidURL))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method.rawValue
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Add authentication token if available
        if let token = KeychainManager.shared.getAccessToken() {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        // Add request body for POST/PUT requests
        if let body = endpoint.body {
            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            } catch {
                completion(.failure(.encodingError))
                return
            }
        }
        
        session.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(.networkError(error.localizedDescription)))
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(.failure(.invalidResponse))
                    return
                }
                
                guard 200...299 ~= httpResponse.statusCode else {
                    let errorMessage = self.parseErrorMessage(from: data)
                    completion(.failure(.serverError(httpResponse.statusCode, errorMessage)))
                    return
                }
                
                guard let data = data else {
                    completion(.failure(.noData))
                    return
                }
                
                do {
                    let decodedResponse = try JSONDecoder().decode(T.self, from: data)
                    completion(.success(decodedResponse))
                } catch {
                    print("Decoding error: \(error)")
                    completion(.failure(.decodingError))
                }
            }
        }.resume()
    }
    
    // MARK: - Combine-based Request Method
    
    func requestPublisher<T: Codable>(
        endpoint: APIEndpoint,
        responseType: T.Type
    ) -> AnyPublisher<T, NetworkError> {
        guard let url = endpoint.url(baseURL: baseURL) else {
            return Fail(error: NetworkError.invalidURL)
                .eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method.rawValue
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        if let token = KeychainManager.shared.getAccessToken() {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        if let body = endpoint.body {
            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            } catch {
                return Fail(error: NetworkError.encodingError)
                    .eraseToAnyPublisher()
            }
        }
        
        return session.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: T.self, decoder: JSONDecoder())
            .mapError { error in
                if error is DecodingError {
                    return NetworkError.decodingError
                } else {
                    return NetworkError.networkError(error.localizedDescription)
                }
            }
            .eraseToAnyPublisher()
    }
    
    private func parseErrorMessage(from data: Data?) -> String {
        guard let data = data else { return "Unknown error" }
        
        do {
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let message = json["message"] as? String {
                return message
            }
        } catch {}
        
        return "Server error"
    }
}

// MARK: - API Endpoints

enum APIEndpoint {
    // Authentication
    case login(email: String, password: String)
    case register(RegisterRequest)
    case refreshToken(String)
    case logout
    
    // Posts
    case getPosts(page: Int, limit: Int, categoryID: String?, search: String?)
    case getPost(id: String)
    case createPost(CreatePostRequest)
    case updatePost(id: String, UpdatePostRequest)
    case deletePost(id: String)
    case likePost(id: String)
    case unlikePost(id: String)
    
    // Users
    case getUser(id: String)
    case updateUser(id: String, UpdateUserRequest)
    case getCurrentUser
    
    // Categories
    case getCategories
    case getCategory(id: String)
    
    var path: String {
        switch self {
        case .login: return "/auth/login"
        case .register: return "/auth/register"
        case .refreshToken: return "/auth/refresh"
        case .logout: return "/auth/logout"
        case .getPosts: return "/posts"
        case .getPost(let id): return "/posts/\(id)"
        case .createPost: return "/posts"
        case .updatePost(let id, _): return "/posts/\(id)"
        case .deletePost(let id): return "/posts/\(id)"
        case .likePost(let id): return "/posts/\(id)/like"
        case .unlikePost(let id): return "/posts/\(id)/like"
        case .getUser(let id): return "/users/\(id)"
        case .updateUser(let id, _): return "/users/\(id)"
        case .getCurrentUser: return "/users/me"
        case .getCategories: return "/categories"
        case .getCategory(let id): return "/categories/\(id)"
        }
    }
    
    var method: HTTPMethod {
        switch self {
        case .login, .register, .refreshToken, .createPost, .likePost:
            return .POST
        case .updatePost, .updateUser:
            return .PUT
        case .deletePost, .logout, .unlikePost:
            return .DELETE
        default:
            return .GET
        }
    }
    
    var queryItems: [URLQueryItem]? {
        switch self {
        case .getPosts(let page, let limit, let categoryID, let search):
            var items = [
                URLQueryItem(name: "page", value: "\(page)"),
                URLQueryItem(name: "limit", value: "\(limit)")
            ]
            if let categoryID = categoryID {
                items.append(URLQueryItem(name: "category", value: categoryID))
            }
            if let search = search {
                items.append(URLQueryItem(name: "search", value: search))
            }
            return items
        default:
            return nil
        }
    }
    
    var body: [String: Any]? {
        switch self {
        case .login(let email, let password):
            return ["email": email, "password": password]
        case .register(let request):
            return try? request.asDictionary()
        case .refreshToken(let token):
            return ["refresh_token": token]
        case .createPost(let request):
            return try? request.asDictionary()
        case .updatePost(_, let request):
            return try? request.asDictionary()
        case .updateUser(_, let request):
            return try? request.asDictionary()
        default:
            return nil
        }
    }
    
    func url(baseURL: String) -> URL? {
        var components = URLComponents(string: baseURL + path)
        components?.queryItems = queryItems
        return components?.url
    }
}

enum HTTPMethod: String {
    case GET = "GET"
    case POST = "POST"
    case PUT = "PUT"
    case DELETE = "DELETE"
}

// MARK: - Network Error

enum NetworkError: Error, LocalizedError {
    case invalidURL
    case noData
    case decodingError
    case encodingError
    case networkError(String)
    case serverError(Int, String)
    case invalidResponse
    case unauthorized
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .noData:
            return "No data received"
        case .decodingError:
            return "Failed to decode response"
        case .encodingError:
            return "Failed to encode request"
        case .networkError(let message):
            return "Network error: \(message)"
        case .serverError(let code, let message):
            return "Server error \(code): \(message)"
        case .invalidResponse:
            return "Invalid response"
        case .unauthorized:
            return "Unauthorized access"
        }
    }
}

// MARK: - Request Models

struct CreatePostRequest: Codable {
    let title: String
    let content: String
    let excerpt: String?
    let categoryID: String
    let tags: [String]
    let isPublished: Bool
    
    private enum CodingKeys: String, CodingKey {
        case title, content, excerpt, tags
        case categoryID = "category_id"
        case isPublished = "is_published"
    }
}

struct UpdatePostRequest: Codable {
    let title: String?
    let content: String?
    let excerpt: String?
    let categoryID: String?
    let tags: [String]?
    let isPublished: Bool?
    
    private enum CodingKeys: String, CodingKey {
        case title, content, excerpt, tags
        case categoryID = "category_id"
        case isPublished = "is_published"
    }
}

struct UpdateUserRequest: Codable {
    let firstName: String?
    let lastName: String?
    let username: String?
    let bio: String?
    
    private enum CodingKeys: String, CodingKey {
        case username, bio
        case firstName = "first_name"
        case lastName = "last_name"
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           7. CORE DATA INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

import CoreData

// MARK: - Core Data Stack

class CoreDataManager: ObservableObject {
    static let shared = CoreDataManager()
    
    private init() {}
    
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "DataModel")
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("Core Data error: \(error.localizedDescription)")
            }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
        return container
    }()
    
    var viewContext: NSManagedObjectContext {
        persistentContainer.viewContext
    }
    
    func save() {
        let context = viewContext
        
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                print("Save error: \(error)")
            }
        }
    }
    
    func delete(_ object: NSManagedObject) {
        viewContext.delete(object)
        save()
    }
    
    // MARK: - User Operations
    
    func saveUser(_ user: User) {
        let userEntity = UserEntity(context: viewContext)
        userEntity.id = user.id
        userEntity.email = user.email
        userEntity.username = user.username
        userEntity.firstName = user.firstName
        userEntity.lastName = user.lastName
        userEntity.avatarURL = user.avatarURL?.absoluteString
        userEntity.bio = user.bio
        userEntity.isActive = user.isActive
        userEntity.createdAt = user.createdAt
        userEntity.updatedAt = user.updatedAt
        
        save()
    }
    
    func fetchUser(withID id: String) -> User? {
        let request: NSFetchRequest<UserEntity> = UserEntity.fetchRequest()
        request.predicate = NSPredicate(format: "id == %@", id)
        request.fetchLimit = 1
        
        do {
            let userEntities = try viewContext.fetch(request)
            return userEntities.first?.toDomainModel()
        } catch {
            print("Fetch user error: \(error)")
            return nil
        }
    }
    
    // MARK: - Post Operations
    
    func savePosts(_ posts: [Post]) {
        posts.forEach { post in
            let postEntity = PostEntity(context: viewContext)
            postEntity.id = post.id
            postEntity.title = post.title
            postEntity.content = post.content
            postEntity.excerpt = post.excerpt
            postEntity.imageURL = post.imageURL?.absoluteString
            postEntity.authorID = post.authorID
            postEntity.categoryID = post.categoryID
            postEntity.tags = post.tags.joined(separator: ",")
            postEntity.isPublished = post.isPublished
            postEntity.viewCount = Int32(post.viewCount)
            postEntity.likeCount = Int32(post.likeCount)
            postEntity.commentCount = Int32(post.commentCount)
            postEntity.isLiked = post.isLiked
            postEntity.publishedAt = post.publishedAt
            postEntity.createdAt = post.createdAt
            postEntity.updatedAt = post.updatedAt
        }
        
        save()
    }
    
    func fetchPosts() -> [Post] {
        let request: NSFetchRequest<PostEntity> = PostEntity.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]
        
        do {
            let postEntities = try viewContext.fetch(request)
            return postEntities.compactMap { $0.toDomainModel