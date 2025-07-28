"""
AI Travel Planner - 数据模型模块
统一导入所有数据模型
"""

# 用户域模型
from .user import (
    # 枚举
    TravelStyle,
    Language,
    Currency,
    UserStatus,
    PaymentMethodType,
    LoyaltyTier,
    
    # 核心模型
    User,
    UserPreferences,
    LoyaltyProgram,
    PaymentMethod,
    
    # 请求/响应模型
    UserCreate,
    UserUpdate,
    UserResponse,
    UserPreferencesUpdate,
    PaymentMethodCreate,
    UserListResponse,
    UserStats,
    
    # 基础模型
    BaseUser,
)

# 旅行计划域模型
from .travel import (
    # 枚举
    PlanStatus,
    BookingStatus,
    FlightClass,
    AccommodationType,
    ActivityType,
    WeatherCondition,
    
    # 核心模型
    TravelPlan,
    Destination,
    TravelerInfo,
    BudgetBreakdown,
    FlightBooking,
    Layover,
    AccommodationBooking,
    ItineraryDay,
    ItineraryActivity,
    ActivityBooking,
    WeatherInfo,
    
    # 请求/响应模型
    TravelPlanCreate,
    TravelPlanUpdate,
    TravelPlanResponse,
)

# 对话域模型
from .conversation import (
    # 枚举
    ConversationStatus,
    MessageRole,
    MessageType,
    AttachmentType,
    ToolCallStatus,
    
    # 核心模型
    Conversation,
    Message,
    MessageAttachment,
    ToolCall,
    ConversationSummary,
    ConversationMetrics,
    
    # 实时消息模型
    StreamingMessage,
    ChatEvent,
    WebSocketMessage,
    WebSocketResponse,
    
    # 请求/响应模型
    ConversationCreate,
    MessageCreate,
    MessageUpdate,
    ConversationResponse,
    MessageResponse,
    ConversationListResponse,
)

# 知识库域模型
from .knowledge import (
    # 枚举
    DocumentType,
    DocumentStatus,
    ChunkStrategy,
    EmbeddingModel,
    SearchType,
    
    # 核心模型
    KnowledgeDocument,
    DocumentChunk,
    VectorSearchQuery,
    VectorSearchResult,
    SearchResponse,
    RAGContext,
    RAGResponse,
    UserVector,
    KnowledgeEntity,
    KnowledgeRelation,
    
    # 请求/响应模型
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    SearchRequest,
    RAGRequest,
)

# 智能体域模型
from .agent import (
    # 枚举
    AgentType,
    AgentStatus,
    TaskStatus,
    TaskPriority,
    CollaborationType,
    
    # 核心模型
    Agent,
    AgentSession,
    AgentInteraction,
    Task,
    TaskResult,
    AgentTeam,
    CollaborationSession,
    AgentMessage,
    AgentPerformanceMetrics,
    
    # 请求/响应模型
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    TaskCreate,
    TaskResponse,
    AgentSessionCreate,
    InteractionRequest,
    InteractionResponse,
)

# 导出所有模型
__all__ = [
    # 用户域
    "TravelStyle", "Language", "Currency", "UserStatus", "PaymentMethodType", "LoyaltyTier",
    "User", "UserPreferences", "LoyaltyProgram", "PaymentMethod",
    "UserCreate", "UserUpdate", "UserResponse", "UserPreferencesUpdate", 
    "PaymentMethodCreate", "UserListResponse", "UserStats", "BaseUser",
    
    # 旅行计划域
    "PlanStatus", "BookingStatus", "FlightClass", "AccommodationType", "ActivityType", "WeatherCondition",
    "TravelPlan", "Destination", "TravelerInfo", "BudgetBreakdown", "FlightBooking", "Layover",
    "AccommodationBooking", "ItineraryDay", "ItineraryActivity", "ActivityBooking", "WeatherInfo",
    "TravelPlanCreate", "TravelPlanUpdate", "TravelPlanResponse",
    
    # 对话域
    "ConversationStatus", "MessageRole", "MessageType", "AttachmentType", "ToolCallStatus",
    "Conversation", "Message", "MessageAttachment", "ToolCall", "ConversationSummary", "ConversationMetrics",
    "StreamingMessage", "ChatEvent", "WebSocketMessage", "WebSocketResponse",
    "ConversationCreate", "MessageCreate", "MessageUpdate", "ConversationResponse", 
    "MessageResponse", "ConversationListResponse",
    
    # 知识库域
    "DocumentType", "DocumentStatus", "ChunkStrategy", "EmbeddingModel", "SearchType",
    "KnowledgeDocument", "DocumentChunk", "VectorSearchQuery", "VectorSearchResult", "SearchResponse",
    "RAGContext", "RAGResponse", "UserVector", "KnowledgeEntity", "KnowledgeRelation",
    "DocumentCreate", "DocumentUpdate", "DocumentResponse", "SearchRequest", "RAGRequest",
    
    # 智能体域
    "AgentType", "AgentStatus", "TaskStatus", "TaskPriority", "CollaborationType",
    "Agent", "AgentSession", "AgentInteraction", "Task", "TaskResult", "AgentTeam",
    "CollaborationSession", "AgentMessage", "AgentPerformanceMetrics",
    "AgentCreate", "AgentUpdate", "AgentResponse", "TaskCreate", "TaskResponse",
    "AgentSessionCreate", "InteractionRequest", "InteractionResponse",
] 