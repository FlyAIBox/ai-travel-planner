"""
数据库 ORM 模型统一导入
"""

# 基础模型
from shared.database.connection import Base

# 用户域 ORM 模型
from .user import (
    UserORM,
    UserPreferencesORM,
    LoyaltyProgramORM,
    PaymentMethodORM,
)

# 旅行计划域 ORM 模型
from .travel import (
    TravelPlanORM,
    DestinationORM,
    TravelerInfoORM,
    BudgetBreakdownORM,
    FlightBookingORM,
    LayoverORM,
    AccommodationBookingORM,
    ItineraryDayORM,
    ItineraryActivityORM,
    ActivityBookingORM,
    WeatherInfoORM,
)

# 对话域 ORM 模型
from .conversation import (
    ConversationORM,
    MessageORM,
    MessageAttachmentORM,
    ToolCallORM,
    ConversationSummaryORM,
    ConversationMetricsORM,
)

# 智能体域 ORM 模型
from .agent import (
    AgentORM,
    AgentSessionORM,
    AgentInteractionORM,
    TaskORM,
    TaskResultORM,
    AgentTeamORM,
    CollaborationSessionORM,
    AgentMessageORM,
    AgentPerformanceMetricsORM,
)

# 知识库域 ORM 模型
from .knowledge import (
    KnowledgeDocumentORM,
    DocumentChunkORM,
    UserVectorORM,
    KnowledgeEntityORM,
    KnowledgeRelationORM,
    document_entity_associations,
)

# 导出所有 ORM 模型
__all__ = [
    # 基础
    "Base",
    
    # 用户域
    "UserORM",
    "UserPreferencesORM", 
    "LoyaltyProgramORM",
    "PaymentMethodORM",
    
    # 旅行计划域
    "TravelPlanORM",
    "DestinationORM",
    "TravelerInfoORM",
    "BudgetBreakdownORM",
    "FlightBookingORM",
    "LayoverORM",
    "AccommodationBookingORM",
    "ItineraryDayORM",
    "ItineraryActivityORM",
    "ActivityBookingORM",
    "WeatherInfoORM",
    
    # 对话域
    "ConversationORM",
    "MessageORM",
    "MessageAttachmentORM",
    "ToolCallORM",
    "ConversationSummaryORM",
    "ConversationMetricsORM",
    
    # 智能体域
    "AgentORM",
    "AgentSessionORM",
    "AgentInteractionORM",
    "TaskORM",
    "TaskResultORM",
    "AgentTeamORM",
    "CollaborationSessionORM",
    "AgentMessageORM",
    "AgentPerformanceMetricsORM",
    
    # 知识库域
    "KnowledgeDocumentORM",
    "DocumentChunkORM", 
    "UserVectorORM",
    "KnowledgeEntityORM",
    "KnowledgeRelationORM",
    "document_entity_associations",
] 