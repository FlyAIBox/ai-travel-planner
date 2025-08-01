"""
AI Travel Planner - 共享数据模型
定义系统中所有领域的数据模型和验证规则
"""

from .user import *
from .travel import *
from .conversation import *
from .knowledge import *
from .agent import *
from .common import *

__all__ = [
    # 用户模型
    "User", "UserPreferences", "LoyaltyProgram", "UserProfile",
    
    # 旅行模型
    "TravelPlan", "Destination", "FlightBooking", "AccommodationBooking",
    "Activity", "Transportation", "Budget", "Itinerary",
    
    # 对话模型
    "Conversation", "Message", "MessageAttachment", "ChatSession",
    
    # 知识库模型
    "KnowledgeDocument", "VectorSearchQuery", "SearchResult",
    "DocumentChunk", "EmbeddingVector",
    
    # 智能体模型
    "AgentSession", "AgentInteraction", "TaskResult", "AgentState",
    
    # 通用模型
    "BaseModel", "ResponseModel", "PaginationModel", "FilterModel",
    "ErrorResponse", "SuccessResponse", "Location", "DateRange"
] 