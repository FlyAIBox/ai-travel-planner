"""
对话域数据模型
包含对话、消息、附件、工具调用等模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .user import User
from .common import IDMixin, TimestampMixin


# ==================== 枚举类型 ====================
class ConversationStatus(str, Enum):
    """对话状态"""
    ACTIVE = "active"        # 活跃
    PAUSED = "paused"        # 暂停
    COMPLETED = "completed"  # 完成
    ARCHIVED = "archived"    # 归档
    DELETED = "deleted"      # 删除


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"           # 用户
    ASSISTANT = "assistant" # 助手
    SYSTEM = "system"       # 系统
    TOOL = "tool"          # 工具


class MessageType(str, Enum):
    """消息类型"""
    TEXT = "text"               # 文本
    IMAGE = "image"             # 图片
    AUDIO = "audio"             # 音频
    VIDEO = "video"             # 视频
    FILE = "file"               # 文件
    LOCATION = "location"       # 位置
    TOOL_CALL = "tool_call"     # 工具调用
    TOOL_RESULT = "tool_result" # 工具结果


class AttachmentType(str, Enum):
    """附件类型"""
    IMAGE = "image"         # 图片
    AUDIO = "audio"         # 音频
    VIDEO = "video"         # 视频
    DOCUMENT = "document"   # 文档
    LINK = "link"          # 链接
    LOCATION = "location"   # 位置


class ToolCallStatus(str, Enum):
    """工具调用状态"""
    PENDING = "pending"     # 等待中
    RUNNING = "running"     # 运行中
    COMPLETED = "completed" # 完成
    FAILED = "failed"       # 失败
    TIMEOUT = "timeout"     # 超时


# ==================== 对话模型 ====================
class Conversation(IDMixin, TimestampMixin):
    """对话"""
    
    id: UUID = Field(default_factory=uuid4, description="对话ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 基本信息
    title: Optional[str] = Field(None, max_length=200, description="对话标题")
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE, description="对话状态")
    
    # 上下文信息
    context: Dict[str, Any] = Field(default={}, description="对话上下文")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    # 统计信息
    message_count: int = Field(default=0, ge=0, description="消息数量")
    total_tokens: int = Field(default=0, ge=0, description="总令牌数")
    
    # 设置
    language: str = Field(default="zh-cn", description="对话语言")
    model: Optional[str] = Field(None, description="使用的模型")
    temperature: float = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_tokens: int = Field(default=4096, ge=1, description="最大令牌数")
    
    # 标签
    tags: List[str] = Field(default=[], description="标签")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    last_activity_at: datetime = Field(default_factory=datetime.utcnow, description="最后活动时间")


class Message(IDMixin, TimestampMixin):
    """消息"""

    conversation_id: UUID = Field(..., description="对话ID")
    
    # 基本信息
    role: MessageRole = Field(..., description="角色")
    content: str = Field(..., description="消息内容")
    message_type: MessageType = Field(default=MessageType.TEXT, description="消息类型")
    
    # 顺序和版本
    sequence_number: int = Field(..., ge=0, description="序列号")
    version: int = Field(default=1, ge=1, description="版本号")
    parent_id: Optional[UUID] = Field(None, description="父消息ID")
    
    # AI相关信息
    model: Optional[str] = Field(None, description="使用的模型")
    prompt_tokens: int = Field(default=0, ge=0, description="提示令牌数")
    completion_tokens: int = Field(default=0, ge=0, description="完成令牌数")
    total_tokens: int = Field(default=0, ge=0, description="总令牌数")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    reasoning: Optional[str] = Field(None, description="推理过程")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="置信度")
    
    # 状态
    is_edited: bool = Field(default=False, description="是否已编辑")
    is_deleted: bool = Field(default=False, description="是否已删除")
    is_flagged: bool = Field(default=False, description="是否被标记")
    

    
    @validator('total_tokens', always=True)
    def calculate_total_tokens(cls, v, values):
        """计算总令牌数"""
        prompt = values.get('prompt_tokens', 0)
        completion = values.get('completion_tokens', 0)
        return prompt + completion


class MessageAttachment(IDMixin, TimestampMixin):
    """消息附件"""
    message_id: UUID = Field(..., description="消息ID")
    
    # 文件信息
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., ge=0, description="文件大小（字节）")
    file_type: str = Field(..., description="文件类型")
    attachment_type: AttachmentType = Field(..., description="附件类型")
    
    # 存储信息
    file_url: str = Field(..., description="文件URL")
    file_path: Optional[str] = Field(None, description="文件路径")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    alt_text: Optional[str] = Field(None, description="替代文本")
    description: Optional[str] = Field(None, description="描述")
    



class ToolCall(BaseModel):
    """工具调用"""
    
    id: UUID = Field(default_factory=uuid4, description="工具调用ID")
    message_id: UUID = Field(..., description="消息ID")
    
    # 工具信息
    tool_name: str = Field(..., description="工具名称")
    tool_version: Optional[str] = Field(None, description="工具版本")
    function_name: str = Field(..., description="函数名称")
    
    # 调用参数
    arguments: Dict[str, Any] = Field(..., description="调用参数")
    
    # 执行信息
    status: ToolCallStatus = Field(default=ToolCallStatus.PENDING, description="执行状态")
    result: Optional[Dict[str, Any]] = Field(None, description="执行结果")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 性能信息
    execution_time_ms: Optional[int] = Field(None, ge=0, description="执行时间（毫秒）")
    memory_used_mb: Optional[float] = Field(None, ge=0, description="内存使用（MB）")
    
    # 时间戳
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    @validator('completed_at')
    def validate_completion_time(cls, v, values):
        """验证完成时间"""
        if v and 'started_at' in values and v < values['started_at']:
            raise ValueError('完成时间必须晚于开始时间')
        return v


# ==================== 对话分析模型 ====================
class ConversationSummary(BaseModel):
    """对话摘要"""
    
    id: UUID = Field(default_factory=uuid4, description="摘要ID")
    conversation_id: UUID = Field(..., description="对话ID")
    
    # 摘要内容
    summary: str = Field(..., description="摘要内容")
    key_points: List[str] = Field(default=[], description="关键点")
    action_items: List[str] = Field(default=[], description="行动项")
    
    # 情感分析
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="情感分数")
    sentiment_label: Optional[str] = Field(None, description="情感标签")
    
    # 主题分析
    topics: List[str] = Field(default=[], description="主题")
    entities: List[Dict[str, str]] = Field(default=[], description="实体")
    intents: List[str] = Field(default=[], description="意图")
    
    # 质量评估
    clarity_score: Optional[float] = Field(None, ge=0, le=1, description="清晰度评分")
    completeness_score: Optional[float] = Field(None, ge=0, le=1, description="完整性评分")
    satisfaction_score: Optional[float] = Field(None, ge=0, le=1, description="满意度评分")
    
    # 生成信息
    generated_by: str = Field(..., description="生成者")
    model_used: Optional[str] = Field(None, description="使用的模型")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class ConversationMetrics(BaseModel):
    """对话指标"""
    
    id: UUID = Field(default_factory=uuid4, description="指标ID")
    conversation_id: UUID = Field(..., description="对话ID")
    
    # 基础指标
    duration_seconds: int = Field(..., ge=0, description="持续时间（秒）")
    turn_count: int = Field(..., ge=0, description="轮次数")
    user_message_count: int = Field(..., ge=0, description="用户消息数")
    assistant_message_count: int = Field(..., ge=0, description="助手消息数")
    
    # 令牌使用
    total_prompt_tokens: int = Field(..., ge=0, description="总提示令牌")
    total_completion_tokens: int = Field(..., ge=0, description="总完成令牌")
    average_response_time_ms: float = Field(..., ge=0, description="平均响应时间（毫秒）")
    
    # 工具使用
    tool_calls_count: int = Field(default=0, ge=0, description="工具调用次数")
    successful_tool_calls: int = Field(default=0, ge=0, description="成功工具调用次数")
    failed_tool_calls: int = Field(default=0, ge=0, description="失败工具调用次数")
    
    # 用户体验
    user_satisfaction: Optional[float] = Field(None, ge=0, le=5, description="用户满意度")
    task_completion_rate: Optional[float] = Field(None, ge=0, le=1, description="任务完成率")
    
    # 时间戳
    calculated_at: datetime = Field(default_factory=datetime.utcnow, description="计算时间")


# ==================== 请求/响应模型 ====================
class ConversationCreate(BaseModel):
    """创建对话请求"""
    title: Optional[str] = Field(None, max_length=200)
    context: Dict[str, Any] = Field(default={})
    language: str = Field(default="zh-cn")
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)
    tags: List[str] = Field(default=[])


class MessageCreate(BaseModel):
    """创建消息请求"""
    content: str = Field(..., min_length=1)
    message_type: MessageType = Field(default=MessageType.TEXT)
    metadata: Dict[str, Any] = Field(default={})
    attachments: List[Dict[str, Any]] = Field(default=[])


class MessageUpdate(BaseModel):
    """更新消息请求"""
    content: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    is_flagged: Optional[bool] = None


class ConversationResponse(BaseModel):
    """对话响应"""
    id: UUID
    title: Optional[str]
    status: ConversationStatus
    message_count: int
    language: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    



class MessageResponse(BaseModel):
    """消息响应"""
    id: UUID
    role: MessageRole
    content: str
    message_type: MessageType
    sequence_number: int
    total_tokens: int
    created_at: datetime
    is_edited: bool
    is_flagged: bool
    



class ConversationListResponse(BaseModel):
    """对话列表响应"""
    conversations: List[ConversationResponse]
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    size: int = Field(..., ge=1, le=100)
    
    @property
    def total_pages(self) -> int:
        """总页数"""
        return (self.total + self.size - 1) // self.size


# ==================== 实时消息模型 ====================
class StreamingMessage(BaseModel):
    """流式消息"""
    
    conversation_id: UUID = Field(..., description="对话ID")
    message_id: UUID = Field(..., description="消息ID")
    chunk_id: UUID = Field(default_factory=uuid4, description="分块ID")
    
    # 内容信息
    content_delta: str = Field(..., description="内容增量")
    is_complete: bool = Field(default=False, description="是否完成")
    
    # 元数据
    tokens_used: int = Field(default=0, ge=0, description="使用的令牌数")
    finish_reason: Optional[str] = Field(None, description="完成原因")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


class ChatEvent(BaseModel):
    """聊天事件"""
    
    event_id: UUID = Field(default_factory=uuid4, description="事件ID")
    conversation_id: UUID = Field(..., description="对话ID")
    
    # 事件信息
    event_type: str = Field(..., description="事件类型")
    event_data: Dict[str, Any] = Field(..., description="事件数据")
    
    # 用户信息
    user_id: Optional[UUID] = Field(None, description="用户ID")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


# ==================== WebSocket消息模型 ====================
class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    
    # 消息类型
    type: str = Field(..., description="消息类型")
    
    # 数据载荷
    data: Dict[str, Any] = Field(..., description="数据载荷")
    
    # 元数据
    conversation_id: Optional[UUID] = Field(None, description="对话ID")
    message_id: Optional[UUID] = Field(None, description="消息ID")
    request_id: Optional[str] = Field(None, description="请求ID")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


class WebSocketResponse(BaseModel):
    """WebSocket响应"""
    
    success: bool = Field(..., description="是否成功")
    message: Optional[str] = Field(None, description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    error: Optional[Dict[str, Any]] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳") 