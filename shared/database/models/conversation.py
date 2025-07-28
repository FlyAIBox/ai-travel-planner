"""
对话域 SQLAlchemy ORM 模型
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Boolean, DateTime, String, Text, Integer, Numeric, JSON, 
    Enum as SQLEnum, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.dialects.mysql import CHAR, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from shared.database.connection import Base
from shared.models.conversation import (
    ConversationStatus, MessageRole, MessageType, AttachmentType, ToolCallStatus
)


# ==================== 对话模型 ====================
class ConversationORM(Base):
    """对话表"""
    __tablename__ = "conversations"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="对话ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        comment="用户ID"
    )
    
    # 基本信息
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="对话标题")
    status: Mapped[ConversationStatus] = mapped_column(
        SQLEnum(ConversationStatus), 
        nullable=False, 
        default=ConversationStatus.ACTIVE,
        comment="对话状态"
    )
    
    # 上下文信息
    context: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="对话上下文JSON")
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    
    # 统计信息
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="消息数量")
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="总令牌数")
    
    # 设置
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="zh-cn", comment="对话语言")
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="使用的模型")
    temperature: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, default=0.7, comment="温度参数")
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=4096, comment="最大令牌数")
    
    # 标签
    tags: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="标签JSON")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="最后活动时间"
    )
    
    # 关联关系
    messages: Mapped[List["MessageORM"]] = relationship(
        "MessageORM", 
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageORM.sequence_number"
    )
    summaries: Mapped[List["ConversationSummaryORM"]] = relationship(
        "ConversationSummaryORM", 
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    metrics: Mapped[List["ConversationMetricsORM"]] = relationship(
        "ConversationMetricsORM", 
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_conversations_user_id", "user_id"),
        Index("idx_conversations_status", "status"),
        Index("idx_conversations_created_at", "created_at"),
        Index("idx_conversations_last_activity", "last_activity_at"),
        Index("idx_conversations_language", "language"),
        CheckConstraint("message_count >= 0", name="check_message_count_positive"),
        CheckConstraint("total_tokens >= 0", name="check_total_tokens_positive"),
        CheckConstraint("temperature >= 0 AND temperature <= 2", name="check_temperature_range"),
        CheckConstraint("max_tokens >= 1", name="check_max_tokens_positive"),
    )


class MessageORM(Base):
    """消息表"""
    __tablename__ = "messages"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="消息ID")
    conversation_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False,
        comment="对话ID"
    )
    
    # 基本信息
    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(MessageRole), 
        nullable=False,
        comment="角色"
    )
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="消息内容")
    message_type: Mapped[MessageType] = mapped_column(
        SQLEnum(MessageType), 
        nullable=False, 
        default=MessageType.TEXT,
        comment="消息类型"
    )
    
    # 顺序和版本
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False, comment="序列号")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="版本号")
    parent_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("messages.id", ondelete="SET NULL"), 
        nullable=True,
        comment="父消息ID"
    )
    
    # AI相关信息
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="使用的模型")
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="提示令牌数")
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="完成令牌数")
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="总令牌数")
    
    # 元数据
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="推理过程")
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="置信度"
    )
    
    # 状态
    is_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否已编辑")
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否已删除")
    is_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否被标记")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    
    # 关联关系
    conversation: Mapped["ConversationORM"] = relationship("ConversationORM", back_populates="messages")
    parent: Mapped[Optional["MessageORM"]] = relationship("MessageORM", remote_side=[id])
    children: Mapped[List["MessageORM"]] = relationship("MessageORM", back_populates="parent")
    attachments: Mapped[List["MessageAttachmentORM"]] = relationship(
        "MessageAttachmentORM", 
        back_populates="message",
        cascade="all, delete-orphan"
    )
    tool_calls: Mapped[List["ToolCallORM"]] = relationship(
        "ToolCallORM", 
        back_populates="message",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_messages_conversation_id", "conversation_id"),
        Index("idx_messages_role", "role"),
        Index("idx_messages_type", "message_type"),
        Index("idx_messages_sequence", "conversation_id", "sequence_number"),
        Index("idx_messages_created_at", "created_at"),
        Index("idx_messages_parent_id", "parent_id"),
        CheckConstraint("sequence_number >= 0", name="check_sequence_number_positive"),
        CheckConstraint("version >= 1", name="check_version_positive"),
        CheckConstraint("prompt_tokens >= 0", name="check_prompt_tokens_positive"),
        CheckConstraint("completion_tokens >= 0", name="check_completion_tokens_positive"),
        CheckConstraint("total_tokens >= 0", name="check_total_tokens_positive"),
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)", name="check_confidence_range"),
    )


class MessageAttachmentORM(Base):
    """消息附件表"""
    __tablename__ = "message_attachments"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="附件ID")
    message_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("messages.id", ondelete="CASCADE"), 
        nullable=False,
        comment="消息ID"
    )
    
    # 文件信息
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, comment="文件名")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小（字节）")
    file_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="文件类型")
    attachment_type: Mapped[AttachmentType] = mapped_column(
        SQLEnum(AttachmentType), 
        nullable=False,
        comment="附件类型"
    )
    
    # 存储信息
    file_url: Mapped[str] = mapped_column(String(500), nullable=False, comment="文件URL")
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="文件路径")
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="缩略图URL")
    
    # 元数据
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    alt_text: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="替代文本")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="描述")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    message: Mapped["MessageORM"] = relationship("MessageORM", back_populates="attachments")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_message_attachments_message_id", "message_id"),
        Index("idx_message_attachments_type", "attachment_type"),
        Index("idx_message_attachments_created_at", "created_at"),
        CheckConstraint("file_size >= 0", name="check_file_size_positive"),
    )


class ToolCallORM(Base):
    """工具调用表"""
    __tablename__ = "tool_calls"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="工具调用ID")
    message_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("messages.id", ondelete="CASCADE"), 
        nullable=False,
        comment="消息ID"
    )
    
    # 工具信息
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="工具名称")
    tool_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="工具版本")
    function_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="函数名称")
    
    # 调用参数
    arguments: Mapped[str] = mapped_column(JSON, nullable=False, comment="调用参数JSON")
    
    # 执行信息
    status: Mapped[ToolCallStatus] = mapped_column(
        SQLEnum(ToolCallStatus), 
        nullable=False, 
        default=ToolCallStatus.PENDING,
        comment="执行状态"
    )
    result: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="执行结果JSON")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="错误信息")
    
    # 性能信息
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="执行时间（毫秒）")
    memory_used_mb: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True, comment="内存使用（MB）")
    
    # 时间戳
    started_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="开始时间"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="完成时间")
    
    # 关联关系
    message: Mapped["MessageORM"] = relationship("MessageORM", back_populates="tool_calls")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_tool_calls_message_id", "message_id"),
        Index("idx_tool_calls_tool_name", "tool_name"),
        Index("idx_tool_calls_status", "status"),
        Index("idx_tool_calls_started_at", "started_at"),
        CheckConstraint("execution_time_ms IS NULL OR execution_time_ms >= 0", name="check_execution_time_positive"),
        CheckConstraint("memory_used_mb IS NULL OR memory_used_mb >= 0", name="check_memory_used_positive"),
    )


# ==================== 对话分析模型 ====================
class ConversationSummaryORM(Base):
    """对话摘要表"""
    __tablename__ = "conversation_summaries"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="摘要ID")
    conversation_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False,
        comment="对话ID"
    )
    
    # 摘要内容
    summary: Mapped[str] = mapped_column(Text, nullable=False, comment="摘要内容")
    key_points: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="关键点JSON")
    action_items: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="行动项JSON")
    
    # 情感分析
    sentiment_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="情感分数"
    )
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="情感标签")
    
    # 主题分析
    topics: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="主题JSON")
    entities: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="实体JSON")
    intents: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="意图JSON")
    
    # 质量评估
    clarity_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="清晰度评分"
    )
    completeness_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="完整性评分"
    )
    satisfaction_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="满意度评分"
    )
    
    # 生成信息
    generated_by: Mapped[str] = mapped_column(String(100), nullable=False, comment="生成者")
    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="使用的模型")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    conversation: Mapped["ConversationORM"] = relationship("ConversationORM", back_populates="summaries")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_conversation_summaries_conversation_id", "conversation_id"),
        Index("idx_conversation_summaries_created_at", "created_at"),
        CheckConstraint("sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)", name="check_sentiment_score_range"),
        CheckConstraint("clarity_score IS NULL OR (clarity_score >= 0 AND clarity_score <= 1)", name="check_clarity_score_range"),
        CheckConstraint("completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 1)", name="check_completeness_score_range"),
        CheckConstraint("satisfaction_score IS NULL OR (satisfaction_score >= 0 AND satisfaction_score <= 1)", name="check_satisfaction_score_range"),
    )


class ConversationMetricsORM(Base):
    """对话指标表"""
    __tablename__ = "conversation_metrics"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="指标ID")
    conversation_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("conversations.id", ondelete="CASCADE"), 
        nullable=False,
        comment="对话ID"
    )
    
    # 基础指标
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, comment="持续时间（秒）")
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮次数")
    user_message_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="用户消息数")
    assistant_message_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="助手消息数")
    
    # 令牌使用
    total_prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, comment="总提示令牌")
    total_completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, comment="总完成令牌")
    average_response_time_ms: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False, comment="平均响应时间（毫秒）")
    
    # 工具使用
    tool_calls_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="工具调用次数")
    successful_tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="成功工具调用次数")
    failed_tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="失败工具调用次数")
    
    # 用户体验
    user_satisfaction: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="用户满意度"
    )
    task_completion_rate: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="任务完成率"
    )
    
    # 时间戳
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="计算时间"
    )
    
    # 关联关系
    conversation: Mapped["ConversationORM"] = relationship("ConversationORM", back_populates="metrics")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_conversation_metrics_conversation_id", "conversation_id"),
        Index("idx_conversation_metrics_calculated_at", "calculated_at"),
        CheckConstraint("duration_seconds >= 0", name="check_duration_positive"),
        CheckConstraint("turn_count >= 0", name="check_turn_count_positive"),
        CheckConstraint("user_message_count >= 0", name="check_user_message_count_positive"),
        CheckConstraint("assistant_message_count >= 0", name="check_assistant_message_count_positive"),
        CheckConstraint("total_prompt_tokens >= 0", name="check_prompt_tokens_positive"),
        CheckConstraint("total_completion_tokens >= 0", name="check_completion_tokens_positive"),
        CheckConstraint("average_response_time_ms >= 0", name="check_response_time_positive"),
        CheckConstraint("tool_calls_count >= 0", name="check_tool_calls_count_positive"),
        CheckConstraint("successful_tool_calls >= 0", name="check_successful_tool_calls_positive"),
        CheckConstraint("failed_tool_calls >= 0", name="check_failed_tool_calls_positive"),
        CheckConstraint("user_satisfaction IS NULL OR (user_satisfaction >= 0 AND user_satisfaction <= 5)", name="check_user_satisfaction_range"),
        CheckConstraint("task_completion_rate IS NULL OR (task_completion_rate >= 0 AND task_completion_rate <= 1)", name="check_task_completion_rate_range"),
    ) 