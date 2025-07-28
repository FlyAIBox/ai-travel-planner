"""
智能体域 SQLAlchemy ORM 模型
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
from shared.models.agent import (
    AgentType, AgentStatus, TaskStatus, TaskPriority, CollaborationType
)


# ==================== 智能体模型 ====================
class AgentORM(Base):
    """智能体表"""
    __tablename__ = "agents"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="智能体ID")
    
    # 基本信息
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="智能体名称")
    agent_type: Mapped[AgentType] = mapped_column(
        SQLEnum(AgentType), 
        nullable=False,
        comment="智能体类型"
    )
    description: Mapped[str] = mapped_column(Text, nullable=False, comment="描述")
    
    # 状态信息
    status: Mapped[AgentStatus] = mapped_column(
        SQLEnum(AgentStatus), 
        nullable=False, 
        default=AgentStatus.IDLE,
        comment="状态"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否激活")
    
    # 能力配置
    capabilities: Mapped[str] = mapped_column(JSON, nullable=False, comment="能力列表JSON")
    specialties: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="专长JSON")
    languages: Mapped[str] = mapped_column(JSON, nullable=False, comment="支持语言JSON")
    
    # 性格特征
    personality_traits: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="性格特征JSON")
    communication_style: Mapped[str] = mapped_column(
        String(50), 
        nullable=False, 
        default="professional",
        comment="沟通风格"
    )
    
    # 配置参数
    model_config: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="模型配置JSON")
    prompt_template: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="提示词模板")
    temperature: Mapped[float] = mapped_column(
        Numeric(3, 2), 
        nullable=False, 
        default=0.7,
        comment="温度参数"
    )
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=2048, comment="最大令牌数")
    
    # 性能指标
    success_rate: Mapped[float] = mapped_column(
        Numeric(5, 4), 
        nullable=False, 
        default=0.0,
        comment="成功率"
    )
    average_response_time: Mapped[float] = mapped_column(
        Numeric(8, 2), 
        nullable=False, 
        default=0.0,
        comment="平均响应时间"
    )
    total_interactions: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        default=0,
        comment="总交互次数"
    )
    
    # 学习能力
    learning_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否启用学习")
    feedback_score: Mapped[float] = mapped_column(
        Numeric(3, 2), 
        nullable=False, 
        default=0.0,
        comment="反馈评分"
    )
    
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
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="最后活跃时间")
    
    # 关联关系
    sessions: Mapped[List["AgentSessionORM"]] = relationship(
        "AgentSessionORM", 
        back_populates="agent",
        cascade="all, delete-orphan"
    )
    interactions: Mapped[List["AgentInteractionORM"]] = relationship(
        "AgentInteractionORM", 
        back_populates="agent",
        cascade="all, delete-orphan"
    )
    assigned_tasks: Mapped[List["TaskORM"]] = relationship(
        "TaskORM", 
        back_populates="assigned_agent"
    )
    performance_metrics: Mapped[List["AgentPerformanceMetricsORM"]] = relationship(
        "AgentPerformanceMetricsORM", 
        back_populates="agent",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_agents_type", "agent_type"),
        Index("idx_agents_status", "status"),
        Index("idx_agents_active", "is_active"),
        Index("idx_agents_name", "name"),
        Index("idx_agents_created_at", "created_at"),
        CheckConstraint("temperature >= 0 AND temperature <= 2", name="check_temperature_range"),
        CheckConstraint("max_tokens >= 1", name="check_max_tokens_positive"),
        CheckConstraint("success_rate >= 0 AND success_rate <= 1", name="check_success_rate_range"),
        CheckConstraint("average_response_time >= 0", name="check_response_time_positive"),
        CheckConstraint("total_interactions >= 0", name="check_interactions_positive"),
        CheckConstraint("feedback_score >= 0 AND feedback_score <= 5", name="check_feedback_score_range"),
    )


class AgentSessionORM(Base):
    """智能体会话表"""
    __tablename__ = "agent_sessions"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="会话ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        comment="用户ID"
    )
    agent_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="智能体ID"
    )
    
    # 会话信息
    session_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="会话名称")
    conversation_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("conversations.id", ondelete="SET NULL"), 
        nullable=True,
        comment="对话ID"
    )
    
    # 上下文管理
    context: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="会话上下文JSON")
    memory: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="会话记忆JSON")
    
    # 目标和任务
    goals: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="目标列表JSON")
    current_task: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="当前任务")
    
    # 状态管理
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否活跃")
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="交互次数")
    
    # 配置
    settings: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="会话设置JSON")
    
    # 时间戳
    started_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="开始时间"
    )
    last_interaction_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="最后交互时间"
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="结束时间")
    
    # 关联关系
    agent: Mapped["AgentORM"] = relationship("AgentORM", back_populates="sessions")
    interactions: Mapped[List["AgentInteractionORM"]] = relationship(
        "AgentInteractionORM", 
        back_populates="session",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_agent_sessions_user_id", "user_id"),
        Index("idx_agent_sessions_agent_id", "agent_id"),
        Index("idx_agent_sessions_conversation_id", "conversation_id"),
        Index("idx_agent_sessions_active", "is_active"),
        Index("idx_agent_sessions_started_at", "started_at"),
        CheckConstraint("interaction_count >= 0", name="check_interaction_count_positive"),
    )


class AgentInteractionORM(Base):
    """智能体交互表"""
    __tablename__ = "agent_interactions"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="交互ID")
    session_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agent_sessions.id", ondelete="CASCADE"), 
        nullable=False,
        comment="会话ID"
    )
    agent_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="智能体ID"
    )
    
    # 交互内容
    user_input: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="用户输入")
    agent_response: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="智能体响应")
    
    # 交互类型
    interaction_type: Mapped[str] = mapped_column(
        String(50), 
        nullable=False, 
        default="chat",
        comment="交互类型"
    )
    intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="意图识别")
    
    # 工具调用
    tools_used: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="使用的工具JSON")
    tool_results: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="工具结果JSON")
    
    # 性能指标
    response_time_ms: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False, comment="响应时间（毫秒）")
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="使用令牌数")
    
    # 质量评估
    user_satisfaction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="用户满意度")
    accuracy_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="准确性评分"
    )
    helpfulness_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="有用性评分"
    )
    
    # 上下文
    context_before: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="交互前上下文JSON")
    context_after: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="交互后上下文JSON")
    
    # 元数据
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    
    # 时间戳
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="交互时间"
    )
    
    # 关联关系
    session: Mapped["AgentSessionORM"] = relationship("AgentSessionORM", back_populates="interactions")
    agent: Mapped["AgentORM"] = relationship("AgentORM", back_populates="interactions")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_agent_interactions_session_id", "session_id"),
        Index("idx_agent_interactions_agent_id", "agent_id"),
        Index("idx_agent_interactions_type", "interaction_type"),
        Index("idx_agent_interactions_timestamp", "timestamp"),
        Index("idx_agent_interactions_intent", "intent"),
        CheckConstraint("response_time_ms >= 0", name="check_response_time_positive"),
        CheckConstraint("tokens_used >= 0", name="check_tokens_used_positive"),
        CheckConstraint("user_satisfaction IS NULL OR (user_satisfaction >= 1 AND user_satisfaction <= 5)", name="check_user_satisfaction_range"),
        CheckConstraint("accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 1)", name="check_accuracy_score_range"),
        CheckConstraint("helpfulness_score IS NULL OR (helpfulness_score >= 0 AND helpfulness_score <= 1)", name="check_helpfulness_score_range"),
    )


# ==================== 任务模型 ====================
class TaskORM(Base):
    """任务表"""
    __tablename__ = "tasks"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="任务ID")
    
    # 基本信息
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment="任务标题")
    description: Mapped[str] = mapped_column(Text, nullable=False, comment="任务描述")
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="任务类型")
    
    # 状态和优先级
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus), 
        nullable=False, 
        default=TaskStatus.PENDING,
        comment="任务状态"
    )
    priority: Mapped[TaskPriority] = mapped_column(
        SQLEnum(TaskPriority), 
        nullable=False, 
        default=TaskPriority.MEDIUM,
        comment="优先级"
    )
    
    # 分配信息
    assigned_agent_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="SET NULL"), 
        nullable=True,
        comment="分配的智能体ID"
    )
    requester_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        comment="请求者ID"
    )
    
    # 任务参数
    input_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="输入数据JSON")
    expected_output: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="期望输出JSON")
    constraints: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="约束条件JSON")
    
    # 执行信息
    execution_plan: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="执行计划JSON")
    progress: Mapped[float] = mapped_column(
        Numeric(3, 2), 
        nullable=False, 
        default=0.0,
        comment="进度"
    )
    
    # 依赖关系
    dependencies: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="依赖任务ID JSON")
    parent_task_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("tasks.id", ondelete="SET NULL"), 
        nullable=True,
        comment="父任务ID"
    )
    subtasks: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="子任务ID JSON")
    
    # 时间管理
    estimated_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="预估时长（分钟）")
    actual_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="实际时长（分钟）")
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="截止时间")
    
    # 结果
    result: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="任务结果JSON")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="错误信息")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    assigned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="分配时间")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="开始时间")
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="完成时间")
    
    # 关联关系
    assigned_agent: Mapped[Optional["AgentORM"]] = relationship("AgentORM", back_populates="assigned_tasks")
    parent_task: Mapped[Optional["TaskORM"]] = relationship("TaskORM", remote_side=[id])
    children_tasks: Mapped[List["TaskORM"]] = relationship("TaskORM", back_populates="parent_task")
    task_results: Mapped[List["TaskResultORM"]] = relationship(
        "TaskResultORM", 
        back_populates="task",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_tasks_status", "status"),
        Index("idx_tasks_priority", "priority"),
        Index("idx_tasks_type", "task_type"),
        Index("idx_tasks_assigned_agent_id", "assigned_agent_id"),
        Index("idx_tasks_requester_id", "requester_id"),
        Index("idx_tasks_parent_task_id", "parent_task_id"),
        Index("idx_tasks_created_at", "created_at"),
        Index("idx_tasks_deadline", "deadline"),
        CheckConstraint("progress >= 0 AND progress <= 1", name="check_progress_range"),
        CheckConstraint("estimated_duration IS NULL OR estimated_duration >= 0", name="check_estimated_duration_positive"),
        CheckConstraint("actual_duration IS NULL OR actual_duration >= 0", name="check_actual_duration_positive"),
    )


class TaskResultORM(Base):
    """任务结果表"""
    __tablename__ = "task_results"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="结果ID")
    task_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("tasks.id", ondelete="CASCADE"), 
        nullable=False,
        comment="任务ID"
    )
    agent_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="执行智能体ID"
    )
    
    # 结果内容
    output_data: Mapped[str] = mapped_column(JSON, nullable=False, comment="输出数据JSON")
    summary: Mapped[str] = mapped_column(Text, nullable=False, comment="结果摘要")
    
    # 执行信息
    execution_steps: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="执行步骤JSON")
    tools_used: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="使用的工具JSON")
    
    # 质量评估
    quality_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="质量评分"
    )
    completeness: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="完整性")
    accuracy: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="准确性")
    
    # 性能指标
    execution_time_ms: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, comment="执行时间（毫秒）")
    tokens_consumed: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="消耗令牌数")
    api_calls_made: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="API调用次数")
    
    # 错误处理
    errors_encountered: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="遇到的错误JSON")
    warnings: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="警告信息JSON")
    
    # 元数据
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    task: Mapped["TaskORM"] = relationship("TaskORM", back_populates="task_results")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_task_results_task_id", "task_id"),
        Index("idx_task_results_agent_id", "agent_id"),
        Index("idx_task_results_created_at", "created_at"),
        CheckConstraint("quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)", name="check_quality_score_range"),
        CheckConstraint("completeness >= 0 AND completeness <= 1", name="check_completeness_range"),
        CheckConstraint("accuracy >= 0 AND accuracy <= 1", name="check_accuracy_range"),
        CheckConstraint("execution_time_ms >= 0", name="check_execution_time_positive"),
        CheckConstraint("tokens_consumed >= 0", name="check_tokens_consumed_positive"),
        CheckConstraint("api_calls_made >= 0", name="check_api_calls_positive"),
    )


# ==================== 多智能体协作模型 ====================
class AgentTeamORM(Base):
    """智能体团队表"""
    __tablename__ = "agent_teams"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="团队ID")
    
    # 基本信息
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="团队名称")
    description: Mapped[str] = mapped_column(Text, nullable=False, comment="团队描述")
    
    # 团队成员
    leader_agent_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="团队领导ID"
    )
    member_agent_ids: Mapped[str] = mapped_column(JSON, nullable=False, comment="团队成员ID列表JSON")
    
    # 协作配置
    collaboration_type: Mapped[CollaborationType] = mapped_column(
        SQLEnum(CollaborationType), 
        nullable=False,
        comment="协作类型"
    )
    workflow: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="工作流程JSON")
    
    # 团队状态
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否活跃")
    current_project: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="当前项目")
    
    # 性能指标
    team_efficiency: Mapped[float] = mapped_column(
        Numeric(3, 2), 
        nullable=False, 
        default=0.0,
        comment="团队效率"
    )
    collaboration_score: Mapped[float] = mapped_column(
        Numeric(3, 2), 
        nullable=False, 
        default=0.0,
        comment="协作评分"
    )
    
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
    collaboration_sessions: Mapped[List["CollaborationSessionORM"]] = relationship(
        "CollaborationSessionORM", 
        back_populates="team",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_agent_teams_leader_id", "leader_agent_id"),
        Index("idx_agent_teams_active", "is_active"),
        Index("idx_agent_teams_created_at", "created_at"),
        CheckConstraint("team_efficiency >= 0 AND team_efficiency <= 1", name="check_team_efficiency_range"),
        CheckConstraint("collaboration_score >= 0 AND collaboration_score <= 1", name="check_collaboration_score_range"),
    )


class CollaborationSessionORM(Base):
    """协作会话表"""
    __tablename__ = "collaboration_sessions"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="协作会话ID")
    team_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agent_teams.id", ondelete="CASCADE"), 
        nullable=False,
        comment="团队ID"
    )
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        comment="用户ID"
    )
    
    # 会话信息
    objective: Mapped[str] = mapped_column(Text, nullable=False, comment="协作目标")
    plan: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="协作计划JSON")
    
    # 参与智能体
    participating_agents: Mapped[str] = mapped_column(JSON, nullable=False, comment="参与智能体ID JSON")
    current_speaker: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="SET NULL"), 
        nullable=True,
        comment="当前发言者ID"
    )
    
    # 状态管理
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否活跃")
    phase: Mapped[str] = mapped_column(String(50), nullable=False, default="planning", comment="当前阶段")
    
    # 进度跟踪
    progress: Mapped[float] = mapped_column(
        Numeric(3, 2), 
        nullable=False, 
        default=0.0,
        comment="进度"
    )
    milestones: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="里程碑JSON")
    
    # 结果
    outcome: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="协作结果JSON")
    
    # 时间戳
    started_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="开始时间"
    )
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="最后活动时间"
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="结束时间")
    
    # 关联关系
    team: Mapped["AgentTeamORM"] = relationship("AgentTeamORM", back_populates="collaboration_sessions")
    messages: Mapped[List["AgentMessageORM"]] = relationship(
        "AgentMessageORM", 
        back_populates="collaboration_session",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_collaboration_sessions_team_id", "team_id"),
        Index("idx_collaboration_sessions_user_id", "user_id"),
        Index("idx_collaboration_sessions_active", "is_active"),
        Index("idx_collaboration_sessions_started_at", "started_at"),
        CheckConstraint("progress >= 0 AND progress <= 1", name="check_progress_range"),
    )


class AgentMessageORM(Base):
    """智能体间消息表"""
    __tablename__ = "agent_messages"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="消息ID")
    collaboration_session_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("collaboration_sessions.id", ondelete="CASCADE"), 
        nullable=False,
        comment="协作会话ID"
    )
    
    # 发送者和接收者
    sender_agent_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="发送者智能体ID"
    )
    receiver_agent_ids: Mapped[str] = mapped_column(JSON, nullable=False, comment="接收者智能体ID列表JSON")
    
    # 消息内容
    message_type: Mapped[str] = mapped_column(String(50), nullable=False, comment="消息类型")
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="消息内容")
    data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="附加数据JSON")
    
    # 消息属性
    priority: Mapped[TaskPriority] = mapped_column(
        SQLEnum(TaskPriority), 
        nullable=False, 
        default=TaskPriority.MEDIUM,
        comment="优先级"
    )
    requires_response: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否需要回复")
    
    # 状态
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否已读")
    response_received: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否收到回复")
    
    # 时间戳
    sent_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="发送时间"
    )
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="阅读时间")
    
    # 关联关系
    collaboration_session: Mapped["CollaborationSessionORM"] = relationship("CollaborationSessionORM", back_populates="messages")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_agent_messages_session_id", "collaboration_session_id"),
        Index("idx_agent_messages_sender_id", "sender_agent_id"),
        Index("idx_agent_messages_type", "message_type"),
        Index("idx_agent_messages_sent_at", "sent_at"),
        Index("idx_agent_messages_priority", "priority"),
    )


# ==================== 智能体评估模型 ====================
class AgentPerformanceMetricsORM(Base):
    """智能体性能指标表"""
    __tablename__ = "agent_performance_metrics"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="指标ID")
    agent_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("agents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="智能体ID"
    )
    
    # 时间范围
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="统计开始时间")
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="统计结束时间")
    
    # 基础指标
    total_interactions: Mapped[int] = mapped_column(Integer, nullable=False, comment="总交互次数")
    successful_interactions: Mapped[int] = mapped_column(Integer, nullable=False, comment="成功交互次数")
    failed_interactions: Mapped[int] = mapped_column(Integer, nullable=False, comment="失败交互次数")
    
    # 性能指标
    average_response_time: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False, comment="平均响应时间")
    success_rate: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, comment="成功率")
    user_satisfaction_avg: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="平均用户满意度")
    
    # 效率指标
    tasks_completed: Mapped[int] = mapped_column(Integer, nullable=False, comment="完成任务数")
    average_task_duration: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False, comment="平均任务时长")
    resource_utilization: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="资源利用率")
    
    # 质量指标
    accuracy_score: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="准确性评分")
    helpfulness_score: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="有用性评分")
    coherence_score: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="连贯性评分")
    
    # 学习指标
    improvement_rate: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, default=0.0, comment="改进率")
    feedback_incorporation: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, default=0.0, comment="反馈采纳率")
    
    # 时间戳
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="计算时间"
    )
    
    # 关联关系
    agent: Mapped["AgentORM"] = relationship("AgentORM", back_populates="performance_metrics")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_agent_performance_metrics_agent_id", "agent_id"),
        Index("idx_agent_performance_metrics_period", "period_start", "period_end"),
        Index("idx_agent_performance_metrics_calculated_at", "calculated_at"),
        CheckConstraint("period_end >= period_start", name="check_period_range"),
        CheckConstraint("total_interactions >= 0", name="check_total_interactions_positive"),
        CheckConstraint("successful_interactions >= 0", name="check_successful_interactions_positive"),
        CheckConstraint("failed_interactions >= 0", name="check_failed_interactions_positive"),
        CheckConstraint("average_response_time >= 0", name="check_average_response_time_positive"),
        CheckConstraint("success_rate >= 0 AND success_rate <= 1", name="check_success_rate_range"),
        CheckConstraint("user_satisfaction_avg >= 0 AND user_satisfaction_avg <= 5", name="check_user_satisfaction_range"),
        CheckConstraint("tasks_completed >= 0", name="check_tasks_completed_positive"),
        CheckConstraint("average_task_duration >= 0", name="check_average_task_duration_positive"),
        CheckConstraint("resource_utilization >= 0 AND resource_utilization <= 1", name="check_resource_utilization_range"),
        CheckConstraint("accuracy_score >= 0 AND accuracy_score <= 1", name="check_accuracy_score_range"),
        CheckConstraint("helpfulness_score >= 0 AND helpfulness_score <= 1", name="check_helpfulness_score_range"),
        CheckConstraint("coherence_score >= 0 AND coherence_score <= 1", name="check_coherence_score_range"),
        CheckConstraint("improvement_rate >= 0", name="check_improvement_rate_positive"),
        CheckConstraint("feedback_incorporation >= 0 AND feedback_incorporation <= 1", name="check_feedback_incorporation_range"),
    ) 