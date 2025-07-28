"""
智能体域数据模型
包含智能体、会话、交互、任务等模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .user import BaseUser


# ==================== 枚举类型 ====================
class AgentType(str, Enum):
    """智能体类型"""
    PLANNER = "planner"               # 规划师
    RESEARCHER = "researcher"         # 研究员
    BOOKING_SPECIALIST = "booking"    # 预订专家
    BUDGET_ADVISOR = "budget"         # 预算顾问
    LOCAL_GUIDE = "guide"             # 本地向导
    TRANSLATOR = "translator"         # 翻译助手
    EMERGENCY_HELPER = "emergency"    # 紧急助手
    PHOTOGRAPHER = "photographer"     # 摄影师
    FOODIE = "foodie"                # 美食家
    COORDINATOR = "coordinator"       # 协调员


class AgentStatus(str, Enum):
    """智能体状态"""
    IDLE = "idle"                # 空闲
    ACTIVE = "active"            # 活跃
    BUSY = "busy"                # 忙碌
    OFFLINE = "offline"          # 离线
    MAINTENANCE = "maintenance"  # 维护中
    ERROR = "error"              # 错误状态


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"          # 待处理
    ASSIGNED = "assigned"        # 已分配
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消
    TIMEOUT = "timeout"          # 超时


class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"          # 低
    MEDIUM = "medium"    # 中
    HIGH = "high"        # 高
    URGENT = "urgent"    # 紧急
    CRITICAL = "critical" # 关键


class CollaborationType(str, Enum):
    """协作类型"""
    SEQUENTIAL = "sequential"    # 顺序协作
    PARALLEL = "parallel"        # 并行协作
    HIERARCHICAL = "hierarchical" # 层次协作
    PEER_TO_PEER = "peer_to_peer" # 点对点协作


# ==================== 智能体模型 ====================
class Agent(BaseUser):
    """智能体"""
    
    id: UUID = Field(default_factory=uuid4, description="智能体ID")
    
    # 基本信息
    name: str = Field(..., min_length=1, max_length=100, description="智能体名称")
    agent_type: AgentType = Field(..., description="智能体类型")
    description: str = Field(..., max_length=1000, description="描述")
    
    # 状态信息
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="状态")
    is_active: bool = Field(default=True, description="是否激活")
    
    # 能力配置
    capabilities: List[str] = Field(..., description="能力列表")
    specialties: List[str] = Field(default=[], description="专长")
    languages: List[str] = Field(default=["zh-cn"], description="支持语言")
    
    # 性格特征
    personality_traits: Dict[str, float] = Field(default={}, description="性格特征")
    communication_style: str = Field(default="professional", description="沟通风格")
    
    # 配置参数
    model_config: Dict[str, Any] = Field(default={}, description="模型配置")
    prompt_template: str = Field(..., description="提示词模板")
    temperature: float = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_tokens: int = Field(default=2048, ge=1, description="最大令牌数")
    
    # 性能指标
    success_rate: float = Field(default=0.0, ge=0, le=1, description="成功率")
    average_response_time: float = Field(default=0.0, ge=0, description="平均响应时间")
    total_interactions: int = Field(default=0, ge=0, description="总交互次数")
    
    # 学习能力
    learning_enabled: bool = Field(default=True, description="是否启用学习")
    feedback_score: float = Field(default=0.0, ge=0, le=5, description="反馈评分")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    last_active_at: Optional[datetime] = Field(None, description="最后活跃时间")


class AgentSession(BaseUser):
    """智能体会话"""
    
    id: UUID = Field(default_factory=uuid4, description="会话ID")
    user_id: UUID = Field(..., description="用户ID")
    agent_id: UUID = Field(..., description="智能体ID")
    
    # 会话信息
    session_name: Optional[str] = Field(None, max_length=200, description="会话名称")
    conversation_id: Optional[UUID] = Field(None, description="对话ID")
    
    # 上下文管理
    context: Dict[str, Any] = Field(default={}, description="会话上下文")
    memory: Dict[str, Any] = Field(default={}, description="会话记忆")
    
    # 目标和任务
    goals: List[str] = Field(default=[], description="目标列表")
    current_task: Optional[str] = Field(None, description="当前任务")
    
    # 状态管理
    is_active: bool = Field(default=True, description="是否活跃")
    interaction_count: int = Field(default=0, ge=0, description="交互次数")
    
    # 配置
    settings: Dict[str, Any] = Field(default={}, description="会话设置")
    
    # 时间戳
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    last_interaction_at: datetime = Field(default_factory=datetime.utcnow, description="最后交互时间")
    ended_at: Optional[datetime] = Field(None, description="结束时间")


class AgentInteraction(BaseUser):
    """智能体交互"""
    
    id: UUID = Field(default_factory=uuid4, description="交互ID")
    session_id: UUID = Field(..., description="会话ID")
    agent_id: UUID = Field(..., description="智能体ID")
    
    # 交互内容
    user_input: str = Field(..., description="用户输入")
    agent_response: str = Field(..., description="智能体响应")
    
    # 交互类型
    interaction_type: str = Field(default="chat", description="交互类型")
    intent: Optional[str] = Field(None, description="意图识别")
    
    # 工具调用
    tools_used: List[str] = Field(default=[], description="使用的工具")
    tool_results: Dict[str, Any] = Field(default={}, description="工具结果")
    
    # 性能指标
    response_time_ms: float = Field(..., ge=0, description="响应时间（毫秒）")
    tokens_used: int = Field(default=0, ge=0, description="使用令牌数")
    
    # 质量评估
    user_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="用户满意度")
    accuracy_score: Optional[float] = Field(None, ge=0, le=1, description="准确性评分")
    helpfulness_score: Optional[float] = Field(None, ge=0, le=1, description="有用性评分")
    
    # 上下文
    context_before: Dict[str, Any] = Field(default={}, description="交互前上下文")
    context_after: Dict[str, Any] = Field(default={}, description="交互后上下文")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="交互时间")


class Task(BaseUser):
    """任务"""
    
    id: UUID = Field(default_factory=uuid4, description="任务ID")
    
    # 基本信息
    title: str = Field(..., min_length=1, max_length=200, description="任务标题")
    description: str = Field(..., max_length=2000, description="任务描述")
    task_type: str = Field(..., description="任务类型")
    
    # 状态和优先级
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")
    
    # 分配信息
    assigned_agent_id: Optional[UUID] = Field(None, description="分配的智能体ID")
    requester_id: Optional[UUID] = Field(None, description="请求者ID")
    
    # 任务参数
    input_data: Dict[str, Any] = Field(default={}, description="输入数据")
    expected_output: Optional[Dict[str, Any]] = Field(None, description="期望输出")
    constraints: Dict[str, Any] = Field(default={}, description="约束条件")
    
    # 执行信息
    execution_plan: Optional[List[Dict[str, Any]]] = Field(None, description="执行计划")
    progress: float = Field(default=0.0, ge=0, le=1, description="进度")
    
    # 依赖关系
    dependencies: List[UUID] = Field(default=[], description="依赖任务ID")
    parent_task_id: Optional[UUID] = Field(None, description="父任务ID")
    subtasks: List[UUID] = Field(default=[], description="子任务ID")
    
    # 时间管理
    estimated_duration: Optional[int] = Field(None, ge=0, description="预估时长（分钟）")
    actual_duration: Optional[int] = Field(None, ge=0, description="实际时长（分钟）")
    deadline: Optional[datetime] = Field(None, description="截止时间")
    
    # 结果
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    assigned_at: Optional[datetime] = Field(None, description="分配时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")


class TaskResult(BaseUser):
    """任务结果"""
    
    id: UUID = Field(default_factory=uuid4, description="结果ID")
    task_id: UUID = Field(..., description="任务ID")
    agent_id: UUID = Field(..., description="执行智能体ID")
    
    # 结果内容
    output_data: Dict[str, Any] = Field(..., description="输出数据")
    summary: str = Field(..., description="结果摘要")
    
    # 执行信息
    execution_steps: List[Dict[str, Any]] = Field(default=[], description="执行步骤")
    tools_used: List[str] = Field(default=[], description="使用的工具")
    
    # 质量评估
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="质量评分")
    completeness: float = Field(..., ge=0, le=1, description="完整性")
    accuracy: float = Field(..., ge=0, le=1, description="准确性")
    
    # 性能指标
    execution_time_ms: float = Field(..., ge=0, description="执行时间（毫秒）")
    tokens_consumed: int = Field(default=0, ge=0, description="消耗令牌数")
    api_calls_made: int = Field(default=0, ge=0, description="API调用次数")
    
    # 错误处理
    errors_encountered: List[str] = Field(default=[], description="遇到的错误")
    warnings: List[str] = Field(default=[], description="警告信息")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


# ==================== 多智能体协作模型 ====================
class AgentTeam(BaseUser):
    """智能体团队"""
    
    id: UUID = Field(default_factory=uuid4, description="团队ID")
    
    # 基本信息
    name: str = Field(..., min_length=1, max_length=100, description="团队名称")
    description: str = Field(..., max_length=1000, description="团队描述")
    
    # 团队成员
    leader_agent_id: UUID = Field(..., description="团队领导ID")
    member_agent_ids: List[UUID] = Field(..., description="团队成员ID列表")
    
    # 协作配置
    collaboration_type: CollaborationType = Field(..., description="协作类型")
    workflow: Dict[str, Any] = Field(default={}, description="工作流程")
    
    # 团队状态
    is_active: bool = Field(default=True, description="是否活跃")
    current_project: Optional[str] = Field(None, description="当前项目")
    
    # 性能指标
    team_efficiency: float = Field(default=0.0, ge=0, le=1, description="团队效率")
    collaboration_score: float = Field(default=0.0, ge=0, le=1, description="协作评分")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")


class CollaborationSession(BaseUser):
    """协作会话"""
    
    id: UUID = Field(default_factory=uuid4, description="协作会话ID")
    team_id: UUID = Field(..., description="团队ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 会话信息
    objective: str = Field(..., description="协作目标")
    plan: Dict[str, Any] = Field(default={}, description="协作计划")
    
    # 参与智能体
    participating_agents: List[UUID] = Field(..., description="参与智能体ID")
    current_speaker: Optional[UUID] = Field(None, description="当前发言者ID")
    
    # 状态管理
    is_active: bool = Field(default=True, description="是否活跃")
    phase: str = Field(default="planning", description="当前阶段")
    
    # 进度跟踪
    progress: float = Field(default=0.0, ge=0, le=1, description="进度")
    milestones: List[Dict[str, Any]] = Field(default=[], description="里程碑")
    
    # 结果
    outcome: Optional[Dict[str, Any]] = Field(None, description="协作结果")
    
    # 时间戳
    started_at: datetime = Field(default_factory=datetime.utcnow, description="开始时间")
    last_activity_at: datetime = Field(default_factory=datetime.utcnow, description="最后活动时间")
    ended_at: Optional[datetime] = Field(None, description="结束时间")


class AgentMessage(BaseUser):
    """智能体间消息"""
    
    id: UUID = Field(default_factory=uuid4, description="消息ID")
    collaboration_session_id: UUID = Field(..., description="协作会话ID")
    
    # 发送者和接收者
    sender_agent_id: UUID = Field(..., description="发送者智能体ID")
    receiver_agent_ids: List[UUID] = Field(..., description="接收者智能体ID列表")
    
    # 消息内容
    message_type: str = Field(..., description="消息类型")
    content: str = Field(..., description="消息内容")
    data: Dict[str, Any] = Field(default={}, description="附加数据")
    
    # 消息属性
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")
    requires_response: bool = Field(default=False, description="是否需要回复")
    
    # 状态
    is_read: bool = Field(default=False, description="是否已读")
    response_received: bool = Field(default=False, description="是否收到回复")
    
    # 时间戳
    sent_at: datetime = Field(default_factory=datetime.utcnow, description="发送时间")
    read_at: Optional[datetime] = Field(None, description="阅读时间")


# ==================== 智能体评估模型 ====================
class AgentPerformanceMetrics(BaseUser):
    """智能体性能指标"""
    
    id: UUID = Field(default_factory=uuid4, description="指标ID")
    agent_id: UUID = Field(..., description="智能体ID")
    
    # 时间范围
    period_start: datetime = Field(..., description="统计开始时间")
    period_end: datetime = Field(..., description="统计结束时间")
    
    # 基础指标
    total_interactions: int = Field(..., ge=0, description="总交互次数")
    successful_interactions: int = Field(..., ge=0, description="成功交互次数")
    failed_interactions: int = Field(..., ge=0, description="失败交互次数")
    
    # 性能指标
    average_response_time: float = Field(..., ge=0, description="平均响应时间")
    success_rate: float = Field(..., ge=0, le=1, description="成功率")
    user_satisfaction_avg: float = Field(..., ge=0, le=5, description="平均用户满意度")
    
    # 效率指标
    tasks_completed: int = Field(..., ge=0, description="完成任务数")
    average_task_duration: float = Field(..., ge=0, description="平均任务时长")
    resource_utilization: float = Field(..., ge=0, le=1, description="资源利用率")
    
    # 质量指标
    accuracy_score: float = Field(..., ge=0, le=1, description="准确性评分")
    helpfulness_score: float = Field(..., ge=0, le=1, description="有用性评分")
    coherence_score: float = Field(..., ge=0, le=1, description="连贯性评分")
    
    # 学习指标
    improvement_rate: float = Field(default=0.0, description="改进率")
    feedback_incorporation: float = Field(default=0.0, ge=0, le=1, description="反馈采纳率")
    
    # 时间戳
    calculated_at: datetime = Field(default_factory=datetime.utcnow, description="计算时间")


# ==================== 请求/响应模型 ====================
class AgentCreate(BaseUser):
    """创建智能体请求"""
    name: str = Field(..., min_length=1, max_length=100)
    agent_type: AgentType
    description: str = Field(..., max_length=1000)
    capabilities: List[str]
    specialties: List[str] = Field(default=[])
    languages: List[str] = Field(default=["zh-cn"])
    prompt_template: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    personality_traits: Dict[str, float] = Field(default={})
    model_config: Dict[str, Any] = Field(default={})


class AgentUpdate(BaseUser):
    """更新智能体请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    status: Optional[AgentStatus] = None
    is_active: Optional[bool] = None
    capabilities: Optional[List[str]] = None
    specialties: Optional[List[str]] = None
    prompt_template: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    personality_traits: Optional[Dict[str, float]] = None


class AgentResponse(BaseUser):
    """智能体响应"""
    id: UUID
    name: str
    agent_type: AgentType
    description: str
    status: AgentStatus
    is_active: bool
    capabilities: List[str]
    specialties: List[str]
    languages: List[str]
    success_rate: float
    total_interactions: int
    created_at: datetime
    last_active_at: Optional[datetime]
    
    class Config(BaseUser.Config):
        orm_mode = True


class TaskCreate(BaseUser):
    """创建任务请求"""
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    task_type: str
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    assigned_agent_id: Optional[UUID] = None
    input_data: Dict[str, Any] = Field(default={})
    constraints: Dict[str, Any] = Field(default={})
    deadline: Optional[datetime] = None


class TaskResponse(BaseUser):
    """任务响应"""
    id: UUID
    title: str
    description: str
    task_type: str
    status: TaskStatus
    priority: TaskPriority
    assigned_agent_id: Optional[UUID]
    progress: float
    estimated_duration: Optional[int]
    actual_duration: Optional[int]
    deadline: Optional[datetime]
    created_at: datetime
    completed_at: Optional[datetime]
    
    class Config(BaseUser.Config):
        orm_mode = True


class AgentSessionCreate(BaseUser):
    """创建智能体会话请求"""
    agent_id: UUID
    session_name: Optional[str] = Field(None, max_length=200)
    goals: List[str] = Field(default=[])
    context: Dict[str, Any] = Field(default={})
    settings: Dict[str, Any] = Field(default={})


class InteractionRequest(BaseUser):
    """交互请求"""
    session_id: UUID
    user_input: str = Field(..., min_length=1)
    interaction_type: str = Field(default="chat")
    context: Dict[str, Any] = Field(default={})


class InteractionResponse(BaseUser):
    """交互响应"""
    id: UUID
    agent_response: str
    intent: Optional[str]
    tools_used: List[str]
    response_time_ms: float
    tokens_used: int
    context_after: Dict[str, Any]
    timestamp: datetime 