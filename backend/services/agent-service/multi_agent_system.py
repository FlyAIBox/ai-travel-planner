"""
多智能体系统框架
基于LangChain实现智能体协作、任务调度、状态管理等功能
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque

try:
    from langchain.agents import Agent, AgentExecutor, Tool
    from langchain.agents.agent import AgentAction, AgentFinish
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.callbacks.manager import CallbackManagerForChainRun
    from langchain.prompts import PromptTemplate
    from langchain.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # 创建模拟类
    class BaseMessage: pass
    class HumanMessage(BaseMessage): 
        def __init__(self, content: str): 
            self.content = content
    class AIMessage(BaseMessage): 
        def __init__(self, content: str): 
            self.content = content
    class SystemMessage(BaseMessage): 
        def __init__(self, content: str): 
            self.content = content
    class BaseTool:
        def __init__(self, name: str, description: str, func: Callable = None):
            self.name = name
            self.description = description
            self.func = func
        def run(self, input_str: str) -> str:
            if self.func:
                return self.func(input_str)
            return f"执行工具 {self.name}: {input_str}"

import structlog
from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AgentRole(Enum):
    """智能体角色"""
    COORDINATOR = "coordinator"      # 协调者
    FLIGHT_EXPERT = "flight_expert"  # 航班专家
    HOTEL_EXPERT = "hotel_expert"    # 酒店专家
    ITINERARY_PLANNER = "itinerary_planner"  # 行程规划师
    BUDGET_ANALYST = "budget_analyst"        # 预算分析师
    LOCAL_GUIDE = "local_guide"      # 当地向导
    TRANSLATOR = "translator"        # 翻译专家
    WEATHER_EXPERT = "weather_expert"  # 天气专家


class AgentTask(Enum):
    """智能体任务类型"""
    SEARCH_FLIGHTS = "search_flights"
    SEARCH_HOTELS = "search_hotels"
    PLAN_ITINERARY = "plan_itinerary"
    ANALYZE_BUDGET = "analyze_budget"
    PROVIDE_LOCAL_INFO = "provide_local_info"
    TRANSLATE_CONTENT = "translate_content"
    CHECK_WEATHER = "check_weather"
    COORDINATE_AGENTS = "coordinate_agents"
    MAKE_RECOMMENDATION = "make_recommendation"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentMessage:
    """智能体消息"""
    id: str
    sender_id: str
    receiver_id: str
    content: str
    message_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now()


@dataclass
class AgentTaskRequest:
    """智能体任务请求"""
    id: str
    task_type: AgentTask
    requester_id: str
    assignee_id: Optional[str]
    priority: TaskPriority
    parameters: Dict[str, Any]
    deadline: Optional[datetime]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass 
class AgentCapability:
    """智能体能力"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    execution_time_estimate: float  # 预估执行时间（秒）
    success_rate: float = 0.95      # 成功率
    cost_estimate: float = 0.0      # 成本估算


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    role: AgentRole
    status: str  # idle, busy, error, offline
    current_tasks: List[str]
    completed_tasks: int
    failed_tasks: int
    last_activity: datetime
    load_factor: float  # 负载因子 0-1
    capabilities: List[AgentCapability]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: deque = deque()
        self.message_history: List[AgentMessage] = []
        self.processing = False
        self.max_history = 1000
        self._lock = threading.Lock()
    
    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]):
        """订阅消息"""
        with self._lock:
            self.subscribers[agent_id].append(callback)
    
    def unsubscribe(self, agent_id: str, callback: Callable = None):
        """取消订阅"""
        with self._lock:
            if callback:
                if callback in self.subscribers[agent_id]:
                    self.subscribers[agent_id].remove(callback)
            else:
                self.subscribers[agent_id] = []
    
    async def publish(self, message: AgentMessage):
        """发布消息"""
        with self._lock:
            self.message_queue.append(message)
            self.message_history.append(message)
            
            # 限制历史记录大小
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history//2:]
        
        # 处理消息队列
        if not self.processing:
            await self._process_message_queue()
    
    async def _process_message_queue(self):
        """处理消息队列"""
        self.processing = True
        
        try:
            while self.message_queue:
                with self._lock:
                    if not self.message_queue:
                        break
                    message = self.message_queue.popleft()
                
                # 分发消息
                await self._distribute_message(message)
                
        except Exception as e:
            logger.error(f"消息处理失败: {e}")
        finally:
            self.processing = False
    
    async def _distribute_message(self, message: AgentMessage):
        """分发消息"""
        receiver_id = message.receiver_id
        
        # 广播消息
        if receiver_id == "*":
            for agent_id, callbacks in self.subscribers.items():
                for callback in callbacks:
                    try:
                        await self._safe_callback(callback, message)
                    except Exception as e:
                        logger.error(f"广播消息到 {agent_id} 失败: {e}")
        
        # 单播消息
        elif receiver_id in self.subscribers:
            for callback in self.subscribers[receiver_id]:
                try:
                    await self._safe_callback(callback, message)
                except Exception as e:
                    logger.error(f"发送消息到 {receiver_id} 失败: {e}")
    
    async def _safe_callback(self, callback: Callable, message: AgentMessage):
        """安全执行回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.error(f"回调执行失败: {e}")
    
    def get_message_history(self, agent_id: str = None, 
                          message_type: str = None,
                          limit: int = 100) -> List[AgentMessage]:
        """获取消息历史"""
        with self._lock:
            messages = self.message_history
            
            # 过滤条件
            if agent_id:
                messages = [m for m in messages 
                          if m.sender_id == agent_id or m.receiver_id == agent_id]
            
            if message_type:
                messages = [m for m in messages if m.message_type == message_type]
            
            return messages[-limit:] if limit else messages


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.pending_tasks: Dict[str, AgentTaskRequest] = {}
        self.running_tasks: Dict[str, AgentTaskRequest] = {}
        self.completed_tasks: Dict[str, AgentTaskRequest] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
    
    def submit_task(self, task: AgentTaskRequest) -> str:
        """提交任务"""
        with self._lock:
            self.pending_tasks[task.id] = task
            
            # 处理依赖关系
            if task.dependencies:
                self.task_dependencies[task.id] = task.dependencies.copy()
        
        logger.info(f"任务 {task.id} 已提交，类型: {task.task_type.value}")
        return task.id
    
    def get_ready_tasks(self, agent_capabilities: List[AgentCapability]) -> List[AgentTaskRequest]:
        """获取可执行的任务"""
        ready_tasks = []
        capability_names = [cap.name for cap in agent_capabilities]
        
        with self._lock:
            for task_id, task in self.pending_tasks.items():
                # 检查依赖是否完成
                if self._dependencies_satisfied(task_id):
                    # 检查智能体能力
                    if task.task_type.value in capability_names:
                        ready_tasks.append(task)
        
        # 按优先级和创建时间排序
        ready_tasks.sort(key=lambda t: (-t.priority.value, t.created_at))
        return ready_tasks
    
    def _dependencies_satisfied(self, task_id: str) -> bool:
        """检查任务依赖是否满足"""
        if task_id not in self.task_dependencies:
            return True
        
        for dep_id in self.task_dependencies[task_id]:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def start_task(self, task_id: str, agent_id: str) -> bool:
        """开始执行任务"""
        with self._lock:
            if task_id not in self.pending_tasks:
                return False
            
            task = self.pending_tasks.pop(task_id)
            task.assignee_id = agent_id
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.now()
            self.running_tasks[task_id] = task
        
        logger.info(f"任务 {task_id} 开始执行，分配给智能体 {agent_id}")
        return True
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """完成任务"""
        with self._lock:
            if task_id not in self.running_tasks:
                return False
            
            task = self.running_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.updated_at = datetime.now()
            self.completed_tasks[task_id] = task
            
            # 清理依赖关系
            if task_id in self.task_dependencies:
                del self.task_dependencies[task_id]
        
        logger.info(f"任务 {task_id} 执行完成")
        return True
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """任务失败"""
        with self._lock:
            if task_id not in self.running_tasks:
                return False
            
            task = self.running_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            task.error_message = error_message
            task.updated_at = datetime.now()
            self.completed_tasks[task_id] = task
        
        logger.error(f"任务 {task_id} 执行失败: {error_message}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[AgentTaskRequest]:
        """获取任务状态"""
        for task_dict in [self.pending_tasks, self.running_tasks, self.completed_tasks]:
            if task_id in task_dict:
                return task_dict[task_id]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取调度统计"""
        with self._lock:
            return {
                "pending_tasks": len(self.pending_tasks),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "total_tasks": len(self.pending_tasks) + len(self.running_tasks) + len(self.completed_tasks)
            }


class BaseAgent(ABC):
    """基础智能体类"""
    
    def __init__(self, agent_id: str, role: AgentRole, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.state = AgentState(
            agent_id=agent_id,
            role=role,
            status="idle",
            current_tasks=[],
            completed_tasks=0,
            failed_tasks=0,
            last_activity=datetime.now(),
            load_factor=0.0,
            capabilities=capabilities
        )
        
        # 组件
        self.message_bus: Optional[MessageBus] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        
        # 工具
        self.tools: List[BaseTool] = []
        
        # 状态管理
        self.is_running = False
        self.task_executor = ThreadPoolExecutor(max_workers=3)
        
        # 初始化工具
        self._initialize_tools()
    
    @abstractmethod
    def _initialize_tools(self):
        """初始化智能体工具"""
        pass
    
    @abstractmethod
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理任务"""
        pass
    
    def set_message_bus(self, message_bus: MessageBus):
        """设置消息总线"""
        self.message_bus = message_bus
        self.message_bus.subscribe(self.agent_id, self._handle_message)
    
    def set_task_scheduler(self, scheduler: TaskScheduler):
        """设置任务调度器"""
        self.task_scheduler = scheduler
    
    async def start(self):
        """启动智能体"""
        self.is_running = True
        self.state.status = "idle"
        logger.info(f"智能体 {self.agent_id} ({self.role.value}) 已启动")
        
        # 启动任务处理循环
        asyncio.create_task(self._task_processing_loop())
    
    async def stop(self):
        """停止智能体"""
        self.is_running = False
        self.state.status = "offline"
        logger.info(f"智能体 {self.agent_id} 已停止")
    
    async def _task_processing_loop(self):
        """任务处理循环"""
        while self.is_running:
            try:
                if self.task_scheduler and self.state.status == "idle":
                    # 获取可执行任务
                    ready_tasks = self.task_scheduler.get_ready_tasks(self.capabilities)
                    
                    if ready_tasks:
                        task = ready_tasks[0]  # 选择优先级最高的任务
                        
                        # 开始执行任务
                        if self.task_scheduler.start_task(task.id, self.agent_id):
                            await self._execute_task(task)
                
                # 等待一段时间再检查
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"智能体 {self.agent_id} 任务处理循环出错: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: AgentTaskRequest):
        """执行任务"""
        self.state.status = "busy"
        self.state.current_tasks.append(task.id)
        self.state.last_activity = datetime.now()
        
        try:
            logger.info(f"智能体 {self.agent_id} 开始执行任务 {task.id}")
            
            # 处理任务
            result = await self.process_task(task)
            
            # 任务完成
            if self.task_scheduler:
                self.task_scheduler.complete_task(task.id, result)
            
            self.state.completed_tasks += 1
            
            # 发送完成消息
            if self.message_bus:
                completion_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=task.requester_id,
                    content=f"任务 {task.id} 执行完成",
                    message_type="task_completed",
                    timestamp=datetime.now(),
                    metadata={"task_id": task.id, "result": result}
                )
                await self.message_bus.publish(completion_message)
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 执行任务 {task.id} 失败: {e}")
            
            # 任务失败
            if self.task_scheduler:
                self.task_scheduler.fail_task(task.id, str(e))
            
            self.state.failed_tasks += 1
            
            # 发送失败消息
            if self.message_bus:
                failure_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=task.requester_id,
                    content=f"任务 {task.id} 执行失败: {str(e)}",
                    message_type="task_failed",
                    timestamp=datetime.now(),
                    metadata={"task_id": task.id, "error": str(e)}
                )
                await self.message_bus.publish(failure_message)
        
        finally:
            # 更新状态
            self.state.current_tasks.remove(task.id)
            self.state.status = "idle"
            self.state.last_activity = datetime.now()
            
            # 更新负载因子
            self.state.load_factor = len(self.state.current_tasks) / max(len(self.capabilities), 1)
    
    async def _handle_message(self, message: AgentMessage):
        """处理消息"""
        try:
            logger.debug(f"智能体 {self.agent_id} 收到消息: {message.message_type}")
            
            if message.message_type == "task_request":
                # 处理任务请求
                await self._handle_task_request(message)
            elif message.message_type == "collaboration_request":
                # 处理协作请求
                await self._handle_collaboration_request(message)
            elif message.message_type == "status_query":
                # 处理状态查询
                await self._handle_status_query(message)
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 处理消息失败: {e}")
    
    async def _handle_task_request(self, message: AgentMessage):
        """处理任务请求"""
        # 可以在这里实现直接任务分配逻辑
        pass
    
    async def _handle_collaboration_request(self, message: AgentMessage):
        """处理协作请求"""
        # 智能体间协作逻辑
        pass
    
    async def _handle_status_query(self, message: AgentMessage):
        """处理状态查询"""
        if self.message_bus:
            status_response = AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=json.dumps(asdict(self.state)),
                message_type="status_response",
                timestamp=datetime.now()
            )
            await self.message_bus.publish(status_response)
    
    def send_message(self, receiver_id: str, content: str, 
                    message_type: str = "general") -> bool:
        """发送消息"""
        if not self.message_bus:
            return False
        
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            timestamp=datetime.now()
        )
        
        asyncio.create_task(self.message_bus.publish(message))
        return True
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def get_state(self) -> AgentState:
        """获取智能体状态"""
        return self.state
    
    def update_load_factor(self):
        """更新负载因子"""
        max_tasks = len(self.capabilities) * 2  # 每个能力最多处理2个任务
        self.state.load_factor = len(self.state.current_tasks) / max(max_tasks, 1)


class MultiAgentSystem:
    """多智能体系统"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = MessageBus()
        self.task_scheduler = TaskScheduler()
        
        # 系统状态
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # 性能监控
        self.performance_metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_task_time": 0.0,
            "system_load": 0.0
        }
        
        # 启动监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
    
    def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        agent.set_message_bus(self.message_bus)
        agent.set_task_scheduler(self.task_scheduler)
        self.agents[agent.agent_id] = agent
        logger.info(f"智能体 {agent.agent_id} ({agent.role.value}) 已注册")
    
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            self.message_bus.unsubscribe(agent_id)
            del self.agents[agent_id]
            logger.info(f"智能体 {agent_id} 已注销")
    
    async def start_system(self):
        """启动系统"""
        self.is_running = True
        self.start_time = datetime.now()
        
        # 启动所有智能体
        for agent in self.agents.values():
            await agent.start()
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("多智能体系统已启动")
    
    async def stop_system(self):
        """停止系统"""
        self.is_running = False
        
        # 停止所有智能体
        for agent in self.agents.values():
            await agent.stop()
        
        # 停止监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("多智能体系统已停止")
    
    async def submit_task(self, task_type: AgentTask, parameters: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         requester_id: str = "system") -> str:
        """提交任务"""
        task = AgentTaskRequest(
            id=str(uuid.uuid4()),
            task_type=task_type,
            requester_id=requester_id,
            assignee_id=None,
            priority=priority,
            parameters=parameters,
            deadline=None
        )
        
        task_id = self.task_scheduler.submit_task(task)
        logger.info(f"任务 {task_id} 已提交到系统")
        return task_id
    
    def get_agent_by_role(self, role: AgentRole) -> Optional[BaseAgent]:
        """根据角色获取智能体"""
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        return None
    
    def get_agents_by_capability(self, capability_name: str) -> List[BaseAgent]:
        """根据能力获取智能体"""
        matching_agents = []
        for agent in self.agents.values():
            for capability in agent.capabilities:
                if capability.name == capability_name:
                    matching_agents.append(agent)
                    break
        return matching_agents
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        agent_statuses = {}
        total_load = 0.0
        
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                "role": agent.role.value,
                "status": agent.state.status,
                "load_factor": agent.state.load_factor,
                "completed_tasks": agent.state.completed_tasks,
                "failed_tasks": agent.state.failed_tasks,
                "current_tasks": len(agent.state.current_tasks)
            }
            total_load += agent.state.load_factor
        
        system_load = total_load / max(len(self.agents), 1)
        
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None,
            "agents": agent_statuses,
            "system_load": system_load,
            "task_statistics": self.task_scheduler.get_statistics(),
            "performance_metrics": self.performance_metrics
        }
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 更新性能指标
                await self._update_performance_metrics()
                
                # 检查智能体健康状态
                await self._health_check()
                
                # 等待下次监控
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(5)
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        # 计算系统负载
        total_load = sum(agent.state.load_factor for agent in self.agents.values())
        self.performance_metrics["system_load"] = total_load / max(len(self.agents), 1)
        
        # 更新任务统计
        task_stats = self.task_scheduler.get_statistics()
        self.performance_metrics["total_tasks_processed"] = task_stats["completed_tasks"]
        
        # 计算成功率
        total_completed = sum(agent.state.completed_tasks for agent in self.agents.values())
        total_failed = sum(agent.state.failed_tasks for agent in self.agents.values())
        
        if total_completed + total_failed > 0:
            success_rate = total_completed / (total_completed + total_failed)
            self.performance_metrics["success_rate"] = success_rate
    
    async def _health_check(self):
        """健康检查"""
        for agent_id, agent in self.agents.items():
            # 检查智能体是否响应
            last_activity = agent.state.last_activity
            if datetime.now() - last_activity > timedelta(minutes=5):
                logger.warning(f"智能体 {agent_id} 可能无响应，上次活动: {last_activity}")
            
            # 检查负载
            if agent.state.load_factor > 0.9:
                logger.warning(f"智能体 {agent_id} 负载过高: {agent.state.load_factor}")
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """协调智能体"""
        # 这里可以实现复杂的智能体协调逻辑
        # 例如：任务分解、资源分配、冲突解决等
        
        coordinator = self.get_agent_by_role(AgentRole.COORDINATOR)
        if coordinator:
            # 使用协调者智能体处理
            task = AgentTaskRequest(
                id=str(uuid.uuid4()),
                task_type=AgentTask.COORDINATE_AGENTS,
                requester_id="system",
                assignee_id=coordinator.agent_id,
                priority=TaskPriority.HIGH,
                parameters=coordination_request
            )
            
            result = await coordinator.process_task(task)
            return result
        else:
            # 简单的协调逻辑
            return {"status": "no_coordinator", "message": "没有可用的协调者智能体"}


# 全局多智能体系统实例
_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """获取多智能体系统实例"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system 