"""
多角色智能体系统
实现LangChain智能体框架、MultiAgentSystem协调器架构、智能体基类和消息总线机制、任务分发和结果聚合逻辑
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent import AgentAction, AgentFinish
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import redis.asyncio as redis

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AgentRole(Enum):
    """智能体角色"""
    COORDINATOR = "coordinator"           # 主控协调者
    FLIGHT_EXPERT = "flight_expert"      # 航班搜索专家
    HOTEL_EXPERT = "hotel_expert"        # 酒店推荐专家
    ITINERARY_PLANNER = "itinerary_planner"  # 行程规划师
    BUDGET_ANALYST = "budget_analyst"    # 预算分析师
    LOCAL_GUIDE = "local_guide"          # 当地向导
    WEATHER_ADVISOR = "weather_advisor"  # 天气顾问


class TaskType(Enum):
    """任务类型"""
    SEARCH = "search"                    # 搜索任务
    ANALYZE = "analyze"                  # 分析任务
    PLAN = "plan"                       # 规划任务
    RECOMMEND = "recommend"             # 推荐任务
    VALIDATE = "validate"               # 验证任务
    COORDINATE = "coordinate"           # 协调任务


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"                 # 待处理
    ASSIGNED = "assigned"               # 已分配
    IN_PROGRESS = "in_progress"         # 进行中
    COMPLETED = "completed"             # 已完成
    FAILED = "failed"                   # 失败
    CANCELLED = "cancelled"             # 已取消


class MessageType(Enum):
    """消息类型"""
    TASK_REQUEST = "task_request"       # 任务请求
    TASK_RESPONSE = "task_response"     # 任务响应
    COLLABORATION = "collaboration"     # 协作消息
    STATUS_UPDATE = "status_update"     # 状态更新
    ERROR = "error"                     # 错误消息
    BROADCAST = "broadcast"             # 广播消息


@dataclass
class AgentTask:
    """智能体任务"""
    task_id: str
    task_type: TaskType
    content: str
    context: Dict[str, Any]
    requester_id: str
    assignee_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # 0-10，10最高
    deadline: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['task_type'] = self.task_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.deadline:
            data['deadline'] = self.deadline.isoformat()
        return data


@dataclass
class AgentMessage:
    """智能体消息"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: Optional[str]  # None表示广播
    content: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AgentCallback(BaseCallbackHandler):
    """智能体回调处理器"""
    
    def __init__(self, agent_id: str, message_bus: 'MessageBus'):
        self.agent_id = agent_id
        self.message_bus = message_bus
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """智能体执行动作时的回调"""
        logger.info(f"Agent {self.agent_id} executing action: {action.tool}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """智能体完成任务时的回调"""
        logger.info(f"Agent {self.agent_id} finished task")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """工具开始执行时的回调"""
        logger.info(f"Agent {self.agent_id} starting tool: {serialized.get('name', 'unknown')}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """工具执行完成时的回调"""
        logger.info(f"Agent {self.agent_id} tool output: {output[:100]}...")


class MessageBus:
    """消息总线"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history = 1000
    
    async def initialize(self):
        """初始化消息总线"""
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
        
        logger.info("消息总线初始化完成")
    
    async def send_message(self, message: AgentMessage):
        """发送消息"""
        # 保存到历史记录
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
        # 发布到Redis
        if self.redis_client:
            channel = f"agent_messages_{message.receiver_id}" if message.receiver_id else "agent_messages_broadcast"
            await self.redis_client.publish(channel, json.dumps(message.to_dict()))
        
        # 通知本地订阅者
        if message.receiver_id in self.subscribers:
            for callback in self.subscribers[message.receiver_id]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"消息处理回调失败: {e}")
        
        # 广播消息
        if message.receiver_id is None and "broadcast" in self.subscribers:
            for callback in self.subscribers["broadcast"]:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"广播消息处理回调失败: {e}")
        
        logger.debug(f"消息已发送: {message.message_id}")
    
    def subscribe(self, agent_id: str, callback: Callable):
        """订阅消息"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        logger.debug(f"智能体 {agent_id} 已订阅消息")
    
    def unsubscribe(self, agent_id: str, callback: Callable):
        """取消订阅"""
        if agent_id in self.subscribers:
            if callback in self.subscribers[agent_id]:
                self.subscribers[agent_id].remove(callback)
                logger.debug(f"智能体 {agent_id} 已取消订阅")
    
    async def get_message_history(self, 
                                 agent_id: Optional[str] = None,
                                 message_type: Optional[MessageType] = None,
                                 limit: int = 100) -> List[AgentMessage]:
        """获取消息历史"""
        filtered_messages = self.message_history
        
        if agent_id:
            filtered_messages = [msg for msg in filtered_messages 
                               if msg.sender_id == agent_id or msg.receiver_id == agent_id]
        
        if message_type:
            filtered_messages = [msg for msg in filtered_messages 
                               if msg.message_type == message_type]
        
        return filtered_messages[-limit:]
    
    async def cleanup(self):
        """清理资源"""
        if self.redis_client:
            await self.redis_client.close()


class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, 
                 agent_id: str,
                 role: AgentRole,
                 message_bus: MessageBus,
                 tools: List[BaseTool] = None,
                 memory_window: int = 10):
        self.agent_id = agent_id
        self.role = role
        self.message_bus = message_bus
        self.tools = tools or []
        
        # LangChain组件
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.callback_handler = AgentCallback(agent_id, message_bus)
        
        # 状态管理
        self.is_active = False
        self.current_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[str] = []
        self.capabilities: List[TaskType] = []
        
        # 性能统计
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "last_active": None
        }
        
        # 订阅消息
        self.message_bus.subscribe(self.agent_id, self._handle_message)
        self.message_bus.subscribe("broadcast", self._handle_broadcast)
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理任务 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[TaskType]:
        """获取智能体能力 - 子类必须实现"""
        pass
    
    async def start(self):
        """启动智能体"""
        self.is_active = True
        self.stats["last_active"] = datetime.now()
        
        # 发送上线通知
        await self._send_status_update("online")
        logger.info(f"智能体 {self.agent_id} ({self.role.value}) 已启动")
    
    async def stop(self):
        """停止智能体"""
        self.is_active = False
        
        # 取消所有进行中的任务
        for task_id, task in self.current_tasks.items():
            task.status = TaskStatus.CANCELLED
            await self._send_task_response(task, {"error": "Agent stopped"})
        
        self.current_tasks.clear()
        
        # 发送下线通知
        await self._send_status_update("offline")
        logger.info(f"智能体 {self.agent_id} 已停止")
    
    async def assign_task(self, task: AgentTask) -> bool:
        """分配任务"""
        if not self.is_active:
            return False
        
        if task.task_type not in self.get_capabilities():
            return False
        
        task.assignee_id = self.agent_id
        task.status = TaskStatus.ASSIGNED
        task.updated_at = datetime.now()
        
        self.current_tasks[task.task_id] = task
        
        # 异步处理任务
        asyncio.create_task(self._execute_task(task))
        
        return True
    
    async def _execute_task(self, task: AgentTask):
        """执行任务"""
        start_time = datetime.now()
        
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.updated_at = datetime.now()
            
            logger.info(f"智能体 {self.agent_id} 开始执行任务 {task.task_id}")
            
            # 处理任务
            result = await self.process_task(task)
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.updated_at = datetime.now()
            
            # 发送任务响应
            await self._send_task_response(task, result)
            
            # 更新统计信息
            self.stats["tasks_completed"] += 1
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_average_response_time(response_time)
            
            logger.info(f"智能体 {self.agent_id} 完成任务 {task.task_id}，耗时 {response_time:.2f}s")
            
        except Exception as e:
            # 任务失败
            task.status = TaskStatus.FAILED
            task.result = {"error": str(e)}
            task.updated_at = datetime.now()
            
            await self._send_task_response(task, {"error": str(e)})
            
            self.stats["tasks_failed"] += 1
            logger.error(f"智能体 {self.agent_id} 执行任务 {task.task_id} 失败: {e}")
        
        finally:
            # 清理当前任务
            if task.task_id in self.current_tasks:
                del self.current_tasks[task.task_id]
            
            self.completed_tasks.append(task.task_id)
            self.stats["last_active"] = datetime.now()
    
    async def _handle_message(self, message: AgentMessage):
        """处理接收到的消息"""
        if message.message_type == MessageType.TASK_REQUEST:
            # 处理任务请求
            task_data = message.content.get("task")
            if task_data:
                task = AgentTask(**task_data)
                await self.assign_task(task)
        
        elif message.message_type == MessageType.COLLABORATION:
            # 处理协作消息
            await self._handle_collaboration(message)
    
    async def _handle_broadcast(self, message: AgentMessage):
        """处理广播消息"""
        if message.sender_id == self.agent_id:
            return  # 忽略自己发送的广播
        
        # 处理广播消息逻辑
        logger.debug(f"智能体 {self.agent_id} 收到广播: {message.content}")
    
    async def _handle_collaboration(self, message: AgentMessage):
        """处理协作消息"""
        # 子类可以重写此方法来处理特定的协作逻辑
        logger.debug(f"智能体 {self.agent_id} 收到协作消息: {message.content}")
    
    async def _send_task_response(self, task: AgentTask, result: Dict[str, Any]):
        """发送任务响应"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_RESPONSE,
            sender_id=self.agent_id,
            receiver_id=task.requester_id,
            content={
                "task_id": task.task_id,
                "status": task.status.value,
                "result": result
            }
        )
        
        await self.message_bus.send_message(message)
    
    async def _send_status_update(self, status: str):
        """发送状态更新"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.STATUS_UPDATE,
            sender_id=self.agent_id,
            receiver_id=None,  # 广播
            content={
                "status": status,
                "capabilities": [cap.value for cap in self.get_capabilities()],
                "current_tasks": len(self.current_tasks),
                "stats": self.stats
            }
        )
        
        await self.message_bus.send_message(message)
    
    async def send_collaboration_message(self, receiver_id: str, content: Dict[str, Any]):
        """发送协作消息"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COLLABORATION,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content
        )
        
        await self.message_bus.send_message(message)
    
    def _update_average_response_time(self, response_time: float):
        """更新平均响应时间"""
        completed = self.stats["tasks_completed"]
        current_avg = self.stats["average_response_time"]
        
        if completed > 1:
            self.stats["average_response_time"] = (current_avg * (completed - 1) + response_time) / completed
        else:
            self.stats["average_response_time"] = response_time
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "is_active": self.is_active,
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "current_tasks": len(self.current_tasks),
            "completed_tasks": len(self.completed_tasks),
            "stats": self.stats
        }


class TaskQueue:
    """任务队列"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.pending_tasks: List[AgentTask] = []
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
    async def initialize(self):
        """初始化任务队列"""
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
        
        logger.info("任务队列初始化完成")
    
    async def add_task(self, task: AgentTask):
        """添加任务到队列"""
        self.pending_tasks.append(task)
        
        # 按优先级和创建时间排序
        self.pending_tasks.sort(
            key=lambda t: (-t.priority, t.created_at)
        )
        
        # 持久化到Redis
        if self.redis_client:
            await self.redis_client.lpush(
                "agent_tasks_pending",
                json.dumps(task.to_dict())
            )
        
        logger.info(f"任务 {task.task_id} 已添加到队列")
    
    async def get_next_task(self, agent_capabilities: List[TaskType]) -> Optional[AgentTask]:
        """获取下一个合适的任务"""
        for i, task in enumerate(self.pending_tasks):
            if task.task_type in agent_capabilities:
                # 移除并返回任务
                return self.pending_tasks.pop(i)
        
        return None
    
    async def assign_task(self, task_id: str, agent_id: str):
        """分配任务给智能体"""
        self.task_assignments[task_id] = agent_id
        
        # 更新Redis
        if self.redis_client:
            await self.redis_client.hset(
                "agent_task_assignments",
                task_id,
                agent_id
            )
    
    async def complete_task(self, task_id: str):
        """标记任务完成"""
        if task_id in self.task_assignments:
            del self.task_assignments[task_id]
        
        # 从Redis清理
        if self.redis_client:
            await self.redis_client.hdel("agent_task_assignments", task_id)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "pending_tasks": len(self.pending_tasks),
            "assigned_tasks": len(self.task_assignments),
            "task_types": [task.task_type.value for task in self.pending_tasks]
        }


class MultiAgentSystem:
    """多智能体系统协调器"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = MessageBus()
        self.task_queue = TaskQueue()
        
        # 系统状态
        self.is_running = False
        self.coordinator_id = "coordinator_001"
        
        # 配置
        self.config = {
            "max_concurrent_tasks": 10,
            "task_timeout": 300,  # 5分钟
            "health_check_interval": 30,  # 30秒
            "load_balancing": True
        }
        
        # 统计信息
        self.system_stats = {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_task_time": 0.0,
            "system_uptime": None
        }
    
    async def initialize(self):
        """初始化多智能体系统"""
        # 初始化消息总线和任务队列
        await self.message_bus.initialize()
        await self.task_queue.initialize()
        
        # 订阅系统消息
        self.message_bus.subscribe("broadcast", self._handle_system_message)
        
        self.system_stats["system_uptime"] = datetime.now()
        logger.info("多智能体系统初始化完成")
    
    async def start(self):
        """启动系统"""
        self.is_running = True
        
        # 启动所有智能体
        for agent in self.agents.values():
            await agent.start()
        
        # 启动任务调度循环
        asyncio.create_task(self._task_scheduling_loop())
        
        # 启动健康检查循环
        asyncio.create_task(self._health_check_loop())
        
        logger.info("多智能体系统已启动")
    
    async def stop(self):
        """停止系统"""
        self.is_running = False
        
        # 停止所有智能体
        for agent in self.agents.values():
            await agent.stop()
        
        # 清理资源
        await self.message_bus.cleanup()
        
        logger.info("多智能体系统已停止")
    
    def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        self.agents[agent.agent_id] = agent
        self.system_stats["total_agents"] += 1
        
        logger.info(f"智能体 {agent.agent_id} ({agent.role.value}) 已注册")
    
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.system_stats["total_agents"] -= 1
            logger.info(f"智能体 {agent_id} 已注销")
    
    async def submit_task(self, 
                         task_type: TaskType,
                         content: str,
                         context: Dict[str, Any] = None,
                         priority: int = 0,
                         deadline: Optional[datetime] = None) -> str:
        """提交任务"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            content=content,
            context=context or {},
            requester_id=self.coordinator_id,
            priority=priority,
            deadline=deadline
        )
        
        await self.task_queue.add_task(task)
        self.system_stats["total_tasks"] += 1
        
        logger.info(f"任务 {task.task_id} ({task_type.value}) 已提交")
        return task.task_id
    
    async def _task_scheduling_loop(self):
        """任务调度循环"""
        while self.is_running:
            try:
                # 获取可用的智能体
                available_agents = [
                    agent for agent in self.agents.values()
                    if agent.is_active and len(agent.current_tasks) < 3  # 每个智能体最多3个并发任务
                ]
                
                if available_agents:
                    # 为每个可用智能体分配任务
                    for agent in available_agents:
                        task = await self.task_queue.get_next_task(agent.get_capabilities())
                        if task:
                            success = await agent.assign_task(task)
                            if success:
                                await self.task_queue.assign_task(task.task_id, agent.agent_id)
                                logger.info(f"任务 {task.task_id} 已分配给智能体 {agent.agent_id}")
                
                # 更新活跃智能体数量
                self.system_stats["active_agents"] = len([a for a in self.agents.values() if a.is_active])
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"任务调度循环出错: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                # 检查智能体健康状态
                unhealthy_agents = []
                
                for agent_id, agent in self.agents.items():
                    if agent.is_active and agent.stats["last_active"]:
                        last_active = agent.stats["last_active"]
                        if isinstance(last_active, str):
                            last_active = datetime.fromisoformat(last_active)
                        
                        # 检查是否超过健康检查间隔的2倍时间没有活动
                        if datetime.now() - last_active > timedelta(seconds=self.config["health_check_interval"] * 2):
                            unhealthy_agents.append(agent_id)
                
                # 处理不健康的智能体
                for agent_id in unhealthy_agents:
                    logger.warning(f"智能体 {agent_id} 可能不健康，重启中...")
                    agent = self.agents[agent_id]
                    await agent.stop()
                    await agent.start()
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"健康检查循环出错: {e}")
                await asyncio.sleep(30)
    
    async def _handle_system_message(self, message: AgentMessage):
        """处理系统消息"""
        if message.message_type == MessageType.STATUS_UPDATE:
            # 更新智能体状态统计
            logger.debug(f"收到状态更新: {message.sender_id} - {message.content.get('status')}")
        
        elif message.message_type == MessageType.TASK_RESPONSE:
            # 更新任务统计
            task_id = message.content.get("task_id")
            status = message.content.get("status")
            
            if status == "completed":
                self.system_stats["completed_tasks"] += 1
                await self.task_queue.complete_task(task_id)
            elif status == "failed":
                self.system_stats["failed_tasks"] += 1
                await self.task_queue.complete_task(task_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 更新统计信息
        if self.system_stats["system_uptime"]:
            uptime = datetime.now() - self.system_stats["system_uptime"]
            self.system_stats["uptime_seconds"] = uptime.total_seconds()
        
        # 获取智能体状态
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        # 获取任务队列状态
        queue_status = await self.task_queue.get_queue_status()
        
        return {
            "system_stats": self.system_stats,
            "agents": agent_statuses,
            "task_queue": queue_status,
            "is_running": self.is_running,
            "config": self.config
        }
    
    async def broadcast_message(self, content: Dict[str, Any]):
        """广播消息到所有智能体"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            sender_id=self.coordinator_id,
            receiver_id=None,
            content=content
        )
        
        await self.message_bus.send_message(message)
    
    async def send_collaboration_request(self, 
                                       requester_id: str,
                                       target_id: str,
                                       collaboration_type: str,
                                       content: Dict[str, Any]):
        """发送协作请求"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COLLABORATION,
            sender_id=requester_id,
            receiver_id=target_id,
            content={
                "collaboration_type": collaboration_type,
                "content": content
            }
        )
        
        await self.message_bus.send_message(message)
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新系统配置"""
        self.config.update(new_config)
        logger.info(f"系统配置已更新: {new_config}")


# 全局多智能体系统实例
_multi_agent_system: Optional[MultiAgentSystem] = None


async def get_multi_agent_system() -> MultiAgentSystem:
    """获取多智能体系统实例"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
        await _multi_agent_system.initialize()
    return _multi_agent_system 