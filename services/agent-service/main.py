"""
智能体服务
实现多角色智能体协作、任务分发、结果聚合等功能
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from enum import Enum
import uuid

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# 枚举和数据类
class AgentType(Enum):
    """智能体类型"""
    COORDINATOR = "coordinator"
    FLIGHT_AGENT = "flight_agent"
    HOTEL_AGENT = "hotel_agent"
    ITINERARY_AGENT = "itinerary_agent"
    BUDGET_AGENT = "budget_agent"
    LOCAL_GUIDE_AGENT = "local_guide_agent"
    RECOMMENDATION_AGENT = "recommendation_agent"


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
class AgentCapability:
    """智能体能力"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


@dataclass
class Task:
    """任务"""
    task_id: str
    agent_type: AgentType
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


# Pydantic模型
class TaskCreate(BaseModel):
    """创建任务模型"""
    agent_type: str
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: int = Field(default=2, ge=1, le=4)
    dependencies: List[str] = []


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    agent_type: str
    task_type: str
    description: str
    status: str
    priority: int
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentStatus(BaseModel):
    """智能体状态"""
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_activity: str


class CollaborationRequest(BaseModel):
    """协作请求模型"""
    user_query: str
    context: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}


# 智能体基类
class BaseAgent:
    """智能体基类"""
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = "idle"
        self.current_task = None
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.last_activity = datetime.now()
        self.capabilities = self._define_capabilities()
    
    def _define_capabilities(self) -> List[AgentCapability]:
        """定义智能体能力"""
        return []
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行任务"""
        self.status = "busy"
        self.current_task = task.task_id
        self.last_activity = datetime.now()
        
        try:
            result = await self._process_task(task)
            self.completed_tasks += 1
            self.status = "idle"
            self.current_task = None
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            self.status = "idle"
            self.current_task = None
            raise e
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理具体任务（子类实现）"""
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status,
            "current_task": self.current_task,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "last_activity": self.last_activity.isoformat(),
            "capabilities": [cap.name for cap in self.capabilities]
        }


# 协调器智能体
class CoordinatorAgent(BaseAgent):
    """协调器智能体"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.COORDINATOR)
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="task_decomposition",
                description="将复杂任务分解为子任务",
                input_schema={"user_query": "str", "context": "dict"},
                output_schema={"subtasks": "list", "execution_plan": "dict"}
            ),
            AgentCapability(
                name="result_aggregation",
                description="聚合各智能体的结果",
                input_schema={"subtask_results": "list"},
                output_schema={"final_result": "dict", "summary": "str"}
            )
        ]
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理协调任务"""
        if task.task_type == "decompose_query":
            return await self._decompose_query(task.input_data)
        elif task.task_type == "aggregate_results":
            return await self._aggregate_results(task.input_data)
        else:
            raise ValueError(f"未知任务类型: {task.task_type}")
    
    async def _decompose_query(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """分解用户查询"""
        user_query = input_data.get("user_query", "")
        context = input_data.get("context", {})
        
        # 模拟任务分解逻辑
        subtasks = []
        
        # 根据查询内容确定需要的智能体
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ["机票", "航班", "飞机"]):
            subtasks.append({
                "agent_type": "flight_agent",
                "task_type": "search_flights",
                "description": "搜索航班信息",
                "input_data": {"query": user_query, "context": context}
            })
        
        if any(word in query_lower for word in ["酒店", "住宿", "宾馆"]):
            subtasks.append({
                "agent_type": "hotel_agent",
                "task_type": "search_hotels",
                "description": "搜索酒店信息",
                "input_data": {"query": user_query, "context": context}
            })
        
        if any(word in query_lower for word in ["行程", "计划", "安排", "旅游"]):
            subtasks.append({
                "agent_type": "itinerary_agent",
                "task_type": "create_itinerary",
                "description": "制定旅行行程",
                "input_data": {"query": user_query, "context": context}
            })
        
        if any(word in query_lower for word in ["预算", "费用", "价格", "花费"]):
            subtasks.append({
                "agent_type": "budget_agent",
                "task_type": "calculate_budget",
                "description": "计算旅行预算",
                "input_data": {"query": user_query, "context": context}
            })
        
        # 如果没有明确指定，添加推荐智能体
        if not subtasks:
            subtasks.append({
                "agent_type": "recommendation_agent",
                "task_type": "general_recommendation",
                "description": "提供旅行推荐",
                "input_data": {"query": user_query, "context": context}
            })
        
        return {
            "subtasks": subtasks,
            "execution_plan": {
                "parallel_execution": True,
                "estimated_duration": len(subtasks) * 30,  # 秒
                "dependencies": {}
            }
        }
    
    async def _aggregate_results(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """聚合子任务结果"""
        subtask_results = input_data.get("subtask_results", [])
        
        # 按智能体类型组织结果
        organized_results = {}
        for result in subtask_results:
            agent_type = result.get("agent_type")
            if agent_type not in organized_results:
                organized_results[agent_type] = []
            organized_results[agent_type].append(result.get("result", {}))
        
        # 生成综合建议
        summary_parts = []
        for agent_type, results in organized_results.items():
            if results:
                summary_parts.append(f"{agent_type}: {len(results)}个建议")
        
        summary = f"为您找到了 {', '.join(summary_parts)}"
        
        return {
            "final_result": organized_results,
            "summary": summary,
            "total_results": len(subtask_results),
            "successful_agents": len(organized_results)
        }


# 航班智能体
class FlightAgent(BaseAgent):
    """航班搜索智能体"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.FLIGHT_AGENT)
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="flight_search",
                description="搜索航班信息",
                input_schema={"departure": "str", "arrival": "str", "date": "str"},
                output_schema={"flights": "list", "total_results": "int"}
            )
        ]
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理航班搜索任务"""
        # 模拟航班搜索
        await asyncio.sleep(1.0)  # 模拟API调用延迟
        
        return {
            "flights": [
                {
                    "flight_number": "CA1234",
                    "airline": "中国国际航空",
                    "departure_time": "08:00",
                    "arrival_time": "10:30",
                    "price": 800,
                    "duration": "2小时30分钟"
                },
                {
                    "flight_number": "CZ5678",
                    "airline": "中国南方航空",
                    "departure_time": "14:00",
                    "arrival_time": "16:45",
                    "price": 750,
                    "duration": "2小时45分钟"
                }
            ],
            "total_results": 2,
            "search_criteria": task.input_data
        }


# 酒店智能体
class HotelAgent(BaseAgent):
    """酒店搜索智能体"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.HOTEL_AGENT)
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="hotel_search",
                description="搜索酒店信息",
                input_schema={"location": "str", "checkin": "str", "checkout": "str"},
                output_schema={"hotels": "list", "total_results": "int"}
            )
        ]
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理酒店搜索任务"""
        # 模拟酒店搜索
        await asyncio.sleep(0.8)  # 模拟API调用延迟
        
        return {
            "hotels": [
                {
                    "name": "豪华酒店",
                    "star_rating": 5,
                    "price_per_night": 680,
                    "rating": 4.5,
                    "amenities": ["WiFi", "健身房", "游泳池"]
                },
                {
                    "name": "经济型酒店",
                    "star_rating": 3,
                    "price_per_night": 280,
                    "rating": 4.2,
                    "amenities": ["WiFi", "早餐"]
                }
            ],
            "total_results": 2,
            "search_criteria": task.input_data
        }


# 行程智能体
class ItineraryAgent(BaseAgent):
    """行程规划智能体"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.ITINERARY_AGENT)
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="create_itinerary",
                description="创建旅行行程",
                input_schema={"destination": "str", "duration": "int", "interests": "list"},
                output_schema={"itinerary": "dict", "total_days": "int"}
            )
        ]
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理行程规划任务"""
        # 模拟行程规划
        await asyncio.sleep(1.2)  # 模拟处理延迟
        
        return {
            "itinerary": {
                "day_1": {
                    "morning": "到达目的地，入住酒店",
                    "afternoon": "参观著名景点",
                    "evening": "品尝当地美食"
                },
                "day_2": {
                    "morning": "博物馆参观",
                    "afternoon": "购物和休闲",
                    "evening": "夜景观赏"
                }
            },
            "total_days": 2,
            "estimated_cost": 1500,
            "recommendations": ["建议提前预订门票", "注意天气变化"]
        }


# 预算智能体
class BudgetAgent(BaseAgent):
    """预算计算智能体"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.BUDGET_AGENT)
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="calculate_budget",
                description="计算旅行预算",
                input_schema={"destination": "str", "duration": "int", "style": "str"},
                output_schema={"budget_breakdown": "dict", "total_budget": "float"}
            )
        ]
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理预算计算任务"""
        # 模拟预算计算
        await asyncio.sleep(0.6)
        
        return {
            "budget_breakdown": {
                "交通费": 1000,
                "住宿费": 800,
                "餐饮费": 500,
                "景点门票": 300,
                "购物娱乐": 400,
                "其他费用": 200
            },
            "total_budget": 3200,
            "budget_tips": ["提前预订可节省20%费用", "选择当地交通更经济"]
        }


# 推荐智能体
class RecommendationAgent(BaseAgent):
    """推荐智能体"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.RECOMMENDATION_AGENT)
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="general_recommendation",
                description="提供旅行推荐",
                input_schema={"query": "str", "preferences": "dict"},
                output_schema={"recommendations": "list", "categories": "list"}
            )
        ]
    
    async def _process_task(self, task: Task) -> Dict[str, Any]:
        """处理推荐任务"""
        # 模拟推荐生成
        await asyncio.sleep(0.9)
        
        return {
            "recommendations": [
                {
                    "type": "destination",
                    "title": "热门目的地推荐",
                    "items": ["北京", "上海", "西安", "成都"]
                },
                {
                    "type": "activity",
                    "title": "推荐活动",
                    "items": ["文化古迹游览", "美食品尝", "购物体验"]
                }
            ],
            "categories": ["destination", "activity"],
            "personalized": True
        }


# 智能体管理器
class AgentManager:
    """智能体管理器"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, Task] = {}
        self.running = False
    
    def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        self.agents[agent.agent_id] = agent
        logger.info(f"注册智能体: {agent.agent_id} ({agent.agent_type.value})")
    
    def get_available_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """获取可用的智能体"""
        for agent in self.agents.values():
            if agent.agent_type == agent_type and agent.status == "idle":
                return agent
        return None
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        self.task_queue.append(task)
        logger.info(f"任务已提交: {task.task_id}")
        return task.task_id
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """执行单个任务"""
        agent = self.get_available_agent(task.agent_type)
        if not agent:
            raise ValueError(f"没有可用的 {task.agent_type.value} 智能体")
        
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_to = agent.agent_id
        task.updated_at = datetime.now()
        
        try:
            result = await agent.execute_task(task)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.updated_at = datetime.now()
            
            self.completed_tasks[task.task_id] = task
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.now()
            logger.error(f"任务执行失败: {task.task_id}, 错误: {e}")
            raise
    
    async def process_collaboration_request(self, request: CollaborationRequest) -> Dict[str, Any]:
        """处理协作请求"""
        # 创建协调任务
        coordinator_task = Task(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.COORDINATOR,
            task_type="decompose_query",
            description="分解用户查询",
            input_data={
                "user_query": request.user_query,
                "context": request.context,
                "preferences": request.preferences
            },
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 执行协调任务
        decomposition_result = await self.execute_task(coordinator_task)
        subtasks_data = decomposition_result.get("subtasks", [])
        
        # 创建子任务
        subtasks = []
        for subtask_data in subtasks_data:
            subtask = Task(
                task_id=str(uuid.uuid4()),
                agent_type=AgentType(subtask_data["agent_type"]),
                task_type=subtask_data["task_type"],
                description=subtask_data["description"],
                input_data=subtask_data["input_data"],
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            subtasks.append(subtask)
        
        # 并行执行子任务
        subtask_results = []
        for subtask in subtasks:
            try:
                result = await self.execute_task(subtask)
                subtask_results.append({
                    "task_id": subtask.task_id,
                    "agent_type": subtask.agent_type.value,
                    "result": result,
                    "status": "completed"
                })
            except Exception as e:
                subtask_results.append({
                    "task_id": subtask.task_id,
                    "agent_type": subtask.agent_type.value,
                    "error": str(e),
                    "status": "failed"
                })
        
        # 聚合结果
        aggregation_task = Task(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.COORDINATOR,
            task_type="aggregate_results",
            description="聚合子任务结果",
            input_data={"subtask_results": subtask_results},
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        final_result = await self.execute_task(aggregation_task)
        
        return {
            "collaboration_id": str(uuid.uuid4()),
            "user_query": request.user_query,
            "decomposition": decomposition_result,
            "subtask_results": subtask_results,
            "final_result": final_result,
            "execution_summary": {
                "total_subtasks": len(subtasks),
                "successful_subtasks": len([r for r in subtask_results if r.get("status") == "completed"]),
                "failed_subtasks": len([r for r in subtask_results if r.get("status") == "failed"]),
                "total_execution_time": sum([2.5] * len(subtasks))  # 模拟执行时间
            }
        }
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """获取所有智能体状态"""
        return [agent.get_status() for agent in self.agents.values()]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 查找已完成任务
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        # 查找队列中的任务
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None


# 依赖注入
async def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True
    )


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动智能体服务...")
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 创建智能体管理器
    agent_manager = AgentManager()
    
    # 注册各类智能体
    agent_manager.register_agent(CoordinatorAgent("coordinator_001"))
    agent_manager.register_agent(FlightAgent("flight_001"))
    agent_manager.register_agent(HotelAgent("hotel_001"))
    agent_manager.register_agent(ItineraryAgent("itinerary_001"))
    agent_manager.register_agent(BudgetAgent("budget_001"))
    agent_manager.register_agent(RecommendationAgent("recommendation_001"))
    
    # 存储到应用状态
    app.state.redis_client = redis_client
    app.state.agent_manager = agent_manager
    
    logger.info("智能体服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭智能体服务...")
    await redis_client.close()
    logger.info("智能体服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner Agent Service",
    description="智能体服务，提供多角色智能体协作功能",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 智能体协作端点
@app.post("/api/v1/agents/collaborate")
async def collaborate(request: CollaborationRequest):
    """智能体协作"""
    try:
        agent_manager = app.state.agent_manager
        result = await agent_manager.process_collaboration_request(request)
        
        return {
            "success": True,
            "collaboration_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"智能体协作失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/tasks", response_model=TaskResponse)
async def create_task(task_data: TaskCreate):
    """创建任务"""
    try:
        agent_manager = app.state.agent_manager
        
        task = Task(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType(task_data.agent_type),
            task_type=task_data.task_type,
            description=task_data.description,
            input_data=task_data.input_data,
            priority=TaskPriority(task_data.priority),
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            dependencies=task_data.dependencies
        )
        
        # 提交任务
        task_id = await agent_manager.submit_task(task)
        
        # 异步执行任务
        asyncio.create_task(execute_task_async(task))
        
        return TaskResponse(
            task_id=task.task_id,
            agent_type=task.agent_type.value,
            task_type=task.task_type,
            description=task.description,
            status=task.status.value,
            priority=task.priority.value,
            created_at=task.created_at.isoformat(),
            updated_at=task.updated_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        agent_manager = app.state.agent_manager
        task_data = agent_manager.get_task_status(task_id)
        
        if not task_data:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return TaskResponse(
            task_id=task_data["task_id"],
            agent_type=task_data["agent_type"],
            task_type=task_data["task_type"],
            description=task_data["description"],
            status=task_data["status"],
            priority=task_data["priority"],
            created_at=task_data["created_at"],
            updated_at=task_data["updated_at"],
            result=task_data.get("result"),
            error=task_data.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/status")
async def get_agents_status():
    """获取所有智能体状态"""
    try:
        agent_manager = app.state.agent_manager
        agents_status = agent_manager.get_agent_status()
        
        return {
            "agents": agents_status,
            "total_agents": len(agents_status),
            "active_agents": len([a for a in agents_status if a["status"] == "busy"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取智能体状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/capabilities")
async def get_agent_capabilities():
    """获取智能体能力列表"""
    try:
        agent_manager = app.state.agent_manager
        
        capabilities = {}
        for agent in agent_manager.agents.values():
            capabilities[agent.agent_type.value] = [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_schema": cap.input_schema,
                    "output_schema": cap.output_schema
                }
                for cap in agent.capabilities
            ]
        
        return {
            "capabilities": capabilities,
            "total_agent_types": len(capabilities),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取智能体能力失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查
@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    try:
        redis_client = app.state.redis_client
        await redis_client.ping()
        
        agent_manager = app.state.agent_manager
        total_agents = len(agent_manager.agents)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "agent-service",
            "version": "1.0.0",
            "total_agents": total_agents
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# 后台任务
async def execute_task_async(task: Task):
    """异步执行任务"""
    try:
        agent_manager = app.state.agent_manager
        await agent_manager.execute_task(task)
        logger.info(f"任务执行完成: {task.task_id}")
    except Exception as e:
        logger.error(f"异步任务执行失败: {task.task_id}, 错误: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    ) 