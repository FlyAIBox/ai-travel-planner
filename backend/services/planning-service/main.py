"""
旅行规划服务
实现TravelPlanningEngine规划引擎架构、约束求解器和多目标优化算法、行程路径优化和时间安排算法、动态重规划和方案调整功能
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import math

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TravelMode(Enum):
    """出行方式"""
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR = "car"
    WALKING = "walking"
    BIKE = "bike"


class ActivityType(Enum):
    """活动类型"""
    SIGHTSEEING = "sightseeing"
    DINING = "dining"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    CULTURE = "culture"
    NATURE = "nature"
    RELAXATION = "relaxation"
    ADVENTURE = "adventure"


class OptimizationGoal(Enum):
    """优化目标"""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_EXPERIENCE = "maximize_experience"
    BALANCED = "balanced"


@dataclass
class Location:
    """位置信息"""
    id: str
    name: str
    latitude: float
    longitude: float
    address: str
    category: str
    rating: Optional[float] = None
    price_level: Optional[int] = None
    opening_hours: Optional[Dict[str, str]] = None
    visit_duration: Optional[int] = None  # 建议游览时间（分钟）


@dataclass
class Activity:
    """活动"""
    id: str
    name: str
    location: Location
    activity_type: ActivityType
    duration: int  # 持续时间（分钟）
    cost: float
    rating: float
    description: str
    requirements: List[str] = None
    best_time: Optional[str] = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []


@dataclass
class Transportation:
    """交通信息"""
    mode: TravelMode
    from_location: Location
    to_location: Location
    duration: int  # 时间（分钟）
    cost: float
    distance: float  # 距离（公里）
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    provider: Optional[str] = None
    booking_info: Optional[Dict[str, Any]] = None


@dataclass
class Constraint:
    """约束条件"""
    type: str
    value: Any
    priority: int  # 优先级 1-10
    description: str


@dataclass
class TravelPlan:
    """旅行计划"""
    id: str
    user_id: str
    title: str
    destination: str
    start_date: datetime
    end_date: datetime
    budget: float
    activities: List[Activity]
    transportations: List[Transportation]
    total_cost: float
    total_duration: int
    optimization_score: float
    constraints: List[Constraint]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


# Pydantic模型
class PlanningRequest(BaseModel):
    """规划请求"""
    user_id: str
    destination: str
    start_date: str
    end_date: str
    budget: float
    preferences: Dict[str, Any] = {}
    constraints: List[Dict[str, Any]] = []
    optimization_goal: str = "balanced"


class PlanningResponse(BaseModel):
    """规划响应"""
    plan_id: str
    success: bool
    plan: Optional[Dict[str, Any]] = None
    alternatives: List[Dict[str, Any]] = []
    optimization_details: Dict[str, Any] = {}
    message: str


class OptimizationRequest(BaseModel):
    """优化请求"""
    plan_id: str
    optimization_goal: str
    new_constraints: List[Dict[str, Any]] = []


class DistanceCalculator:
    """距离计算器"""
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间的haversine距离（公里）"""
        R = 6371  # 地球半径（公里）
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    @staticmethod
    def travel_time_estimate(distance: float, mode: TravelMode) -> int:
        """估算旅行时间（分钟）"""
        # 平均速度（公里/小时）
        speeds = {
            TravelMode.WALKING: 5,
            TravelMode.BIKE: 15,
            TravelMode.BUS: 30,
            TravelMode.CAR: 50,
            TravelMode.TRAIN: 80,
            TravelMode.FLIGHT: 500
        }
        
        speed = speeds.get(mode, 30)
        return int((distance / speed) * 60)  # 转换为分钟


class ConstraintSolver:
    """约束求解器"""
    
    def __init__(self):
        self.constraints = []
    
    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)
    
    def validate_plan(self, plan: TravelPlan) -> Tuple[bool, List[str]]:
        """验证计划是否满足约束"""
        violations = []
        
        for constraint in self.constraints:
            if not self._check_constraint(plan, constraint):
                violations.append(f"违反约束: {constraint.description}")
        
        return len(violations) == 0, violations
    
    def _check_constraint(self, plan: TravelPlan, constraint: Constraint) -> bool:
        """检查单个约束"""
        if constraint.type == "budget":
            return plan.total_cost <= constraint.value
        elif constraint.type == "duration":
            return plan.total_duration <= constraint.value
        elif constraint.type == "activity_count":
            return len(plan.activities) <= constraint.value
        elif constraint.type == "transport_mode":
            allowed_modes = constraint.value
            for transport in plan.transportations:
                if transport.mode.value not in allowed_modes:
                    return False
            return True
        elif constraint.type == "activity_type":
            required_types = constraint.value
            plan_types = {activity.activity_type.value for activity in plan.activities}
            return all(req_type in plan_types for req_type in required_types)
        
        return True


class RouteOptimizer:
    """路径优化器"""
    
    def __init__(self):
        self.distance_calculator = DistanceCalculator()
    
    def optimize_route(self, activities: List[Activity], start_location: Location) -> List[Activity]:
        """使用TSP算法优化路径"""
        if len(activities) <= 2:
            return activities
        
        # 构建距离矩阵
        locations = [start_location] + [activity.location for activity in activities]
        distance_matrix = self._build_distance_matrix(locations)
        
        # 使用OR-Tools求解TSP
        try:
            optimized_indices = self._solve_tsp(distance_matrix)
            # 移除起始点索引，返回优化后的活动顺序
            activity_indices = [idx - 1 for idx in optimized_indices[1:] if idx > 0]
            return [activities[idx] for idx in activity_indices]
        except Exception as e:
            logger.warning(f"TSP优化失败，使用贪心算法: {e}")
            return self._greedy_route_optimization(activities, start_location)
    
    def _build_distance_matrix(self, locations: List[Location]) -> List[List[int]]:
        """构建距离矩阵"""
        n = len(locations)
        matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = self.distance_calculator.haversine_distance(
                        locations[i].latitude, locations[i].longitude,
                        locations[j].latitude, locations[j].longitude
                    )
                    matrix[i][j] = int(distance * 1000)  # 转换为米
        
        return matrix
    
    def _solve_tsp(self, distance_matrix: List[List[int]]) -> List[int]:
        """使用OR-Tools求解TSP"""
        # 创建路由模型
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # 定义距离回调函数
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 设置搜索参数
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # 求解
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # 提取路径
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            return route
        
        raise Exception("TSP求解失败")
    
    def _greedy_route_optimization(self, activities: List[Activity], start_location: Location) -> List[Activity]:
        """贪心算法路径优化"""
        if not activities:
            return []
        
        optimized = []
        remaining = activities.copy()
        current_location = start_location
        
        while remaining:
            # 找到距离当前位置最近的活动
            min_distance = float('inf')
            nearest_activity = None
            nearest_index = -1
            
            for i, activity in enumerate(remaining):
                distance = self.distance_calculator.haversine_distance(
                    current_location.latitude, current_location.longitude,
                    activity.location.latitude, activity.location.longitude
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_activity = activity
                    nearest_index = i
            
            if nearest_activity:
                optimized.append(nearest_activity)
                current_location = nearest_activity.location
                remaining.pop(nearest_index)
        
        return optimized


class TravelPlanningEngine:
    """旅行规划引擎"""
    
    def __init__(self):
        self.constraint_solver = ConstraintSolver()
        self.route_optimizer = RouteOptimizer()
        self.distance_calculator = DistanceCalculator()
        self.activity_database = []  # 模拟活动数据库
        self.plans = {}  # 存储计划
    
    async def initialize(self):
        """初始化规划引擎"""
        # 加载活动数据库
        await self._load_activity_database()
        logger.info("旅行规划引擎初始化完成")
    
    async def create_plan(self, request: PlanningRequest) -> TravelPlan:
        """创建旅行计划"""
        logger.info(f"开始创建旅行计划: {request.destination}")
        
        # 解析日期
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # 计算天数
        days = (end_date - start_date).days + 1
        
        # 添加约束
        self._setup_constraints(request)
        
        # 选择活动
        activities = await self._select_activities(
            request.destination, 
            days, 
            request.budget, 
            request.preferences
        )
        
        # 优化路径
        start_location = self._get_city_center(request.destination)
        optimized_activities = self.route_optimizer.optimize_route(activities, start_location)
        
        # 生成交通安排
        transportations = await self._plan_transportation(optimized_activities, start_location)
        
        # 计算总成本和时间
        total_cost = sum(activity.cost for activity in optimized_activities)
        total_cost += sum(transport.cost for transport in transportations)
        
        total_duration = sum(activity.duration for activity in optimized_activities)
        total_duration += sum(transport.duration for transport in transportations)
        
        # 创建计划
        plan = TravelPlan(
            id=str(uuid.uuid4()),
            user_id=request.user_id,
            title=f"{request.destination}旅行计划",
            destination=request.destination,
            start_date=start_date,
            end_date=end_date,
            budget=request.budget,
            activities=optimized_activities,
            transportations=transportations,
            total_cost=total_cost,
            total_duration=total_duration,
            optimization_score=self._calculate_optimization_score(
                optimized_activities, 
                transportations, 
                request.optimization_goal
            ),
            constraints=[Constraint(**c) for c in request.constraints],
            metadata={
                "optimization_goal": request.optimization_goal,
                "preferences": request.preferences,
                "days": days
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 验证约束
        is_valid, violations = self.constraint_solver.validate_plan(plan)
        if not is_valid:
            logger.warning(f"计划违反约束: {violations}")
            # 可以在这里进行调整
        
        # 存储计划
        self.plans[plan.id] = plan
        
        logger.info(f"旅行计划创建完成: {plan.id}")
        return plan
    
    async def optimize_plan(self, plan_id: str, optimization_goal: OptimizationGoal) -> TravelPlan:
        """优化现有计划"""
        if plan_id not in self.plans:
            raise ValueError(f"计划不存在: {plan_id}")
        
        plan = self.plans[plan_id]
        
        # 根据优化目标重新安排
        if optimization_goal == OptimizationGoal.MINIMIZE_COST:
            plan = await self._optimize_for_cost(plan)
        elif optimization_goal == OptimizationGoal.MINIMIZE_TIME:
            plan = await self._optimize_for_time(plan)
        elif optimization_goal == OptimizationGoal.MAXIMIZE_EXPERIENCE:
            plan = await self._optimize_for_experience(plan)
        else:  # BALANCED
            plan = await self._optimize_balanced(plan)
        
        plan.updated_at = datetime.now()
        plan.optimization_score = self._calculate_optimization_score(
            plan.activities, 
            plan.transportations, 
            optimization_goal.value
        )
        
        self.plans[plan_id] = plan
        return plan
    
    def _setup_constraints(self, request: PlanningRequest):
        """设置约束条件"""
        self.constraint_solver.constraints = []
        
        # 预算约束
        self.constraint_solver.add_constraint(Constraint(
            type="budget",
            value=request.budget,
            priority=9,
            description=f"总预算不超过 {request.budget} 元"
        ))
        
        # 添加用户自定义约束
        for constraint_data in request.constraints:
            constraint = Constraint(**constraint_data)
            self.constraint_solver.add_constraint(constraint)
    
    async def _load_activity_database(self):
        """加载活动数据库"""
        # 模拟活动数据
        self.activity_database = [
            Activity(
                id="act_001",
                name="天安门广场",
                location=Location(
                    id="loc_001",
                    name="天安门广场",
                    latitude=39.9042,
                    longitude=116.4074,
                    address="北京市东城区天安门广场",
                    category="历史文化"
                ),
                activity_type=ActivityType.CULTURE,
                duration=120,
                cost=0,
                rating=4.8,
                description="中国的象征性地标"
            ),
            Activity(
                id="act_002", 
                name="故宫博物院",
                location=Location(
                    id="loc_002",
                    name="故宫博物院",
                    latitude=39.9163,
                    longitude=116.3972,
                    address="北京市东城区景山前街4号",
                    category="历史文化"
                ),
                activity_type=ActivityType.CULTURE,
                duration=180,
                cost=60,
                rating=4.9,
                description="明清两朝的皇家宫殿"
            ),
            # 可以添加更多活动...
        ]
    
    async def _select_activities(self, destination: str, days: int, budget: float, preferences: Dict[str, Any]) -> List[Activity]:
        """选择活动"""
        # 根据目的地、天数、预算和偏好选择活动
        selected = []
        available_budget = budget * 0.7  # 70%预算用于活动
        
        # 按评分排序
        sorted_activities = sorted(self.activity_database, key=lambda x: x.rating, reverse=True)
        
        current_cost = 0
        for activity in sorted_activities:
            if len(selected) >= days * 3:  # 每天最多3个活动
                break
            
            if current_cost + activity.cost <= available_budget:
                selected.append(activity)
                current_cost += activity.cost
        
        return selected[:days * 2]  # 限制总活动数量
    
    def _get_city_center(self, city: str) -> Location:
        """获取城市中心位置"""
        # 简化实现，返回城市中心
        city_centers = {
            "北京": Location(
                id="center_beijing",
                name="北京市中心",
                latitude=39.9042,
                longitude=116.4074,
                address="北京市中心",
                category="地标"
            ),
            "上海": Location(
                id="center_shanghai",
                name="上海市中心",
                latitude=31.2304,
                longitude=121.4737,
                address="上海市中心",
                category="地标"
            ),
        }
        
        return city_centers.get(city, city_centers["北京"])
    
    async def _plan_transportation(self, activities: List[Activity], start_location: Location) -> List[Transportation]:
        """规划交通"""
        transportations = []
        current_location = start_location
        
        for activity in activities:
            if current_location.id != activity.location.id:
                # 计算距离和时间
                distance = self.distance_calculator.haversine_distance(
                    current_location.latitude, current_location.longitude,
                    activity.location.latitude, activity.location.longitude
                )
                
                # 选择交通方式
                if distance <= 2:  # 2公里内步行
                    mode = TravelMode.WALKING
                    cost = 0
                elif distance <= 10:  # 10公里内公交或出租车
                    mode = TravelMode.BUS
                    cost = 5
                else:  # 远距离选择地铁或出租车
                    mode = TravelMode.CAR
                    cost = distance * 2
                
                duration = self.distance_calculator.travel_time_estimate(distance, mode)
                
                transport = Transportation(
                    mode=mode,
                    from_location=current_location,
                    to_location=activity.location,
                    duration=duration,
                    cost=cost,
                    distance=distance
                )
                
                transportations.append(transport)
                current_location = activity.location
        
        return transportations
    
    def _calculate_optimization_score(self, activities: List[Activity], transportations: List[Transportation], goal: str) -> float:
        """计算优化分数"""
        if not activities:
            return 0.0
        
        # 基础分数计算
        avg_rating = sum(activity.rating for activity in activities) / len(activities)
        total_cost = sum(activity.cost for activity in activities) + sum(t.cost for t in transportations)
        total_time = sum(activity.duration for activity in activities) + sum(t.duration for t in transportations)
        
        if goal == "minimize_cost":
            return max(0, 1 - (total_cost / 10000))  # 成本越低分数越高
        elif goal == "minimize_time":
            return max(0, 1 - (total_time / (24 * 60)))  # 时间越短分数越高
        elif goal == "maximize_experience":
            return avg_rating / 5.0  # 评分越高分数越高
        else:  # balanced
            cost_score = max(0, 1 - (total_cost / 5000))
            time_score = max(0, 1 - (total_time / (12 * 60)))
            experience_score = avg_rating / 5.0
            return (cost_score + time_score + experience_score) / 3
    
    async def _optimize_for_cost(self, plan: TravelPlan) -> TravelPlan:
        """成本优化"""
        # 移除最昂贵的活动，选择更便宜的替代方案
        sorted_activities = sorted(plan.activities, key=lambda x: x.cost)
        plan.activities = sorted_activities[:len(sorted_activities)//2]  # 保留一半较便宜的
        
        # 重新计算交通
        start_location = self._get_city_center(plan.destination)
        plan.transportations = await self._plan_transportation(plan.activities, start_location)
        
        # 更新总成本
        plan.total_cost = sum(activity.cost for activity in plan.activities)
        plan.total_cost += sum(transport.cost for transport in plan.transportations)
        
        return plan
    
    async def _optimize_for_time(self, plan: TravelPlan) -> TravelPlan:
        """时间优化"""
        # 移除耗时最长的活动
        sorted_activities = sorted(plan.activities, key=lambda x: x.duration)
        plan.activities = sorted_activities[:len(sorted_activities)//2]
        
        # 重新优化路径
        start_location = self._get_city_center(plan.destination)
        plan.activities = self.route_optimizer.optimize_route(plan.activities, start_location)
        plan.transportations = await self._plan_transportation(plan.activities, start_location)
        
        # 更新总时间
        plan.total_duration = sum(activity.duration for activity in plan.activities)
        plan.total_duration += sum(transport.duration for transport in plan.transportations)
        
        return plan
    
    async def _optimize_for_experience(self, plan: TravelPlan) -> TravelPlan:
        """体验优化"""
        # 保留评分最高的活动
        sorted_activities = sorted(plan.activities, key=lambda x: x.rating, reverse=True)
        plan.activities = sorted_activities
        
        # 重新规划
        start_location = self._get_city_center(plan.destination)
        plan.activities = self.route_optimizer.optimize_route(plan.activities, start_location)
        plan.transportations = await self._plan_transportation(plan.activities, start_location)
        
        return plan
    
    async def _optimize_balanced(self, plan: TravelPlan) -> TravelPlan:
        """平衡优化"""
        # 综合考虑成本、时间和体验
        # 计算每个活动的综合分数
        for activity in plan.activities:
            cost_score = max(0, 1 - (activity.cost / 200))  # 成本分数
            experience_score = activity.rating / 5.0  # 体验分数
            time_score = max(0, 1 - (activity.duration / 300))  # 时间分数
            activity.composite_score = (cost_score + experience_score + time_score) / 3
        
        # 按综合分数排序并选择
        sorted_activities = sorted(plan.activities, key=lambda x: getattr(x, 'composite_score', 0), reverse=True)
        plan.activities = sorted_activities[:int(len(sorted_activities) * 0.8)]  # 保留80%
        
        # 重新规划
        start_location = self._get_city_center(plan.destination)
        plan.activities = self.route_optimizer.optimize_route(plan.activities, start_location)
        plan.transportations = await self._plan_transportation(plan.activities, start_location)
        
        return plan


# 依赖注入
async def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB_CACHE,
        decode_responses=True
    )


# 全局规划引擎实例
planning_engine = TravelPlanningEngine()


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动旅行规划服务...")
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 初始化规划引擎
    await planning_engine.initialize()
    
    # 存储到应用状态
    app.state.redis_client = redis_client
    app.state.planning_engine = planning_engine
    
    logger.info("旅行规划服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭旅行规划服务...")
    await redis_client.close()
    logger.info("旅行规划服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner Planning Service",
    description="旅行规划服务，提供智能行程规划和路径优化",
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


# 规划端点
@app.post("/api/v1/plans", response_model=PlanningResponse)
async def create_travel_plan(request: PlanningRequest):
    """创建旅行计划"""
    try:
        engine = app.state.planning_engine
        plan = await engine.create_plan(request)
        
        return PlanningResponse(
            plan_id=plan.id,
            success=True,
            plan=asdict(plan),
            message="旅行计划创建成功"
        )
        
    except Exception as e:
        logger.error(f"创建旅行计划失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/plans/{plan_id}/optimize")
async def optimize_travel_plan(plan_id: str, request: OptimizationRequest):
    """优化旅行计划"""
    try:
        engine = app.state.planning_engine
        
        optimization_goal = OptimizationGoal(request.optimization_goal)
        optimized_plan = await engine.optimize_plan(plan_id, optimization_goal)
        
        return PlanningResponse(
            plan_id=optimized_plan.id,
            success=True,
            plan=asdict(optimized_plan),
            message="计划优化成功"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"优化旅行计划失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/plans/{plan_id}")
async def get_travel_plan(plan_id: str):
    """获取旅行计划"""
    try:
        engine = app.state.planning_engine
        
        if plan_id not in engine.plans:
            raise HTTPException(status_code=404, detail="计划不存在")
        
        plan = engine.plans[plan_id]
        return {
            "success": True,
            "plan": asdict(plan)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取旅行计划失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/plans")
async def list_travel_plans(user_id: Optional[str] = None):
    """列出旅行计划"""
    try:
        engine = app.state.planning_engine
        
        plans = list(engine.plans.values())
        if user_id:
            plans = [plan for plan in plans if plan.user_id == user_id]
        
        return {
            "success": True,
            "plans": [asdict(plan) for plan in plans],
            "total": len(plans)
        }
        
    except Exception as e:
        logger.error(f"列出旅行计划失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/plans/{plan_id}")
async def delete_travel_plan(plan_id: str):
    """删除旅行计划"""
    try:
        engine = app.state.planning_engine
        
        if plan_id not in engine.plans:
            raise HTTPException(status_code=404, detail="计划不存在")
        
        del engine.plans[plan_id]
        
        return {
            "success": True,
            "message": "计划删除成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除旅行计划失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查
@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    try:
        redis_client = app.state.redis_client
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "planning-service",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    ) 