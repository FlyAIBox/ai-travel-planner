"""
旅行规划引擎
实现约束求解、多目标优化、路径规划、时间调度、动态重规划等核心算法
"""

import asyncio
import json
import math
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict, deque
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

import structlog
from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Location:
    """地点信息"""
    id: str
    name: str
    latitude: float
    longitude: float
    category: str  # 景点、酒店、餐厅、交通枢纽等
    visit_duration: int  # 建议游览时长（分钟）
    opening_hours: Dict[str, str] = field(default_factory=dict)  # {"monday": "09:00-18:00"}
    rating: float = 0.0
    price_level: int = 1  # 1-5价格等级
    accessibility: Dict[str, bool] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """活动信息"""
    id: str
    name: str
    location: Location
    start_time: datetime
    end_time: datetime
    activity_type: str  # 观光、餐饮、购物、休息等
    priority: int = 1  # 1-5优先级
    cost: float = 0.0
    prerequisites: List[str] = field(default_factory=list)  # 前置活动ID
    conflicts: List[str] = field(default_factory=list)  # 冲突活动ID
    participants: int = 1
    booking_required: bool = False
    weather_dependent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transportation:
    """交通信息"""
    id: str
    mode: str  # 步行、公交、地铁、出租车、飞机等
    origin: Location
    destination: Location
    duration: int  # 分钟
    cost: float
    distance: float  # 公里
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    schedule: List[str] = field(default_factory=list)  # 班次时间
    comfort_level: int = 1  # 1-5舒适度
    environmental_impact: float = 0.0  # 环境影响评分
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """约束条件"""
    id: str
    type: str  # 时间、预算、地点、活动等约束
    description: str
    parameters: Dict[str, Any]
    weight: float = 1.0  # 约束权重
    hard: bool = True  # 硬约束或软约束
    violation_penalty: float = 1000.0  # 违反约束的惩罚分数


@dataclass
class Objective:
    """优化目标"""
    id: str
    name: str
    description: str
    weight: float = 1.0
    maximize: bool = True  # True表示最大化，False表示最小化
    evaluation_func: str = ""  # 评估函数名称


@dataclass
class DayPlan:
    """单日计划"""
    date: datetime
    activities: List[Activity]
    transportations: List[Transportation]
    total_cost: float = 0.0
    total_duration: int = 0  # 分钟
    score: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self._calculate_totals()
    
    def _calculate_totals(self):
        """计算总计信息"""
        self.total_cost = sum(activity.cost for activity in self.activities)
        self.total_cost += sum(transport.cost for transport in self.transportations)
        
        if self.activities:
            start_time = min(activity.start_time for activity in self.activities)
            end_time = max(activity.end_time for activity in self.activities)
            self.total_duration = int((end_time - start_time).total_seconds() / 60)


@dataclass
class TravelPlan:
    """旅行计划"""
    id: str
    name: str
    destinations: List[str]
    start_date: datetime
    end_date: datetime
    daily_plans: List[DayPlan]
    participants: int = 1
    budget_limit: float = 0.0
    total_cost: float = 0.0
    total_score: float = 0.0
    constraints: List[Constraint] = field(default_factory=list)
    objectives: List[Objective] = field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._calculate_totals()
    
    def _calculate_totals(self):
        """计算总计信息"""
        self.total_cost = sum(day.total_cost for day in self.daily_plans)
        self.total_score = sum(day.score for day in self.daily_plans)


class ConstraintSolver:
    """约束求解器"""
    
    def __init__(self):
        self.constraints = []
        self.variables = {}
        self.domains = {}
        self.solution_cache = {}
    
    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)
    
    def solve(self, plan: TravelPlan) -> Tuple[bool, Dict[str, Any]]:
        """求解约束问题"""
        try:
            # 检查所有约束
            violations = []
            total_penalty = 0.0
            
            for constraint in plan.constraints:
                violation = self._check_constraint(constraint, plan)
                if violation:
                    violations.append(violation)
                    total_penalty += constraint.violation_penalty * constraint.weight
            
            # 判断是否满足所有硬约束
            hard_violations = [v for v in violations if v["is_hard"]]
            feasible = len(hard_violations) == 0
            
            result = {
                "feasible": feasible,
                "violations": violations,
                "total_penalty": total_penalty,
                "satisfaction_score": max(0, 1.0 - total_penalty / 10000.0)
            }
            
            return feasible, result
            
        except Exception as e:
            logger.error(f"约束求解失败: {e}")
            return False, {"error": str(e)}
    
    def _check_constraint(self, constraint: Constraint, plan: TravelPlan) -> Optional[Dict[str, Any]]:
        """检查单个约束"""
        constraint_type = constraint.type
        parameters = constraint.parameters
        
        if constraint_type == "budget":
            return self._check_budget_constraint(constraint, plan)
        elif constraint_type == "time":
            return self._check_time_constraint(constraint, plan)
        elif constraint_type == "location":
            return self._check_location_constraint(constraint, plan)
        elif constraint_type == "activity":
            return self._check_activity_constraint(constraint, plan)
        elif constraint_type == "transportation":
            return self._check_transportation_constraint(constraint, plan)
        
        return None
    
    def _check_budget_constraint(self, constraint: Constraint, plan: TravelPlan) -> Optional[Dict[str, Any]]:
        """检查预算约束"""
        max_budget = constraint.parameters.get("max_budget", float('inf'))
        
        if plan.total_cost > max_budget:
            violation_amount = plan.total_cost - max_budget
            return {
                "constraint_id": constraint.id,
                "type": "budget",
                "is_hard": constraint.hard,
                "violation_amount": violation_amount,
                "description": f"超出预算 {violation_amount:.2f} 元"
            }
        
        return None
    
    def _check_time_constraint(self, constraint: Constraint, plan: TravelPlan) -> Optional[Dict[str, Any]]:
        """检查时间约束"""
        parameters = constraint.parameters
        
        # 检查总时长约束
        if "max_duration_days" in parameters:
            max_days = parameters["max_duration_days"]
            actual_days = len(plan.daily_plans)
            
            if actual_days > max_days:
                return {
                    "constraint_id": constraint.id,
                    "type": "time",
                    "is_hard": constraint.hard,
                    "violation_amount": actual_days - max_days,
                    "description": f"超出最大行程天数 {actual_days - max_days} 天"
                }
        
        # 检查每日时长约束
        if "max_daily_hours" in parameters:
            max_hours = parameters["max_daily_hours"]
            
            for day_plan in plan.daily_plans:
                daily_hours = day_plan.total_duration / 60.0
                if daily_hours > max_hours:
                    return {
                        "constraint_id": constraint.id,
                        "type": "time",
                        "is_hard": constraint.hard,
                        "violation_amount": daily_hours - max_hours,
                        "description": f"单日行程超时 {daily_hours - max_hours:.1f} 小时"
                    }
        
        return None
    
    def _check_location_constraint(self, constraint: Constraint, plan: TravelPlan) -> Optional[Dict[str, Any]]:
        """检查地点约束"""
        parameters = constraint.parameters
        
        # 检查必访地点
        if "required_locations" in parameters:
            required = set(parameters["required_locations"])
            visited = set()
            
            for day_plan in plan.daily_plans:
                for activity in day_plan.activities:
                    visited.add(activity.location.name)
            
            missing = required - visited
            if missing:
                return {
                    "constraint_id": constraint.id,
                    "type": "location",
                    "is_hard": constraint.hard,
                    "violation_amount": len(missing),
                    "description": f"未包含必访地点: {', '.join(missing)}"
                }
        
        # 检查禁止地点
        if "forbidden_locations" in parameters:
            forbidden = set(parameters["forbidden_locations"])
            visited = set()
            
            for day_plan in plan.daily_plans:
                for activity in day_plan.activities:
                    visited.add(activity.location.name)
            
            violations = forbidden & visited
            if violations:
                return {
                    "constraint_id": constraint.id,
                    "type": "location",
                    "is_hard": constraint.hard,
                    "violation_amount": len(violations),
                    "description": f"包含禁止地点: {', '.join(violations)}"
                }
        
        return None
    
    def _check_activity_constraint(self, constraint: Constraint, plan: TravelPlan) -> Optional[Dict[str, Any]]:
        """检查活动约束"""
        parameters = constraint.parameters
        
        # 检查活动类型限制
        if "max_activities_per_type" in parameters:
            type_limits = parameters["max_activities_per_type"]
            type_counts = defaultdict(int)
            
            for day_plan in plan.daily_plans:
                for activity in day_plan.activities:
                    type_counts[activity.activity_type] += 1
            
            for activity_type, limit in type_limits.items():
                if type_counts[activity_type] > limit:
                    violation = type_counts[activity_type] - limit
                    return {
                        "constraint_id": constraint.id,
                        "type": "activity",
                        "is_hard": constraint.hard,
                        "violation_amount": violation,
                        "description": f"{activity_type}活动超出限制 {violation} 个"
                    }
        
        return None
    
    def _check_transportation_constraint(self, constraint: Constraint, plan: TravelPlan) -> Optional[Dict[str, Any]]:
        """检查交通约束"""
        parameters = constraint.parameters
        
        # 检查交通方式限制
        if "preferred_modes" in parameters and "forbidden_modes" in parameters:
            preferred = set(parameters["preferred_modes"])
            forbidden = set(parameters["forbidden_modes"])
            
            for day_plan in plan.daily_plans:
                for transport in day_plan.transportations:
                    if transport.mode in forbidden:
                        return {
                            "constraint_id": constraint.id,
                            "type": "transportation",
                            "is_hard": constraint.hard,
                            "violation_amount": 1,
                            "description": f"使用了禁止的交通方式: {transport.mode}"
                        }
        
        return None


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self):
        self.objectives = []
        self.pareto_front = []
        self.optimization_history = []
    
    def add_objective(self, objective: Objective):
        """添加优化目标"""
        self.objectives.append(objective)
    
    def optimize(self, plan: TravelPlan, population_size: int = 50, generations: int = 100) -> TravelPlan:
        """多目标优化"""
        try:
            # 使用简化的遗传算法进行多目标优化
            population = self._initialize_population(plan, population_size)
            
            for generation in range(generations):
                # 评估种群
                fitness_scores = [self._evaluate_fitness(individual) for individual in population]
                
                # 选择和繁殖
                population = self._evolve_population(population, fitness_scores)
                
                # 记录最佳个体
                best_individual = max(zip(population, fitness_scores), key=lambda x: x[1]["total_score"])[0]
                
                self.optimization_history.append({
                    "generation": generation,
                    "best_score": max(fitness_scores, key=lambda x: x["total_score"])["total_score"],
                    "average_score": sum(fs["total_score"] for fs in fitness_scores) / len(fitness_scores)
                })
            
            # 返回最优解
            final_fitness = [self._evaluate_fitness(individual) for individual in population]
            best_plan = max(zip(population, final_fitness), key=lambda x: x[1]["total_score"])[0]
            
            return best_plan
            
        except Exception as e:
            logger.error(f"多目标优化失败: {e}")
            return plan
    
    def _initialize_population(self, base_plan: TravelPlan, size: int) -> List[TravelPlan]:
        """初始化种群"""
        population = [base_plan]
        
        # 生成变异个体
        for _ in range(size - 1):
            mutated_plan = self._mutate_plan(base_plan)
            population.append(mutated_plan)
        
        return population
    
    def _mutate_plan(self, plan: TravelPlan) -> TravelPlan:
        """变异计划"""
        # 创建计划的副本
        mutated_plan = TravelPlan(
            id=str(uuid.uuid4()),
            name=f"{plan.name}_mutated",
            destinations=plan.destinations.copy(),
            start_date=plan.start_date,
            end_date=plan.end_date,
            daily_plans=[],
            participants=plan.participants,
            budget_limit=plan.budget_limit,
            constraints=plan.constraints.copy(),
            objectives=plan.objectives.copy()
        )
        
        # 简单的变异策略：调整活动顺序
        for day_plan in plan.daily_plans:
            new_activities = day_plan.activities.copy()
            if len(new_activities) > 1:
                # 随机交换两个活动
                i, j = random.sample(range(len(new_activities)), 2)
                new_activities[i], new_activities[j] = new_activities[j], new_activities[i]
                
                # 重新调整时间
                self._adjust_activity_times(new_activities)
            
            new_day_plan = DayPlan(
                date=day_plan.date,
                activities=new_activities,
                transportations=day_plan.transportations.copy()
            )
            mutated_plan.daily_plans.append(new_day_plan)
        
        return mutated_plan
    
    def _adjust_activity_times(self, activities: List[Activity]):
        """调整活动时间"""
        if not activities:
            return
        
        # 简单的时间调整策略
        current_time = activities[0].start_time.replace(hour=9, minute=0)
        
        for activity in activities:
            activity.start_time = current_time
            activity.end_time = current_time + timedelta(minutes=activity.location.visit_duration)
            current_time = activity.end_time + timedelta(minutes=30)  # 30分钟间隔
    
    def _evaluate_fitness(self, plan: TravelPlan) -> Dict[str, Any]:
        """评估适应度"""
        fitness_scores = {}
        total_score = 0.0
        
        for objective in self.objectives:
            score = self._evaluate_objective(objective, plan)
            fitness_scores[objective.id] = score
            total_score += score * objective.weight
        
        fitness_scores["total_score"] = total_score
        return fitness_scores
    
    def _evaluate_objective(self, objective: Objective, plan: TravelPlan) -> float:
        """评估单个目标"""
        objective_name = objective.name
        
        if objective_name == "minimize_cost":
            # 成本最小化（转换为最大化问题）
            max_possible_cost = plan.budget_limit or 10000
            return max(0, max_possible_cost - plan.total_cost) / max_possible_cost
        
        elif objective_name == "maximize_attractions":
            # 景点数量最大化
            attraction_count = 0
            for day_plan in plan.daily_plans:
                for activity in day_plan.activities:
                    if activity.location.category == "景点":
                        attraction_count += 1
            return attraction_count / 10.0  # 标准化
        
        elif objective_name == "maximize_rating":
            # 评分最大化
            total_rating = 0.0
            count = 0
            for day_plan in plan.daily_plans:
                for activity in day_plan.activities:
                    if activity.location.rating > 0:
                        total_rating += activity.location.rating
                        count += 1
            return (total_rating / max(count, 1)) / 10.0  # 标准化到0-1
        
        elif objective_name == "minimize_travel_time":
            # 交通时间最小化
            total_travel_time = 0
            for day_plan in plan.daily_plans:
                for transport in day_plan.transportations:
                    total_travel_time += transport.duration
            max_travel_time = len(plan.daily_plans) * 480  # 每天最多8小时交通
            return max(0, max_travel_time - total_travel_time) / max_travel_time
        
        else:
            return 0.5  # 默认分数
    
    def _evolve_population(self, population: List[TravelPlan], fitness_scores: List[Dict[str, Any]]) -> List[TravelPlan]:
        """进化种群"""
        # 选择优秀个体（精英主义）
        sorted_population = sorted(zip(population, fitness_scores), 
                                 key=lambda x: x[1]["total_score"], reverse=True)
        
        # 保留前50%
        elite_size = len(population) // 2
        new_population = [individual for individual, _ in sorted_population[:elite_size]]
        
        # 生成新个体填充种群
        while len(new_population) < len(population):
            # 选择两个父代
            parent1 = random.choice(new_population)
            parent2 = random.choice(new_population)
            
            # 简单的交叉和变异
            child = self._crossover(parent1, parent2)
            child = self._mutate_plan(child)
            
            new_population.append(child)
        
        return new_population
    
    def _crossover(self, parent1: TravelPlan, parent2: TravelPlan) -> TravelPlan:
        """交叉操作"""
        # 简单的交叉策略：随机选择每日计划
        child = TravelPlan(
            id=str(uuid.uuid4()),
            name=f"crossover_{random.randint(1000, 9999)}",
            destinations=parent1.destinations.copy(),
            start_date=parent1.start_date,
            end_date=parent1.end_date,
            daily_plans=[],
            participants=parent1.participants,
            budget_limit=parent1.budget_limit,
            constraints=parent1.constraints.copy(),
            objectives=parent1.objectives.copy()
        )
        
        for i in range(len(parent1.daily_plans)):
            if random.random() < 0.5:
                child.daily_plans.append(parent1.daily_plans[i])
            else:
                child.daily_plans.append(parent2.daily_plans[i])
        
        return child


class PathOptimizer:
    """路径优化器"""
    
    def __init__(self):
        self.distance_cache = {}
        self.route_cache = {}
    
    def optimize_route(self, locations: List[Location], 
                      algorithm: str = "nearest_neighbor") -> List[Location]:
        """优化路线"""
        if len(locations) <= 2:
            return locations
        
        try:
            if algorithm == "nearest_neighbor":
                return self._nearest_neighbor_algorithm(locations)
            elif algorithm == "genetic_algorithm":
                return self._genetic_algorithm(locations)
            elif algorithm == "simulated_annealing":
                return self._simulated_annealing(locations)
            elif algorithm == "two_opt":
                return self._two_opt_algorithm(locations)
            else:
                return self._nearest_neighbor_algorithm(locations)
                
        except Exception as e:
            logger.error(f"路径优化失败: {e}")
            return locations
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """计算两点间距离（使用哈弗辛公式）"""
        cache_key = (loc1.id, loc2.id)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # 地球半径（公里）
        R = 6371.0
        
        # 转换为弧度
        lat1_rad = math.radians(loc1.latitude)
        lon1_rad = math.radians(loc1.longitude)
        lat2_rad = math.radians(loc2.latitude)
        lon2_rad = math.radians(loc2.longitude)
        
        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # 哈弗辛公式
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # 缓存结果
        self.distance_cache[cache_key] = distance
        self.distance_cache[(loc2.id, loc1.id)] = distance
        
        return distance
    
    def _calculate_route_distance(self, route: List[Location]) -> float:
        """计算路线总距离"""
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self._calculate_distance(route[i], route[i + 1])
        return total_distance
    
    def _nearest_neighbor_algorithm(self, locations: List[Location]) -> List[Location]:
        """最近邻算法"""
        if not locations:
            return []
        
        unvisited = locations[1:].copy()  # 从第一个位置开始
        route = [locations[0]]
        current = locations[0]
        
        while unvisited:
            # 找到最近的未访问位置
            nearest = min(unvisited, key=lambda loc: self._calculate_distance(current, loc))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return route
    
    def _genetic_algorithm(self, locations: List[Location], 
                          population_size: int = 100, generations: int = 500) -> List[Location]:
        """遗传算法优化TSP"""
        if len(locations) <= 2:
            return locations
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            route = locations.copy()
            random.shuffle(route[1:])  # 保持起点不变
            population.append(route)
        
        for generation in range(generations):
            # 评估适应度（距离越短适应度越高）
            fitness_scores = []
            for route in population:
                distance = self._calculate_route_distance(route)
                fitness = 1.0 / (1.0 + distance)  # 避免除零
                fitness_scores.append(fitness)
            
            # 选择和繁殖
            new_population = []
            
            # 精英主义：保留最好的个体
            best_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(population[best_idx])
            
            # 生成新个体
            while len(new_population) < population_size:
                # 轮盘赌选择
                parent1 = self._roulette_selection(population, fitness_scores)
                parent2 = self._roulette_selection(population, fitness_scores)
                
                # 交叉
                child = self._order_crossover(parent1, parent2)
                
                # 变异
                if random.random() < 0.02:  # 2%变异率
                    child = self._mutate_route(child)
                
                new_population.append(child)
            
            population = new_population
        
        # 返回最佳路线
        final_fitness = [1.0 / (1.0 + self._calculate_route_distance(route)) for route in population]
        best_route = population[final_fitness.index(max(final_fitness))]
        
        return best_route
    
    def _roulette_selection(self, population: List[List[Location]], 
                           fitness_scores: List[float]) -> List[Location]:
        """轮盘赌选择"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return population[i]
        
        return population[-1]
    
    def _order_crossover(self, parent1: List[Location], parent2: List[Location]) -> List[Location]:
        """顺序交叉"""
        size = len(parent1)
        if size <= 2:
            return parent1.copy()
        
        # 选择交叉区间
        start = random.randint(1, size - 2)  # 保持起点不变
        end = random.randint(start + 1, size)
        
        # 创建子代
        child = [None] * size
        child[0] = parent1[0]  # 保持起点不变
        
        # 复制交叉区间
        for i in range(start, end):
            child[i] = parent1[i]
        
        # 从parent2填充剩余位置
        parent2_ptr = 1
        for i in range(1, size):
            if child[i] is None:
                while parent2[parent2_ptr] in child:
                    parent2_ptr += 1
                child[i] = parent2[parent2_ptr]
                parent2_ptr += 1
        
        return child
    
    def _mutate_route(self, route: List[Location]) -> List[Location]:
        """路线变异"""
        mutated = route.copy()
        
        if len(mutated) <= 3:
            return mutated
        
        # 随机交换两个位置（不包括起点）
        i = random.randint(1, len(mutated) - 1)
        j = random.randint(1, len(mutated) - 1)
        
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def _simulated_annealing(self, locations: List[Location], 
                           initial_temp: float = 1000.0, 
                           cooling_rate: float = 0.995,
                           min_temp: float = 1.0) -> List[Location]:
        """模拟退火算法"""
        # 初始解
        current_route = locations.copy()
        random.shuffle(current_route[1:])  # 保持起点不变
        current_distance = self._calculate_route_distance(current_route)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = initial_temp
        
        while temperature > min_temp:
            # 生成邻居解（2-opt交换）
            neighbor_route = self._two_opt_swap(current_route)
            neighbor_distance = self._calculate_route_distance(neighbor_route)
            
            # 计算接受概率
            if neighbor_distance < current_distance:
                # 更好的解，直接接受
                current_route = neighbor_route
                current_distance = neighbor_distance
                
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            else:
                # 较差的解，以一定概率接受
                delta = neighbor_distance - current_distance
                probability = math.exp(-delta / temperature)
                
                if random.random() < probability:
                    current_route = neighbor_route
                    current_distance = neighbor_distance
            
            # 降温
            temperature *= cooling_rate
        
        return best_route
    
    def _two_opt_algorithm(self, locations: List[Location]) -> List[Location]:
        """2-opt算法"""
        best_route = locations.copy()
        best_distance = self._calculate_route_distance(best_route)
        improved = True
        
        while improved:
            improved = False
            
            for i in range(1, len(locations) - 1):
                for j in range(i + 1, len(locations)):
                    if j - i == 1:
                        continue  # 跳过相邻边
                    
                    # 尝试2-opt交换
                    new_route = best_route.copy()
                    new_route[i:j] = reversed(new_route[i:j])
                    
                    new_distance = self._calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
        
        return best_route
    
    def _two_opt_swap(self, route: List[Location]) -> List[Location]:
        """2-opt交换操作"""
        new_route = route.copy()
        
        if len(new_route) <= 3:
            return new_route
        
        # 随机选择两个位置进行2-opt交换
        i = random.randint(1, len(new_route) - 2)
        j = random.randint(i + 1, len(new_route) - 1)
        
        # 反转i到j之间的路径
        new_route[i:j+1] = reversed(new_route[i:j+1])
        
        return new_route


class TimeScheduler:
    """时间调度器"""
    
    def __init__(self):
        self.scheduling_cache = {}
    
    def schedule_activities(self, activities: List[Activity], 
                          start_time: datetime,
                          end_time: datetime,
                          buffer_time: int = 30) -> List[Activity]:
        """安排活动时间"""
        try:
            scheduled_activities = []
            current_time = start_time
            
            # 按优先级排序活动
            sorted_activities = sorted(activities, key=lambda a: a.priority, reverse=True)
            
            for activity in sorted_activities:
                # 检查是否有足够时间
                activity_duration = timedelta(minutes=activity.location.visit_duration)
                activity_end = current_time + activity_duration
                
                if activity_end <= end_time:
                    # 安排活动
                    scheduled_activity = Activity(
                        id=activity.id,
                        name=activity.name,
                        location=activity.location,
                        start_time=current_time,
                        end_time=activity_end,
                        activity_type=activity.activity_type,
                        priority=activity.priority,
                        cost=activity.cost,
                        prerequisites=activity.prerequisites,
                        conflicts=activity.conflicts,
                        participants=activity.participants,
                        booking_required=activity.booking_required,
                        weather_dependent=activity.weather_dependent,
                        metadata=activity.metadata
                    )
                    
                    scheduled_activities.append(scheduled_activity)
                    
                    # 更新当前时间（包括缓冲时间）
                    current_time = activity_end + timedelta(minutes=buffer_time)
                else:
                    logger.warning(f"活动 {activity.name} 无法安排在指定时间内")
            
            return scheduled_activities
            
        except Exception as e:
            logger.error(f"活动调度失败: {e}")
            return activities
    
    def optimize_schedule(self, activities: List[Activity], 
                         constraints: List[Constraint] = None) -> List[Activity]:
        """优化时间安排"""
        if not activities:
            return []
        
        try:
            # 检查时间依赖关系
            dependency_graph = self._build_dependency_graph(activities)
            
            # 拓扑排序
            sorted_activity_ids = self._topological_sort(dependency_graph)
            
            # 考虑营业时间约束
            constrained_activities = self._apply_time_constraints(sorted_activity_ids, constraints)
            
            # 最小化等待时间
            optimized_activities = self._minimize_waiting_time(constrained_activities)
            
            return optimized_activities
            
        except Exception as e:
            logger.error(f"时间安排优化失败: {e}")
            return activities
    
    def _build_dependency_graph(self, activities: List[Activity]) -> Dict[str, List[str]]:
        """构建依赖关系图"""
        graph = {activity.id: [] for activity in activities}
        
        for activity in activities:
            for prerequisite in activity.prerequisites:
                if prerequisite in graph:
                    graph[prerequisite].append(activity.id)
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """拓扑排序"""
        in_degree = {node: 0 for node in graph}
        
        # 计算入度
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # 使用队列进行拓扑排序
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def _apply_time_constraints(self, activity_ids: List[str], 
                               constraints: List[Constraint]) -> List[str]:
        """应用时间约束"""
        # 这里可以根据具体的时间约束调整活动顺序
        # 例如：营业时间、特殊时间要求等
        return activity_ids
    
    def _minimize_waiting_time(self, activity_ids: List[str]) -> List[Activity]:
        """最小化等待时间"""
        # 简化实现：保持原有顺序
        # 实际应用中可以使用更复杂的调度算法
        return activity_ids


class TravelPlanningEngine:
    """旅行规划引擎"""
    
    def __init__(self):
        self.constraint_solver = ConstraintSolver()
        self.optimizer = MultiObjectiveOptimizer()
        self.path_optimizer = PathOptimizer()
        self.time_scheduler = TimeScheduler()
        
        # 地点数据库（简化版）
        self.locations_db = self._initialize_locations_db()
        
        # 规划历史
        self.planning_history = []
    
    def _initialize_locations_db(self) -> Dict[str, List[Location]]:
        """初始化地点数据库"""
        return {
            "北京": [
                Location("loc_001", "故宫", 39.9163, 116.3972, "景点", 240, {"monday": "08:30-17:00"}, 9.2, 3, {}, ["历史", "文化"]),
                Location("loc_002", "长城", 40.4319, 116.5704, "景点", 360, {"monday": "07:00-18:00"}, 9.0, 2, {}, ["历史", "世界遗产"]),
                Location("loc_003", "天坛", 39.8828, 116.4066, "景点", 180, {"monday": "06:00-18:00"}, 8.8, 2, {}, ["历史", "建筑"]),
                Location("loc_004", "颐和园", 39.9999, 116.2753, "景点", 240, {"monday": "06:30-18:00"}, 8.9, 2, {}, ["园林", "历史"]),
                Location("loc_005", "全聚德", 39.9075, 116.3975, "餐厅", 90, {"monday": "11:00-21:00"}, 8.5, 4, {}, ["烤鸭", "特色菜"])
            ],
            "上海": [
                Location("loc_101", "外滩", 31.2397, 121.4900, "景点", 120, {"monday": "00:00-23:59"}, 9.1, 1, {}, ["观光", "建筑"]),
                Location("loc_102", "东方明珠", 31.2397, 121.4994, "景点", 120, {"monday": "08:00-21:30"}, 8.7, 3, {}, ["观光", "地标"]),
                Location("loc_103", "豫园", 31.2272, 121.4921, "景点", 180, {"monday": "08:30-17:00"}, 8.6, 2, {}, ["园林", "历史"]),
                Location("loc_104", "南京路", 31.2342, 121.4733, "购物", 240, {"monday": "10:00-22:00"}, 8.4, 2, {}, ["购物", "步行街"])
            ]
        }
    
    async def create_travel_plan(self, requirements: Dict[str, Any]) -> TravelPlan:
        """创建旅行计划"""
        try:
            # 解析需求
            destinations = requirements.get("destinations", [])
            start_date = requirements.get("start_date")
            end_date = requirements.get("end_date")
            participants = requirements.get("participants", 1)
            budget_limit = requirements.get("budget_limit", 0.0)
            preferences = requirements.get("preferences", {})
            
            # 转换日期
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            
            # 计算行程天数
            duration = (end_date - start_date).days + 1
            
            # 创建基础计划
            plan = TravelPlan(
                id=str(uuid.uuid4()),
                name=f"{'、'.join(destinations)}旅行计划",
                destinations=destinations,
                start_date=start_date,
                end_date=end_date,
                daily_plans=[],
                participants=participants,
                budget_limit=budget_limit
            )
            
            # 添加约束
            await self._add_constraints(plan, requirements)
            
            # 添加目标
            await self._add_objectives(plan, preferences)
            
            # 生成每日计划
            await self._generate_daily_plans(plan, preferences)
            
            # 优化计划
            optimized_plan = await self._optimize_plan(plan)
            
            # 记录规划历史
            self.planning_history.append({
                "timestamp": datetime.now(),
                "plan_id": optimized_plan.id,
                "requirements": requirements,
                "result": "success"
            })
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"创建旅行计划失败: {e}")
            
            # 记录失败
            self.planning_history.append({
                "timestamp": datetime.now(),
                "requirements": requirements,
                "result": "failed",
                "error": str(e)
            })
            
            raise
    
    async def _add_constraints(self, plan: TravelPlan, requirements: Dict[str, Any]):
        """添加约束条件"""
        constraints = []
        
        # 预算约束
        if plan.budget_limit > 0:
            budget_constraint = Constraint(
                id="budget_limit",
                type="budget",
                description=f"预算不超过{plan.budget_limit}元",
                parameters={"max_budget": plan.budget_limit},
                hard=True
            )
            constraints.append(budget_constraint)
        
        # 时间约束
        duration = (plan.end_date - plan.start_date).days + 1
        time_constraint = Constraint(
            id="time_limit",
            type="time",
            description=f"行程不超过{duration}天",
            parameters={"max_duration_days": duration, "max_daily_hours": 12},
            hard=True
        )
        constraints.append(time_constraint)
        
        # 地点约束
        if "must_visit" in requirements:
            location_constraint = Constraint(
                id="must_visit_locations",
                type="location",
                description="必须访问指定地点",
                parameters={"required_locations": requirements["must_visit"]},
                hard=True
            )
            constraints.append(location_constraint)
        
        plan.constraints = constraints
    
    async def _add_objectives(self, plan: TravelPlan, preferences: Dict[str, Any]):
        """添加优化目标"""
        objectives = []
        
        # 默认目标
        objectives.extend([
            Objective("minimize_cost", "成本最小化", "降低总体费用", 0.3, False),
            Objective("maximize_attractions", "景点最大化", "增加景点数量", 0.2, True),
            Objective("maximize_rating", "评分最大化", "选择高评分地点", 0.3, True),
            Objective("minimize_travel_time", "交通时间最小化", "减少路上时间", 0.2, False)
        ])
        
        # 根据偏好调整权重
        travel_type = preferences.get("travel_type", "leisure")
        if travel_type == "budget":
            objectives[0].weight = 0.5  # 更重视成本
            objectives[1].weight = 0.15
        elif travel_type == "luxury":
            objectives[2].weight = 0.4  # 更重视质量
            objectives[0].weight = 0.1
        
        plan.objectives = objectives
    
    async def _generate_daily_plans(self, plan: TravelPlan, preferences: Dict[str, Any]):
        """生成每日计划"""
        duration = (plan.end_date - plan.start_date).days + 1
        daily_plans = []
        
        for day in range(duration):
            current_date = plan.start_date + timedelta(days=day)
            
            # 选择当天的目的地
            destination = plan.destinations[day % len(plan.destinations)]
            
            # 获取该目的地的活动
            activities = await self._generate_activities(destination, current_date, preferences)
            
            # 优化路线
            locations = [activity.location for activity in activities]
            optimized_locations = self.path_optimizer.optimize_route(locations)
            
            # 重新排序活动
            optimized_activities = []
            for location in optimized_locations:
                for activity in activities:
                    if activity.location.id == location.id:
                        optimized_activities.append(activity)
                        break
            
            # 调度时间
            start_time = current_date.replace(hour=9, minute=0)
            end_time = current_date.replace(hour=18, minute=0)
            
            scheduled_activities = self.time_scheduler.schedule_activities(
                optimized_activities, start_time, end_time
            )
            
            # 生成交通计划
            transportations = await self._generate_transportations(scheduled_activities)
            
            # 创建每日计划
            day_plan = DayPlan(
                date=current_date,
                activities=scheduled_activities,
                transportations=transportations
            )
            
            daily_plans.append(day_plan)
        
        plan.daily_plans = daily_plans
    
    async def _generate_activities(self, destination: str, date: datetime, 
                                 preferences: Dict[str, Any]) -> List[Activity]:
        """生成活动"""
        activities = []
        locations = self.locations_db.get(destination, [])
        
        # 根据偏好筛选地点
        travel_type = preferences.get("travel_type", "leisure")
        interests = preferences.get("interests", [])
        
        filtered_locations = []
        for location in locations:
            # 根据旅行类型筛选
            if travel_type == "cultural" and "文化" not in location.tags:
                continue
            if travel_type == "nature" and "自然" not in location.tags:
                continue
            
            # 根据兴趣筛选
            if interests and not any(interest in location.tags for interest in interests):
                continue
            
            filtered_locations.append(location)
        
        # 如果筛选后没有地点，使用所有地点
        if not filtered_locations:
            filtered_locations = locations
        
        # 创建活动
        for i, location in enumerate(filtered_locations[:5]):  # 每天最多5个活动
            activity = Activity(
                id=f"activity_{date.strftime('%Y%m%d')}_{i:02d}",
                name=f"游览{location.name}",
                location=location,
                start_time=date,  # 临时时间，后续会调整
                end_time=date,    # 临时时间，后续会调整
                activity_type=location.category,
                priority=5 - i,   # 优先级递减
                cost=location.price_level * 50,  # 简化的费用计算
                participants=1,
                booking_required=location.category == "景点",
                weather_dependent=location.category == "景点"
            )
            activities.append(activity)
        
        return activities
    
    async def _generate_transportations(self, activities: List[Activity]) -> List[Transportation]:
        """生成交通计划"""
        transportations = []
        
        for i in range(len(activities) - 1):
            current_activity = activities[i]
            next_activity = activities[i + 1]
            
            # 计算交通时间和费用
            distance = self.path_optimizer._calculate_distance(
                current_activity.location, next_activity.location
            )
            
            # 选择交通方式
            if distance < 1.0:  # 1公里内步行
                mode = "步行"
                duration = int(distance * 15)  # 每公里15分钟
                cost = 0.0
            elif distance < 5.0:  # 5公里内地铁/公交
                mode = "地铁"
                duration = int(distance * 8 + 10)  # 每公里8分钟+等车时间
                cost = 5.0
            else:  # 长距离出租车
                mode = "出租车"
                duration = int(distance * 5 + 5)  # 每公里5分钟+起步时间
                cost = distance * 3.0 + 10.0  # 每公里3元+起步费
            
            transportation = Transportation(
                id=f"transport_{i:02d}",
                mode=mode,
                origin=current_activity.location,
                destination=next_activity.location,
                duration=duration,
                cost=cost,
                distance=distance,
                departure_time=current_activity.end_time,
                arrival_time=next_activity.start_time
            )
            
            transportations.append(transportation)
        
        return transportations
    
    async def _optimize_plan(self, plan: TravelPlan) -> TravelPlan:
        """优化计划"""
        try:
            # 约束检查
            feasible, constraint_result = self.constraint_solver.solve(plan)
            
            if not feasible:
                logger.warning(f"计划不满足约束: {constraint_result}")
                # 可以在这里实施修复策略
            
            # 多目标优化
            optimized_plan = self.optimizer.optimize(plan, population_size=20, generations=50)
            
            # 更新优化历史
            optimized_plan.optimization_history.append({
                "timestamp": datetime.now().isoformat(),
                "constraint_satisfaction": constraint_result,
                "optimization_method": "genetic_algorithm",
                "improvement": optimized_plan.total_score - plan.total_score
            })
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"计划优化失败: {e}")
            return plan
    
    async def replan(self, original_plan: TravelPlan, 
                    changes: Dict[str, Any]) -> TravelPlan:
        """动态重规划"""
        try:
            # 克隆原计划
            new_plan = TravelPlan(
                id=str(uuid.uuid4()),
                name=f"{original_plan.name}_重规划",
                destinations=original_plan.destinations.copy(),
                start_date=original_plan.start_date,
                end_date=original_plan.end_date,
                daily_plans=[],
                participants=original_plan.participants,
                budget_limit=original_plan.budget_limit,
                constraints=original_plan.constraints.copy(),
                objectives=original_plan.objectives.copy()
            )
            
            # 应用变更
            await self._apply_changes(new_plan, changes)
            
            # 重新生成计划
            preferences = changes.get("preferences", {})
            await self._generate_daily_plans(new_plan, preferences)
            
            # 优化新计划
            optimized_plan = await self._optimize_plan(new_plan)
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"动态重规划失败: {e}")
            return original_plan
    
    async def _apply_changes(self, plan: TravelPlan, changes: Dict[str, Any]):
        """应用变更"""
        # 更新基本信息
        if "start_date" in changes:
            plan.start_date = datetime.fromisoformat(changes["start_date"])
        
        if "end_date" in changes:
            plan.end_date = datetime.fromisoformat(changes["end_date"])
        
        if "budget_limit" in changes:
            plan.budget_limit = changes["budget_limit"]
            
            # 更新预算约束
            for constraint in plan.constraints:
                if constraint.id == "budget_limit":
                    constraint.parameters["max_budget"] = plan.budget_limit
        
        if "destinations" in changes:
            plan.destinations = changes["destinations"]
        
        if "participants" in changes:
            plan.participants = changes["participants"]
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        if not self.planning_history:
            return {"total_plans": 0, "success_rate": 0.0}
        
        total_plans = len(self.planning_history)
        successful_plans = sum(1 for entry in self.planning_history if entry["result"] == "success")
        success_rate = successful_plans / total_plans
        
        # 分析常见目的地
        destinations_count = defaultdict(int)
        for entry in self.planning_history:
            if "requirements" in entry and "destinations" in entry["requirements"]:
                for dest in entry["requirements"]["destinations"]:
                    destinations_count[dest] += 1
        
        popular_destinations = sorted(destinations_count.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_plans": total_plans,
            "successful_plans": successful_plans,
            "success_rate": success_rate,
            "popular_destinations": popular_destinations,
            "average_optimization_time": self._calculate_average_optimization_time()
        }
    
    def _calculate_average_optimization_time(self) -> float:
        """计算平均优化时间"""
        # 简化实现，返回估计值
        return 2.5  # 2.5秒


# 全局旅行规划引擎实例
_travel_planning_engine: Optional[TravelPlanningEngine] = None


def get_travel_planning_engine() -> TravelPlanningEngine:
    """获取旅行规划引擎实例"""
    global _travel_planning_engine
    if _travel_planning_engine is None:
        _travel_planning_engine = TravelPlanningEngine()
    return _travel_planning_engine 