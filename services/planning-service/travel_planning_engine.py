"""
旅行规划引擎
实现TravelPlanningEngine规划引擎架构、约束求解器和多目标优化算法、
行程路径优化和时间安排算法、动态重规划和方案调整功能
"""

import asyncio
import json
import math
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import heapq
import random
from collections import defaultdict
import numpy as np

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ConstraintType(Enum):
    """约束类型"""
    BUDGET = "budget"                   # 预算约束
    TIME = "time"                      # 时间约束
    DISTANCE = "distance"              # 距离约束
    AVAILABILITY = "availability"       # 可用性约束
    PREFERENCE = "preference"          # 偏好约束
    CAPACITY = "capacity"              # 容量约束


class ObjectiveType(Enum):
    """优化目标类型"""
    MINIMIZE_COST = "minimize_cost"     # 最小化成本
    MINIMIZE_TIME = "minimize_time"     # 最小化时间
    MAXIMIZE_SATISFACTION = "maximize_satisfaction"  # 最大化满意度
    MINIMIZE_DISTANCE = "minimize_distance"  # 最小化距离
    MAXIMIZE_EXPERIENCES = "maximize_experiences"  # 最大化体验数量


class PlanStatus(Enum):
    """计划状态"""
    DRAFT = "draft"                    # 草案
    OPTIMIZING = "optimizing"          # 优化中
    READY = "ready"                    # 就绪
    CONFIRMED = "confirmed"            # 已确认
    IN_PROGRESS = "in_progress"        # 进行中
    COMPLETED = "completed"            # 已完成
    CANCELLED = "cancelled"            # 已取消


@dataclass
class Location:
    """位置信息"""
    id: str
    name: str
    latitude: float
    longitude: float
    city: str
    country: str
    category: str = "general"
    rating: float = 0.0
    cost_level: int = 1  # 1-5级别
    visit_duration: int = 120  # 建议游览时间（分钟）
    opening_hours: Dict[str, str] = field(default_factory=dict)
    seasonal_availability: List[str] = field(default_factory=list)
    
    def calculate_distance(self, other: 'Location') -> float:
        """计算与另一个位置的距离（公里）"""
        R = 6371  # 地球半径（公里）
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


@dataclass
class Activity:
    """活动信息"""
    id: str
    name: str
    location: Location
    category: str
    duration: int  # 分钟
    cost: float
    rating: float
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    best_time: List[str] = field(default_factory=list)  # ["morning", "afternoon", "evening"]
    prerequisites: List[str] = field(default_factory=list)
    
    def is_available(self, date: datetime, time_slot: str) -> bool:
        """检查在指定时间是否可用"""
        # 简化的可用性检查
        if self.best_time and time_slot not in self.best_time:
            return False
        
        # 检查位置的开放时间
        weekday = date.strftime("%A").lower()
        if weekday in self.location.opening_hours:
            opening_time = self.location.opening_hours[weekday]
            if opening_time == "closed":
                return False
        
        return True


@dataclass
class Transportation:
    """交通方式"""
    mode: str  # "flight", "train", "bus", "car", "walk"
    from_location: Location
    to_location: Location
    duration: int  # 分钟
    cost: float
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    booking_required: bool = False
    
    @property
    def distance(self) -> float:
        """获取交通距离"""
        return self.from_location.calculate_distance(self.to_location)


@dataclass
class Constraint:
    """约束条件"""
    type: ConstraintType
    description: str
    value: Any
    priority: int = 1  # 1-10，10最高
    is_hard: bool = True  # 硬约束vs软约束
    
    def validate(self, plan: 'TravelPlan') -> Tuple[bool, str]:
        """验证计划是否满足约束"""
        if self.type == ConstraintType.BUDGET:
            total_cost = plan.calculate_total_cost()
            if total_cost > self.value:
                return False, f"预算超出：{total_cost} > {self.value}"
        
        elif self.type == ConstraintType.TIME:
            total_duration = plan.calculate_total_duration()
            if total_duration > self.value:
                return False, f"时间超出：{total_duration} > {self.value}分钟"
        
        elif self.type == ConstraintType.DISTANCE:
            total_distance = plan.calculate_total_distance()
            if total_distance > self.value:
                return False, f"距离超出：{total_distance} > {self.value}公里"
        
        return True, "约束满足"


@dataclass
class Objective:
    """优化目标"""
    type: ObjectiveType
    weight: float = 1.0
    target_value: Optional[float] = None
    
    def evaluate(self, plan: 'TravelPlan') -> float:
        """评估计划在此目标下的得分"""
        if self.type == ObjectiveType.MINIMIZE_COST:
            return -plan.calculate_total_cost() * self.weight
        
        elif self.type == ObjectiveType.MINIMIZE_TIME:
            return -plan.calculate_total_duration() * self.weight
        
        elif self.type == ObjectiveType.MAXIMIZE_SATISFACTION:
            return plan.calculate_satisfaction_score() * self.weight
        
        elif self.type == ObjectiveType.MINIMIZE_DISTANCE:
            return -plan.calculate_total_distance() * self.weight
        
        elif self.type == ObjectiveType.MAXIMIZE_EXPERIENCES:
            return len(plan.get_all_activities()) * self.weight
        
        return 0.0


@dataclass
class DayPlan:
    """单日计划"""
    date: datetime
    activities: List[Activity] = field(default_factory=list)
    transportations: List[Transportation] = field(default_factory=list)
    start_time: time = time(9, 0)
    end_time: time = time(18, 0)
    
    def add_activity(self, activity: Activity, start_time: Optional[time] = None):
        """添加活动"""
        self.activities.append(activity)
    
    def calculate_daily_cost(self) -> float:
        """计算当日总费用"""
        activity_cost = sum(activity.cost for activity in self.activities)
        transport_cost = sum(transport.cost for transport in self.transportations)
        return activity_cost + transport_cost
    
    def calculate_daily_duration(self) -> int:
        """计算当日总时长（分钟）"""
        activity_duration = sum(activity.duration for activity in self.activities)
        transport_duration = sum(transport.duration for transport in self.transportations)
        return activity_duration + transport_duration
    
    def get_locations(self) -> List[Location]:
        """获取当日所有位置"""
        locations = []
        for activity in self.activities:
            if activity.location not in locations:
                locations.append(activity.location)
        return locations


@dataclass
class TravelPlan:
    """旅行计划"""
    id: str
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    daily_plans: List[DayPlan] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    objectives: List[Objective] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_days(self) -> int:
        """获取计划天数"""
        return (self.end_date - self.start_date).days + 1
    
    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)
    
    def add_objective(self, objective: Objective):
        """添加目标"""
        self.objectives.append(objective)
    
    def calculate_total_cost(self) -> float:
        """计算总费用"""
        return sum(day.calculate_daily_cost() for day in self.daily_plans)
    
    def calculate_total_duration(self) -> int:
        """计算总时长（分钟）"""
        return sum(day.calculate_daily_duration() for day in self.daily_plans)
    
    def calculate_total_distance(self) -> float:
        """计算总距离"""
        total_distance = 0.0
        for day in self.daily_plans:
            locations = day.get_locations()
            for i in range(len(locations) - 1):
                total_distance += locations[i].calculate_distance(locations[i + 1])
        return total_distance
    
    def calculate_satisfaction_score(self) -> float:
        """计算满意度分数"""
        activities = self.get_all_activities()
        if not activities:
            return 0.0
        
        # 基于活动评分计算满意度
        total_rating = sum(activity.rating for activity in activities)
        avg_rating = total_rating / len(activities)
        
        # 考虑活动数量的影响
        quantity_bonus = min(len(activities) / 20, 1.0)  # 最多20个活动
        
        return avg_rating * (1 + quantity_bonus)
    
    def get_all_activities(self) -> List[Activity]:
        """获取所有活动"""
        activities = []
        for day in self.daily_plans:
            activities.extend(day.activities)
        return activities
    
    def get_all_locations(self) -> List[Location]:
        """获取所有位置"""
        locations = []
        for day in self.daily_plans:
            for location in day.get_locations():
                if location not in locations:
                    locations.append(location)
        return locations
    
    def validate_constraints(self) -> Tuple[bool, List[str]]:
        """验证所有约束"""
        violations = []
        for constraint in self.constraints:
            is_valid, message = constraint.validate(self)
            if not is_valid:
                violations.append(message)
        
        return len(violations) == 0, violations
    
    def calculate_objective_score(self) -> float:
        """计算综合目标分数"""
        if not self.objectives:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for objective in self.objectives:
            score = objective.evaluate(self)
            total_score += score
            total_weight += objective.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class ConstraintSolver:
    """约束求解器"""
    
    def __init__(self):
        self.max_iterations = 1000
        self.tolerance = 1e-6
    
    async def solve(self, 
                   plan: TravelPlan,
                   available_activities: List[Activity],
                   available_transportations: List[Transportation]) -> Tuple[bool, TravelPlan]:
        """求解约束优化问题"""
        logger.info(f"开始求解计划 {plan.id} 的约束优化问题")
        
        # 初始化求解
        current_plan = self._initialize_plan(plan, available_activities)
        
        # 迭代优化
        iteration = 0
        best_plan = current_plan
        best_score = current_plan.calculate_objective_score()
        
        while iteration < self.max_iterations:
            # 生成候选解
            candidate_plan = await self._generate_candidate_solution(
                current_plan, available_activities, available_transportations
            )
            
            # 评估候选解
            if await self._is_better_solution(candidate_plan, best_plan):
                best_plan = candidate_plan
                best_score = candidate_plan.calculate_objective_score()
                logger.debug(f"迭代 {iteration}: 找到更好解，分数: {best_score}")
            
            # 检查收敛条件
            if await self._check_convergence(best_plan):
                break
            
            current_plan = candidate_plan
            iteration += 1
        
        logger.info(f"约束求解完成，迭代次数: {iteration}，最终分数: {best_score}")
        return True, best_plan
    
    def _initialize_plan(self, plan: TravelPlan, activities: List[Activity]) -> TravelPlan:
        """初始化计划"""
        # 为每一天分配初始活动
        activities_by_category = defaultdict(list)
        for activity in activities:
            activities_by_category[activity.category].append(activity)
        
        # 简单的初始分配策略
        for i, day_plan in enumerate(plan.daily_plans):
            # 每天分配2-4个活动
            num_activities = min(random.randint(2, 4), len(activities))
            selected_activities = random.sample(activities, num_activities)
            
            for activity in selected_activities:
                day_plan.add_activity(activity)
        
        return plan
    
    async def _generate_candidate_solution(self,
                                         current_plan: TravelPlan,
                                         activities: List[Activity],
                                         transportations: List[Transportation]) -> TravelPlan:
        """生成候选解"""
        # 复制当前计划
        candidate = self._deep_copy_plan(current_plan)
        
        # 随机选择改进策略
        strategies = [
            self._swap_activities,
            self._add_activity,
            self._remove_activity,
            self._reorder_activities,
            self._optimize_transportation
        ]
        
        strategy = random.choice(strategies)
        await strategy(candidate, activities, transportations)
        
        return candidate
    
    async def _swap_activities(self, plan: TravelPlan, activities: List[Activity], transportations: List[Transportation]):
        """交换活动策略"""
        if len(plan.daily_plans) < 2:
            return
        
        # 随机选择两天
        day1, day2 = random.sample(plan.daily_plans, 2)
        
        if day1.activities and day2.activities:
            # 交换一个活动
            activity1 = random.choice(day1.activities)
            activity2 = random.choice(day2.activities)
            
            day1.activities.remove(activity1)
            day2.activities.remove(activity2)
            
            day1.activities.append(activity2)
            day2.activities.append(activity1)
    
    async def _add_activity(self, plan: TravelPlan, activities: List[Activity], transportations: List[Transportation]):
        """添加活动策略"""
        # 选择一个随机的日子和活动
        day = random.choice(plan.daily_plans)
        available_activities = [a for a in activities if a not in day.activities]
        
        if available_activities:
            new_activity = random.choice(available_activities)
            day.add_activity(new_activity)
    
    async def _remove_activity(self, plan: TravelPlan, activities: List[Activity], transportations: List[Transportation]):
        """移除活动策略"""
        day = random.choice(plan.daily_plans)
        if day.activities:
            activity_to_remove = random.choice(day.activities)
            day.activities.remove(activity_to_remove)
    
    async def _reorder_activities(self, plan: TravelPlan, activities: List[Activity], transportations: List[Transportation]):
        """重新排序活动策略"""
        day = random.choice(plan.daily_plans)
        if len(day.activities) > 1:
            random.shuffle(day.activities)
    
    async def _optimize_transportation(self, plan: TravelPlan, activities: List[Activity], transportations: List[Transportation]):
        """优化交通策略"""
        for day in plan.daily_plans:
            # 根据位置重新排序活动以减少移动距离
            if len(day.activities) > 1:
                day.activities = self._optimize_route(day.activities)
    
    def _optimize_route(self, activities: List[Activity]) -> List[Activity]:
        """优化路线顺序（简化的TSP）"""
        if len(activities) <= 2:
            return activities
        
        # 使用最近邻算法
        optimized = [activities[0]]
        remaining = activities[1:]
        
        while remaining:
            current_location = optimized[-1].location
            nearest_activity = min(remaining, 
                                 key=lambda a: current_location.calculate_distance(a.location))
            optimized.append(nearest_activity)
            remaining.remove(nearest_activity)
        
        return optimized
    
    async def _is_better_solution(self, candidate: TravelPlan, current_best: TravelPlan) -> bool:
        """判断候选解是否更好"""
        # 首先检查约束
        candidate_valid, _ = candidate.validate_constraints()
        current_valid, _ = current_best.validate_constraints()
        
        # 如果候选解满足约束而当前解不满足，则候选解更好
        if candidate_valid and not current_valid:
            return True
        
        # 如果候选解不满足约束而当前解满足，则候选解更差
        if not candidate_valid and current_valid:
            return False
        
        # 如果都满足或都不满足约束，比较目标函数值
        candidate_score = candidate.calculate_objective_score()
        current_score = current_best.calculate_objective_score()
        
        return candidate_score > current_score
    
    async def _check_convergence(self, plan: TravelPlan) -> bool:
        """检查是否收敛"""
        # 简单的收敛检查：如果满足所有硬约束则认为收敛
        is_valid, _ = plan.validate_constraints()
        return is_valid
    
    def _deep_copy_plan(self, plan: TravelPlan) -> TravelPlan:
        """深度复制计划"""
        # 简化的深度复制
        new_plan = TravelPlan(
            id=plan.id + "_copy",
            name=plan.name,
            description=plan.description,
            start_date=plan.start_date,
            end_date=plan.end_date,
            constraints=plan.constraints.copy(),
            objectives=plan.objectives.copy(),
            status=plan.status,
            metadata=plan.metadata.copy()
        )
        
        # 复制每日计划
        for day_plan in plan.daily_plans:
            new_day = DayPlan(
                date=day_plan.date,
                activities=day_plan.activities.copy(),
                transportations=day_plan.transportations.copy(),
                start_time=day_plan.start_time,
                end_time=day_plan.end_time
            )
            new_plan.daily_plans.append(new_day)
        
        return new_plan


class PathOptimizer:
    """路径优化器"""
    
    def __init__(self):
        self.optimization_methods = ["nearest_neighbor", "genetic_algorithm", "simulated_annealing"]
    
    async def optimize_path(self, 
                          locations: List[Location],
                          start_location: Optional[Location] = None,
                          method: str = "nearest_neighbor") -> List[Location]:
        """优化访问路径"""
        if len(locations) <= 2:
            return locations
        
        if method == "nearest_neighbor":
            return await self._nearest_neighbor_optimization(locations, start_location)
        elif method == "genetic_algorithm":
            return await self._genetic_algorithm_optimization(locations, start_location)
        elif method == "simulated_annealing":
            return await self._simulated_annealing_optimization(locations, start_location)
        else:
            return locations
    
    async def _nearest_neighbor_optimization(self, 
                                           locations: List[Location],
                                           start_location: Optional[Location] = None) -> List[Location]:
        """最近邻优化"""
        if not locations:
            return []
        
        # 选择起始位置
        current = start_location or locations[0]
        remaining = [loc for loc in locations if loc != current]
        optimized_path = [current]
        
        while remaining:
            # 找到最近的位置
            nearest = min(remaining, key=lambda loc: current.calculate_distance(loc))
            optimized_path.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return optimized_path
    
    async def _genetic_algorithm_optimization(self,
                                            locations: List[Location],
                                            start_location: Optional[Location] = None) -> List[Location]:
        """遗传算法优化"""
        if len(locations) <= 3:
            return await self._nearest_neighbor_optimization(locations, start_location)
        
        population_size = min(50, len(locations) * 2)
        generations = 100
        mutation_rate = 0.1
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = locations.copy()
            random.shuffle(individual)
            if start_location and start_location in individual:
                # 确保起始位置在开头
                individual.remove(start_location)
                individual.insert(0, start_location)
            population.append(individual)
        
        # 进化过程
        for generation in range(generations):
            # 评估适应度
            fitness_scores = [1 / (self._calculate_path_distance(individual) + 1) for individual in population]
            
            # 选择
            new_population = []
            for _ in range(population_size):
                # 轮盘赌选择
                parent1 = self._roulette_wheel_selection(population, fitness_scores)
                parent2 = self._roulette_wheel_selection(population, fitness_scores)
                
                # 交叉
                child = self._crossover(parent1, parent2)
                
                # 变异
                if random.random() < mutation_rate:
                    child = self._mutate(child, start_location)
                
                new_population.append(child)
            
            population = new_population
        
        # 返回最好的解
        best_individual = min(population, key=self._calculate_path_distance)
        return best_individual
    
    async def _simulated_annealing_optimization(self,
                                              locations: List[Location],
                                              start_location: Optional[Location] = None) -> List[Location]:
        """模拟退火优化"""
        current_solution = locations.copy()
        if start_location and start_location in current_solution:
            current_solution.remove(start_location)
            current_solution.insert(0, start_location)
        else:
            random.shuffle(current_solution)
        
        current_distance = self._calculate_path_distance(current_solution)
        best_solution = current_solution.copy()
        best_distance = current_distance
        
        # 模拟退火参数
        initial_temperature = 1000
        cooling_rate = 0.95
        min_temperature = 1
        
        temperature = initial_temperature
        
        while temperature > min_temperature:
            # 生成邻域解
            new_solution = self._generate_neighbor(current_solution, start_location)
            new_distance = self._calculate_path_distance(new_solution)
            
            # 接受准则
            if new_distance < current_distance or random.random() < math.exp(-(new_distance - current_distance) / temperature):
                current_solution = new_solution
                current_distance = new_distance
                
                if new_distance < best_distance:
                    best_solution = new_solution.copy()
                    best_distance = new_distance
            
            temperature *= cooling_rate
        
        return best_solution
    
    def _calculate_path_distance(self, path: List[Location]) -> float:
        """计算路径总距离"""
        if len(path) <= 1:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += path[i].calculate_distance(path[i + 1])
        
        return total_distance
    
    def _roulette_wheel_selection(self, population: List[List[Location]], fitness_scores: List[float]) -> List[Location]:
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
    
    def _crossover(self, parent1: List[Location], parent2: List[Location]) -> List[Location]:
        """交叉操作（顺序交叉）"""
        if len(parent1) <= 2:
            return parent1.copy()
        
        start_idx = random.randint(0, len(parent1) - 2)
        end_idx = random.randint(start_idx + 1, len(parent1) - 1)
        
        child = [None] * len(parent1)
        
        # 复制父代1的片段
        for i in range(start_idx, end_idx + 1):
            child[i] = parent1[i]
        
        # 从父代2填充剩余位置
        parent2_remaining = [loc for loc in parent2 if loc not in child]
        j = 0
        for i in range(len(child)):
            if child[i] is None and j < len(parent2_remaining):
                child[i] = parent2_remaining[j]
                j += 1
        
        return child
    
    def _mutate(self, individual: List[Location], start_location: Optional[Location] = None) -> List[Location]:
        """变异操作"""
        mutated = individual.copy()
        
        if len(mutated) <= 2:
            return mutated
        
        # 确保起始位置不被变异
        mutable_range = (1, len(mutated)) if start_location and mutated[0] == start_location else (0, len(mutated))
        
        if mutable_range[1] - mutable_range[0] >= 2:
            # 随机交换两个位置
            idx1 = random.randint(mutable_range[0], mutable_range[1] - 1)
            idx2 = random.randint(mutable_range[0], mutable_range[1] - 1)
            
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def _generate_neighbor(self, solution: List[Location], start_location: Optional[Location] = None) -> List[Location]:
        """生成邻域解"""
        neighbor = solution.copy()
        
        if len(neighbor) <= 2:
            return neighbor
        
        # 确保起始位置不被移动
        mutable_range = (1, len(neighbor)) if start_location and neighbor[0] == start_location else (0, len(neighbor))
        
        if mutable_range[1] - mutable_range[0] >= 2:
            # 2-opt 移动
            i = random.randint(mutable_range[0], mutable_range[1] - 2)
            j = random.randint(i + 1, mutable_range[1] - 1)
            
            # 反转子序列
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        return neighbor


class TimeScheduler:
    """时间调度器"""
    
    def __init__(self):
        self.time_slots = {
            "morning": (time(8, 0), time(12, 0)),
            "afternoon": (time(12, 0), time(17, 0)),
            "evening": (time(17, 0), time(21, 0))
        }
    
    async def schedule_activities(self, 
                                day_plan: DayPlan,
                                preferences: Dict[str, Any] = None) -> DayPlan:
        """为一天的活动安排时间"""
        preferences = preferences or {}
        
        # 按照优先级和时间偏好排序活动
        sorted_activities = self._sort_activities_by_priority(day_plan.activities, preferences)
        
        # 分配时间槽
        scheduled_activities = []
        current_time = day_plan.start_time
        
        for activity in sorted_activities:
            # 检查活动的时间偏好
            if self._is_activity_suitable_for_time(activity, current_time):
                # 安排活动
                scheduled_activities.append(activity)
                
                # 更新当前时间
                activity_duration = timedelta(minutes=activity.duration)
                current_datetime = datetime.combine(day_plan.date.date(), current_time)
                next_datetime = current_datetime + activity_duration
                current_time = next_datetime.time()
                
                # 检查是否超过结束时间
                if current_time > day_plan.end_time:
                    break
            else:
                # 找到下一个合适的时间槽
                next_suitable_time = self._find_next_suitable_time(activity, current_time)
                if next_suitable_time and next_suitable_time <= day_plan.end_time:
                    current_time = next_suitable_time
                    scheduled_activities.append(activity)
                    
                    # 更新时间
                    activity_duration = timedelta(minutes=activity.duration)
                    current_datetime = datetime.combine(day_plan.date.date(), current_time)
                    next_datetime = current_datetime + activity_duration
                    current_time = next_datetime.time()
        
        # 更新日计划
        day_plan.activities = scheduled_activities
        return day_plan
    
    def _sort_activities_by_priority(self, 
                                   activities: List[Activity],
                                   preferences: Dict[str, Any]) -> List[Activity]:
        """按优先级排序活动"""
        def priority_score(activity: Activity) -> float:
            score = activity.rating  # 基础分数
            
            # 根据偏好调整分数
            preferred_categories = preferences.get("preferred_categories", [])
            if activity.category in preferred_categories:
                score += 2.0
            
            # 根据时间偏好调整
            preferred_times = preferences.get("preferred_times", [])
            if any(pref_time in activity.best_time for pref_time in preferred_times):
                score += 1.0
            
            return score
        
        return sorted(activities, key=priority_score, reverse=True)
    
    def _is_activity_suitable_for_time(self, activity: Activity, current_time: time) -> bool:
        """检查活动是否适合当前时间"""
        if not activity.best_time:
            return True
        
        for time_slot in activity.best_time:
            start_time, end_time = self.time_slots.get(time_slot, (time(0, 0), time(23, 59)))
            if start_time <= current_time <= end_time:
                return True
        
        return False
    
    def _find_next_suitable_time(self, activity: Activity, current_time: time) -> Optional[time]:
        """找到活动的下一个合适时间"""
        if not activity.best_time:
            return current_time
        
        suitable_times = []
        for time_slot in activity.best_time:
            start_time, _ = self.time_slots.get(time_slot, (time(0, 0), time(23, 59)))
            if start_time >= current_time:
                suitable_times.append(start_time)
        
        return min(suitable_times) if suitable_times else None


class TravelPlanningEngine:
    """旅行规划引擎"""
    
    def __init__(self):
        self.constraint_solver = ConstraintSolver()
        self.path_optimizer = PathOptimizer()
        self.time_scheduler = TimeScheduler()
        
        # 规划统计
        self.planning_stats = {
            "total_plans_created": 0,
            "successful_optimizations": 0,
            "average_optimization_time": 0.0,
            "constraint_violations": 0
        }
    
    async def create_travel_plan(self,
                                requirements: Dict[str, Any],
                                available_activities: List[Activity],
                                available_transportations: List[Transportation] = None) -> TravelPlan:
        """创建旅行计划"""
        start_time = datetime.now()
        
        logger.info("开始创建旅行计划")
        
        # 解析需求
        plan_requirements = self._parse_requirements(requirements)
        
        # 创建基础计划
        travel_plan = await self._create_base_plan(plan_requirements)
        
        # 添加约束和目标
        await self._add_constraints_and_objectives(travel_plan, plan_requirements)
        
        # 初始化日计划
        await self._initialize_daily_plans(travel_plan, available_activities)
        
        # 约束求解和优化
        success, optimized_plan = await self.constraint_solver.solve(
            travel_plan, available_activities, available_transportations or []
        )
        
        if success:
            # 路径优化
            await self._optimize_daily_routes(optimized_plan)
            
            # 时间调度
            await self._schedule_daily_activities(optimized_plan, plan_requirements)
            
            # 添加交通安排
            await self._add_transportation(optimized_plan, available_transportations or [])
            
            optimized_plan.status = PlanStatus.READY
            self.planning_stats["successful_optimizations"] += 1
        else:
            optimized_plan.status = PlanStatus.DRAFT
        
        # 更新统计
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._update_planning_stats(optimization_time)
        
        logger.info(f"旅行计划创建完成，状态: {optimized_plan.status}")
        return optimized_plan
    
    async def replan(self,
                    existing_plan: TravelPlan,
                    changes: Dict[str, Any],
                    available_activities: List[Activity],
                    available_transportations: List[Transportation] = None) -> TravelPlan:
        """重新规划"""
        logger.info(f"开始重新规划计划 {existing_plan.id}")
        
        # 应用变更
        updated_plan = await self._apply_changes(existing_plan, changes)
        
        # 重新优化
        success, replanned = await self.constraint_solver.solve(
            updated_plan, available_activities, available_transportations or []
        )
        
        if success:
            await self._optimize_daily_routes(replanned)
            await self._schedule_daily_activities(replanned, changes)
            await self._add_transportation(replanned, available_transportations or [])
            replanned.status = PlanStatus.READY
        
        return replanned
    
    async def optimize_existing_plan(self,
                                   plan: TravelPlan,
                                   optimization_goals: List[str] = None) -> TravelPlan:
        """优化现有计划"""
        optimization_goals = optimization_goals or ["minimize_cost", "maximize_satisfaction"]
        
        logger.info(f"开始优化计划 {plan.id}")
        
        # 更新优化目标
        plan.objectives.clear()
        for goal in optimization_goals:
            if goal == "minimize_cost":
                plan.add_objective(Objective(ObjectiveType.MINIMIZE_COST, weight=1.0))
            elif goal == "maximize_satisfaction":
                plan.add_objective(Objective(ObjectiveType.MAXIMIZE_SATISFACTION, weight=1.0))
            elif goal == "minimize_time":
                plan.add_objective(Objective(ObjectiveType.MINIMIZE_TIME, weight=1.0))
            elif goal == "minimize_distance":
                plan.add_objective(Objective(ObjectiveType.MINIMIZE_DISTANCE, weight=1.0))
        
        # 重新优化
        activities = plan.get_all_activities()
        success, optimized = await self.constraint_solver.solve(plan, activities, [])
        
        if success:
            await self._optimize_daily_routes(optimized)
            optimized.status = PlanStatus.READY
        
        return optimized
    
    def _parse_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """解析规划需求"""
        return {
            "destinations": requirements.get("destinations", []),
            "start_date": requirements.get("start_date"),
            "end_date": requirements.get("end_date"),
            "budget": requirements.get("budget", 0),
            "travelers": requirements.get("travelers", 1),
            "preferences": requirements.get("preferences", {}),
            "constraints": requirements.get("constraints", []),
            "objectives": requirements.get("objectives", ["maximize_satisfaction"])
        }
    
    async def _create_base_plan(self, requirements: Dict[str, Any]) -> TravelPlan:
        """创建基础计划"""
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        plan = TravelPlan(
            id=plan_id,
            name=f"旅行计划_{plan_id}",
            description="自动生成的旅行计划",
            start_date=requirements["start_date"],
            end_date=requirements["end_date"],
            metadata={
                "travelers": requirements["travelers"],
                "destinations": requirements["destinations"],
                "preferences": requirements["preferences"]
            }
        )
        
        return plan
    
    async def _add_constraints_and_objectives(self, plan: TravelPlan, requirements: Dict[str, Any]):
        """添加约束和目标"""
        # 添加预算约束
        if requirements["budget"] > 0:
            budget_constraint = Constraint(
                type=ConstraintType.BUDGET,
                description=f"总预算不超过 {requirements['budget']} 元",
                value=requirements["budget"],
                priority=9
            )
            plan.add_constraint(budget_constraint)
        
        # 添加时间约束
        total_hours = plan.duration_days * 10  # 每天10小时活动时间
        time_constraint = Constraint(
            type=ConstraintType.TIME,
            description=f"总活动时间不超过 {total_hours} 小时",
            value=total_hours * 60,  # 转换为分钟
            priority=8
        )
        plan.add_constraint(time_constraint)
        
        # 添加目标
        for objective_name in requirements["objectives"]:
            if objective_name == "minimize_cost":
                plan.add_objective(Objective(ObjectiveType.MINIMIZE_COST, weight=1.0))
            elif objective_name == "maximize_satisfaction":
                plan.add_objective(Objective(ObjectiveType.MAXIMIZE_SATISFACTION, weight=2.0))
            elif objective_name == "minimize_distance":
                plan.add_objective(Objective(ObjectiveType.MINIMIZE_DISTANCE, weight=0.5))
    
    async def _initialize_daily_plans(self, plan: TravelPlan, activities: List[Activity]):
        """初始化日计划"""
        current_date = plan.start_date
        
        while current_date <= plan.end_date:
            day_plan = DayPlan(
                date=current_date,
                start_time=time(9, 0),
                end_time=time(18, 0)
            )
            plan.daily_plans.append(day_plan)
            current_date += timedelta(days=1)
    
    async def _optimize_daily_routes(self, plan: TravelPlan):
        """优化每日路线"""
        for day_plan in plan.daily_plans:
            if len(day_plan.activities) > 1:
                # 提取活动位置
                locations = [activity.location for activity in day_plan.activities]
                
                # 优化路径
                optimized_locations = await self.path_optimizer.optimize_path(locations)
                
                # 重新排序活动
                optimized_activities = []
                for location in optimized_locations:
                    for activity in day_plan.activities:
                        if activity.location == location and activity not in optimized_activities:
                            optimized_activities.append(activity)
                            break
                
                day_plan.activities = optimized_activities
    
    async def _schedule_daily_activities(self, plan: TravelPlan, requirements: Dict[str, Any]):
        """安排每日活动时间"""
        preferences = requirements.get("preferences", {})
        
        for day_plan in plan.daily_plans:
            await self.time_scheduler.schedule_activities(day_plan, preferences)
    
    async def _add_transportation(self, plan: TravelPlan, transportations: List[Transportation]):
        """添加交通安排"""
        for day_plan in plan.daily_plans:
            if len(day_plan.activities) > 1:
                for i in range(len(day_plan.activities) - 1):
                    from_location = day_plan.activities[i].location
                    to_location = day_plan.activities[i + 1].location
                    
                    # 寻找合适的交通方式
                    transport = self._find_best_transportation(
                        from_location, to_location, transportations
                    )
                    
                    if transport:
                        day_plan.transportations.append(transport)
    
    def _find_best_transportation(self,
                                from_location: Location,
                                to_location: Location,
                                transportations: List[Transportation]) -> Optional[Transportation]:
        """找到最佳交通方式"""
        # 简化实现：根据距离选择交通方式
        distance = from_location.calculate_distance(to_location)
        
        if distance < 1:  # 1公里内步行
            return Transportation(
                mode="walk",
                from_location=from_location,
                to_location=to_location,
                duration=int(distance * 12),  # 假设步行速度5km/h
                cost=0
            )
        elif distance < 10:  # 10公里内打车
            return Transportation(
                mode="taxi",
                from_location=from_location,
                to_location=to_location,
                duration=int(distance * 4),  # 假设车速15km/h（城市）
                cost=distance * 3  # 每公里3元
            )
        else:  # 长距离使用公共交通
            return Transportation(
                mode="public",
                from_location=from_location,
                to_location=to_location,
                duration=int(distance * 6),  # 假设公共交通速度10km/h
                cost=distance * 1  # 每公里1元
            )
    
    async def _apply_changes(self, plan: TravelPlan, changes: Dict[str, Any]) -> TravelPlan:
        """应用变更到计划"""
        updated_plan = self.constraint_solver._deep_copy_plan(plan)
        
        # 应用预算变更
        if "budget" in changes:
            # 更新预算约束
            for constraint in updated_plan.constraints:
                if constraint.type == ConstraintType.BUDGET:
                    constraint.value = changes["budget"]
        
        # 应用活动变更
        if "add_activities" in changes:
            for activity_data in changes["add_activities"]:
                # 这里需要根据activity_data创建Activity对象
                # 简化实现
                pass
        
        if "remove_activities" in changes:
            activities_to_remove = changes["remove_activities"]
            for day_plan in updated_plan.daily_plans:
                day_plan.activities = [
                    activity for activity in day_plan.activities 
                    if activity.id not in activities_to_remove
                ]
        
        return updated_plan
    
    def _update_planning_stats(self, optimization_time: float):
        """更新规划统计"""
        self.planning_stats["total_plans_created"] += 1
        
        total_plans = self.planning_stats["total_plans_created"]
        current_avg = self.planning_stats["average_optimization_time"]
        
        self.planning_stats["average_optimization_time"] = (
            (current_avg * (total_plans - 1) + optimization_time) / total_plans
        )
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        return self.planning_stats.copy()
    
    async def validate_plan(self, plan: TravelPlan) -> Dict[str, Any]:
        """验证计划"""
        validation_result = {
            "is_valid": True,
            "constraint_violations": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 验证约束
        is_valid, violations = plan.validate_constraints()
        validation_result["is_valid"] = is_valid
        validation_result["constraint_violations"] = violations
        
        # 检查警告
        total_cost = plan.calculate_total_cost()
        if total_cost == 0:
            validation_result["warnings"].append("计划中没有任何费用，可能缺少活动")
        
        # 生成建议
        satisfaction_score = plan.calculate_satisfaction_score()
        if satisfaction_score < 3.0:
            validation_result["suggestions"].append("考虑添加更多高评分的活动来提升满意度")
        
        return validation_result


# 全局规划引擎实例
_travel_planning_engine: Optional[TravelPlanningEngine] = None


def get_travel_planning_engine() -> TravelPlanningEngine:
    """获取旅行规划引擎实例"""
    global _travel_planning_engine
    if _travel_planning_engine is None:
        _travel_planning_engine = TravelPlanningEngine()
    return _travel_planning_engine 