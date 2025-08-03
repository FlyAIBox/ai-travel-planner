"""
专业智能体角色实现
实现各种专业化的旅行规划智能体
"""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import random

try:
    from langchain.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class BaseTool:
        def __init__(self, name: str, description: str, func=None):
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
from .multi_agent_system import (
    BaseAgent, AgentRole, AgentTask, AgentTaskRequest, AgentCapability, AgentMessage
)

logger = get_logger(__name__)
settings = get_settings()


class CoordinatorAgent(BaseAgent):
    """协调者智能体 - 统筹管理其他智能体的协作"""
    
    def __init__(self, agent_id: str = "coordinator_001"):
        capabilities = [
            AgentCapability(
                name="coordinate_agents",
                description="协调多个智能体完成复杂任务",
                input_schema={"task_description": "str", "requirements": "dict"},
                output_schema={"coordination_plan": "dict", "agent_assignments": "list"},
                execution_time_estimate=5.0
            ),
            AgentCapability(
                name="task_decomposition",
                description="将复杂任务分解为子任务",
                input_schema={"complex_task": "str", "constraints": "dict"},
                output_schema={"subtasks": "list", "execution_order": "list"},
                execution_time_estimate=3.0
            ),
            AgentCapability(
                name="conflict_resolution",
                description="解决智能体间的冲突和资源争用",
                input_schema={"conflicts": "list", "agents": "list"},
                output_schema={"resolution_strategy": "dict", "reassignments": "list"},
                execution_time_estimate=4.0
            )
        ]
        
        super().__init__(agent_id, AgentRole.COORDINATOR, capabilities)
        
        # 协调状态
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        self.agent_workload: Dict[str, float] = {}
    
    def _initialize_tools(self):
        """初始化协调工具"""
        self.tools = [
            BaseTool(
                name="analyze_task_complexity",
                description="分析任务复杂度并制定分解策略",
                func=self._analyze_task_complexity
            ),
            BaseTool(
                name="assign_tasks_to_agents",
                description="根据智能体能力分配任务",
                func=self._assign_tasks_to_agents
            ),
            BaseTool(
                name="monitor_progress",
                description="监控任务执行进度",
                func=self._monitor_progress
            ),
            BaseTool(
                name="optimize_workflow",
                description="优化工作流程",
                func=self._optimize_workflow
            )
        ]
    
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理协调任务"""
        task_type = task.task_type
        parameters = task.parameters
        
        if task_type == AgentTask.COORDINATE_AGENTS:
            return await self._coordinate_travel_planning(parameters)
        elif task_type == AgentTask.MAKE_RECOMMENDATION:
            return await self._make_travel_recommendation(parameters)
        else:
            return {"error": f"不支持的任务类型: {task_type}"}
    
    async def _coordinate_travel_planning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """协调旅行规划"""
        user_query = parameters.get("user_query", "")
        requirements = parameters.get("requirements", {})
        
        coordination_id = str(uuid.uuid4())
        
        try:
            # 分析用户需求
            analysis = self._analyze_user_requirements(user_query, requirements)
            
            # 任务分解
            subtasks = self._decompose_travel_planning_task(analysis)
            
            # 分配任务给专业智能体
            agent_assignments = await self._assign_subtasks(subtasks)
            
            # 监控执行
            execution_plan = {
                "coordination_id": coordination_id,
                "analysis": analysis,
                "subtasks": subtasks,
                "agent_assignments": agent_assignments,
                "status": "initiated",
                "created_at": datetime.now().isoformat()
            }
            
            self.active_coordinations[coordination_id] = execution_plan
            
            return {
                "coordination_id": coordination_id,
                "status": "success",
                "execution_plan": execution_plan,
                "estimated_completion_time": self._estimate_completion_time(subtasks)
            }
            
        except Exception as e:
            logger.error(f"旅行规划协调失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_user_requirements(self, query: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """分析用户需求"""
        analysis = {
            "destinations": [],
            "travel_dates": {},
            "budget": {},
            "preferences": {},
            "group_size": 1,
            "special_requirements": []
        }
        
        # 提取目的地
        destinations = re.findall(r'[\u4e00-\u9fff]+(?:市|省|县|区)', query)
        if destinations:
            analysis["destinations"] = list(set(destinations))
        
        # 提取时间信息
        dates = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日', query)
        if dates:
            analysis["travel_dates"]["mentioned_dates"] = dates
        
        # 提取预算信息
        budgets = re.findall(r'(\d+)元|(\d+)万|预算.*?(\d+)', query)
        if budgets:
            analysis["budget"]["mentioned_amounts"] = budgets
        
        # 分析旅行类型
        if any(word in query for word in ["家庭", "亲子", "孩子"]):
            analysis["preferences"]["travel_type"] = "family"
        elif any(word in query for word in ["情侣", "蜜月", "浪漫"]):
            analysis["preferences"]["travel_type"] = "romantic"
        elif any(word in query for word in ["商务", "会议", "出差"]):
            analysis["preferences"]["travel_type"] = "business"
        else:
            analysis["preferences"]["travel_type"] = "leisure"
        
        # 合并显式需求
        analysis.update(requirements)
        
        return analysis
    
    def _decompose_travel_planning_task(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分解旅行规划任务"""
        subtasks = []
        
        # 航班搜索任务
        if analysis.get("destinations"):
            subtasks.append({
                "task_type": "search_flights",
                "priority": "high",
                "parameters": {
                    "destinations": analysis["destinations"],
                    "travel_dates": analysis.get("travel_dates", {}),
                    "budget_constraint": analysis.get("budget", {})
                },
                "assigned_agent_role": "flight_expert",
                "estimated_time": 3.0
            })
        
        # 酒店搜索任务
        if analysis.get("destinations"):
            subtasks.append({
                "task_type": "search_hotels",
                "priority": "high",
                "parameters": {
                    "destinations": analysis["destinations"],
                    "travel_dates": analysis.get("travel_dates", {}),
                    "preferences": analysis.get("preferences", {}),
                    "budget_constraint": analysis.get("budget", {})
                },
                "assigned_agent_role": "hotel_expert",
                "estimated_time": 4.0
            })
        
        # 行程规划任务
        subtasks.append({
            "task_type": "plan_itinerary",
            "priority": "medium",
            "parameters": {
                "destinations": analysis.get("destinations", []),
                "travel_dates": analysis.get("travel_dates", {}),
                "preferences": analysis.get("preferences", {}),
                "group_size": analysis.get("group_size", 1)
            },
            "assigned_agent_role": "itinerary_planner",
            "estimated_time": 5.0,
            "dependencies": ["search_flights", "search_hotels"]
        })
        
        # 预算分析任务
        if analysis.get("budget"):
            subtasks.append({
                "task_type": "analyze_budget",
                "priority": "medium",
                "parameters": {
                    "budget_range": analysis["budget"],
                    "destinations": analysis.get("destinations", []),
                    "travel_dates": analysis.get("travel_dates", {}),
                    "group_size": analysis.get("group_size", 1)
                },
                "assigned_agent_role": "budget_analyst",
                "estimated_time": 2.0
            })
        
        # 当地信息任务
        if analysis.get("destinations"):
            subtasks.append({
                "task_type": "provide_local_info",
                "priority": "low",
                "parameters": {
                    "destinations": analysis["destinations"],
                    "travel_type": analysis.get("preferences", {}).get("travel_type", "leisure")
                },
                "assigned_agent_role": "local_guide",
                "estimated_time": 3.0
            })
        
        return subtasks
    
    async def _assign_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分配子任务给智能体"""
        assignments = []
        
        for subtask in subtasks:
            assignment = {
                "subtask_id": str(uuid.uuid4()),
                "task_type": subtask["task_type"],
                "assigned_agent_role": subtask["assigned_agent_role"],
                "parameters": subtask["parameters"],
                "priority": subtask["priority"],
                "estimated_time": subtask["estimated_time"],
                "dependencies": subtask.get("dependencies", []),
                "status": "assigned",
                "assigned_at": datetime.now().isoformat()
            }
            assignments.append(assignment)
        
        return assignments
    
    def _estimate_completion_time(self, subtasks: List[Dict[str, Any]]) -> float:
        """估算完成时间"""
        # 考虑并行执行和依赖关系
        total_time = sum(task["estimated_time"] for task in subtasks)
        
        # 考虑并行执行的优化
        parallel_factor = 0.6  # 假设60%的任务可以并行执行
        estimated_time = total_time * parallel_factor
        
        return estimated_time
    
    def _analyze_task_complexity(self, task_description: str) -> str:
        """分析任务复杂度"""
        complexity_score = 0
        
        # 基于关键词计算复杂度
        complex_keywords = ["多城市", "长途", "定制", "团体", "商务", "会议"]
        complexity_score += sum(1 for keyword in complex_keywords if keyword in task_description)
        
        if complexity_score <= 1:
            return "简单任务，可以直接分配给单个智能体"
        elif complexity_score <= 3:
            return "中等复杂度，需要2-3个智能体协作"
        else:
            return "高复杂度任务，需要多智能体协调配合"
    
    def _assign_tasks_to_agents(self, task_info: str) -> str:
        """分配任务给智能体"""
        # 简化的任务分配逻辑
        assignments = []
        
        if "航班" in task_info or "机票" in task_info:
            assignments.append("flight_expert")
        if "酒店" in task_info or "住宿" in task_info:
            assignments.append("hotel_expert")
        if "行程" in task_info or "路线" in task_info:
            assignments.append("itinerary_planner")
        if "预算" in task_info or "费用" in task_info:
            assignments.append("budget_analyst")
        
        return f"建议分配给以下智能体: {', '.join(assignments)}"
    
    def _monitor_progress(self, coordination_id: str) -> str:
        """监控进度"""
        if coordination_id in self.active_coordinations:
            coordination = self.active_coordinations[coordination_id]
            return f"协调任务 {coordination_id} 当前状态: {coordination['status']}"
        else:
            return f"未找到协调任务 {coordination_id}"
    
    def _optimize_workflow(self, workflow_info: str) -> str:
        """优化工作流程"""
        optimizations = [
            "建议并行执行航班和酒店搜索",
            "预算分析可以在搜索完成后进行",
            "行程规划依赖于交通和住宿信息",
            "当地信息可以并行获取"
        ]
        
        return "工作流程优化建议: " + "; ".join(optimizations)
    
    async def _make_travel_recommendation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """制作旅行推荐"""
        try:
            # 整合其他智能体的结果
            recommendations = {
                "destinations": parameters.get("destinations", []),
                "flights": parameters.get("flight_options", []),
                "hotels": parameters.get("hotel_options", []),
                "itinerary": parameters.get("suggested_itinerary", {}),
                "budget_analysis": parameters.get("budget_breakdown", {}),
                "local_tips": parameters.get("local_information", [])
            }
            
            # 生成综合推荐
            summary = self._generate_recommendation_summary(recommendations)
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "summary": summary,
                "confidence_score": 0.85
            }
            
        except Exception as e:
            logger.error(f"生成推荐失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_recommendation_summary(self, recommendations: Dict[str, Any]) -> str:
        """生成推荐摘要"""
        summary_parts = []
        
        if recommendations.get("destinations"):
            destinations = ", ".join(recommendations["destinations"])
            summary_parts.append(f"推荐目的地: {destinations}")
        
        if recommendations.get("budget_analysis"):
            budget = recommendations["budget_analysis"]
            if "total_cost" in budget:
                summary_parts.append(f"预估总费用: {budget['total_cost']}元")
        
        if recommendations.get("itinerary"):
            itinerary = recommendations["itinerary"]
            if "duration" in itinerary:
                summary_parts.append(f"建议行程时长: {itinerary['duration']}天")
        
        return "；".join(summary_parts) if summary_parts else "正在为您准备个性化推荐..."


class FlightAgent(BaseAgent):
    """航班专家智能体 - 专门处理航班搜索和预订"""
    
    def __init__(self, agent_id: str = "flight_expert_001"):
        capabilities = [
            AgentCapability(
                name="search_flights",
                description="搜索航班信息",
                input_schema={"origin": "str", "destination": "str", "date": "str"},
                output_schema={"flights": "list", "best_options": "list"},
                execution_time_estimate=3.0
            ),
            AgentCapability(
                name="compare_flight_prices",
                description="比较航班价格",
                input_schema={"flight_options": "list", "criteria": "dict"},
                output_schema={"comparison": "dict", "recommendations": "list"},
                execution_time_estimate=2.0
            ),
            AgentCapability(
                name="analyze_flight_routes",
                description="分析航线选择",
                input_schema={"destinations": "list", "constraints": "dict"},
                output_schema={"route_analysis": "dict", "optimal_routes": "list"},
                execution_time_estimate=2.5
            )
        ]
        
        super().__init__(agent_id, AgentRole.FLIGHT_EXPERT, capabilities)
        
        # 航班数据缓存
        self.flight_cache: Dict[str, Any] = {}
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def _initialize_tools(self):
        """初始化航班工具"""
        self.tools = [
            BaseTool(
                name="search_flight_api",
                description="调用航班搜索API",
                func=self._search_flight_api
            ),
            BaseTool(
                name="analyze_price_trends",
                description="分析价格趋势",
                func=self._analyze_price_trends
            ),
            BaseTool(
                name="filter_flights",
                description="根据条件过滤航班",
                func=self._filter_flights
            ),
            BaseTool(
                name="calculate_total_cost",
                description="计算总费用",
                func=self._calculate_total_cost
            )
        ]
    
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理航班相关任务"""
        task_type = task.task_type
        parameters = task.parameters
        
        if task_type == AgentTask.SEARCH_FLIGHTS:
            return await self._search_flights(parameters)
        else:
            return {"error": f"不支持的任务类型: {task_type}"}
    
    async def _search_flights(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """搜索航班"""
        try:
            destinations = parameters.get("destinations", [])
            travel_dates = parameters.get("travel_dates", {})
            budget_constraint = parameters.get("budget_constraint", {})
            
            # 模拟航班搜索
            flight_results = []
            
            for destination in destinations:
                flights = self._mock_flight_search(destination, travel_dates, budget_constraint)
                flight_results.extend(flights)
            
            # 分析和排序
            best_options = self._analyze_and_rank_flights(flight_results, budget_constraint)
            
            return {
                "status": "success",
                "total_flights_found": len(flight_results),
                "all_flights": flight_results,
                "best_options": best_options,
                "search_timestamp": datetime.now().isoformat(),
                "price_analysis": self._generate_price_analysis(flight_results)
            }
            
        except Exception as e:
            logger.error(f"航班搜索失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _mock_flight_search(self, destination: str, travel_dates: Dict[str, Any], 
                           budget_constraint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """模拟航班搜索"""
        flights = []
        
        # 生成模拟航班数据
        airlines = ["中国国航", "东方航空", "南方航空", "海南航空", "厦门航空"]
        
        for i in range(5):  # 每个目的地生成5个航班选项
            flight = {
                "flight_id": f"FL{random.randint(1000, 9999)}",
                "airline": random.choice(airlines),
                "origin": "北京首都机场",
                "destination": f"{destination}机场",
                "departure_time": "08:00",
                "arrival_time": "11:30",
                "duration": "3小时30分钟",
                "price": random.randint(800, 2500),
                "seat_class": random.choice(["经济舱", "商务舱"]),
                "stops": random.choice([0, 1]),
                "aircraft_type": random.choice(["波音737", "空客A320", "波音777"]),
                "availability": random.choice(["充足", "紧张", "仅剩少量"]),
                "booking_class": random.choice(["Y", "B", "H", "K"]),
                "baggage_allowance": "23kg",
                "cancellation_policy": "免费退改",
                "search_date": datetime.now().isoformat()
            }
            flights.append(flight)
        
        return flights
    
    def _analyze_and_rank_flights(self, flights: List[Dict[str, Any]], 
                                 budget_constraint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析和排序航班"""
        # 评分算法
        for flight in flights:
            score = 0.0
            
            # 价格评分 (权重: 40%)
            price = flight["price"]
            max_price = max(f["price"] for f in flights)
            min_price = min(f["price"] for f in flights)
            price_score = 1.0 - (price - min_price) / (max_price - min_price) if max_price > min_price else 1.0
            score += price_score * 0.4
            
            # 时间评分 (权重: 30%)
            # 简化处理，偏好上午出发
            if "08:" in flight["departure_time"]:
                score += 0.3
            elif "09:" in flight["departure_time"] or "10:" in flight["departure_time"]:
                score += 0.25
            else:
                score += 0.15
            
            # 直飞优先 (权重: 20%)
            if flight["stops"] == 0:
                score += 0.2
            else:
                score += 0.1
            
            # 航空公司评分 (权重: 10%)
            preferred_airlines = ["中国国航", "东方航空", "南方航空"]
            if flight["airline"] in preferred_airlines:
                score += 0.1
            else:
                score += 0.05
            
            flight["recommendation_score"] = score
        
        # 按评分排序
        sorted_flights = sorted(flights, key=lambda x: x["recommendation_score"], reverse=True)
        
        return sorted_flights[:3]  # 返回前3个最佳选项
    
    def _generate_price_analysis(self, flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成价格分析"""
        prices = [flight["price"] for flight in flights]
        
        if not prices:
            return {}
        
        return {
            "average_price": sum(prices) / len(prices),
            "min_price": min(prices),
            "max_price": max(prices),
            "price_range": max(prices) - min(prices),
            "recommended_booking_time": "建议提前2-4周预订以获得更好价格",
            "price_trend": "价格相对稳定" if max(prices) - min(prices) < 500 else "价格波动较大"
        }
    
    def _search_flight_api(self, search_params: str) -> str:
        """模拟调用航班搜索API"""
        return f"已搜索航班，参数: {search_params}，找到 {random.randint(5, 20)} 个结果"
    
    def _analyze_price_trends(self, route: str) -> str:
        """分析价格趋势"""
        trends = [
            "该航线价格在过去一月内上涨了15%",
            "价格相对稳定，建议尽快预订",
            "价格有下降趋势，可以等待更好时机",
            "周末价格通常比工作日贵20-30%"
        ]
        return random.choice(trends)
    
    def _filter_flights(self, filter_criteria: str) -> str:
        """过滤航班"""
        return f"根据条件 '{filter_criteria}' 过滤后剩余 {random.randint(3, 10)} 个航班选项"
    
    def _calculate_total_cost(self, flight_details: str) -> str:
        """计算总费用"""
        base_price = random.randint(1000, 3000)
        taxes = base_price * 0.15
        total = base_price + taxes
        return f"机票费用: {base_price}元，税费: {taxes:.0f}元，总计: {total:.0f}元"


class HotelAgent(BaseAgent):
    """酒店专家智能体 - 专门处理酒店搜索和预订"""
    
    def __init__(self, agent_id: str = "hotel_expert_001"):
        capabilities = [
            AgentCapability(
                name="search_hotels",
                description="搜索酒店信息",
                input_schema={"destination": "str", "checkin": "str", "checkout": "str"},
                output_schema={"hotels": "list", "recommendations": "list"},
                execution_time_estimate=4.0
            ),
            AgentCapability(
                name="analyze_hotel_amenities",
                description="分析酒店设施",
                input_schema={"hotel_list": "list", "preferences": "dict"},
                output_schema={"amenity_analysis": "dict", "matched_hotels": "list"},
                execution_time_estimate=2.0
            )
        ]
        
        super().__init__(agent_id, AgentRole.HOTEL_EXPERT, capabilities)
        
        # 酒店数据
        self.hotel_database = self._initialize_hotel_database()
    
    def _initialize_tools(self):
        """初始化酒店工具"""
        self.tools = [
            BaseTool(
                name="search_hotel_api",
                description="调用酒店搜索API",
                func=self._search_hotel_api
            ),
            BaseTool(
                name="analyze_reviews",
                description="分析酒店评价",
                func=self._analyze_reviews
            ),
            BaseTool(
                name="compare_amenities",
                description="比较酒店设施",
                func=self._compare_amenities
            ),
            BaseTool(
                name="check_availability",
                description="检查可用性",
                func=self._check_availability
            )
        ]
    
    def _initialize_hotel_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """初始化酒店数据库"""
        return {
            "北京": [
                {
                    "name": "北京饭店",
                    "star_rating": 5,
                    "price_range": "800-1500",
                    "location": "王府井",
                    "amenities": ["免费WiFi", "健身房", "游泳池", "餐厅", "会议室"]
                },
                {
                    "name": "如家快捷酒店",
                    "star_rating": 3,
                    "price_range": "200-400",
                    "location": "朝阳区",
                    "amenities": ["免费WiFi", "24小时前台", "空调"]
                }
            ],
            "上海": [
                {
                    "name": "上海和平饭店",
                    "star_rating": 5,
                    "price_range": "1000-2000",
                    "location": "外滩",
                    "amenities": ["免费WiFi", "健身房", "spa", "餐厅", "商务中心"]
                }
            ]
        }
    
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理酒店相关任务"""
        task_type = task.task_type
        parameters = task.parameters
        
        if task_type == AgentTask.SEARCH_HOTELS:
            return await self._search_hotels(parameters)
        else:
            return {"error": f"不支持的任务类型: {task_type}"}
    
    async def _search_hotels(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """搜索酒店"""
        try:
            destinations = parameters.get("destinations", [])
            travel_dates = parameters.get("travel_dates", {})
            preferences = parameters.get("preferences", {})
            budget_constraint = parameters.get("budget_constraint", {})
            
            hotel_results = []
            
            for destination in destinations:
                hotels = self._mock_hotel_search(destination, travel_dates, preferences, budget_constraint)
                hotel_results.extend(hotels)
            
            # 分析和推荐
            recommendations = self._analyze_and_recommend_hotels(hotel_results, preferences, budget_constraint)
            
            return {
                "status": "success",
                "total_hotels_found": len(hotel_results),
                "all_hotels": hotel_results,
                "recommendations": recommendations,
                "search_timestamp": datetime.now().isoformat(),
                "booking_tips": self._generate_booking_tips(hotel_results)
            }
            
        except Exception as e:
            logger.error(f"酒店搜索失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _mock_hotel_search(self, destination: str, travel_dates: Dict[str, Any],
                          preferences: Dict[str, Any], budget_constraint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """模拟酒店搜索"""
        base_hotels = self.hotel_database.get(destination, [])
        hotels = []
        
        for base_hotel in base_hotels:
            # 生成具体的搜索结果
            hotel = {
                "hotel_id": f"HT{random.randint(1000, 9999)}",
                "name": base_hotel["name"],
                "star_rating": base_hotel["star_rating"],
                "location": base_hotel["location"],
                "distance_to_center": f"{random.uniform(0.5, 5.0):.1f}公里",
                "price_per_night": random.randint(200, 1500),
                "total_price": random.randint(600, 4500),
                "amenities": base_hotel["amenities"],
                "guest_rating": random.uniform(7.5, 9.5),
                "review_count": random.randint(100, 2000),
                "room_type": random.choice(["标准间", "大床房", "套房", "商务房"]),
                "breakfast_included": random.choice([True, False]),
                "free_cancellation": random.choice([True, False]),
                "wifi_free": "免费WiFi" in base_hotel["amenities"],
                "parking_available": random.choice([True, False]),
                "availability": random.choice(["可预订", "仅剩少量", "需确认"]),
                "special_offers": random.choice([None, "早鸟优惠", "连住优惠", "会员折扣"])
            }
            hotels.append(hotel)
        
        # 如果数据库中没有该目的地的酒店，生成一些通用的
        if not hotels:
            for i in range(3):
                hotel = {
                    "hotel_id": f"HT{random.randint(1000, 9999)}",
                    "name": f"{destination}大酒店{i+1}",
                    "star_rating": random.randint(3, 5),
                    "location": f"{destination}市中心",
                    "distance_to_center": f"{random.uniform(0.5, 3.0):.1f}公里",
                    "price_per_night": random.randint(300, 1200),
                    "total_price": random.randint(900, 3600),
                    "amenities": ["免费WiFi", "空调", "24小时前台"],
                    "guest_rating": random.uniform(7.0, 9.0),
                    "review_count": random.randint(50, 1500),
                    "room_type": "标准间",
                    "breakfast_included": random.choice([True, False]),
                    "free_cancellation": True,
                    "wifi_free": True,
                    "parking_available": random.choice([True, False]),
                    "availability": "可预订"
                }
                hotels.append(hotel)
        
        return hotels
    
    def _analyze_and_recommend_hotels(self, hotels: List[Dict[str, Any]], 
                                    preferences: Dict[str, Any], 
                                    budget_constraint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析和推荐酒店"""
        # 评分算法
        for hotel in hotels:
            score = 0.0
            
            # 价格评分 (权重: 35%)
            price = hotel["price_per_night"]
            if "mentioned_amounts" in budget_constraint:
                # 根据预算评分
                budget_limit = 1000  # 默认预算
                price_score = max(0, 1.0 - (price - budget_limit) / budget_limit) if price > budget_limit else 1.0
            else:
                # 相对价格评分
                prices = [h["price_per_night"] for h in hotels]
                max_price = max(prices)
                min_price = min(prices)
                price_score = 1.0 - (price - min_price) / (max_price - min_price) if max_price > min_price else 1.0
            score += price_score * 0.35
            
            # 评分和评价数量 (权重: 25%)
            rating_score = hotel["guest_rating"] / 10.0
            review_bonus = min(hotel["review_count"] / 1000.0, 0.1)
            score += (rating_score + review_bonus) * 0.25
            
            # 位置评分 (权重: 20%)
            distance = float(hotel["distance_to_center"].replace("公里", ""))
            location_score = max(0, 1.0 - distance / 5.0)  # 5公里内满分
            score += location_score * 0.20
            
            # 设施评分 (权重: 15%)
            amenity_score = len(hotel["amenities"]) / 10.0  # 假设10个设施为满分
            score += min(amenity_score, 1.0) * 0.15
            
            # 其他优势 (权重: 5%)
            bonus = 0.0
            if hotel["free_cancellation"]:
                bonus += 0.02
            if hotel["breakfast_included"]:
                bonus += 0.02
            if hotel["special_offers"]:
                bonus += 0.01
            score += bonus
            
            hotel["recommendation_score"] = min(score, 1.0)
        
        # 按评分排序
        sorted_hotels = sorted(hotels, key=lambda x: x["recommendation_score"], reverse=True)
        
        return sorted_hotels[:5]  # 返回前5个推荐
    
    def _generate_booking_tips(self, hotels: List[Dict[str, Any]]) -> List[str]:
        """生成预订建议"""
        tips = [
            "建议选择可免费取消的酒店，以便行程调整",
            "查看酒店最近的评价，了解当前服务质量",
            "考虑酒店位置，选择交通便利的区域",
            "比较不同平台的价格，可能有优惠差异"
        ]
        
        # 根据搜索结果添加特定建议
        avg_rating = sum(h["guest_rating"] for h in hotels) / len(hotels) if hotels else 0
        if avg_rating > 8.5:
            tips.append("该地区酒店总体评价很高，可以放心选择")
        
        avg_price = sum(h["price_per_night"] for h in hotels) / len(hotels) if hotels else 0
        if avg_price > 800:
            tips.append("该地区酒店价格偏高，建议提前预订或考虑稍远的区域")
        
        return tips[:4]  # 返回最多4个建议
    
    def _search_hotel_api(self, search_params: str) -> str:
        """模拟调用酒店搜索API"""
        return f"已搜索酒店，参数: {search_params}，找到 {random.randint(8, 25)} 个结果"
    
    def _analyze_reviews(self, hotel_name: str) -> str:
        """分析酒店评价"""
        review_aspects = [
            f"{hotel_name} 整体评价良好，服务态度友好",
            "位置便利，交通方便",
            "房间干净整洁，设施完善",
            "性价比较高，值得推荐"
        ]
        return "；".join(review_aspects)
    
    def _compare_amenities(self, hotel_list: str) -> str:
        """比较酒店设施"""
        return "已比较酒店设施，高星级酒店设施更完善，经济型酒店基础设施齐全"
    
    def _check_availability(self, hotel_dates: str) -> str:
        """检查可用性"""
        return f"查询日期 {hotel_dates} 的酒店可用性，大部分酒店有房间"


class ItineraryAgent(BaseAgent):
    """行程规划师智能体 - 专门制定旅行行程"""
    
    def __init__(self, agent_id: str = "itinerary_planner_001"):
        capabilities = [
            AgentCapability(
                name="plan_itinerary",
                description="制定详细的旅行行程",
                input_schema={"destinations": "list", "duration": "int", "preferences": "dict"},
                output_schema={"itinerary": "dict", "daily_plans": "list"},
                execution_time_estimate=5.0
            ),
            AgentCapability(
                name="optimize_routes",
                description="优化旅行路线",
                input_schema={"locations": "list", "transportation": "dict"},
                output_schema={"optimized_route": "list", "time_savings": "float"},
                execution_time_estimate=3.0
            )
        ]
        
        super().__init__(agent_id, AgentRole.ITINERARY_PLANNER, capabilities)
        
        # 景点数据库
        self.attractions_database = self._initialize_attractions_database()
    
    def _initialize_tools(self):
        """初始化行程规划工具"""
        self.tools = [
            BaseTool(
                name="find_attractions",
                description="查找景点信息",
                func=self._find_attractions
            ),
            BaseTool(
                name="calculate_travel_time",
                description="计算景点间交通时间",
                func=self._calculate_travel_time
            ),
            BaseTool(
                name="suggest_restaurants",
                description="推荐餐厅",
                func=self._suggest_restaurants
            ),
            BaseTool(
                name="optimize_schedule",
                description="优化时间安排",
                func=self._optimize_schedule
            )
        ]
    
    def _initialize_attractions_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """初始化景点数据库"""
        return {
            "北京": [
                {"name": "故宫", "type": "历史", "duration": 4, "rating": 9.2, "description": "明清皇宫"},
                {"name": "长城", "type": "历史", "duration": 6, "rating": 9.0, "description": "世界文化遗产"},
                {"name": "天坛", "type": "历史", "duration": 3, "rating": 8.8, "description": "皇家祭坛"},
                {"name": "颐和园", "type": "园林", "duration": 4, "rating": 8.9, "description": "皇家园林"},
                {"name": "798艺术区", "type": "文化", "duration": 3, "rating": 8.5, "description": "当代艺术区"}
            ],
            "上海": [
                {"name": "外滩", "type": "观光", "duration": 2, "rating": 9.1, "description": "上海地标"},
                {"name": "东方明珠", "type": "观光", "duration": 2, "rating": 8.7, "description": "电视塔"},
                {"name": "豫园", "type": "园林", "duration": 3, "rating": 8.6, "description": "古典园林"},
                {"name": "南京路", "type": "购物", "duration": 4, "rating": 8.4, "description": "购物街"},
                {"name": "田子坊", "type": "文化", "duration": 3, "rating": 8.3, "description": "创意园区"}
            ]
        }
    
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理行程规划任务"""
        task_type = task.task_type
        parameters = task.parameters
        
        if task_type == AgentTask.PLAN_ITINERARY:
            return await self._plan_itinerary(parameters)
        else:
            return {"error": f"不支持的任务类型: {task_type}"}
    
    async def _plan_itinerary(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """制定行程计划"""
        try:
            destinations = parameters.get("destinations", [])
            travel_dates = parameters.get("travel_dates", {})
            preferences = parameters.get("preferences", {})
            group_size = parameters.get("group_size", 1)
            
            # 确定行程天数
            duration = self._calculate_duration(travel_dates)
            
            # 为每个目的地制定行程
            itinerary = {
                "total_duration": duration,
                "destinations": destinations,
                "daily_plans": [],
                "recommendations": [],
                "travel_tips": []
            }
            
            for destination in destinations:
                daily_plans = self._create_daily_plans(destination, duration, preferences)
                itinerary["daily_plans"].extend(daily_plans)
            
            # 生成整体建议
            itinerary["recommendations"] = self._generate_itinerary_recommendations(itinerary, preferences)
            itinerary["travel_tips"] = self._generate_travel_tips(destinations, preferences)
            
            return {
                "status": "success",
                "itinerary": itinerary,
                "planning_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"行程规划失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_duration(self, travel_dates: Dict[str, Any]) -> int:
        """计算行程天数"""
        # 简化处理，默认3天
        if "mentioned_dates" in travel_dates:
            return len(travel_dates["mentioned_dates"])
        return 3
    
    def _create_daily_plans(self, destination: str, duration: int, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为目的地创建每日计划"""
        attractions = self.attractions_database.get(destination, [])
        daily_plans = []
        
        # 根据偏好筛选景点
        travel_type = preferences.get("travel_type", "leisure")
        filtered_attractions = self._filter_attractions_by_preference(attractions, travel_type)
        
        # 分配景点到每天
        attractions_per_day = max(1, len(filtered_attractions) // duration)
        
        for day in range(1, duration + 1):
            start_idx = (day - 1) * attractions_per_day
            end_idx = start_idx + attractions_per_day
            day_attractions = filtered_attractions[start_idx:end_idx]
            
            daily_plan = {
                "day": day,
                "destination": destination,
                "theme": self._determine_day_theme(day_attractions),
                "activities": []
            }
            
            # 创建时间安排
            current_time = "09:00"
            for attraction in day_attractions:
                activity = {
                    "time": current_time,
                    "activity": f"参观{attraction['name']}",
                    "duration": f"{attraction['duration']}小时",
                    "description": attraction['description'],
                    "type": attraction['type'],
                    "rating": attraction['rating']
                }
                daily_plan["activities"].append(activity)
                
                # 更新时间
                current_time = self._add_hours(current_time, attraction['duration'] + 1)  # 包括交通时间
            
            # 添加餐饮安排
            self._add_meal_suggestions(daily_plan, destination)
            
            daily_plans.append(daily_plan)
        
        return daily_plans
    
    def _filter_attractions_by_preference(self, attractions: List[Dict[str, Any]], travel_type: str) -> List[Dict[str, Any]]:
        """根据偏好筛选景点"""
        if travel_type == "family":
            # 家庭旅行偏好适合儿童的景点
            preferred_types = ["园林", "观光", "文化"]
        elif travel_type == "romantic":
            # 情侣旅行偏好浪漫景点
            preferred_types = ["园林", "观光"]
        elif travel_type == "business":
            # 商务旅行偏好便捷景点
            preferred_types = ["观光", "购物"]
        else:
            # 休闲旅行包含所有类型
            preferred_types = ["历史", "园林", "观光", "文化", "购物"]
        
        filtered = [attr for attr in attractions if attr["type"] in preferred_types]
        
        # 按评分排序
        filtered.sort(key=lambda x: x["rating"], reverse=True)
        
        return filtered[:6]  # 最多选择6个景点
    
    def _determine_day_theme(self, attractions: List[Dict[str, Any]]) -> str:
        """确定每日主题"""
        if not attractions:
            return "休闲游览"
        
        types = [attr["type"] for attr in attractions]
        type_count = {}
        for t in types:
            type_count[t] = type_count.get(t, 0) + 1
        
        main_type = max(type_count, key=type_count.get)
        
        theme_mapping = {
            "历史": "历史文化之旅",
            "园林": "园林风光之旅", 
            "观光": "城市观光之旅",
            "文化": "文化体验之旅",
            "购物": "购物休闲之旅"
        }
        
        return theme_mapping.get(main_type, "综合游览")
    
    def _add_hours(self, time_str: str, hours: int) -> str:
        """时间加法"""
        hour, minute = map(int, time_str.split(":"))
        hour += hours
        if hour >= 24:
            hour = 23
        return f"{hour:02d}:{minute:02d}"
    
    def _add_meal_suggestions(self, daily_plan: Dict[str, Any], destination: str):
        """添加餐饮建议"""
        meal_suggestions = [
            {
                "time": "12:00",
                "activity": "午餐时间",
                "duration": "1小时",
                "description": f"在{destination}品尝当地特色美食",
                "type": "餐饮",
                "suggestions": self._get_restaurant_suggestions(destination)
            },
            {
                "time": "18:00", 
                "activity": "晚餐时间",
                "duration": "1.5小时",
                "description": f"享用{destination}特色晚餐",
                "type": "餐饮",
                "suggestions": self._get_restaurant_suggestions(destination, meal_type="dinner")
            }
        ]
        
        daily_plan["activities"].extend(meal_suggestions)
        
        # 按时间排序
        daily_plan["activities"].sort(key=lambda x: x["time"])
    
    def _get_restaurant_suggestions(self, destination: str, meal_type: str = "lunch") -> List[str]:
        """获取餐厅建议"""
        restaurant_db = {
            "北京": {
                "lunch": ["全聚德烤鸭店", "老北京炸酱面", "护国寺小吃"],
                "dinner": ["东来顺涮羊肉", "便宜坊烤鸭", "四季民福"]
            },
            "上海": {
                "lunch": ["南翔小笼包", "小杨生煎", "上海本帮菜"],
                "dinner": ["外滩茂悦大酒店", "新荣记", "鹿园"]
            }
        }
        
        return restaurant_db.get(destination, {}).get(meal_type, ["当地特色餐厅"])
    
    def _generate_itinerary_recommendations(self, itinerary: Dict[str, Any], preferences: Dict[str, Any]) -> List[str]:
        """生成行程建议"""
        recommendations = [
            "建议提前预订热门景点门票，避免排队",
            "携带舒适的步行鞋，部分景点需要较多步行",
            "注意天气变化，准备相应的衣物",
            "保持手机电量充足，用于导航和拍照"
        ]
        
        # 根据行程特点添加特定建议
        total_duration = itinerary.get("total_duration", 0)
        if total_duration >= 5:
            recommendations.append("行程较长，建议安排1-2天的休闲时间")
        
        travel_type = preferences.get("travel_type", "")
        if travel_type == "family":
            recommendations.append("带儿童出行时，安排好休息时间和儿童友好的活动")
        
        return recommendations
    
    def _generate_travel_tips(self, destinations: List[str], preferences: Dict[str, Any]) -> List[str]:
        """生成旅行贴士"""
        tips = [
            "提前了解当地的文化习俗和注意事项",
            "准备必要的证件和旅行保险",
            "下载离线地图和翻译软件",
            "保存紧急联系方式和重要信息"
        ]
        
        # 根据目的地添加特定贴士
        for destination in destinations:
            if destination in ["北京", "上海"]:
                tips.append(f"{destination}地铁系统发达，建议使用公共交通")
            
        return tips
    
    def _find_attractions(self, destination: str) -> str:
        """查找景点信息"""
        attractions = self.attractions_database.get(destination, [])
        if attractions:
            names = [attr["name"] for attr in attractions[:3]]
            return f"{destination}主要景点: {', '.join(names)}"
        else:
            return f"暂无{destination}的景点信息"
    
    def _calculate_travel_time(self, route: str) -> str:
        """计算交通时间"""
        # 模拟计算
        travel_time = random.randint(15, 60)
        return f"预计交通时间: {travel_time}分钟"
    
    def _suggest_restaurants(self, location: str) -> str:
        """推荐餐厅"""
        return f"在{location}附近推荐以下餐厅: 当地特色餐厅、连锁品牌、小吃街"
    
    def _optimize_schedule(self, schedule_info: str) -> str:
        """优化时间安排"""
        return "建议合理安排时间，避免过于紧密的行程，预留交通和休息时间"


class BudgetAgent(BaseAgent):
    """预算分析师智能体 - 专门进行旅行预算分析"""
    
    def __init__(self, agent_id: str = "budget_analyst_001"):
        capabilities = [
            AgentCapability(
                name="analyze_budget",
                description="分析旅行预算",
                input_schema={"budget_range": "dict", "destinations": "list", "duration": "int"},
                output_schema={"budget_breakdown": "dict", "cost_analysis": "dict"},
                execution_time_estimate=2.0
            ),
            AgentCapability(
                name="optimize_costs",
                description="优化旅行成本",
                input_schema={"current_plan": "dict", "budget_limit": "float"},
                output_schema={"optimized_plan": "dict", "savings": "float"},
                execution_time_estimate=3.0
            )
        ]
        
        super().__init__(agent_id, AgentRole.BUDGET_ANALYST, capabilities)
        
        # 费用数据库
        self.cost_database = self._initialize_cost_database()
    
    def _initialize_tools(self):
        """初始化预算工具"""
        self.tools = [
            BaseTool(
                name="calculate_accommodation_cost",
                description="计算住宿费用",
                func=self._calculate_accommodation_cost
            ),
            BaseTool(
                name="estimate_meal_cost",
                description="估算餐饮费用",
                func=self._estimate_meal_cost
            ),
            BaseTool(
                name="analyze_transportation_cost",
                description="分析交通费用",
                func=self._analyze_transportation_cost
            ),
            BaseTool(
                name="suggest_cost_savings",
                description="建议节省费用的方法",
                func=self._suggest_cost_savings
            )
        ]
    
    def _initialize_cost_database(self) -> Dict[str, Dict[str, Any]]:
        """初始化费用数据库"""
        return {
            "北京": {
                "accommodation": {"budget": 150, "mid": 400, "luxury": 800},
                "meals": {"budget": 80, "mid": 150, "luxury": 300},
                "local_transport": {"daily": 30, "taxi": 50},
                "attractions": {"average": 60, "premium": 120},
                "shopping": {"souvenirs": 100, "luxury": 500}
            },
            "上海": {
                "accommodation": {"budget": 180, "mid": 450, "luxury": 900},
                "meals": {"budget": 90, "mid": 180, "luxury": 350},
                "local_transport": {"daily": 35, "taxi": 60},
                "attractions": {"average": 70, "premium": 140},
                "shopping": {"souvenirs": 120, "luxury": 600}
            }
        }
    
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理预算分析任务"""
        task_type = task.task_type
        parameters = task.parameters
        
        if task_type == AgentTask.ANALYZE_BUDGET:
            return await self._analyze_budget(parameters)
        else:
            return {"error": f"不支持的任务类型: {task_type}"}
    
    async def _analyze_budget(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """分析旅行预算"""
        try:
            budget_range = parameters.get("budget_range", {})
            destinations = parameters.get("destinations", [])
            travel_dates = parameters.get("travel_dates", {})
            group_size = parameters.get("group_size", 1)
            
            # 计算行程天数
            duration = len(travel_dates.get("mentioned_dates", [])) or 3
            
            # 分析预算
            budget_analysis = {
                "total_budget_estimate": 0,
                "per_person_cost": 0,
                "cost_breakdown": {},
                "budget_recommendations": [],
                "cost_optimization_tips": []
            }
            
            total_cost = 0
            cost_breakdown = {}
            
            for destination in destinations:
                dest_costs = self._calculate_destination_costs(destination, duration, group_size)
                cost_breakdown[destination] = dest_costs
                total_cost += dest_costs["total"]
            
            # 添加交通费用
            transportation_cost = self._estimate_transportation_cost(destinations, group_size)
            cost_breakdown["transportation"] = {"total": transportation_cost}
            total_cost += transportation_cost
            
            budget_analysis["total_budget_estimate"] = total_cost
            budget_analysis["per_person_cost"] = total_cost / group_size
            budget_analysis["cost_breakdown"] = cost_breakdown
            
            # 生成预算建议
            budget_analysis["budget_recommendations"] = self._generate_budget_recommendations(
                total_cost, budget_range, group_size
            )
            budget_analysis["cost_optimization_tips"] = self._generate_cost_optimization_tips(
                cost_breakdown, destinations
            )
            
            return {
                "status": "success",
                "budget_analysis": budget_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"预算分析失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_destination_costs(self, destination: str, duration: int, group_size: int) -> Dict[str, Any]:
        """计算目的地费用"""
        cost_data = self.cost_database.get(destination, self.cost_database["北京"])  # 默认使用北京数据
        
        # 选择消费水平（这里简化为中等水平）
        accommodation_cost = cost_data["accommodation"]["mid"] * duration
        meal_cost = cost_data["meals"]["mid"] * duration
        transport_cost = cost_data["local_transport"]["daily"] * duration
        attraction_cost = cost_data["attractions"]["average"] * max(1, duration // 2)  # 假设每两天参观一个付费景点
        shopping_cost = cost_data["shopping"]["souvenirs"]
        
        total_cost = (accommodation_cost + meal_cost + transport_cost + attraction_cost + shopping_cost) * group_size
        
        return {
            "accommodation": accommodation_cost * group_size,
            "meals": meal_cost * group_size,
            "local_transport": transport_cost * group_size,
            "attractions": attraction_cost * group_size,
            "shopping": shopping_cost * group_size,
            "total": total_cost
        }
    
    def _estimate_transportation_cost(self, destinations: List[str], group_size: int) -> float:
        """估算交通费用"""
        if len(destinations) == 1:
            # 单目的地往返
            base_cost = 1500  # 基础往返交通费
        else:
            # 多目的地
            base_cost = 2000 + (len(destinations) - 1) * 500
        
        return base_cost * group_size
    
    def _generate_budget_recommendations(self, total_cost: float, budget_range: Dict[str, Any], group_size: int) -> List[str]:
        """生成预算建议"""
        recommendations = []
        
        per_person_cost = total_cost / group_size
        
        # 分析预算充足性
        if "mentioned_amounts" in budget_range:
            # 这里简化处理预算分析
            recommendations.append(f"预估总费用为每人{per_person_cost:.0f}元")
        else:
            recommendations.append(f"建议每人准备{per_person_cost:.0f}元预算")
        
        # 费用分级建议
        if per_person_cost < 2000:
            recommendations.append("属于经济型旅行，建议关注性价比")
        elif per_person_cost < 5000:
            recommendations.append("属于中等消费水平，可以享受较好的服务")
        else:
            recommendations.append("属于高端旅行，可以选择优质的酒店和服务")
        
        recommendations.extend([
            "建议预留10-20%的应急资金",
            "提前预订可以获得更好的价格",
            "关注各种优惠活动和折扣信息"
        ])
        
        return recommendations
    
    def _generate_cost_optimization_tips(self, cost_breakdown: Dict[str, Any], destinations: List[str]) -> List[str]:
        """生成成本优化建议"""
        tips = [
            "住宿: 考虑民宿或连锁酒店，性价比更高",
            "餐饮: 尝试当地小吃和特色餐厅，价格实惠",
            "交通: 使用公共交通，购买当地交通卡",
            "景点: 寻找免费或优惠的景点和活动",
            "购物: 比较价格，避免冲动消费"
        ]
        
        # 根据目的地添加特定建议
        for destination in destinations:
            if destination in ["北京", "上海"]:
                tips.append(f"{destination}地铁很发达，建议多使用地铁出行")
        
        return tips
    
    def _calculate_accommodation_cost(self, accommodation_info: str) -> str:
        """计算住宿费用"""
        # 解析住宿信息并计算费用
        cost_range = random.randint(200, 800)
        return f"住宿费用预估: {cost_range}-{cost_range + 200}元/晚"
    
    def _estimate_meal_cost(self, meal_info: str) -> str:
        """估算餐饮费用"""
        daily_cost = random.randint(80, 200)
        return f"餐饮费用预估: {daily_cost}元/天"
    
    def _analyze_transportation_cost(self, transport_info: str) -> str:
        """分析交通费用"""
        cost = random.randint(500, 2000)
        return f"交通费用预估: {cost}元"
    
    def _suggest_cost_savings(self, current_budget: str) -> str:
        """建议节省费用的方法"""
        savings_tips = [
            "选择淡季出行可节省20-30%费用",
            "提前预订酒店和机票有更多优惠",
            "使用团购和优惠券",
            "选择当地公共交通替代出租车"
        ]
        return "节省费用建议: " + "; ".join(savings_tips)


class LocalGuideAgent(BaseAgent):
    """当地向导智能体 - 提供当地信息和建议"""
    
    def __init__(self, agent_id: str = "local_guide_001"):
        capabilities = [
            AgentCapability(
                name="provide_local_info",
                description="提供当地信息和建议",
                input_schema={"destination": "str", "interest_type": "str"},
                output_schema={"local_info": "dict", "recommendations": "list"},
                execution_time_estimate=3.0
            ),
            AgentCapability(
                name="cultural_guidance",
                description="提供文化指导",
                input_schema={"destination": "str", "activities": "list"},
                output_schema={"cultural_tips": "list", "etiquette": "list"},
                execution_time_estimate=2.0
            )
        ]
        
        super().__init__(agent_id, AgentRole.LOCAL_GUIDE, capabilities)
        
        # 当地信息数据库
        self.local_info_database = self._initialize_local_info_database()
    
    def _initialize_tools(self):
        """初始化当地向导工具"""
        self.tools = [
            BaseTool(
                name="get_weather_info",
                description="获取天气信息",
                func=self._get_weather_info
            ),
            BaseTool(
                name="find_local_events",
                description="查找当地活动",
                func=self._find_local_events
            ),
            BaseTool(
                name="recommend_hidden_gems",
                description="推荐小众景点",
                func=self._recommend_hidden_gems
            ),
            BaseTool(
                name="provide_safety_tips",
                description="提供安全建议",
                func=self._provide_safety_tips
            )
        ]
    
    def _initialize_local_info_database(self) -> Dict[str, Dict[str, Any]]:
        """初始化当地信息数据库"""
        return {
            "北京": {
                "weather": "四季分明，春秋季节最适宜旅游",
                "culture": {
                    "customs": ["尊重传统文化", "参观寺庙时保持安静", "拍照前询问许可"],
                    "language": "普通话，基本英语在景区可用",
                    "currency": "人民币(CNY)"
                },
                "transportation": {
                    "metro": "地铁系统发达，覆盖主要景区",
                    "taxi": "出租车较多，建议使用滴滴等打车软件",
                    "bus": "公交网络完善，但可能比较复杂"
                },
                "food": {
                    "specialties": ["北京烤鸭", "炸酱面", "豆汁", "驴打滚"],
                    "restaurants": ["全聚德", "便宜坊", "东来顺"],
                    "street_food": ["王府井小吃街", "护国寺小吃"]
                },
                "shopping": {
                    "areas": ["王府井", "西单", "三里屯", "798艺术区"],
                    "souvenirs": ["京剧脸谱", "景泰蓝", "毛笔字画", "中国结"]
                },
                "hidden_gems": [
                    "南锣鼓巷胡同游",
                    "雍和宫祈福",
                    "什刹海酒吧街",
                    "钟鼓楼观夜景"
                ],
                "safety_tips": [
                    "注意保管财物，避免在人群密集处露财",
                    "使用正规交通工具，避免黑车",
                    "在景区注意防范强制消费",
                    "保存紧急联系电话"
                ]
            },
            "上海": {
                "weather": "亚热带季风气候，春秋最佳",
                "culture": {
                    "customs": ["海纳百川的城市文化", "中西合璧的建筑风格"],
                    "language": "普通话和上海话，英语普及度较高",
                    "currency": "人民币(CNY)"
                },
                "transportation": {
                    "metro": "地铁网络非常发达，是最佳出行方式",
                    "taxi": "出租车便利，也可使用网约车",
                    "bus": "公交系统完善"
                },
                "food": {
                    "specialties": ["小笼包", "生煎包", "上海菜", "白切鸡"],
                    "restaurants": ["南翔小笼", "小杨生煎", "老正兴"],
                    "street_food": ["城隍庙小吃", "七宝老街"]
                },
                "shopping": {
                    "areas": ["南京路", "淮海路", "新天地", "田子坊"],
                    "souvenirs": ["海派文化产品", "丝绸制品", "茶叶", "工艺品"]
                },
                "hidden_gems": [
                    "1933老场坊创意园",
                    "多伦路文化街",
                    "朱家角古镇",
                    "泰康路艺术街"
                ],
                "safety_tips": [
                    "外滩等热门景点人多，注意安全",
                    "乘坐地铁时注意高峰期拥挤",
                    "在商业区注意防范推销",
                    "保持通讯畅通"
                ]
            }
        }
    
    async def process_task(self, task: AgentTaskRequest) -> Dict[str, Any]:
        """处理当地向导任务"""
        task_type = task.task_type
        parameters = task.parameters
        
        if task_type == AgentTask.PROVIDE_LOCAL_INFO:
            return await self._provide_local_info(parameters)
        else:
            return {"error": f"不支持的任务类型: {task_type}"}
    
    async def _provide_local_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """提供当地信息"""
        try:
            destinations = parameters.get("destinations", [])
            travel_type = parameters.get("travel_type", "leisure")
            
            local_info = {
                "destinations_info": {},
                "general_tips": [],
                "cultural_guidance": [],
                "practical_information": {}
            }
            
            for destination in destinations:
                dest_info = self._get_destination_info(destination, travel_type)
                local_info["destinations_info"][destination] = dest_info
            
            # 生成通用建议
            local_info["general_tips"] = self._generate_general_tips(destinations, travel_type)
            local_info["cultural_guidance"] = self._generate_cultural_guidance(destinations)
            local_info["practical_information"] = self._generate_practical_info(destinations)
            
            return {
                "status": "success",
                "local_information": local_info,
                "guide_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"当地信息提供失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_destination_info(self, destination: str, travel_type: str) -> Dict[str, Any]:
        """获取目的地信息"""
        base_info = self.local_info_database.get(destination, {})
        
        destination_info = {
            "weather_info": base_info.get("weather", "请关注当地天气预报"),
            "cultural_highlights": base_info.get("culture", {}),
            "transportation_guide": base_info.get("transportation", {}),
            "food_recommendations": base_info.get("food", {}),
            "shopping_guide": base_info.get("shopping", {}),
            "hidden_gems": base_info.get("hidden_gems", []),
            "safety_tips": base_info.get("safety_tips", [])
        }
        
        # 根据旅行类型定制信息
        if travel_type == "family":
            destination_info["family_friendly_tips"] = [
                "选择适合儿童的景点和活动",
                "注意餐厅的儿童设施",
                "准备常用儿童药品",
                "规划充足的休息时间"
            ]
        elif travel_type == "business":
            destination_info["business_facilities"] = [
                "会议中心和商务酒店信息",
                "商务餐厅推荐",
                "交通便利的办公区域",
                "商务服务设施"
            ]
        
        return destination_info
    
    def _generate_general_tips(self, destinations: List[str], travel_type: str) -> List[str]:
        """生成通用建议"""
        tips = [
            "提前了解当地的天气情况，准备合适的衣物",
            "学习几句基本的当地用语，方便沟通", 
            "下载离线地图和翻译软件",
            "了解当地的紧急联系方式"
        ]
        
        # 根据目的地添加特定建议
        if any(dest in ["北京", "上海"] for dest in destinations):
            tips.append("这些城市地铁很方便，建议办理交通卡")
        
        # 根据旅行类型添加建议
        if travel_type == "family":
            tips.append("带儿童出行时，准备足够的零食和娱乐用品")
        elif travel_type == "romantic":
            tips.append("寻找适合情侣的浪漫餐厅和景点")
        
        return tips
    
    def _generate_cultural_guidance(self, destinations: List[str]) -> List[str]:
        """生成文化指导"""
        guidance = [
            "尊重当地的文化传统和习俗",
            "在宗教场所保持适当的礼貌和安静",
            "拍照时注意是否被允许，特别是在博物馆和寺庙",
            "尝试当地美食时保持开放的心态"
        ]
        
        # 添加特定的文化建议
        for destination in destinations:
            if destination == "北京":
                guidance.append("北京是历史文化名城，参观历史景点时保持敬畏之心")
            elif destination == "上海":
                guidance.append("上海融合中西文化，可以体验不同的文化氛围")
        
        return guidance
    
    def _generate_practical_info(self, destinations: List[str]) -> Dict[str, Any]:
        """生成实用信息"""
        practical_info = {
            "emergency_contacts": {
                "police": "110",
                "fire": "119", 
                "medical": "120",
                "traffic_accident": "122"
            },
            "useful_apps": [
                "高德地图/百度地图 - 导航",
                "支付宝/微信 - 支付",
                "滴滴出行 - 打车",
                "美团/饿了么 - 订餐"
            ],
            "payment_methods": [
                "移动支付（支付宝、微信）非常普及",
                "现金仍然被接受",
                "主要信用卡在大型商场和酒店可用"
            ],
            "communication": [
                "购买当地手机卡或使用国际漫游",
                "WiFi在酒店、餐厅、商场普遍可用",
                "下载翻译软件以备不时之需"
            ]
        }
        
        return practical_info
    
    def _get_weather_info(self, location: str) -> str:
        """获取天气信息"""
        weather_conditions = ["晴", "多云", "小雨", "阴"]
        temperature = random.randint(15, 30)
        condition = random.choice(weather_conditions)
        return f"{location}天气: {condition}，温度约{temperature}°C，建议携带适当衣物"
    
    def _find_local_events(self, location_date: str) -> str:
        """查找当地活动"""
        events = [
            "当地文化节庆活动",
            "博物馆特展", 
            "音乐会或演出",
            "传统市集",
            "艺术展览"
        ]
        selected_events = random.sample(events, 2)
        return f"推荐当地活动: {', '.join(selected_events)}"
    
    def _recommend_hidden_gems(self, destination: str) -> str:
        """推荐小众景点"""
        hidden_gems = self.local_info_database.get(destination, {}).get("hidden_gems", [])
        if hidden_gems:
            selected = random.sample(hidden_gems, min(2, len(hidden_gems)))
            return f"{destination}小众推荐: {', '.join(selected)}"
        else:
            return f"建议探索{destination}的当地街区，发现独特的魅力"
    
    def _provide_safety_tips(self, destination: str) -> str:
        """提供安全建议"""
        safety_tips = self.local_info_database.get(destination, {}).get("safety_tips", [])
        if safety_tips:
            return f"{destination}安全提示: " + "; ".join(safety_tips[:3])
        else:
            return "保持警觉，注意个人财物和人身安全"


# 全局智能体实例管理
_specialized_agents: Dict[str, BaseAgent] = {}


def create_all_specialized_agents() -> Dict[str, BaseAgent]:
    """创建所有专业智能体"""
    global _specialized_agents
    
    if not _specialized_agents:
        _specialized_agents = {
            "coordinator": CoordinatorAgent(),
            "flight_expert": FlightAgent(),
            "hotel_expert": HotelAgent(),
            "itinerary_planner": ItineraryAgent(),
            "budget_analyst": BudgetAgent(),
            "local_guide": LocalGuideAgent()
        }
        
        logger.info(f"已创建 {len(_specialized_agents)} 个专业智能体")
    
    return _specialized_agents


def get_agent_by_role(role: AgentRole) -> Optional[BaseAgent]:
    """根据角色获取智能体"""
    role_mapping = {
        AgentRole.COORDINATOR: "coordinator",
        AgentRole.FLIGHT_EXPERT: "flight_expert", 
        AgentRole.HOTEL_EXPERT: "hotel_expert",
        AgentRole.ITINERARY_PLANNER: "itinerary_planner",
        AgentRole.BUDGET_ANALYST: "budget_analyst",
        AgentRole.LOCAL_GUIDE: "local_guide"
    }
    
    agent_key = role_mapping.get(role)
    if agent_key:
        agents = create_all_specialized_agents()
        return agents.get(agent_key)
    
    return None