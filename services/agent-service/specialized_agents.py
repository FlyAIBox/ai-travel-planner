"""
专业智能体角色实现
创建CoordinatorAgent主控智能体、FlightAgent航班搜索专家、HotelAgent酒店推荐专家、
ItineraryAgent行程规划师、BudgetAgent预算分析师和LocalGuideAgent当地向导
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import random

from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from langchain_core.tools import tool

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .multi_agent_system import BaseAgent, AgentRole, AgentTask, TaskType, TaskStatus, MessageType

logger = get_logger(__name__)
settings = get_settings()


# 定义专业工具
@tool
def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None) -> Dict[str, Any]:
    """搜索航班信息"""
    # 模拟航班搜索API调用
    flights = [
        {
            "flight_id": f"FL{random.randint(1000, 9999)}",
            "airline": random.choice(["中国国航", "东方航空", "南方航空", "海南航空"]),
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure_date} {random.randint(6, 22):02d}:{random.randint(0, 59):02d}",
            "arrival_time": f"{departure_date} {random.randint(8, 23):02d}:{random.randint(0, 59):02d}",
            "price": random.randint(800, 3000),
            "duration": f"{random.randint(1, 8)}小时{random.randint(0, 59)}分钟",
            "stops": random.randint(0, 2)
        }
        for _ in range(3)
    ]
    
    return {
        "status": "success",
        "flights": flights,
        "search_time": datetime.now().isoformat()
    }


@tool
def search_hotels(city: str, checkin_date: str, checkout_date: str, guests: int = 2) -> Dict[str, Any]:
    """搜索酒店信息"""
    # 模拟酒店搜索API调用
    hotels = [
        {
            "hotel_id": f"HT{random.randint(1000, 9999)}",
            "name": f"{city}{random.choice(['大酒店', '国际酒店', '商务酒店', '度假村', '精品酒店'])}",
            "rating": random.randint(3, 5),
            "price_per_night": random.randint(200, 1500),
            "location": f"{city}{random.choice(['市中心', '机场附近', '商业区', '风景区', '交通枢纽'])}",
            "amenities": random.sample(["免费WiFi", "早餐", "停车场", "健身房", "游泳池", "spa"], 3),
            "distance_to_center": f"{random.randint(1, 20)}公里",
            "guest_rating": round(random.uniform(7.5, 9.5), 1)
        }
        for _ in range(4)
    ]
    
    return {
        "status": "success",
        "hotels": hotels,
        "checkin": checkin_date,
        "checkout": checkout_date,
        "search_time": datetime.now().isoformat()
    }


@tool
def get_weather_forecast(city: str, date: str) -> Dict[str, Any]:
    """获取天气预报"""
    # 模拟天气API调用
    weather_conditions = ["晴", "多云", "阴", "小雨", "中雨", "雪"]
    
    return {
        "status": "success",
        "city": city,
        "date": date,
        "temperature": {
            "high": random.randint(15, 35),
            "low": random.randint(5, 25)
        },
        "condition": random.choice(weather_conditions),
        "humidity": f"{random.randint(30, 90)}%",
        "wind_speed": f"{random.randint(5, 25)}km/h",
        "uv_index": random.randint(1, 10),
        "recommendation": "建议携带雨具" if "雨" in random.choice(weather_conditions) else "天气良好，适合出行"
    }


@tool
def calculate_travel_budget(flights: List[Dict], hotels: List[Dict], days: int, meal_budget_per_day: int = 150) -> Dict[str, Any]:
    """计算旅行预算"""
    flight_cost = sum(flight.get("price", 0) for flight in flights)
    hotel_cost = sum(hotel.get("price_per_night", 0) for hotel in hotels) * days
    meal_cost = meal_budget_per_day * days
    activity_cost = days * 200  # 每天活动费用估算
    transport_cost = days * 50   # 每天交通费用估算
    
    total_cost = flight_cost + hotel_cost + meal_cost + activity_cost + transport_cost
    
    return {
        "status": "success",
        "breakdown": {
            "flights": flight_cost,
            "hotels": hotel_cost,
            "meals": meal_cost,
            "activities": activity_cost,
            "local_transport": transport_cost
        },
        "total": total_cost,
        "currency": "CNY",
        "cost_per_day": round(total_cost / days, 2),
        "recommendations": [
            "考虑预订早鸟优惠航班以节省费用",
            "选择包含早餐的酒店可以减少餐饮支出",
            f"建议每日预算控制在{round(total_cost / days * 1.1, 2)}元以内"
        ]
    }


@tool
def get_local_attractions(city: str, category: str = "all") -> Dict[str, Any]:
    """获取当地景点信息"""
    attraction_types = {
        "历史": ["古建筑", "博物馆", "文化遗址", "历史街区"],
        "自然": ["公园", "山景", "海滩", "湖泊"],
        "娱乐": ["主题公园", "购物中心", "夜市", "酒吧街"],
        "美食": ["特色餐厅", "小吃街", "茶楼", "咖啡厅"]
    }
    
    attractions = []
    for cat, types in attraction_types.items():
        if category == "all" or category == cat:
            for attraction_type in types:
                attractions.append({
                    "name": f"{city}{attraction_type}",
                    "category": cat,
                    "rating": round(random.uniform(4.0, 4.9), 1),
                    "price": random.choice(["免费", f"{random.randint(20, 200)}元"]),
                    "duration": f"{random.randint(1, 6)}小时",
                    "description": f"著名的{city}{attraction_type}，是游客必访之地",
                    "opening_hours": "09:00-18:00",
                    "best_visit_time": random.choice(["上午", "下午", "傍晚", "全天"])
                })
    
    return {
        "status": "success",
        "city": city,
        "attractions": random.sample(attractions, min(6, len(attractions))),
        "total_found": len(attractions)
    }


@tool
def create_itinerary(destinations: List[str], days: int, preferences: List[str] = None) -> Dict[str, Any]:
    """创建行程规划"""
    preferences = preferences or ["观光", "美食", "购物"]
    
    itinerary = []
    activities_per_day = {
        "观光": ["参观博物馆", "游览古迹", "登山观景", "城市漫步"],
        "美食": ["品尝当地特色菜", "访问知名餐厅", "体验街头小吃", "参加美食之旅"],
        "购物": ["逛商业街", "访问特色市场", "购买纪念品", "体验当地工艺品"],
        "休闲": ["公园散步", "咖啡厅休息", "spa体验", "海滩放松"]
    }
    
    for day in range(1, days + 1):
        daily_plan = {
            "day": day,
            "date": (datetime.now() + timedelta(days=day-1)).strftime("%Y-%m-%d"),
            "destination": destinations[(day-1) % len(destinations)],
            "activities": []
        }
        
        # 为每天安排3-4个活动
        for time_slot in ["上午", "下午", "晚上"]:
            pref = random.choice(preferences)
            activity = random.choice(activities_per_day.get(pref, ["自由活动"]))
            daily_plan["activities"].append({
                "time": time_slot,
                "activity": activity,
                "category": pref,
                "duration": f"{random.randint(1, 3)}小时",
                "estimated_cost": f"{random.randint(50, 300)}元"
            })
        
        itinerary.append(daily_plan)
    
    return {
        "status": "success",
        "itinerary": itinerary,
        "total_days": days,
        "destinations": destinations,
        "summary": f"{days}天行程，涵盖{len(destinations)}个目的地，主要关注{', '.join(preferences)}"
    }


class CoordinatorAgent(BaseAgent):
    """主控协调智能体"""
    
    def __init__(self, agent_id: str, message_bus):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            message_bus=message_bus,
            tools=[],
            memory_window=20
        )
        
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.active_plans: Dict[str, Dict[str, Any]] = {}
        
    def get_capabilities(self) -> List[TaskType]:
        return [TaskType.COORDINATE, TaskType.PLAN, TaskType.ANALYZE]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理协调任务"""
        if task.task_type == TaskType.COORDINATE:
            return await self._coordinate_multi_agent_task(task)
        elif task.task_type == TaskType.PLAN:
            return await self._create_travel_plan(task)
        elif task.task_type == TaskType.ANALYZE:
            return await self._analyze_travel_requirements(task)
        else:
            return {"error": f"不支持的任务类型: {task.task_type}"}
    
    async def _coordinate_multi_agent_task(self, task: AgentTask) -> Dict[str, Any]:
        """协调多智能体任务"""
        travel_request = task.context
        plan_id = f"plan_{task.task_id}"
        
        # 分析旅行需求
        requirements = await self._parse_travel_requirements(travel_request)
        
        # 创建协作计划
        collaboration_plan = {
            "plan_id": plan_id,
            "requirements": requirements,
            "subtasks": [],
            "results": {},
            "status": "in_progress",
            "created_at": datetime.now().isoformat()
        }
        
        # 分配子任务给专业智能体
        subtasks = await self._create_subtasks(requirements)
        
        for subtask in subtasks:
            # 发送任务请求给相应的专业智能体
            await self._delegate_task_to_specialist(subtask, plan_id)
            collaboration_plan["subtasks"].append(subtask)
        
        self.active_plans[plan_id] = collaboration_plan
        
        return {
            "status": "coordination_started",
            "plan_id": plan_id,
            "subtasks_created": len(subtasks),
            "estimated_completion": "15-20分钟"
        }
    
    async def _create_travel_plan(self, task: AgentTask) -> Dict[str, Any]:
        """创建综合旅行计划"""
        context = task.context
        
        # 基础信息提取
        destinations = context.get("destinations", [])
        travel_dates = context.get("dates", {})
        budget = context.get("budget", 0)
        preferences = context.get("preferences", [])
        
        # 创建综合计划
        plan = {
            "plan_id": task.task_id,
            "destinations": destinations,
            "dates": travel_dates,
            "budget": budget,
            "preferences": preferences,
            "recommendations": {
                "best_travel_time": "建议在春秋季节出行，天气宜人",
                "budget_advice": "建议提前预订以获得更好价格",
                "packing_tips": ["根据目的地天气准备衣物", "携带必要证件", "准备常用药品"]
            },
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "travel_plan": plan,
            "next_steps": ["详细预订", "行程确认", "出行准备"]
        }
    
    async def _analyze_travel_requirements(self, task: AgentTask) -> Dict[str, Any]:
        """分析旅行需求"""
        content = task.content
        context = task.context
        
        # 需求分析
        analysis = {
            "destination_analysis": self._analyze_destinations(content),
            "budget_analysis": self._analyze_budget(content),
            "time_analysis": self._analyze_travel_time(content),
            "preference_analysis": self._analyze_preferences(content),
            "complexity_score": self._calculate_complexity(context),
            "recommendations": []
        }
        
        # 生成建议
        if analysis["complexity_score"] > 0.7:
            analysis["recommendations"].append("建议寻求专业旅行顾问协助")
        
        if len(analysis["destination_analysis"]["destinations"]) > 3:
            analysis["recommendations"].append("建议适当减少目的地数量以获得更好体验")
        
        return {
            "status": "success",
            "analysis": analysis,
            "confidence": 0.85
        }
    
    async def _parse_travel_requirements(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """解析旅行需求"""
        return {
            "destinations": request.get("destinations", []),
            "dates": request.get("dates", {}),
            "budget": request.get("budget", 0),
            "travelers": request.get("travelers", 1),
            "preferences": request.get("preferences", []),
            "special_requirements": request.get("special_requirements", [])
        }
    
    async def _create_subtasks(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建子任务"""
        subtasks = []
        
        # 航班搜索任务
        if requirements.get("destinations"):
            subtasks.append({
                "task_type": "search_flights",
                "agent_role": "flight_expert",
                "priority": 8,
                "params": {
                    "destinations": requirements["destinations"],
                    "dates": requirements.get("dates", {}),
                    "travelers": requirements.get("travelers", 1)
                }
            })
        
        # 酒店搜索任务
        if requirements.get("destinations"):
            subtasks.append({
                "task_type": "search_hotels",
                "agent_role": "hotel_expert",
                "priority": 7,
                "params": {
                    "destinations": requirements["destinations"],
                    "dates": requirements.get("dates", {}),
                    "travelers": requirements.get("travelers", 1)
                }
            })
        
        # 行程规划任务
        subtasks.append({
            "task_type": "create_itinerary",
            "agent_role": "itinerary_planner",
            "priority": 6,
            "params": {
                "destinations": requirements.get("destinations", []),
                "duration": self._calculate_trip_duration(requirements.get("dates", {})),
                "preferences": requirements.get("preferences", [])
            }
        })
        
        # 预算分析任务
        if requirements.get("budget"):
            subtasks.append({
                "task_type": "analyze_budget",
                "agent_role": "budget_analyst",
                "priority": 5,
                "params": {
                    "budget": requirements["budget"],
                    "duration": self._calculate_trip_duration(requirements.get("dates", {})),
                    "destinations": requirements.get("destinations", [])
                }
            })
        
        return subtasks
    
    async def _delegate_task_to_specialist(self, subtask: Dict[str, Any], plan_id: str):
        """委托任务给专业智能体"""
        # 这里应该发送消息给相应的专业智能体
        # 暂时记录任务分配
        logger.info(f"任务 {subtask['task_type']} 已分配给 {subtask['agent_role']}")
        
        # 发送协作消息
        await self.send_collaboration_message(
            receiver_id=subtask['agent_role'],
            content={
                "task_type": subtask['task_type'],
                "plan_id": plan_id,
                "params": subtask['params'],
                "priority": subtask['priority']
            }
        )
    
    def _analyze_destinations(self, content: str) -> Dict[str, Any]:
        """分析目的地"""
        # 简单的目的地提取
        cities = re.findall(r'[北京|上海|广州|深圳|成都|西安|杭州|南京|苏州|青岛]+', content)
        return {
            "destinations": list(set(cities)),
            "destination_count": len(set(cities)),
            "domestic": len(cities) > 0
        }
    
    def _analyze_budget(self, content: str) -> Dict[str, Any]:
        """分析预算"""
        budget_numbers = re.findall(r'(\d+)(?:元|万)', content)
        if budget_numbers:
            budget = int(budget_numbers[0])
            if '万' in content:
                budget *= 10000
        else:
            budget = 0
        
        return {
            "budget": budget,
            "budget_level": "高" if budget > 10000 else "中" if budget > 5000 else "低",
            "budget_adequacy": "充足" if budget > 8000 else "适中" if budget > 3000 else "紧张"
        }
    
    def _analyze_travel_time(self, content: str) -> Dict[str, Any]:
        """分析旅行时间"""
        days = re.findall(r'(\d+)天', content)
        duration = int(days[0]) if days else 3
        
        return {
            "duration_days": duration,
            "duration_category": "长期" if duration > 7 else "中期" if duration > 3 else "短期",
            "season_recommendation": "春秋季节最佳"
        }
    
    def _analyze_preferences(self, content: str) -> Dict[str, Any]:
        """分析偏好"""
        preference_keywords = {
            "美食": ["美食", "吃", "餐厅", "小吃"],
            "历史": ["历史", "古迹", "文化", "博物馆"],
            "自然": ["自然", "风景", "山", "海"],
            "购物": ["购物", "商场", "特产"],
            "休闲": ["休闲", "放松", "度假"]
        }
        
        detected_preferences = []
        for pref, keywords in preference_keywords.items():
            if any(keyword in content for keyword in keywords):
                detected_preferences.append(pref)
        
        return {
            "preferences": detected_preferences,
            "travel_style": "多元化" if len(detected_preferences) > 2 else "专注型"
        }
    
    def _calculate_complexity(self, context: Dict[str, Any]) -> float:
        """计算复杂度"""
        factors = [
            len(context.get("destinations", [])) / 5,  # 目的地数量
            context.get("travelers", 1) / 10,          # 旅行者数量
            len(context.get("preferences", [])) / 5,   # 偏好数量
            len(context.get("special_requirements", [])) / 3  # 特殊要求
        ]
        
        return min(sum(factors) / len(factors), 1.0)
    
    def _calculate_trip_duration(self, dates: Dict[str, Any]) -> int:
        """计算行程天数"""
        if "start" in dates and "end" in dates:
            try:
                start = datetime.fromisoformat(dates["start"])
                end = datetime.fromisoformat(dates["end"])
                return (end - start).days
            except:
                return 3
        return dates.get("duration", 3)


class FlightAgent(BaseAgent):
    """航班搜索专家"""
    
    def __init__(self, agent_id: str, message_bus):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.FLIGHT_EXPERT,
            message_bus=message_bus,
            tools=[search_flights, get_weather_forecast],
            memory_window=15
        )
        
        self.flight_cache = {}
        self.search_history = []
    
    def get_capabilities(self) -> List[TaskType]:
        return [TaskType.SEARCH, TaskType.RECOMMEND]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理航班相关任务"""
        if task.task_type == TaskType.SEARCH:
            return await self._search_flights(task)
        elif task.task_type == TaskType.RECOMMEND:
            return await self._recommend_flights(task)
        else:
            return {"error": f"不支持的任务类型: {task.task_type}"}
    
    async def _search_flights(self, task: AgentTask) -> Dict[str, Any]:
        """搜索航班"""
        context = task.context
        
        origin = context.get("origin", "北京")
        destination = context.get("destination", "上海")
        departure_date = context.get("departure_date", datetime.now().strftime("%Y-%m-%d"))
        return_date = context.get("return_date")
        
        # 调用航班搜索工具
        flight_results = search_flights(origin, destination, departure_date, return_date)
        
        # 缓存搜索结果
        cache_key = f"{origin}_{destination}_{departure_date}"
        self.flight_cache[cache_key] = flight_results
        self.search_history.append({
            "search_time": datetime.now().isoformat(),
            "origin": origin,
            "destination": destination,
            "results_count": len(flight_results.get("flights", []))
        })
        
        # 分析和排序航班
        analyzed_flights = await self._analyze_flights(flight_results["flights"])
        
        return {
            "status": "success",
            "search_params": {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date
            },
            "flights": analyzed_flights,
            "recommendations": await self._generate_flight_recommendations(analyzed_flights),
            "search_time": flight_results["search_time"]
        }
    
    async def _recommend_flights(self, task: AgentTask) -> Dict[str, Any]:
        """推荐航班"""
        context = task.context
        preferences = context.get("preferences", {})
        budget = context.get("budget", 0)
        
        # 从缓存中获取航班数据
        flights = []
        for cached_flights in self.flight_cache.values():
            flights.extend(cached_flights.get("flights", []))
        
        if not flights:
            return {"error": "没有可用的航班数据"}
        
        # 根据偏好筛选和排序
        filtered_flights = await self._filter_flights_by_preferences(flights, preferences, budget)
        
        return {
            "status": "success",
            "recommended_flights": filtered_flights[:5],  # 返回前5个推荐
            "criteria": preferences,
            "total_options": len(flights)
        }
    
    async def _analyze_flights(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析航班数据"""
        analyzed = []
        
        for flight in flights:
            analysis = flight.copy()
            
            # 价格评级
            price = flight["price"]
            if price < 1000:
                analysis["price_rating"] = "经济"
            elif price < 2000:
                analysis["price_rating"] = "适中"
            else:
                analysis["price_rating"] = "昂贵"
            
            # 时间评级
            departure_hour = int(flight["departure_time"].split()[1].split(":")[0])
            if 6 <= departure_hour <= 9:
                analysis["time_rating"] = "早班"
            elif 10 <= departure_hour <= 14:
                analysis["time_rating"] = "上午"
            elif 15 <= departure_hour <= 18:
                analysis["time_rating"] = "下午"
            else:
                analysis["time_rating"] = "晚班"
            
            # 便利性评级
            if flight["stops"] == 0:
                analysis["convenience"] = "直飞"
            elif flight["stops"] == 1:
                analysis["convenience"] = "一次中转"
            else:
                analysis["convenience"] = "多次中转"
            
            # 综合评分
            analysis["overall_score"] = await self._calculate_flight_score(flight)
            
            analyzed.append(analysis)
        
        # 按综合评分排序
        analyzed.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return analyzed
    
    async def _calculate_flight_score(self, flight: Dict[str, Any]) -> float:
        """计算航班综合评分"""
        # 价格分数 (分数越高越好，价格越低越好)
        price = flight["price"]
        price_score = max(0, (3000 - price) / 3000)
        
        # 时间分数
        departure_hour = int(flight["departure_time"].split()[1].split(":")[0])
        if 8 <= departure_hour <= 10 or 14 <= departure_hour <= 16:
            time_score = 1.0  # 最佳时间
        elif 6 <= departure_hour <= 8 or 10 <= departure_hour <= 14 or 16 <= departure_hour <= 18:
            time_score = 0.8  # 较好时间
        else:
            time_score = 0.5  # 一般时间
        
        # 中转分数
        stops = flight["stops"]
        stop_score = max(0, (2 - stops) / 2)
        
        # 综合评分
        overall_score = (price_score * 0.4 + time_score * 0.3 + stop_score * 0.3)
        
        return round(overall_score, 2)
    
    async def _filter_flights_by_preferences(self, 
                                           flights: List[Dict[str, Any]], 
                                           preferences: Dict[str, Any],
                                           budget: int) -> List[Dict[str, Any]]:
        """根据偏好筛选航班"""
        filtered = flights.copy()
        
        # 预算筛选
        if budget > 0:
            filtered = [f for f in filtered if f["price"] <= budget]
        
        # 时间偏好筛选
        time_pref = preferences.get("time_preference")
        if time_pref:
            if time_pref == "morning":
                filtered = [f for f in filtered if 6 <= int(f["departure_time"].split()[1].split(":")[0]) <= 12]
            elif time_pref == "afternoon":
                filtered = [f for f in filtered if 12 <= int(f["departure_time"].split()[1].split(":")[0]) <= 18]
            elif time_pref == "evening":
                filtered = [f for f in filtered if 18 <= int(f["departure_time"].split()[1].split(":")[0]) <= 23]
        
        # 直飞偏好
        if preferences.get("direct_flight_only"):
            filtered = [f for f in filtered if f["stops"] == 0]
        
        # 按偏好排序
        sort_by = preferences.get("sort_by", "price")
        if sort_by == "price":
            filtered.sort(key=lambda x: x["price"])
        elif sort_by == "duration":
            filtered.sort(key=lambda x: x["duration"])
        elif sort_by == "time":
            filtered.sort(key=lambda x: x["departure_time"])
        
        return filtered
    
    async def _generate_flight_recommendations(self, flights: List[Dict[str, Any]]) -> List[str]:
        """生成航班推荐建议"""
        if not flights:
            return ["暂无航班数据"]
        
        recommendations = []
        
        # 最佳性价比推荐
        best_value = min(flights, key=lambda x: x["price"] / max(x["overall_score"], 0.1))
        recommendations.append(f"性价比最佳：{best_value['airline']} {best_value['flight_id']}，价格{best_value['price']}元")
        
        # 最快航班推荐
        fastest = min(flights, key=lambda x: x["stops"])
        if fastest["stops"] == 0:
            recommendations.append(f"直飞推荐：{fastest['airline']} {fastest['flight_id']}，无需中转")
        
        # 价格建议
        avg_price = sum(f["price"] for f in flights) / len(flights)
        recommendations.append(f"平均价格：{avg_price:.0f}元，建议预算{avg_price * 1.1:.0f}元")
        
        return recommendations


class HotelAgent(BaseAgent):
    """酒店推荐专家"""
    
    def __init__(self, agent_id: str, message_bus):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.HOTEL_EXPERT,
            message_bus=message_bus,
            tools=[search_hotels],
            memory_window=15
        )
        
        self.hotel_cache = {}
        self.recommendation_history = []
    
    def get_capabilities(self) -> List[TaskType]:
        return [TaskType.SEARCH, TaskType.RECOMMEND]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理酒店相关任务"""
        if task.task_type == TaskType.SEARCH:
            return await self._search_hotels(task)
        elif task.task_type == TaskType.RECOMMEND:
            return await self._recommend_hotels(task)
        else:
            return {"error": f"不支持的任务类型: {task.task_type}"}
    
    async def _search_hotels(self, task: AgentTask) -> Dict[str, Any]:
        """搜索酒店"""
        context = task.context
        
        city = context.get("city", "北京")
        checkin_date = context.get("checkin_date", datetime.now().strftime("%Y-%m-%d"))
        checkout_date = context.get("checkout_date", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))
        guests = context.get("guests", 2)
        
        # 调用酒店搜索工具
        hotel_results = search_hotels(city, checkin_date, checkout_date, guests)
        
        # 缓存搜索结果
        cache_key = f"{city}_{checkin_date}_{checkout_date}"
        self.hotel_cache[cache_key] = hotel_results
        
        # 分析和评级酒店
        analyzed_hotels = await self._analyze_hotels(hotel_results["hotels"])
        
        return {
            "status": "success",
            "search_params": {
                "city": city,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "guests": guests
            },
            "hotels": analyzed_hotels,
            "recommendations": await self._generate_hotel_recommendations(analyzed_hotels),
            "search_time": hotel_results["search_time"]
        }
    
    async def _recommend_hotels(self, task: AgentTask) -> Dict[str, Any]:
        """推荐酒店"""
        context = task.context
        preferences = context.get("preferences", {})
        budget_per_night = context.get("budget_per_night", 0)
        
        # 从缓存中获取酒店数据
        hotels = []
        for cached_hotels in self.hotel_cache.values():
            hotels.extend(cached_hotels.get("hotels", []))
        
        if not hotels:
            return {"error": "没有可用的酒店数据"}
        
        # 根据偏好筛选和排序
        filtered_hotels = await self._filter_hotels_by_preferences(hotels, preferences, budget_per_night)
        
        # 记录推荐历史
        self.recommendation_history.append({
            "timestamp": datetime.now().isoformat(),
            "preferences": preferences,
            "recommended_count": len(filtered_hotels)
        })
        
        return {
            "status": "success",
            "recommended_hotels": filtered_hotels[:5],
            "criteria": preferences,
            "total_options": len(hotels)
        }
    
    async def _analyze_hotels(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析酒店数据"""
        analyzed = []
        
        for hotel in hotels:
            analysis = hotel.copy()
            
            # 价格评级
            price = hotel["price_per_night"]
            if price < 300:
                analysis["price_category"] = "经济型"
            elif price < 800:
                analysis["price_category"] = "中档"
            elif price < 1500:
                analysis["price_category"] = "高档"
            else:
                analysis["price_category"] = "豪华"
            
            # 位置评级
            distance = float(hotel["distance_to_center"].replace("公里", ""))
            if distance <= 3:
                analysis["location_rating"] = "市中心"
            elif distance <= 8:
                analysis["location_rating"] = "近市区"
            else:
                analysis["location_rating"] = "郊区"
            
            # 设施评分
            amenity_count = len(hotel["amenities"])
            analysis["amenity_score"] = min(amenity_count / 6.0, 1.0)
            
            # 综合评分
            analysis["overall_score"] = await self._calculate_hotel_score(hotel)
            
            analyzed.append(analysis)
        
        # 按综合评分排序
        analyzed.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return analyzed
    
    async def _calculate_hotel_score(self, hotel: Dict[str, Any]) -> float:
        """计算酒店综合评分"""
        # 评分分数 (0-1)
        rating_score = hotel["rating"] / 5.0
        
        # 用户评分分数 (0-1)
        guest_rating_score = (hotel["guest_rating"] - 5) / 5.0  # 5-10转换为0-1
        
        # 位置分数
        distance = float(hotel["distance_to_center"].replace("公里", ""))
        location_score = max(0, (20 - distance) / 20)
        
        # 设施分数
        amenity_score = len(hotel["amenities"]) / 6.0
        
        # 价格分数 (价格越低越好，但要考虑性价比)
        price = hotel["price_per_night"]
        price_score = max(0, (2000 - price) / 2000)
        
        # 综合评分
        overall_score = (
            rating_score * 0.25 +
            guest_rating_score * 0.25 +
            location_score * 0.25 +
            amenity_score * 0.15 +
            price_score * 0.10
        )
        
        return round(overall_score, 2)
    
    async def _filter_hotels_by_preferences(self, 
                                          hotels: List[Dict[str, Any]], 
                                          preferences: Dict[str, Any],
                                          budget_per_night: int) -> List[Dict[str, Any]]:
        """根据偏好筛选酒店"""
        filtered = hotels.copy()
        
        # 预算筛选
        if budget_per_night > 0:
            filtered = [h for h in filtered if h["price_per_night"] <= budget_per_night]
        
        # 星级偏好
        min_rating = preferences.get("min_rating", 0)
        if min_rating > 0:
            filtered = [h for h in filtered if h["rating"] >= min_rating]
        
        # 位置偏好
        location_pref = preferences.get("location_preference")
        if location_pref == "city_center":
            filtered = [h for h in filtered if float(h["distance_to_center"].replace("公里", "")) <= 5]
        elif location_pref == "airport":
            filtered = [h for h in filtered if "机场" in h["location"]]
        
        # 设施要求
        required_amenities = preferences.get("required_amenities", [])
        if required_amenities:
            filtered = [h for h in filtered 
                       if all(amenity in h["amenities"] for amenity in required_amenities)]
        
        # 排序
        sort_by = preferences.get("sort_by", "overall_score")
        if sort_by == "price":
            filtered.sort(key=lambda x: x["price_per_night"])
        elif sort_by == "rating":
            filtered.sort(key=lambda x: x["guest_rating"], reverse=True)
        elif sort_by == "location":
            filtered.sort(key=lambda x: float(x["distance_to_center"].replace("公里", "")))
        else:
            filtered.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        return filtered
    
    async def _generate_hotel_recommendations(self, hotels: List[Dict[str, Any]]) -> List[str]:
        """生成酒店推荐建议"""
        if not hotels:
            return ["暂无酒店数据"]
        
        recommendations = []
        
        # 最佳性价比推荐
        best_value = max(hotels, key=lambda x: x.get("overall_score", 0))
        recommendations.append(f"综合推荐：{best_value['name']}，{best_value['rating']}星，{best_value['price_per_night']}元/晚")
        
        # 最便宜推荐
        cheapest = min(hotels, key=lambda x: x["price_per_night"])
        recommendations.append(f"经济选择：{cheapest['name']}，{cheapest['price_per_night']}元/晚")
        
        # 最高评分推荐
        highest_rated = max(hotels, key=lambda x: x["guest_rating"])
        recommendations.append(f"高评分推荐：{highest_rated['name']}，用户评分{highest_rated['guest_rating']}分")
        
        # 位置建议
        avg_distance = sum(float(h["distance_to_center"].replace("公里", "")) for h in hotels) / len(hotels)
        recommendations.append(f"平均距离市中心：{avg_distance:.1f}公里")
        
        return recommendations


class ItineraryAgent(BaseAgent):
    """行程规划师"""
    
    def __init__(self, agent_id: str, message_bus):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ITINERARY_PLANNER,
            message_bus=message_bus,
            tools=[create_itinerary, get_local_attractions, get_weather_forecast],
            memory_window=20
        )
        
        self.itinerary_templates = {}
        self.planning_history = []
    
    def get_capabilities(self) -> List[TaskType]:
        return [TaskType.PLAN, TaskType.RECOMMEND]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理行程规划任务"""
        if task.task_type == TaskType.PLAN:
            return await self._create_itinerary_plan(task)
        elif task.task_type == TaskType.RECOMMEND:
            return await self._recommend_activities(task)
        else:
            return {"error": f"不支持的任务类型: {task.task_type}"}
    
    async def _create_itinerary_plan(self, task: AgentTask) -> Dict[str, Any]:
        """创建行程规划"""
        context = task.context
        
        destinations = context.get("destinations", [])
        days = context.get("days", 3)
        preferences = context.get("preferences", ["观光", "美食"])
        budget_per_day = context.get("budget_per_day", 500)
        
        # 调用行程创建工具
        itinerary_result = create_itinerary(destinations, days, preferences)
        
        # 获取目的地景点信息
        attractions_info = {}
        for destination in destinations:
            attractions = get_local_attractions(destination)
            attractions_info[destination] = attractions
        
        # 优化行程安排
        optimized_itinerary = await self._optimize_itinerary(
            itinerary_result["itinerary"], 
            attractions_info,
            budget_per_day
        )
        
        # 记录规划历史
        self.planning_history.append({
            "timestamp": datetime.now().isoformat(),
            "destinations": destinations,
            "days": days,
            "preferences": preferences
        })
        
        return {
            "status": "success",
            "itinerary": optimized_itinerary,
            "attractions_info": attractions_info,
            "planning_summary": {
                "total_days": days,
                "destinations": destinations,
                "activities_per_day": len(optimized_itinerary[0]["activities"]) if optimized_itinerary else 0,
                "estimated_total_cost": budget_per_day * days
            },
            "recommendations": await self._generate_itinerary_recommendations(optimized_itinerary)
        }
    
    async def _recommend_activities(self, task: AgentTask) -> Dict[str, Any]:
        """推荐活动"""
        context = task.context
        
        destination = context.get("destination", "北京")
        activity_type = context.get("activity_type", "all")
        time_available = context.get("time_available", "半天")
        
        # 获取景点信息
        attractions = get_local_attractions(destination, activity_type)
        
        # 根据时间筛选活动
        filtered_activities = await self._filter_activities_by_time(
            attractions["attractions"], 
            time_available
        )
        
        return {
            "status": "success",
            "destination": destination,
            "recommended_activities": filtered_activities,
            "time_available": time_available,
            "activity_type": activity_type
        }
    
    async def _optimize_itinerary(self, 
                                itinerary: List[Dict[str, Any]], 
                                attractions_info: Dict[str, Any],
                                budget_per_day: int) -> List[Dict[str, Any]]:
        """优化行程安排"""
        optimized = []
        
        for day_plan in itinerary:
            optimized_day = day_plan.copy()
            
            # 获取当天目的地的景点信息
            destination = day_plan["destination"]
            if destination in attractions_info:
                local_attractions = attractions_info[destination]["attractions"]
                
                # 为每个活动匹配具体景点
                enhanced_activities = []
                for activity in day_plan["activities"]:
                    enhanced_activity = activity.copy()
                    
                    # 匹配相关景点
                    matching_attractions = [
                        attr for attr in local_attractions
                        if activity["category"] in attr["category"] or 
                           any(keyword in attr["name"] for keyword in activity["activity"].split())
                    ]
                    
                    if matching_attractions:
                        selected_attraction = random.choice(matching_attractions)
                        enhanced_activity.update({
                            "venue": selected_attraction["name"],
                            "venue_rating": selected_attraction["rating"],
                            "venue_price": selected_attraction["price"],
                            "opening_hours": selected_attraction["opening_hours"],
                            "description": selected_attraction["description"]
                        })
                    
                    enhanced_activities.append(enhanced_activity)
                
                optimized_day["activities"] = enhanced_activities
            
            # 预算优化
            daily_cost = sum(
                int(re.findall(r'\d+', act.get("estimated_cost", "0元"))[0]) 
                for act in optimized_day["activities"]
                if re.findall(r'\d+', act.get("estimated_cost", "0元"))
            )
            
            optimized_day["daily_cost"] = daily_cost
            optimized_day["budget_status"] = "超预算" if daily_cost > budget_per_day else "预算内"
            
            optimized.append(optimized_day)
        
        return optimized
    
    async def _filter_activities_by_time(self, 
                                       attractions: List[Dict[str, Any]], 
                                       time_available: str) -> List[Dict[str, Any]]:
        """根据时间筛选活动"""
        if time_available == "半天":
            max_activities = 2
        elif time_available == "一天":
            max_activities = 4
        else:  # 多天
            max_activities = len(attractions)
        
        # 按评分排序并选择前N个
        sorted_attractions = sorted(attractions, key=lambda x: x["rating"], reverse=True)
        return sorted_attractions[:max_activities]
    
    async def _generate_itinerary_recommendations(self, itinerary: List[Dict[str, Any]]) -> List[str]:
        """生成行程推荐建议"""
        if not itinerary:
            return ["暂无行程数据"]
        
        recommendations = []
        
        # 总体建议
        total_days = len(itinerary)
        recommendations.append(f"建议的{total_days}天行程已生成，涵盖多个目的地")
        
        # 预算建议
        total_cost = sum(day.get("daily_cost", 0) for day in itinerary)
        recommendations.append(f"预估总费用：{total_cost}元，平均每天{total_cost/total_days:.0f}元")
        
        # 活动建议
        activity_types = set()
        for day in itinerary:
            for activity in day.get("activities", []):
                activity_types.add(activity.get("category", "其他"))
        
        recommendations.append(f"活动类型包括：{', '.join(activity_types)}")
        
        # 时间建议
        recommendations.append("建议合理安排休息时间，避免行程过于紧密")
        
        return recommendations


class BudgetAgent(BaseAgent):
    """预算分析师"""
    
    def __init__(self, agent_id: str, message_bus):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.BUDGET_ANALYST,
            message_bus=message_bus,
            tools=[calculate_travel_budget],
            memory_window=15
        )
        
        self.budget_analysis_history = []
        self.cost_optimization_tips = [
            "提前预订通常可获得更好的价格",
            "选择淡季出行可节省20-40%费用",
            "考虑中转航班而非直飞可节省费用",
            "选择位置稍远但交通便利的酒店",
            "利用当地公共交通工具"
        ]
    
    def get_capabilities(self) -> List[TaskType]:
        return [TaskType.ANALYZE, TaskType.RECOMMEND]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理预算分析任务"""
        if task.task_type == TaskType.ANALYZE:
            return await self._analyze_budget(task)
        elif task.task_type == TaskType.RECOMMEND:
            return await self._recommend_budget_optimization(task)
        else:
            return {"error": f"不支持的任务类型: {task.task_type}"}
    
    async def _analyze_budget(self, task: AgentTask) -> Dict[str, Any]:
        """分析预算"""
        context = task.context
        
        total_budget = context.get("budget", 0)
        days = context.get("days", 3)
        destinations = context.get("destinations", [])
        travelers = context.get("travelers", 1)
        
        # 获取成本估算
        flights = context.get("flights", [{"price": 1000}])  # 默认航班价格
        hotels = context.get("hotels", [{"price_per_night": 300}])  # 默认酒店价格
        
        # 计算旅行预算
        budget_calculation = calculate_travel_budget(flights, hotels, days)
        
        # 预算分析
        estimated_cost = budget_calculation["total"] * travelers
        budget_difference = total_budget - estimated_cost
        
        analysis = {
            "total_budget": total_budget,
            "estimated_cost": estimated_cost,
            "budget_difference": budget_difference,
            "budget_status": "充足" if budget_difference > 0 else "不足",
            "cost_breakdown": budget_calculation["breakdown"],
            "daily_average": budget_calculation["cost_per_day"],
            "travelers": travelers,
            "destinations": destinations,
            "analysis_date": datetime.now().isoformat()
        }
        
        # 生成预算建议
        budget_recommendations = await self._generate_budget_advice(analysis)
        
        # 记录分析历史
        self.budget_analysis_history.append(analysis)
        
        return {
            "status": "success",
            "budget_analysis": analysis,
            "recommendations": budget_recommendations,
            "optimization_potential": await self._calculate_optimization_potential(analysis)
        }
    
    async def _recommend_budget_optimization(self, task: AgentTask) -> Dict[str, Any]:
        """推荐预算优化方案"""
        context = task.context
        current_budget = context.get("current_budget", {})
        target_reduction = context.get("target_reduction_percent", 20)
        
        optimization_strategies = []
        
        # 交通费优化
        flight_cost = current_budget.get("flights", 0)
        if flight_cost > 0:
            potential_savings = flight_cost * 0.3
            optimization_strategies.append({
                "category": "交通",
                "strategy": "选择中转航班或提前预订",
                "potential_savings": potential_savings,
                "implementation": "考虑1-2次中转的航班，通常比直飞便宜20-30%"
            })
        
        # 住宿费优化
        hotel_cost = current_budget.get("hotels", 0)
        if hotel_cost > 0:
            potential_savings = hotel_cost * 0.25
            optimization_strategies.append({
                "category": "住宿",
                "strategy": "选择经济型酒店或民宿",
                "potential_savings": potential_savings,
                "implementation": "考虑位置稍远但交通便利的住宿，或选择民宿"
            })
        
        # 餐饮费优化
        meal_cost = current_budget.get("meals", 0)
        if meal_cost > 0:
            potential_savings = meal_cost * 0.2
            optimization_strategies.append({
                "category": "餐饮",
                "strategy": "体验当地特色小吃",
                "potential_savings": potential_savings,
                "implementation": "选择当地小吃和中档餐厅，避免高档餐厅"
            })
        
        # 活动费优化
        activity_cost = current_budget.get("activities", 0)
        if activity_cost > 0:
            potential_savings = activity_cost * 0.15
            optimization_strategies.append({
                "category": "活动",
                "strategy": "选择免费或低成本景点",
                "potential_savings": potential_savings,
                "implementation": "多选择公园、海滩等免费景点，购买景点联票"
            })
        
        total_potential_savings = sum(s["potential_savings"] for s in optimization_strategies)
        
        return {
            "status": "success",
            "optimization_strategies": optimization_strategies,
            "total_potential_savings": total_potential_savings,
            "target_reduction": target_reduction,
            "achievable": total_potential_savings >= (sum(current_budget.values()) * target_reduction / 100),
            "additional_tips": random.sample(self.cost_optimization_tips, 3)
        }
    
    async def _generate_budget_advice(self, analysis: Dict[str, Any]) -> List[str]:
        """生成预算建议"""
        advice = []
        
        budget_status = analysis["budget_status"]
        budget_difference = analysis["budget_difference"]
        
        if budget_status == "充足":
            if budget_difference > 1000:
                advice.append("预算充足，可以考虑升级住宿或增加特色体验")
            else:
                advice.append("预算基本充足，建议预留一些应急资金")
        else:
            shortage = abs(budget_difference)
            advice.append(f"预算不足{shortage}元，建议考虑优化方案")
            
            if shortage > 2000:
                advice.append("建议重新评估行程或增加预算")
            else:
                advice.append("可通过选择经济型住宿和交通方式来控制成本")
        
        # 成本结构建议
        breakdown = analysis["cost_breakdown"]
        highest_cost = max(breakdown.items(), key=lambda x: x[1])
        advice.append(f"最大支出项为{highest_cost[0]}（{highest_cost[1]}元），可重点优化")
        
        return advice
    
    async def _calculate_optimization_potential(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算优化潜力"""
        breakdown = analysis["cost_breakdown"]
        
        optimization_potential = {}
        
        # 各项目的优化潜力（百分比）
        optimization_rates = {
            "flights": 0.25,      # 航班25%优化潜力
            "hotels": 0.30,       # 酒店30%优化潜力
            "meals": 0.20,        # 餐饮20%优化潜力
            "activities": 0.15,   # 活动15%优化潜力
            "local_transport": 0.10  # 当地交通10%优化潜力
        }
        
        for category, cost in breakdown.items():
            if category in optimization_rates:
                potential_saving = cost * optimization_rates[category]
                optimization_potential[category] = {
                    "current_cost": cost,
                    "potential_saving": potential_saving,
                    "optimized_cost": cost - potential_saving,
                    "saving_percentage": optimization_rates[category] * 100
                }
        
        total_current = sum(breakdown.values())
        total_potential_saving = sum(p["potential_saving"] for p in optimization_potential.values())
        
        return {
            "by_category": optimization_potential,
            "total_current_cost": total_current,
            "total_potential_saving": total_potential_saving,
            "total_optimized_cost": total_current - total_potential_saving,
            "overall_saving_percentage": (total_potential_saving / total_current * 100) if total_current > 0 else 0
        }


class LocalGuideAgent(BaseAgent):
    """当地向导"""
    
    def __init__(self, agent_id: str, message_bus):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.LOCAL_GUIDE,
            message_bus=message_bus,
            tools=[get_local_attractions, get_weather_forecast],
            memory_window=15
        )
        
        self.local_knowledge = {
            "文化习俗": {
                "北京": ["尊重传统文化", "参观寺庙时保持安静", "品尝正宗京菜"],
                "上海": ["体验国际化氛围", "外滩夜景最佳观赏时间", "石库门建筑特色"],
                "成都": ["慢生活节奏", "茶馆文化体验", "川菜辣度选择"]
            },
            "交通贴士": {
                "北京": ["地铁覆盖广泛", "避开早晚高峰", "故宫需提前预约"],
                "上海": ["磁悬浮快线体验", "外滩交通管制", "地铁换乘便利"],
                "成都": ["打车相对便宜", "景区直达公交", "共享单车普及"]
            },
            "美食推荐": {
                "北京": ["烤鸭", "炸酱面", "豆汁配咸菜", "糖葫芦"],
                "上海": ["小笼包", "生煎包", "本帮菜", "上海菜"],
                "成都": ["火锅", "麻婆豆腐", "担担面", "龙抄手"]
            }
        }
        
        self.seasonal_advice = {
            "春季": "气候宜人，适合户外活动，注意花粉过敏",
            "夏季": "炎热多雨，准备防晒和雨具，避开正午时段",
            "秋季": "天高气爽，最佳旅行季节，早晚温差大",
            "冬季": "寒冷干燥，注意保暖，室内外温差大"
        }
    
    def get_capabilities(self) -> List[TaskType]:
        return [TaskType.RECOMMEND, TaskType.ANALYZE]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """处理当地向导任务"""
        if task.task_type == TaskType.RECOMMEND:
            return await self._provide_local_recommendations(task)
        elif task.task_type == TaskType.ANALYZE:
            return await self._analyze_local_conditions(task)
        else:
            return {"error": f"不支持的任务类型: {task.task_type}"}
    
    async def _provide_local_recommendations(self, task: AgentTask) -> Dict[str, Any]:
        """提供当地推荐"""
        context = task.context
        
        destination = context.get("destination", "北京")
        interest_type = context.get("interest_type", "all")
        travel_date = context.get("travel_date", datetime.now().strftime("%Y-%m-%d"))
        
        # 获取当地景点
        attractions = get_local_attractions(destination, interest_type)
        
        # 获取天气信息
        weather = get_weather_forecast(destination, travel_date)
        
        # 生成本地化建议
        local_tips = await self._generate_local_tips(destination, weather, travel_date)
        
        return {
            "status": "success",
            "destination": destination,
            "travel_date": travel_date,
            "weather_info": weather,
            "local_attractions": attractions["attractions"],
            "local_tips": local_tips,
            "cultural_advice": self._get_cultural_advice(destination),
            "transportation_tips": self._get_transportation_tips(destination),
            "food_recommendations": self._get_food_recommendations(destination)
        }
    
    async def _analyze_local_conditions(self, task: AgentTask) -> Dict[str, Any]:
        """分析当地条件"""
        context = task.context
        
        destination = context.get("destination", "北京")
        travel_dates = context.get("travel_dates", [])
        group_size = context.get("group_size", 2)
        
        analysis = {
            "destination": destination,
            "weather_analysis": {},
            "crowd_analysis": {},
            "cost_analysis": {},
            "accessibility_analysis": {},
            "recommendations": []
        }
        
        # 天气分析
        if travel_dates:
            weather_data = []
            for date in travel_dates:
                weather = get_weather_forecast(destination, date)
                weather_data.append(weather)
            
            analysis["weather_analysis"] = {
                "conditions": weather_data,
                "overall_forecast": await self._summarize_weather(weather_data),
                "packing_suggestions": await self._generate_packing_suggestions(weather_data)
            }
        
        # 人群分析
        analysis["crowd_analysis"] = await self._analyze_crowd_levels(destination, travel_dates)
        
        # 成本分析
        analysis["cost_analysis"] = await self._analyze_local_costs(destination, group_size)
        
        # 可达性分析
        analysis["accessibility_analysis"] = await self._analyze_accessibility(destination)
        
        # 综合建议
        analysis["recommendations"] = await self._generate_comprehensive_recommendations(analysis)
        
        return {
            "status": "success",
            "local_analysis": analysis
        }
    
    async def _generate_local_tips(self, destination: str, weather: Dict[str, Any], travel_date: str) -> Dict[str, Any]:
        """生成本地化贴士"""
        tips = {
            "weather_tips": [],
            "cultural_tips": [],
            "practical_tips": [],
            "safety_tips": []
        }
        
        # 天气相关贴士
        condition = weather.get("condition", "晴")
        if "雨" in condition:
            tips["weather_tips"].extend([
                "建议携带雨具",
                "选择室内景点作为备选",
                "注意路面湿滑"
            ])
        elif condition == "晴":
            tips["weather_tips"].extend([
                "注意防晒",
                "多补充水分",
                "适合户外活动"
            ])
        
        # 文化贴士
        if destination in self.local_knowledge["文化习俗"]:
            tips["cultural_tips"] = self.local_knowledge["文化习俗"][destination]
        
        # 实用贴士
        tips["practical_tips"] = [
            "建议提前下载当地地图应用",
            "了解当地支付方式",
            "保存重要联系方式",
            "购买当地电话卡或开通漫游"
        ]
        
        # 安全贴士
        tips["safety_tips"] = [
            "保管好个人证件和财物",
            "了解当地紧急联系方式",
            "避免深夜独自外出",
            "购买旅行保险"
        ]
        
        return tips
    
    def _get_cultural_advice(self, destination: str) -> List[str]:
        """获取文化建议"""
        return self.local_knowledge["文化习俗"].get(destination, [
            "尊重当地文化和习俗",
            "学习基本的当地礼仪",
            "了解宗教场所的参观规则"
        ])
    
    def _get_transportation_tips(self, destination: str) -> List[str]:
        """获取交通贴士"""
        return self.local_knowledge["交通贴士"].get(destination, [
            "了解当地主要交通方式",
            "下载当地交通应用",
            "了解交通卡使用方法"
        ])
    
    def _get_food_recommendations(self, destination: str) -> List[str]:
        """获取美食推荐"""
        return self.local_knowledge["美食推荐"].get(destination, [
            "品尝当地特色菜",
            "尝试街头小吃",
            "了解当地饮食习惯"
        ])
    
    async def _summarize_weather(self, weather_data: List[Dict[str, Any]]) -> str:
        """总结天气情况"""
        if not weather_data:
            return "天气信息不可用"
        
        conditions = [w.get("condition", "未知") for w in weather_data]
        temps = [w.get("temperature", {}) for w in weather_data]
        
        # 统计天气条件
        from collections import Counter
        condition_counts = Counter(conditions)
        most_common = condition_counts.most_common(1)[0][0]
        
        # 计算平均温度
        high_temps = [t.get("high", 0) for t in temps if t.get("high")]
        low_temps = [t.get("low", 0) for t in temps if t.get("low")]
        
        avg_high = sum(high_temps) / len(high_temps) if high_temps else 0
        avg_low = sum(low_temps) / len(low_temps) if low_temps else 0
        
        return f"主要天气：{most_common}，平均温度：{avg_low:.0f}-{avg_high:.0f}°C"
    
    async def _generate_packing_suggestions(self, weather_data: List[Dict[str, Any]]) -> List[str]:
        """生成打包建议"""
        suggestions = []
        
        if not weather_data:
            return ["根据季节准备适当衣物"]
        
        # 分析温度范围
        all_highs = []
        all_lows = []
        conditions = []
        
        for weather in weather_data:
            temp = weather.get("temperature", {})
            if temp.get("high"):
                all_highs.append(temp["high"])
            if temp.get("low"):
                all_lows.append(temp["low"])
            conditions.append(weather.get("condition", ""))
        
        if all_highs and all_lows:
            max_temp = max(all_highs)
            min_temp = min(all_lows)
            
            if max_temp > 30:
                suggestions.append("准备夏季轻薄衣物和防晒用品")
            elif max_temp > 20:
                suggestions.append("准备春秋季节衣物")
            else:
                suggestions.append("准备保暖衣物")
            
            if min_temp < 10:
                suggestions.append("准备外套或毛衣")
        
        # 根据天气条件
        if any("雨" in c for c in conditions):
            suggestions.append("携带雨伞或雨衣")
        
        if any("雪" in c for c in conditions):
            suggestions.append("准备防滑鞋和厚外套")
        
        return suggestions or ["根据天气预报准备适当衣物"]
    
    async def _analyze_crowd_levels(self, destination: str, travel_dates: List[str]) -> Dict[str, Any]:
        """分析人群水平"""
        # 简化的人群分析
        crowd_analysis = {
            "overall_level": "中等",
            "peak_times": ["周末", "节假日"],
            "recommendations": [
                "避开周末和节假日",
                "选择早上或傍晚时段",
                "提前预订热门景点门票"
            ]
        }
        
        # 根据日期判断
        if travel_dates:
            weekend_dates = 0
            for date_str in travel_dates:
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    if date_obj.weekday() >= 5:  # 周六周日
                        weekend_dates += 1
                except:
                    continue
            
            if weekend_dates > len(travel_dates) / 2:
                crowd_analysis["overall_level"] = "较高"
                crowd_analysis["recommendations"].insert(0, "旅行日期多为周末，预计人流较多")
        
        return crowd_analysis
    
    async def _analyze_local_costs(self, destination: str, group_size: int) -> Dict[str, Any]:
        """分析当地成本"""
        # 基础成本数据（示例）
        base_costs = {
            "北京": {"meal": 80, "transport": 50, "attraction": 100},
            "上海": {"meal": 90, "transport": 60, "attraction": 120},
            "成都": {"meal": 60, "transport": 40, "attraction": 80}
        }
        
        costs = base_costs.get(destination, {"meal": 70, "transport": 50, "attraction": 100})
        
        return {
            "daily_meal_cost": costs["meal"] * group_size,
            "daily_transport_cost": costs["transport"],
            "average_attraction_cost": costs["attraction"],
            "group_discounts": "3人以上可享受团体票优惠" if group_size >= 3 else "无团体优惠",
            "cost_saving_tips": [
                "使用当地公共交通",
                "选择当地特色小吃",
                "购买景点联票"
            ]
        }
    
    async def _analyze_accessibility(self, destination: str) -> Dict[str, Any]:
        """分析可达性"""
        return {
            "public_transport": "便利",
            "taxi_availability": "充足",
            "walking_friendly": "适中",
            "accessibility_features": [
                "主要景点有无障碍设施",
                "地铁站配备电梯",
                "多数酒店有无障碍房间"
            ],
            "special_needs_support": "建议提前联系相关场所确认具体设施"
        }
    
    async def _generate_comprehensive_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 基于天气的建议
        weather_analysis = analysis.get("weather_analysis", {})
        if weather_analysis:
            recommendations.append("根据天气预报合理安排室内外活动")
        
        # 基于人群的建议
        crowd_analysis = analysis.get("crowd_analysis", {})
        if crowd_analysis.get("overall_level") == "较高":
            recommendations.append("建议错峰出行，提前预订门票")
        
        # 基于成本的建议
        cost_analysis = analysis.get("cost_analysis", {})
        if cost_analysis:
            recommendations.append("利用当地交通卡和景点联票节省费用")
        
        # 通用建议
        recommendations.extend([
            "下载当地实用应用程序",
            "了解当地紧急联系方式",
            "保持与当地人的友好互动"
        ])
        
        return recommendations


# 智能体工厂函数
async def create_specialized_agents(message_bus) -> Dict[str, BaseAgent]:
    """创建所有专业智能体"""
    agents = {}
    
    # 创建协调智能体
    coordinator = CoordinatorAgent("coordinator_001", message_bus)
    agents[coordinator.agent_id] = coordinator
    
    # 创建航班专家
    flight_agent = FlightAgent("flight_expert_001", message_bus)
    agents[flight_agent.agent_id] = flight_agent
    
    # 创建酒店专家
    hotel_agent = HotelAgent("hotel_expert_001", message_bus)
    agents[hotel_agent.agent_id] = hotel_agent
    
    # 创建行程规划师
    itinerary_agent = ItineraryAgent("itinerary_planner_001", message_bus)
    agents[itinerary_agent.agent_id] = itinerary_agent
    
    # 创建预算分析师
    budget_agent = BudgetAgent("budget_analyst_001", message_bus)
    agents[budget_agent.agent_id] = budget_agent
    
    # 创建当地向导
    local_guide = LocalGuideAgent("local_guide_001", message_bus)
    agents[local_guide.agent_id] = local_guide
    
    logger.info(f"创建了 {len(agents)} 个专业智能体")
    return agents 