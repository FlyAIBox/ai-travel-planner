"""
MCP工具集成
实现旅行相关的MCP工具集，包括航班搜索、酒店查询、天气获取、路线规划等功能
"""

import asyncio
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import re

from mcp_server import MCPTool, MCPResource, MCPPrompt, get_mcp_server
from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TravelTools:
    """旅行工具集"""
    
    def __init__(self):
        self.session = None
        
    async def get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    # 航班搜索工具
    async def search_flights(self, 
                           origin: str, 
                           destination: str, 
                           departure_date: str,
                           return_date: Optional[str] = None,
                           passengers: int = 1,
                           class_type: str = "economy") -> Dict[str, Any]:
        """搜索航班"""
        try:
            # 模拟航班搜索API调用
            flights = []
            
            # 生成模拟航班数据
            airlines = ["国航", "东航", "南航", "海航", "春秋", "吉祥"]
            for i in range(5):
                flight = {
                    "flight_number": f"{airlines[i % len(airlines)]}{1000 + i}",
                    "airline": airlines[i % len(airlines)],
                    "origin": origin,
                    "destination": destination,
                    "departure_time": f"{departure_date} {8 + i*2}:00",
                    "arrival_time": f"{departure_date} {11 + i*2}:30",
                    "duration": "3小时30分钟",
                    "price": 800 + i * 100,
                    "class": class_type,
                    "seats_available": 50 - i * 5,
                    "aircraft": "波音737" if i % 2 == 0 else "空客A320"
                }
                flights.append(flight)
            
            result = {
                "search_criteria": {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "return_date": return_date,
                    "passengers": passengers,
                    "class": class_type
                },
                "flights": flights,
                "total_results": len(flights),
                "search_time": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"航班搜索失败: {e}")
            return {"error": f"航班搜索失败: {str(e)}"}

    # 酒店查询工具  
    async def search_hotels(self,
                          destination: str,
                          check_in: str,
                          check_out: str,
                          guests: int = 2,
                          rooms: int = 1,
                          price_range: Optional[str] = None) -> Dict[str, Any]:
        """搜索酒店"""
        try:
            # 模拟酒店搜索
            hotels = []
            
            hotel_chains = ["希尔顿", "万豪", "洲际", "凯悦", "香格里拉", "如家"]
            star_ratings = [3, 4, 5, 4, 5, 3]
            
            for i in range(6):
                hotel = {
                    "hotel_id": f"hotel_{i+1}",
                    "name": f"{destination}{hotel_chains[i]}酒店",
                    "star_rating": star_ratings[i],
                    "address": f"{destination}市中心区第{i+1}大街{100+i}号",
                    "distance_to_center": f"{0.5 + i*0.3:.1f}公里",
                    "price_per_night": 300 + i * 150,
                    "total_price": (300 + i * 150) * self._calculate_nights(check_in, check_out),
                    "amenities": ["免费WiFi", "健身房", "游泳池", "餐厅", "停车场"][:3+i%3],
                    "rating": 4.5 - i * 0.1,
                    "reviews_count": 1000 - i * 100,
                    "cancellation": "免费取消" if i % 2 == 0 else "有条件取消",
                    "breakfast_included": i % 3 == 0
                }
                hotels.append(hotel)
            
            result = {
                "search_criteria": {
                    "destination": destination,
                    "check_in": check_in,
                    "check_out": check_out,
                    "guests": guests,
                    "rooms": rooms,
                    "price_range": price_range
                },
                "hotels": hotels,
                "total_results": len(hotels),
                "search_time": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"酒店搜索失败: {e}")
            return {"error": f"酒店搜索失败: {str(e)}"}

    def _calculate_nights(self, check_in: str, check_out: str) -> int:
        """计算住宿夜数"""
        try:
            check_in_date = datetime.strptime(check_in, "%Y-%m-%d")
            check_out_date = datetime.strptime(check_out, "%Y-%m-%d") 
            return (check_out_date - check_in_date).days
        except:
            return 1

    # 天气查询工具
    async def get_weather(self, city: str, days: int = 7) -> Dict[str, Any]:
        """获取天气信息"""
        try:
            # 模拟天气API调用
            weather_conditions = ["晴", "多云", "阴", "小雨", "中雨", "雷阵雨", "雪"]
            
            forecasts = []
            base_temp = 20
            
            for i in range(days):
                date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
                forecast = {
                    "date": date,
                    "condition": weather_conditions[i % len(weather_conditions)],
                    "temperature": {
                        "high": base_temp + i + 5,
                        "low": base_temp + i - 5
                    },
                    "humidity": 60 + i * 5,
                    "wind_speed": 10 + i * 2,
                    "precipitation": 0 if i % 3 != 0 else 20 + i * 10,
                    "uv_index": 3 + i % 5,
                    "sunrise": "06:30",
                    "sunset": "18:30"
                }
                forecasts.append(forecast)
            
            result = {
                "city": city,
                "current_weather": forecasts[0] if forecasts else None,
                "forecast": forecasts,
                "forecast_days": days,
                "last_updated": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"天气查询失败: {e}")
            return {"error": f"天气查询失败: {str(e)}"}

    # 汇率查询工具
    async def get_exchange_rate(self, from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
        """获取汇率信息"""
        try:
            # 模拟汇率数据
            rates = {
                ("USD", "CNY"): 7.2,
                ("CNY", "USD"): 0.14,
                ("EUR", "CNY"): 7.8,
                ("CNY", "EUR"): 0.13,
                ("JPY", "CNY"): 0.05,
                ("CNY", "JPY"): 20.0,
                ("USD", "EUR"): 0.92,
                ("EUR", "USD"): 1.09
            }
            
            key = (from_currency.upper(), to_currency.upper())
            rate = rates.get(key, 1.0)
            
            result = {
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "exchange_rate": rate,
                "amount": amount,
                "converted_amount": amount * rate,
                "last_updated": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"汇率查询失败: {e}")
            return {"error": f"汇率查询失败: {str(e)}"}

    # 路线规划工具
    async def plan_route(self, 
                        origin: str, 
                        destination: str, 
                        waypoints: Optional[List[str]] = None,
                        mode: str = "driving") -> Dict[str, Any]:
        """规划路线"""
        try:
            waypoints = waypoints or []
            
            # 模拟路线规划
            all_points = [origin] + waypoints + [destination]
            segments = []
            total_distance = 0
            total_time = 0
            
            for i in range(len(all_points) - 1):
                start = all_points[i]
                end = all_points[i + 1]
                
                # 模拟距离和时间
                distance = 50 + i * 30  # 公里
                time = distance * 1.2   # 分钟
                
                segment = {
                    "start": start,
                    "end": end,
                    "distance_km": distance,
                    "duration_minutes": int(time),
                    "mode": mode,
                    "instructions": f"从{start}出发，沿主要道路行驶{distance}公里到达{end}"
                }
                segments.append(segment)
                
                total_distance += distance
                total_time += time
            
            result = {
                "origin": origin,
                "destination": destination,
                "waypoints": waypoints,
                "mode": mode,
                "total_distance_km": total_distance,
                "total_duration_minutes": int(total_time),
                "estimated_duration": f"{int(total_time // 60)}小时{int(total_time % 60)}分钟",
                "segments": segments,
                "calculated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"路线规划失败: {e}")
            return {"error": f"路线规划失败: {str(e)}"}

    # 景点推荐工具
    async def recommend_attractions(self, 
                                  city: str, 
                                  category: Optional[str] = None,
                                  limit: int = 10) -> Dict[str, Any]:
        """推荐景点"""
        try:
            # 模拟景点数据
            attraction_types = ["历史文化", "自然风光", "主题公园", "博物馆", "购物中心", "美食街"]
            
            attractions = []
            for i in range(min(limit, 10)):
                attraction = {
                    "name": f"{city}著名景点{i+1}",
                    "category": attraction_types[i % len(attraction_types)],
                    "rating": 4.8 - i * 0.1,
                    "reviews_count": 5000 - i * 300,
                    "description": f"这是{city}最受欢迎的{attraction_types[i % len(attraction_types)]}景点之一",
                    "address": f"{city}市{attraction_types[i % len(attraction_types)]}区第{i+1}街",
                    "opening_hours": "09:00-18:00",
                    "ticket_price": 50 + i * 20,
                    "recommended_duration": f"{2 + i}小时",
                    "best_time_to_visit": "春秋两季",
                    "nearby_attractions": [f"景点{j}" for j in range(i+2, i+4)],
                    "tags": ["必游", "拍照", "家庭友好"][:2+i%2]
                }
                attractions.append(attraction)
            
            result = {
                "city": city,
                "category": category,
                "attractions": attractions,
                "total_results": len(attractions),
                "search_time": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"景点推荐失败: {e}")
            return {"error": f"景点推荐失败: {str(e)}"}

    # 餐厅推荐工具
    async def recommend_restaurants(self, 
                                  city: str, 
                                  cuisine: Optional[str] = None,
                                  price_range: Optional[str] = None,
                                  limit: int = 10) -> Dict[str, Any]:
        """推荐餐厅"""
        try:
            # 模拟餐厅数据
            cuisines = ["中餐", "西餐", "日料", "韩料", "泰餐", "意大利菜", "法餐", "印度菜"]
            price_levels = ["经济", "中档", "高档"]
            
            restaurants = []
            for i in range(min(limit, 10)):
                restaurant = {
                    "name": f"{city}{cuisines[i % len(cuisines)]}餐厅{i+1}",
                    "cuisine": cuisines[i % len(cuisines)],
                    "rating": 4.7 - i * 0.05,
                    "reviews_count": 2000 - i * 150,
                    "price_range": price_levels[i % len(price_levels)],
                    "average_cost": 100 + i * 50,
                    "address": f"{city}市美食街第{i+1}号",
                    "phone": f"138{1000+i}{1000+i}",
                    "opening_hours": "11:00-22:00",
                    "specialties": [f"招牌菜{j}" for j in range(1, 4)],
                    "ambiance": ["温馨", "现代", "传统", "浪漫"][i % 4],
                    "booking_required": i % 3 == 0,
                    "delivery_available": i % 2 == 0
                }
                restaurants.append(restaurant)
            
            result = {
                "city": city,
                "cuisine": cuisine,
                "price_range": price_range,
                "restaurants": restaurants,
                "total_results": len(restaurants),
                "search_time": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"餐厅推荐失败: {e}")
            return {"error": f"餐厅推荐失败: {str(e)}"}

    # 预算计算工具
    async def calculate_budget(self, 
                             destination: str,
                             duration_days: int,
                             travelers: int = 1,
                             comfort_level: str = "standard") -> Dict[str, Any]:
        """计算旅行预算"""
        try:
            # 基础费用（每人每天）
            base_costs = {
                "budget": {"accommodation": 100, "food": 80, "transport": 50, "activities": 60},
                "standard": {"accommodation": 200, "food": 150, "transport": 100, "activities": 120},
                "luxury": {"accommodation": 500, "food": 300, "transport": 200, "activities": 250}
            }
            
            costs = base_costs.get(comfort_level, base_costs["standard"])
            
            # 计算各项费用
            accommodation = costs["accommodation"] * duration_days * travelers
            food = costs["food"] * duration_days * travelers
            local_transport = costs["transport"] * duration_days * travelers
            activities = costs["activities"] * duration_days * travelers
            
            # 额外费用
            flight_cost = 1500 * travelers  # 模拟机票费用
            insurance = 50 * travelers * duration_days  # 保险费
            shopping = 500 * travelers  # 购物预算
            emergency = 200 * travelers  # 应急费用
            
            subtotal = accommodation + food + local_transport + activities
            total = subtotal + flight_cost + insurance + shopping + emergency
            
            result = {
                "destination": destination,
                "duration_days": duration_days,
                "travelers": travelers,
                "comfort_level": comfort_level,
                "breakdown": {
                    "accommodation": accommodation,
                    "food": food,
                    "local_transport": local_transport,
                    "activities": activities,
                    "flights": flight_cost,
                    "insurance": insurance,
                    "shopping": shopping,
                    "emergency": emergency
                },
                "subtotal": subtotal,
                "total_budget": total,
                "per_person_cost": total / travelers,
                "daily_average": total / duration_days,
                "currency": "CNY",
                "calculated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"预算计算失败: {e}")
            return {"error": f"预算计算失败: {str(e)}"}


class TravelResources:
    """旅行资源"""
    
    @staticmethod
    async def get_travel_tips() -> str:
        """获取旅行小贴士"""
        tips = [
            "提前规划行程，预订机票和酒店可以节省费用",
            "购买旅行保险，保障旅途安全", 
            "准备好必要的证件和文件",
            "了解目的地的天气和文化习俗",
            "准备常用药品和急救用品",
            "保持重要联系方式和文件的备份",
            "注意财物安全，使用安全的支付方式",
            "尊重当地文化和环境"
        ]
        return "\n".join(f"• {tip}" for tip in tips)
    
    @staticmethod
    async def get_packing_checklist() -> str:
        """获取打包清单"""
        checklist = {
            "证件类": ["护照", "身份证", "驾照", "签证", "机票", "酒店预订单"],
            "衣物类": ["内衣裤", "外套", "休闲装", "正装", "睡衣", "鞋子", "帽子"],
            "电子产品": ["手机", "充电器", "移动电源", "相机", "转换插头", "耳机"],
            "日用品": ["牙刷", "牙膏", "洗发水", "沐浴露", "护肤品", "防晒霜", "毛巾"],
            "药品": ["常用药", "感冒药", "止痛药", "创可贴", "消毒液", "防蚊液"],
            "其他": ["雨伞", "太阳镜", "背包", "钱包", "零钱", "小锁"]
        }
        
        result = "旅行打包清单：\n"
        for category, items in checklist.items():
            result += f"\n【{category}】\n"
            result += "\n".join(f"□ {item}" for item in items)
            result += "\n"
        
        return result
    
    @staticmethod
    async def get_emergency_contacts() -> str:
        """获取应急联系方式"""
        contacts = {
            "中国": {
                "报警": "110",
                "火警": "119", 
                "急救": "120",
                "交通事故": "122"
            },
            "国际通用": {
                "报警": "911 (美国/加拿大)",
                "欧洲报警": "112",
                "国际救援": "+86-10-12308 (中国领事保护)"
            },
            "重要提醒": [
                "保存当地中国领事馆联系方式",
                "记住酒店地址和电话",
                "告知家人行程安排",
                "开通国际漫游或购买当地电话卡"
            ]
        }
        
        result = "应急联系信息：\n"
        for region, info in contacts.items():
            if region != "重要提醒":
                result += f"\n【{region}】\n"
                for service, number in info.items():
                    result += f"• {service}: {number}\n"
        
        result += "\n【重要提醒】\n"
        for tip in contacts["重要提醒"]:
            result += f"• {tip}\n"
        
        return result


class TravelPrompts:
    """旅行提示词"""
    
    @staticmethod
    async def travel_planning_prompt(destination: str = "目的地", duration: str = "时长") -> List[Dict[str, Any]]:
        """旅行规划提示词"""
        return [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": "你是一个专业的旅行规划师，擅长为用户制定详细的旅行计划。请根据用户的需求，提供个性化的旅行建议。"
                }
            },
            {
                "role": "user", 
                "content": {
                    "type": "text",
                    "text": f"我想去{destination}旅行{duration}，请帮我制定一个详细的旅行计划，包括交通、住宿、景点、美食等安排。"
                }
            }
        ]
    
    @staticmethod
    async def budget_optimization_prompt(budget: str = "预算") -> List[Dict[str, Any]]:
        """预算优化提示词"""
        return [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": "你是一个旅行预算专家，能够帮助用户在有限的预算内获得最佳的旅行体验。"
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "text", 
                    "text": f"我的旅行预算是{budget}，请帮我优化支出，让我能够获得最佳的旅行体验，包括省钱小贴士和性价比推荐。"
                }
            }
        ]
    
    @staticmethod
    async def local_culture_prompt(destination: str = "目的地") -> List[Dict[str, Any]]:
        """当地文化提示词"""
        return [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": "你是一个文化专家和当地向导，对世界各地的文化习俗、历史背景和实用信息非常了解。"
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"我即将前往{destination}，请介绍当地的文化习俗、禁忌、小费文化、重要节日以及与当地人交流的注意事项。"
                }
            }
        ]


def create_mcp_tools():
    """创建并注册MCP工具"""
    mcp_server = get_mcp_server()
    travel_tools = TravelTools()
    
    # 注册工具
    tools_config = [
        {
            "tool": MCPTool(
                name="search_flights",
                description="搜索航班信息，包括价格、时间、航空公司等",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "出发城市"},
                        "destination": {"type": "string", "description": "目的地城市"},
                        "departure_date": {"type": "string", "description": "出发日期 (YYYY-MM-DD)"},
                        "return_date": {"type": "string", "description": "返程日期 (YYYY-MM-DD，可选)"},
                        "passengers": {"type": "integer", "description": "乘客人数", "default": 1},
                        "class_type": {"type": "string", "description": "舱位类型", "enum": ["economy", "business", "first"], "default": "economy"}
                    },
                    "required": ["origin", "destination", "departure_date"]
                }
            ),
            "handler": travel_tools.search_flights
        },
        {
            "tool": MCPTool(
                name="search_hotels",
                description="搜索酒店信息，包括价格、评分、设施等",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "目的地城市"},
                        "check_in": {"type": "string", "description": "入住日期 (YYYY-MM-DD)"},
                        "check_out": {"type": "string", "description": "退房日期 (YYYY-MM-DD)"},
                        "guests": {"type": "integer", "description": "客人数量", "default": 2},
                        "rooms": {"type": "integer", "description": "房间数量", "default": 1},
                        "price_range": {"type": "string", "description": "价格范围", "enum": ["budget", "standard", "luxury"]}
                    },
                    "required": ["destination", "check_in", "check_out"]
                }
            ),
            "handler": travel_tools.search_hotels
        },
        {
            "tool": MCPTool(
                name="get_weather",
                description="获取目的地天气预报信息",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "days": {"type": "integer", "description": "预报天数", "default": 7, "minimum": 1, "maximum": 14}
                    },
                    "required": ["city"]
                }
            ),
            "handler": travel_tools.get_weather
        },
        {
            "tool": MCPTool(
                name="get_exchange_rate",
                description="获取汇率信息和货币转换",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "from_currency": {"type": "string", "description": "源货币代码 (如: USD, CNY, EUR)"},
                        "to_currency": {"type": "string", "description": "目标货币代码 (如: USD, CNY, EUR)"},
                        "amount": {"type": "number", "description": "转换金额", "default": 1.0}
                    },
                    "required": ["from_currency", "to_currency"]
                }
            ),
            "handler": travel_tools.get_exchange_rate
        },
        {
            "tool": MCPTool(
                name="plan_route",
                description="规划旅行路线，包括距离、时间和路径",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "出发地"},
                        "destination": {"type": "string", "description": "目的地"},
                        "waypoints": {"type": "array", "items": {"type": "string"}, "description": "途经点列表"},
                        "mode": {"type": "string", "description": "交通方式", "enum": ["driving", "walking", "transit", "cycling"], "default": "driving"}
                    },
                    "required": ["origin", "destination"]
                }
            ),
            "handler": travel_tools.plan_route
        },
        {
            "tool": MCPTool(
                name="recommend_attractions",
                description="推荐旅游景点和热门目的地",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "category": {"type": "string", "description": "景点类型", "enum": ["历史文化", "自然风光", "主题公园", "博物馆", "购物中心", "美食街"]},
                        "limit": {"type": "integer", "description": "推荐数量", "default": 10, "minimum": 1, "maximum": 20}
                    },
                    "required": ["city"]
                }
            ),
            "handler": travel_tools.recommend_attractions
        },
        {
            "tool": MCPTool(
                name="recommend_restaurants",
                description="推荐餐厅和美食信息",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "cuisine": {"type": "string", "description": "菜系类型", "enum": ["中餐", "西餐", "日料", "韩料", "泰餐", "意大利菜", "法餐", "印度菜"]},
                        "price_range": {"type": "string", "description": "价格范围", "enum": ["经济", "中档", "高档"]},
                        "limit": {"type": "integer", "description": "推荐数量", "default": 10, "minimum": 1, "maximum": 20}
                    },
                    "required": ["city"]
                }
            ),
            "handler": travel_tools.recommend_restaurants
        },
        {
            "tool": MCPTool(
                name="calculate_budget",
                description="计算旅行预算和费用分析",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "目的地"},
                        "duration_days": {"type": "integer", "description": "旅行天数", "minimum": 1},
                        "travelers": {"type": "integer", "description": "旅行人数", "default": 1, "minimum": 1},
                        "comfort_level": {"type": "string", "description": "舒适度级别", "enum": ["budget", "standard", "luxury"], "default": "standard"}
                    },
                    "required": ["destination", "duration_days"]
                }
            ),
            "handler": travel_tools.calculate_budget
        }
    ]
    
    # 注册所有工具
    for config in tools_config:
        mcp_server.register_tool(config["tool"], config["handler"])
    
    # 注册资源
    resources_config = [
        {
            "resource": MCPResource(
                uri="travel://tips",
                name="旅行小贴士",
                description="实用的旅行建议和注意事项",
                mimeType="text/plain"
            ),
            "handler": TravelResources.get_travel_tips
        },
        {
            "resource": MCPResource(
                uri="travel://packing-checklist",
                name="打包清单",
                description="完整的旅行打包清单",
                mimeType="text/plain"
            ),
            "handler": TravelResources.get_packing_checklist
        },
        {
            "resource": MCPResource(
                uri="travel://emergency-contacts",
                name="应急联系方式",
                description="全球应急联系电话和重要信息",
                mimeType="text/plain"
            ),
            "handler": TravelResources.get_emergency_contacts
        }
    ]
    
    # 注册所有资源
    for config in resources_config:
        mcp_server.register_resource(config["resource"], config["handler"])
    
    # 注册提示词
    prompts_config = [
        {
            "prompt": MCPPrompt(
                name="travel_planning",
                description="专业的旅行规划提示词，帮助制定详细旅行计划",
                arguments=[
                    {"name": "destination", "description": "目的地", "required": True},
                    {"name": "duration", "description": "旅行时长", "required": True}
                ]
            ),
            "handler": TravelPrompts.travel_planning_prompt
        },
        {
            "prompt": MCPPrompt(
                name="budget_optimization",
                description="预算优化提示词，帮助在有限预算内获得最佳体验",
                arguments=[
                    {"name": "budget", "description": "旅行预算", "required": True}
                ]
            ),
            "handler": TravelPrompts.budget_optimization_prompt
        },
        {
            "prompt": MCPPrompt(
                name="local_culture",
                description="当地文化介绍提示词，了解目的地文化习俗",
                arguments=[
                    {"name": "destination", "description": "目的地", "required": True}
                ]
            ),
            "handler": TravelPrompts.local_culture_prompt
        }
    ]
    
    # 注册所有提示词
    for config in prompts_config:
        mcp_server.register_prompt(config["prompt"], config["handler"])
    
    logger.info("MCP旅行工具集注册完成")
    return mcp_server 