"""
MCP工具集成
实现旅行相关工具集（航班搜索、酒店查询、天气获取）、执行监控、错误处理、缓存优化
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
import hashlib
from dataclasses import dataclass
import redis.asyncio as redis

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .mcp_server import MCPToolBase

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = None


class ToolCache:
    """工具结果缓存"""
    
    def __init__(self, redis_client=None, default_ttl: int = 3600):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.local_cache = {}  # 本地缓存作为备份
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
    
    def _generate_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 将参数序列化并生成哈希
        args_str = json.dumps(arguments, sort_keys=True)
        hash_obj = hashlib.md5(f"{tool_name}:{args_str}".encode())
        return f"mcp_tool_cache:{hash_obj.hexdigest()}"
    
    async def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """获取缓存结果"""
        cache_key = self._generate_cache_key(tool_name, arguments)
        
        try:
            # 优先从Redis获取
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self.cache_stats["hits"] += 1
                    return json.loads(cached_data)
            
            # 从本地缓存获取
            if cache_key in self.local_cache:
                cache_entry = self.local_cache[cache_key]
                if datetime.now() < cache_entry["expires_at"]:
                    self.cache_stats["hits"] += 1
                    return cache_entry["data"]
                else:
                    del self.local_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"缓存获取失败: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def set(self, tool_name: str, arguments: Dict[str, Any], result: Any, ttl: Optional[int] = None) -> None:
        """设置缓存结果"""
        cache_key = self._generate_cache_key(tool_name, arguments)
        ttl = ttl or self.default_ttl
        
        try:
            # 存储到Redis
            if self.redis_client:
                await self.redis_client.set(
                    cache_key,
                    json.dumps(result, default=str),
                    ex=ttl
                )
            
            # 存储到本地缓存
            self.local_cache[cache_key] = {
                "data": result,
                "expires_at": datetime.now() + timedelta(seconds=ttl)
            }
            
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")
            self.cache_stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "errors": self.cache_stats["errors"],
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache)
        }


class ToolMonitor:
    """工具执行监控"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.execution_stats = {}
        self.error_counts = {}
        self.performance_metrics = {}
    
    def record_execution(self, tool_name: str, execution_time: float, success: bool, error: Optional[str] = None) -> None:
        """记录工具执行"""
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        stats = self.execution_stats[tool_name]
        stats["total_calls"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        
        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
            
            # 记录错误
            if tool_name not in self.error_counts:
                self.error_counts[tool_name] = {}
            
            error_key = error or "unknown_error"
            self.error_counts[tool_name][error_key] = self.error_counts[tool_name].get(error_key, 0) + 1
    
    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """获取工具统计"""
        stats = self.execution_stats.get(tool_name, {})
        errors = self.error_counts.get(tool_name, {})
        
        success_rate = 0.0
        if stats.get("total_calls", 0) > 0:
            success_rate = stats["successful_calls"] / stats["total_calls"]
        
        return {
            "execution_stats": stats,
            "error_counts": errors,
            "success_rate": success_rate
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有工具统计"""
        all_stats = {}
        for tool_name in self.execution_stats:
            all_stats[tool_name] = self.get_tool_stats(tool_name)
        return all_stats


class FlightSearchTool(MCPToolBase):
    """航班搜索工具"""
    
    def __init__(self, cache: ToolCache, monitor: ToolMonitor):
        super().__init__(
            name="flight_search",
            description="搜索航班信息",
            input_schema={
                "type": "object",
                "properties": {
                    "departure_city": {
                        "type": "string",
                        "description": "出发城市"
                    },
                    "arrival_city": {
                        "type": "string",
                        "description": "到达城市"
                    },
                    "departure_date": {
                        "type": "string",
                        "format": "date",
                        "description": "出发日期 (YYYY-MM-DD)"
                    },
                    "return_date": {
                        "type": "string",
                        "format": "date",
                        "description": "返程日期 (YYYY-MM-DD)，可选"
                    },
                    "passengers": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "乘客数量，默认为1"
                    },
                    "class": {
                        "type": "string",
                        "enum": ["economy", "premium", "business", "first"],
                        "description": "舱位等级，默认为economy"
                    }
                },
                "required": ["departure_city", "arrival_city", "departure_date"]
            }
        )
        self.cache = cache
        self.monitor = monitor
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """执行航班搜索"""
        start_time = time.time()
        
        try:
            # 检查缓存
            cached_result = await self.cache.get(self.name, arguments)
            if cached_result:
                execution_time = time.time() - start_time
                self.monitor.record_execution(self.name, execution_time, True)
                return ToolExecutionResult(
                    success=True,
                    result=cached_result,
                    execution_time=execution_time,
                    cache_hit=True
                )
            
            # 执行搜索
            result = await self._search_flights(arguments)
            
            # 缓存结果
            await self.cache.set(self.name, arguments, result, ttl=1800)  # 30分钟缓存
            
            execution_time = time.time() - start_time
            self.monitor.record_execution(self.name, execution_time, True)
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                cache_hit=False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.monitor.record_execution(self.name, execution_time, False, error_msg)
            
            return ToolExecutionResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _search_flights(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """搜索航班（模拟实现）"""
        # 这里应该调用真实的航班搜索API
        # 为了演示，我们返回模拟数据
        
        departure_city = arguments.get("departure_city")
        arrival_city = arguments.get("arrival_city")
        departure_date = arguments.get("departure_date")
        return_date = arguments.get("return_date")
        passengers = arguments.get("passengers", 1)
        flight_class = arguments.get("class", "economy")
        
        # 模拟API延迟
        await asyncio.sleep(1.0)
        
        # 模拟航班数据
        flights = [
            {
                "flight_number": "CA1234",
                "airline": "中国国际航空",
                "departure_city": departure_city,
                "arrival_city": arrival_city,
                "departure_time": f"{departure_date} 08:00",
                "arrival_time": f"{departure_date} 10:30",
                "duration": "2小时30分钟",
                "price": 800,
                "class": flight_class,
                "available_seats": 15
            },
            {
                "flight_number": "CZ5678",
                "airline": "中国南方航空",
                "departure_city": departure_city,
                "arrival_city": arrival_city,
                "departure_time": f"{departure_date} 14:00",
                "arrival_time": f"{departure_date} 16:45",
                "duration": "2小时45分钟",
                "price": 750,
                "class": flight_class,
                "available_seats": 8
            }
        ]
        
        result = {
            "search_params": arguments,
            "flights": flights,
            "total_results": len(flights),
            "search_time": datetime.now().isoformat()
        }
        
        if return_date:
            # 添加返程航班
            return_flights = [
                {
                    "flight_number": "CA5678",
                    "airline": "中国国际航空",
                    "departure_city": arrival_city,
                    "arrival_city": departure_city,
                    "departure_time": f"{return_date} 09:00",
                    "arrival_time": f"{return_date} 11:30",
                    "duration": "2小时30分钟",
                    "price": 820,
                    "class": flight_class,
                    "available_seats": 12
                }
            ]
            result["return_flights"] = return_flights
        
        return result


class HotelSearchTool(MCPToolBase):
    """酒店搜索工具"""
    
    def __init__(self, cache: ToolCache, monitor: ToolMonitor):
        super().__init__(
            name="hotel_search",
            description="搜索酒店信息",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "check_in_date": {
                        "type": "string",
                        "format": "date",
                        "description": "入住日期 (YYYY-MM-DD)"
                    },
                    "check_out_date": {
                        "type": "string",
                        "format": "date",
                        "description": "退房日期 (YYYY-MM-DD)"
                    },
                    "guests": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "客人数量，默认为1"
                    },
                    "price_min": {
                        "type": "number",
                        "description": "最低价格"
                    },
                    "price_max": {
                        "type": "number",
                        "description": "最高价格"
                    },
                    "star_rating": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "星级评定"
                    }
                },
                "required": ["city", "check_in_date", "check_out_date"]
            }
        )
        self.cache = cache
        self.monitor = monitor
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """执行酒店搜索"""
        start_time = time.time()
        
        try:
            # 检查缓存
            cached_result = await self.cache.get(self.name, arguments)
            if cached_result:
                execution_time = time.time() - start_time
                self.monitor.record_execution(self.name, execution_time, True)
                return ToolExecutionResult(
                    success=True,
                    result=cached_result,
                    execution_time=execution_time,
                    cache_hit=True
                )
            
            # 执行搜索
            result = await self._search_hotels(arguments)
            
            # 缓存结果
            await self.cache.set(self.name, arguments, result, ttl=3600)  # 1小时缓存
            
            execution_time = time.time() - start_time
            self.monitor.record_execution(self.name, execution_time, True)
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                cache_hit=False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.monitor.record_execution(self.name, execution_time, False, error_msg)
            
            return ToolExecutionResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _search_hotels(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """搜索酒店（模拟实现）"""
        city = arguments.get("city")
        check_in_date = arguments.get("check_in_date")
        check_out_date = arguments.get("check_out_date")
        guests = arguments.get("guests", 1)
        price_min = arguments.get("price_min")
        price_max = arguments.get("price_max")
        star_rating = arguments.get("star_rating")
        
        # 模拟API延迟
        await asyncio.sleep(0.8)
        
        # 模拟酒店数据
        hotels = [
            {
                "hotel_id": "h001",
                "name": f"{city}万豪酒店",
                "star_rating": 5,
                "address": f"{city}市中心商务区",
                "price_per_night": 680,
                "currency": "CNY",
                "rating": 4.5,
                "review_count": 1250,
                "amenities": ["WiFi", "健身房", "游泳池", "餐厅", "商务中心"],
                "available_rooms": 3,
                "room_type": "高级双人房",
                "cancellation_policy": "免费取消",
                "images": [f"https://example.com/hotel_h001_1.jpg"]
            },
            {
                "hotel_id": "h002",
                "name": f"{city}如家酒店",
                "star_rating": 3,
                "address": f"{city}市火车站附近",
                "price_per_night": 280,
                "currency": "CNY",
                "rating": 4.2,
                "review_count": 890,
                "amenities": ["WiFi", "早餐", "24小时前台"],
                "available_rooms": 8,
                "room_type": "标准双人房",
                "cancellation_policy": "入住前24小时免费取消",
                "images": [f"https://example.com/hotel_h002_1.jpg"]
            },
            {
                "hotel_id": "h003",
                "name": f"{city}希尔顿酒店",
                "star_rating": 5,
                "address": f"{city}市金融区",
                "price_per_night": 920,
                "currency": "CNY",
                "rating": 4.7,
                "review_count": 2100,
                "amenities": ["WiFi", "健身房", "SPA", "多个餐厅", "行政酒廊"],
                "available_rooms": 5,
                "room_type": "豪华海景房",
                "cancellation_policy": "免费取消",
                "images": [f"https://example.com/hotel_h003_1.jpg"]
            }
        ]
        
        # 应用过滤条件
        filtered_hotels = []
        for hotel in hotels:
            if price_min and hotel["price_per_night"] < price_min:
                continue
            if price_max and hotel["price_per_night"] > price_max:
                continue
            if star_rating and hotel["star_rating"] < star_rating:
                continue
            filtered_hotels.append(hotel)
        
        return {
            "search_params": arguments,
            "hotels": filtered_hotels,
            "total_results": len(filtered_hotels),
            "search_time": datetime.now().isoformat()
        }


class WeatherTool(MCPToolBase):
    """天气查询工具"""
    
    def __init__(self, cache: ToolCache, monitor: ToolMonitor):
        super().__init__(
            name="weather_inquiry",
            description="查询天气信息",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "description": "查询天数，默认为1天"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "description": "单位系统，默认为metric"
                    }
                },
                "required": ["city"]
            }
        )
        self.cache = cache
        self.monitor = monitor
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.api_key = settings.WEATHER_API_KEY  # 需要在settings中添加
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """执行天气查询"""
        start_time = time.time()
        
        try:
            # 检查缓存
            cached_result = await self.cache.get(self.name, arguments)
            if cached_result:
                execution_time = time.time() - start_time
                self.monitor.record_execution(self.name, execution_time, True)
                return ToolExecutionResult(
                    success=True,
                    result=cached_result,
                    execution_time=execution_time,
                    cache_hit=True
                )
            
            # 执行查询
            result = await self._get_weather(arguments)
            
            # 缓存结果
            await self.cache.set(self.name, arguments, result, ttl=1800)  # 30分钟缓存
            
            execution_time = time.time() - start_time
            self.monitor.record_execution(self.name, execution_time, True)
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                cache_hit=False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.monitor.record_execution(self.name, execution_time, False, error_msg)
            
            return ToolExecutionResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _get_weather(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """获取天气信息（模拟实现）"""
        city = arguments.get("city")
        days = arguments.get("days", 1)
        units = arguments.get("units", "metric")
        
        # 模拟API延迟
        await asyncio.sleep(0.5)
        
        # 模拟天气数据
        current_weather = {
            "city": city,
            "country": "CN",
            "temperature": 22,
            "feels_like": 24,
            "humidity": 65,
            "pressure": 1013,
            "wind_speed": 5,
            "wind_direction": "东北",
            "visibility": 10,
            "uv_index": 6,
            "weather_condition": "多云",
            "weather_description": "多云转晴",
            "icon": "partly_cloudy",
            "timestamp": datetime.now().isoformat()
        }
        
        forecast = []
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "high_temp": 25 + i,
                "low_temp": 18 + i,
                "weather_condition": "晴" if i % 2 == 0 else "多云",
                "rain_probability": 20 + i * 10,
                "wind_speed": 3 + i,
                "humidity": 60 + i * 5
            })
        
        return {
            "current": current_weather,
            "forecast": forecast,
            "units": units,
            "search_time": datetime.now().isoformat()
        }


class CurrencyConvertTool(MCPToolBase):
    """货币转换工具"""
    
    def __init__(self, cache: ToolCache, monitor: ToolMonitor):
        super().__init__(
            name="currency_convert",
            description="货币汇率转换",
            input_schema={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "金额"
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "源货币代码 (如 USD, CNY, EUR)"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "目标货币代码 (如 USD, CNY, EUR)"
                    }
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        )
        self.cache = cache
        self.monitor = monitor
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """执行货币转换"""
        start_time = time.time()
        
        try:
            # 检查缓存
            cached_result = await self.cache.get(self.name, arguments)
            if cached_result:
                execution_time = time.time() - start_time
                self.monitor.record_execution(self.name, execution_time, True)
                return ToolExecutionResult(
                    success=True,
                    result=cached_result,
                    execution_time=execution_time,
                    cache_hit=True
                )
            
            # 执行转换
            result = await self._convert_currency(arguments)
            
            # 缓存结果
            await self.cache.set(self.name, arguments, result, ttl=3600)  # 1小时缓存
            
            execution_time = time.time() - start_time
            self.monitor.record_execution(self.name, execution_time, True)
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                cache_hit=False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.monitor.record_execution(self.name, execution_time, False, error_msg)
            
            return ToolExecutionResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _convert_currency(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """转换货币（模拟实现）"""
        amount = arguments.get("amount")
        from_currency = arguments.get("from_currency").upper()
        to_currency = arguments.get("to_currency").upper()
        
        # 模拟汇率（实际应该从真实API获取）
        exchange_rates = {
            ("USD", "CNY"): 7.2,
            ("CNY", "USD"): 0.139,
            ("EUR", "CNY"): 7.8,
            ("CNY", "EUR"): 0.128,
            ("USD", "EUR"): 0.85,
            ("EUR", "USD"): 1.18,
            ("GBP", "CNY"): 9.1,
            ("CNY", "GBP"): 0.11,
            ("JPY", "CNY"): 0.048,
            ("CNY", "JPY"): 20.8
        }
        
        if from_currency == to_currency:
            rate = 1.0
        else:
            rate = exchange_rates.get((from_currency, to_currency))
            if rate is None:
                # 尝试反向查找
                reverse_rate = exchange_rates.get((to_currency, from_currency))
                if reverse_rate:
                    rate = 1 / reverse_rate
                else:
                    raise ValueError(f"不支持的货币对: {from_currency} -> {to_currency}")
        
        converted_amount = amount * rate
        
        return {
            "original_amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "exchange_rate": rate,
            "converted_amount": round(converted_amount, 2),
            "conversion_time": datetime.now().isoformat()
        }


def create_mcp_tools(redis_client=None) -> List[MCPToolBase]:
    """创建MCP工具集合"""
    cache = ToolCache(redis_client=redis_client)
    monitor = ToolMonitor(redis_client=redis_client)
    
    tools = [
        FlightSearchTool(cache, monitor),
        HotelSearchTool(cache, monitor),
        WeatherTool(cache, monitor),
        CurrencyConvertTool(cache, monitor)
    ]
    
    return tools, cache, monitor 