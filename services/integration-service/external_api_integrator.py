"""
外部数据源集成服务
提供统一的外部API集成框架，包括航班、酒店、天气、汇率等数据源
"""

import asyncio
import aiohttp
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import random

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from retrying import retry
    RETRYING_AVAILABLE = True
except ImportError:
    RETRYING_AVAILABLE = False

import structlog
from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DataSourceType(Enum):
    """数据源类型"""
    FLIGHT = "flight"
    HOTEL = "hotel"
    WEATHER = "weather"
    EXCHANGE_RATE = "exchange_rate"
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    TRANSPORTATION = "transportation"


class APIStatus(Enum):
    """API状态"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class APICredentials:
    """API凭证"""
    api_key: str
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    endpoint_url: str = ""
    rate_limit: int = 1000  # 每小时请求限制
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """API响应"""
    success: bool
    data: Any
    status_code: int
    response_time: float
    error_message: Optional[str] = None
    cached: bool = False
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class APIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, credentials: APICredentials, cache_ttl: int = 3600):
        self.credentials = credentials
        self.cache_ttl = cache_ttl
        self.status = APIStatus.AVAILABLE
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.rate_limiter = RateLimiter(credentials.rate_limit)
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_response_time": 0.0
        }
    
    @abstractmethod
    async def search(self, query: Dict[str, Any]) -> APIResponse:
        """搜索数据"""
        pass
    
    @abstractmethod
    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        """构建请求URL"""
        pass
    
    async def _make_request(self, url: str, method: str = "GET", 
                           headers: Dict[str, str] = None,
                           data: Dict[str, Any] = None) -> APIResponse:
        """发送HTTP请求"""
        start_time = time.time()
        
        try:
            # 速率限制检查
            await self.rate_limiter.acquire()
            
            # 默认请求头
            default_headers = {
                "User-Agent": "AI-Travel-Planner/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            if headers:
                default_headers.update(headers)
            
            # 添加认证信息
            auth_headers = self._get_auth_headers()
            default_headers.update(auth_headers)
            
            # 发送请求
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.credentials.timeout)) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=default_headers,
                    json=data if method in ["POST", "PUT"] else None,
                    params=data if method == "GET" else None
                ) as response:
                    response_time = time.time() - start_time
                    response_data = await response.json()
                    
                    # 更新统计
                    self._update_stats(response.status, response_time, success=True)
                    
                    return APIResponse(
                        success=response.status == 200,
                        data=response_data,
                        status_code=response.status,
                        response_time=response_time,
                        source=self.__class__.__name__
                    )
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self._update_stats(408, response_time, success=False)
            
            return APIResponse(
                success=False,
                data=None,
                status_code=408,
                response_time=response_time,
                error_message="请求超时",
                source=self.__class__.__name__
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(500, response_time, success=False)
            
            logger.error(f"API请求失败: {e}")
            
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                response_time=response_time,
                error_message=str(e),
                source=self.__class__.__name__
            )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证请求头"""
        headers = {}
        
        if self.credentials.api_key:
            headers["Authorization"] = f"Bearer {self.credentials.api_key}"
        
        if self.credentials.access_token:
            headers["X-Access-Token"] = self.credentials.access_token
        
        return headers
    
    def _update_stats(self, status_code: int, response_time: float, success: bool):
        """更新统计信息"""
        self.stats["total_requests"] += 1
        
        if success and status_code == 200:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # 更新平均响应时间
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        self.last_request_time = datetime.now()
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        total_requests = self.stats["total_requests"]
        success_rate = (
            self.stats["successful_requests"] / total_requests * 100
            if total_requests > 0 else 100
        )
        
        # 判断健康状态
        if success_rate >= 95:
            health = "healthy"
        elif success_rate >= 80:
            health = "warning"
        else:
            health = "unhealthy"
        
        return {
            "status": self.status.value,
            "health": health,
            "success_rate": success_rate,
            "stats": self.stats.copy(),
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None
        }


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, rate_limit: int, time_window: int = 3600):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取请求许可"""
        async with self._lock:
            now = time.time()
            
            # 清理过期的请求记录
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # 检查是否超过限制
            if len(self.requests) >= self.rate_limit:
                # 计算需要等待的时间
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                
                if wait_time > 0:
                    logger.warning(f"达到速率限制，等待 {wait_time:.2f} 秒")
                    await asyncio.sleep(wait_time)
            
            # 记录新请求
            self.requests.append(now)


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_cache: Dict[str, CacheEntry] = {}
        self.max_local_cache_size = 1000
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        # 先检查Redis缓存
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    return data
            except Exception as e:
                logger.error(f"Redis缓存读取失败: {e}")
        
        # 检查本地缓存
        if key in self.local_cache:
            entry = self.local_cache[key]
            
            # 检查是否过期
            if datetime.now() < entry.expires_at:
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                return entry.data
            else:
                # 删除过期条目
                del self.local_cache[key]
        
        return None
    
    async def set(self, key: str, data: Any, ttl: int = 3600):
        """设置缓存数据"""
        # 设置Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(data, default=str))
            except Exception as e:
                logger.error(f"Redis缓存写入失败: {e}")
        
        # 设置本地缓存
        if len(self.local_cache) >= self.max_local_cache_size:
            # 清理最少使用的条目
            self._cleanup_local_cache()
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self.local_cache[key] = CacheEntry(
            key=key,
            data=data,
            created_at=datetime.now(),
            expires_at=expires_at
        )
    
    def _cleanup_local_cache(self):
        """清理本地缓存"""
        # 删除过期条目
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.local_cache.items()
            if now >= entry.expires_at
        ]
        
        for key in expired_keys:
            del self.local_cache[key]
        
        # 如果还是太多，删除最少使用的
        if len(self.local_cache) >= self.max_local_cache_size:
            # 按访问次数排序，删除最少使用的50%
            sorted_entries = sorted(
                self.local_cache.items(),
                key=lambda x: x[1].access_count
            )
            
            remove_count = len(sorted_entries) // 2
            for key, _ in sorted_entries[:remove_count]:
                del self.local_cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_entries = len(self.local_cache)
        total_access = sum(entry.access_count for entry in self.local_cache.values())
        
        return {
            "local_cache_size": total_entries,
            "total_access_count": total_access,
            "average_access_per_entry": total_access / max(total_entries, 1)
        }


class FlightAPIClient(APIClient):
    """航班API客户端"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials, cache_ttl=1800)  # 30分钟缓存
        self.data_source_type = DataSourceType.FLIGHT
    
    async def search(self, query: Dict[str, Any]) -> APIResponse:
        """搜索航班"""
        try:
            # 构建查询参数
            params = {
                "origin": query.get("origin", ""),
                "destination": query.get("destination", ""),
                "departure_date": query.get("departure_date", ""),
                "return_date": query.get("return_date", ""),
                "passengers": query.get("passengers", 1),
                "class": query.get("class", "economy")
            }
            
            # 生成缓存键
            cache_key = self._generate_cache_key("flight_search", params)
            
            # 检查缓存
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    status_code=200,
                    response_time=0.0,
                    cached=True,
                    source=self.__class__.__name__
                )
            
            # 构建请求URL
            url = self._build_url("flights/search", params)
            
            # 发送请求
            response = await self._make_request(url, method="GET", data=params)
            
            if response.success:
                # 处理响应数据
                processed_data = self._process_flight_data(response.data)
                
                # 缓存结果
                await self._save_to_cache(cache_key, processed_data)
                
                response.data = processed_data
            
            return response
            
        except Exception as e:
            logger.error(f"航班搜索失败: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                response_time=0.0,
                error_message=str(e),
                source=self.__class__.__name__
            )
    
    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        """构建航班API URL"""
        base_url = self.credentials.endpoint_url or "https://api.example-flight.com/v1"
        return f"{base_url}/{endpoint}"
    
    def _process_flight_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """处理航班数据"""
        # 模拟数据处理
        if isinstance(raw_data, dict) and "flights" in raw_data:
            flights = raw_data["flights"]
        else:
            # 生成模拟数据
            flights = self._generate_mock_flight_data()
        
        processed_flights = []
        for flight in flights[:10]:  # 限制返回数量
            processed_flight = {
                "flight_number": flight.get("flight_number", f"CA{random.randint(1000, 9999)}"),
                "airline": flight.get("airline", "中国国航"),
                "departure_airport": flight.get("departure_airport", "PEK"),
                "arrival_airport": flight.get("arrival_airport", "SHA"),
                "departure_time": flight.get("departure_time", "2024-06-01T08:00:00"),
                "arrival_time": flight.get("arrival_time", "2024-06-01T11:30:00"),
                "duration": flight.get("duration", "3h 30m"),
                "price": flight.get("price", random.randint(800, 2500)),
                "currency": "CNY",
                "stops": flight.get("stops", 0),
                "aircraft": flight.get("aircraft", "Boeing 737"),
                "booking_class": flight.get("booking_class", "Y"),
                "availability": flight.get("availability", "Available"),
                "baggage_allowance": flight.get("baggage_allowance", "23kg"),
                "meal_service": flight.get("meal_service", True),
                "wifi_available": flight.get("wifi_available", True)
            }
            processed_flights.append(processed_flight)
        
        return processed_flights
    
    def _generate_mock_flight_data(self) -> List[Dict[str, Any]]:
        """生成模拟航班数据"""
        airlines = ["中国国航", "东方航空", "南方航空", "海南航空", "厦门航空"]
        aircraft_types = ["Boeing 737", "Airbus A320", "Boeing 777", "Airbus A330"]
        
        flights = []
        for i in range(15):
            flight = {
                "flight_number": f"CA{random.randint(1000, 9999)}",
                "airline": random.choice(airlines),
                "departure_airport": "PEK",
                "arrival_airport": "SHA",
                "departure_time": f"2024-06-01T{8 + i//2:02d}:{(i%2)*30:02d}:00",
                "arrival_time": f"2024-06-01T{11 + i//2:02d}:{(i%2)*30:02d}:00",
                "duration": f"{3 + random.randint(0, 2)}h {random.randint(0, 5)*10}m",
                "price": random.randint(800, 2500),
                "stops": random.choice([0, 1]),
                "aircraft": random.choice(aircraft_types),
                "booking_class": random.choice(["Y", "B", "H", "K"]),
                "availability": "Available"
            }
            flights.append(flight)
        
        return flights
    
    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = f"{operation}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        # 这里应该集成实际的缓存系统
        return None
    
    async def _save_to_cache(self, key: str, data: Any):
        """保存数据到缓存"""
        # 这里应该集成实际的缓存系统
        pass


class HotelAPIClient(APIClient):
    """酒店API客户端"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials, cache_ttl=3600)  # 1小时缓存
        self.data_source_type = DataSourceType.HOTEL
    
    async def search(self, query: Dict[str, Any]) -> APIResponse:
        """搜索酒店"""
        try:
            params = {
                "destination": query.get("destination", ""),
                "check_in": query.get("check_in", ""),
                "check_out": query.get("check_out", ""),
                "guests": query.get("guests", 1),
                "rooms": query.get("rooms", 1),
                "min_price": query.get("min_price", 0),
                "max_price": query.get("max_price", 10000),
                "star_rating": query.get("star_rating", "")
            }
            
            cache_key = self._generate_cache_key("hotel_search", params)
            
            # 检查缓存
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    status_code=200,
                    response_time=0.0,
                    cached=True,
                    source=self.__class__.__name__
                )
            
            url = self._build_url("hotels/search", params)
            response = await self._make_request(url, method="GET", data=params)
            
            if response.success:
                processed_data = self._process_hotel_data(response.data)
                await self._save_to_cache(cache_key, processed_data)
                response.data = processed_data
            
            return response
            
        except Exception as e:
            logger.error(f"酒店搜索失败: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                response_time=0.0,
                error_message=str(e),
                source=self.__class__.__name__
            )
    
    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        """构建酒店API URL"""
        base_url = self.credentials.endpoint_url or "https://api.example-hotel.com/v1"
        return f"{base_url}/{endpoint}"
    
    def _process_hotel_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """处理酒店数据"""
        if isinstance(raw_data, dict) and "hotels" in raw_data:
            hotels = raw_data["hotels"]
        else:
            hotels = self._generate_mock_hotel_data()
        
        processed_hotels = []
        for hotel in hotels[:10]:
            processed_hotel = {
                "hotel_id": hotel.get("id", str(uuid.uuid4())),
                "name": hotel.get("name", f"酒店{random.randint(1, 100)}"),
                "star_rating": hotel.get("star_rating", random.randint(3, 5)),
                "address": hotel.get("address", "市中心"),
                "latitude": hotel.get("latitude", 39.9 + random.uniform(-0.1, 0.1)),
                "longitude": hotel.get("longitude", 116.4 + random.uniform(-0.1, 0.1)),
                "price_per_night": hotel.get("price", random.randint(200, 1500)),
                "currency": "CNY",
                "guest_rating": hotel.get("rating", random.uniform(7.5, 9.5)),
                "review_count": hotel.get("reviews", random.randint(100, 2000)),
                "amenities": hotel.get("amenities", ["免费WiFi", "空调", "24小时前台"]),
                "room_type": hotel.get("room_type", "标准间"),
                "breakfast_included": hotel.get("breakfast", random.choice([True, False])),
                "free_cancellation": hotel.get("cancellation", True),
                "distance_to_center": hotel.get("distance", f"{random.uniform(0.5, 5.0):.1f}km"),
                "availability": "Available",
                "booking_url": hotel.get("booking_url", "https://example.com/book")
            }
            processed_hotels.append(processed_hotel)
        
        return processed_hotels
    
    def _generate_mock_hotel_data(self) -> List[Dict[str, Any]]:
        """生成模拟酒店数据"""
        hotel_names = ["豪华大酒店", "商务酒店", "精品酒店", "如家快捷", "汉庭酒店"]
        amenities_options = [
            ["免费WiFi", "空调", "24小时前台"],
            ["免费WiFi", "健身房", "游泳池", "spa", "餐厅"],
            ["免费WiFi", "空调", "停车场", "会议室"],
            ["免费WiFi", "空调", "24小时前台", "自助早餐"]
        ]
        
        hotels = []
        for i in range(12):
            hotel = {
                "id": str(uuid.uuid4()),
                "name": f"{random.choice(hotel_names)}{i+1}",
                "star_rating": random.randint(3, 5),
                "address": f"中心商务区第{i+1}街",
                "price": random.randint(200, 1500),
                "rating": random.uniform(7.5, 9.5),
                "reviews": random.randint(100, 2000),
                "amenities": random.choice(amenities_options),
                "room_type": random.choice(["标准间", "大床房", "套房"]),
                "breakfast": random.choice([True, False]),
                "cancellation": True
            }
            hotels.append(hotel)
        
        return hotels
    
    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = f"{operation}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        return None
    
    async def _save_to_cache(self, key: str, data: Any):
        """保存数据到缓存"""
        pass


class WeatherAPIClient(APIClient):
    """天气API客户端"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials, cache_ttl=1800)  # 30分钟缓存
        self.data_source_type = DataSourceType.WEATHER
    
    async def search(self, query: Dict[str, Any]) -> APIResponse:
        """获取天气信息"""
        try:
            params = {
                "location": query.get("location", ""),
                "date": query.get("date", ""),
                "days": query.get("days", 7)
            }
            
            cache_key = self._generate_cache_key("weather_forecast", params)
            
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    status_code=200,
                    response_time=0.0,
                    cached=True,
                    source=self.__class__.__name__
                )
            
            url = self._build_url("weather/forecast", params)
            response = await self._make_request(url, method="GET", data=params)
            
            if response.success:
                processed_data = self._process_weather_data(response.data)
                await self._save_to_cache(cache_key, processed_data)
                response.data = processed_data
            
            return response
            
        except Exception as e:
            logger.error(f"天气查询失败: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                response_time=0.0,
                error_message=str(e),
                source=self.__class__.__name__
            )
    
    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        """构建天气API URL"""
        base_url = self.credentials.endpoint_url or "https://api.example-weather.com/v1"
        return f"{base_url}/{endpoint}"
    
    def _process_weather_data(self, raw_data: Any) -> Dict[str, Any]:
        """处理天气数据"""
        if isinstance(raw_data, dict) and "forecast" in raw_data:
            forecast = raw_data["forecast"]
        else:
            forecast = self._generate_mock_weather_data()
        
        return {
            "location": forecast.get("location", "北京"),
            "current": {
                "temperature": forecast.get("current_temp", random.randint(15, 30)),
                "condition": forecast.get("condition", random.choice(["晴", "多云", "小雨", "阴"])),
                "humidity": forecast.get("humidity", random.randint(30, 80)),
                "wind_speed": forecast.get("wind_speed", random.randint(5, 20)),
                "visibility": forecast.get("visibility", random.randint(5, 15))
            },
            "daily_forecast": [
                {
                    "date": f"2024-06-{i+1:02d}",
                    "high_temp": random.randint(20, 35),
                    "low_temp": random.randint(10, 25),
                    "condition": random.choice(["晴", "多云", "小雨", "阴"]),
                    "precipitation_chance": random.randint(0, 80),
                    "wind_speed": random.randint(5, 20)
                }
                for i in range(7)
            ],
            "alerts": forecast.get("alerts", []),
            "air_quality": {
                "aqi": random.randint(50, 200),
                "level": random.choice(["优", "良", "轻度污染", "中度污染"])
            }
        }
    
    def _generate_mock_weather_data(self) -> Dict[str, Any]:
        """生成模拟天气数据"""
        return {
            "location": "北京",
            "current_temp": random.randint(15, 30),
            "condition": random.choice(["晴", "多云", "小雨", "阴"]),
            "humidity": random.randint(30, 80),
            "wind_speed": random.randint(5, 20),
            "visibility": random.randint(5, 15),
            "alerts": []
        }
    
    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = f"{operation}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        return None
    
    async def _save_to_cache(self, key: str, data: Any):
        """保存数据到缓存"""
        pass


class ExchangeRateAPIClient(APIClient):
    """汇率API客户端"""
    
    def __init__(self, credentials: APICredentials):
        super().__init__(credentials, cache_ttl=3600)  # 1小时缓存
        self.data_source_type = DataSourceType.EXCHANGE_RATE
    
    async def search(self, query: Dict[str, Any]) -> APIResponse:
        """获取汇率信息"""
        try:
            params = {
                "base_currency": query.get("base_currency", "USD"),
                "target_currencies": query.get("target_currencies", ["CNY"]),
                "amount": query.get("amount", 1)
            }
            
            cache_key = self._generate_cache_key("exchange_rates", params)
            
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    status_code=200,
                    response_time=0.0,
                    cached=True,
                    source=self.__class__.__name__
                )
            
            url = self._build_url("exchange-rates/latest", params)
            response = await self._make_request(url, method="GET", data=params)
            
            if response.success:
                processed_data = self._process_exchange_rate_data(response.data)
                await self._save_to_cache(cache_key, processed_data)
                response.data = processed_data
            
            return response
            
        except Exception as e:
            logger.error(f"汇率查询失败: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=500,
                response_time=0.0,
                error_message=str(e),
                source=self.__class__.__name__
            )
    
    def _build_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        """构建汇率API URL"""
        base_url = self.credentials.endpoint_url or "https://api.example-exchange.com/v1"
        return f"{base_url}/{endpoint}"
    
    def _process_exchange_rate_data(self, raw_data: Any) -> Dict[str, Any]:
        """处理汇率数据"""
        if isinstance(raw_data, dict) and "rates" in raw_data:
            rates = raw_data["rates"]
        else:
            rates = self._generate_mock_exchange_rates()
        
        return {
            "base_currency": "USD",
            "timestamp": datetime.now().isoformat(),
            "rates": rates,
            "popular_pairs": {
                "USD_CNY": rates.get("CNY", 7.2),
                "EUR_CNY": rates.get("CNY", 7.2) * 1.1,
                "JPY_CNY": rates.get("CNY", 7.2) / 15,
                "GBP_CNY": rates.get("CNY", 7.2) * 1.3
            }
        }
    
    def _generate_mock_exchange_rates(self) -> Dict[str, float]:
        """生成模拟汇率数据"""
        return {
            "CNY": round(random.uniform(7.0, 7.5), 4),
            "EUR": round(random.uniform(0.85, 0.95), 4),
            "GBP": round(random.uniform(0.75, 0.85), 4),
            "JPY": round(random.uniform(110, 130), 2),
            "KRW": round(random.uniform(1100, 1300), 2),
            "HKD": round(random.uniform(7.7, 7.9), 4)
        }
    
    def _generate_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = f"{operation}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        return None
    
    async def _save_to_cache(self, key: str, data: Any):
        """保存数据到缓存"""
        pass


class DataIntegrator:
    """数据集成器 - 统一的外部数据访问接口"""
    
    def __init__(self):
        self.clients: Dict[DataSourceType, APIClient] = {}
        self.cache_manager = CacheManager()
        self.circuit_breaker = CircuitBreaker()
        self.fallback_data = FallbackDataProvider()
        
        # 统计信息
        self.integration_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "fallback_uses": 0
        }
        
        # 初始化客户端
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化API客户端"""
        # 航班API
        flight_credentials = APICredentials(
            api_key=settings.FLIGHT_API_KEY or "demo_key",
            endpoint_url="https://api.flight-search.com/v1",
            rate_limit=1000
        )
        self.clients[DataSourceType.FLIGHT] = FlightAPIClient(flight_credentials)
        
        # 酒店API
        hotel_credentials = APICredentials(
            api_key=settings.HOTEL_API_KEY or "demo_key",
            endpoint_url="https://api.hotel-booking.com/v1",
            rate_limit=500
        )
        self.clients[DataSourceType.HOTEL] = HotelAPIClient(hotel_credentials)
        
        # 天气API
        weather_credentials = APICredentials(
            api_key=settings.WEATHER_API_KEY or "demo_key",
            endpoint_url="https://api.weather-service.com/v1",
            rate_limit=2000
        )
        self.clients[DataSourceType.WEATHER] = WeatherAPIClient(weather_credentials)
        
        # 汇率API
        exchange_credentials = APICredentials(
            api_key=settings.EXCHANGE_RATE_API_KEY or "demo_key",
            endpoint_url="https://api.exchange-rates.com/v1",
            rate_limit=1000
        )
        self.clients[DataSourceType.EXCHANGE_RATE] = ExchangeRateAPIClient(exchange_credentials)
    
    async def search_flights(self, query: Dict[str, Any]) -> APIResponse:
        """搜索航班"""
        return await self._execute_with_fallback(
            DataSourceType.FLIGHT, "search_flights", query
        )
    
    async def search_hotels(self, query: Dict[str, Any]) -> APIResponse:
        """搜索酒店"""
        return await self._execute_with_fallback(
            DataSourceType.HOTEL, "search_hotels", query
        )
    
    async def get_weather_forecast(self, query: Dict[str, Any]) -> APIResponse:
        """获取天气预报"""
        return await self._execute_with_fallback(
            DataSourceType.WEATHER, "get_weather", query
        )
    
    async def get_exchange_rates(self, query: Dict[str, Any]) -> APIResponse:
        """获取汇率"""
        return await self._execute_with_fallback(
            DataSourceType.EXCHANGE_RATE, "get_exchange_rates", query
        )
    
    async def _execute_with_fallback(self, data_source: DataSourceType, 
                                   operation: str, query: Dict[str, Any]) -> APIResponse:
        """带降级的执行请求"""
        self.integration_stats["total_requests"] += 1
        
        try:
            # 检查熔断器状态
            if not self.circuit_breaker.can_execute(data_source):
                logger.warning(f"数据源 {data_source.value} 熔断中，使用降级数据")
                fallback_data = await self.fallback_data.get_fallback_data(data_source, query)
                self.integration_stats["fallback_uses"] += 1
                
                return APIResponse(
                    success=True,
                    data=fallback_data,
                    status_code=200,
                    response_time=0.0,
                    cached=True,
                    source="fallback"
                )
            
            # 获取客户端
            client = self.clients.get(data_source)
            if not client:
                raise ValueError(f"不支持的数据源: {data_source}")
            
            # 执行请求
            response = await client.search(query)
            
            # 更新熔断器状态
            if response.success:
                self.circuit_breaker.record_success(data_source)
                self.integration_stats["successful_requests"] += 1
                
                if response.cached:
                    self.integration_stats["cache_hits"] += 1
            else:
                self.circuit_breaker.record_failure(data_source)
                self.integration_stats["failed_requests"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"数据集成执行失败: {e}")
            self.circuit_breaker.record_failure(data_source)
            self.integration_stats["failed_requests"] += 1
            
            # 使用降级数据
            fallback_data = await self.fallback_data.get_fallback_data(data_source, query)
            self.integration_stats["fallback_uses"] += 1
            
            return APIResponse(
                success=False,
                data=fallback_data,
                status_code=500,
                response_time=0.0,
                error_message=str(e),
                source="fallback"
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取数据集成健康状态"""
        client_health = {}
        for data_source, client in self.clients.items():
            client_health[data_source.value] = client.get_health_status()
        
        total_requests = self.integration_stats["total_requests"]
        success_rate = (
            self.integration_stats["successful_requests"] / total_requests * 100
            if total_requests > 0 else 100
        )
        
        cache_hit_rate = (
            self.integration_stats["cache_hits"] / total_requests * 100
            if total_requests > 0 else 0
        )
        
        return {
            "overall_health": "healthy" if success_rate >= 90 else "degraded",
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "fallback_usage_rate": (
                self.integration_stats["fallback_uses"] / total_requests * 100
                if total_requests > 0 else 0
            ),
            "clients": client_health,
            "circuit_breaker_status": self.circuit_breaker.get_status(),
            "stats": self.integration_stats.copy()
        }


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        # 每个数据源的状态
        self.states: Dict[DataSourceType, Dict[str, Any]] = {}
        
        for data_source in DataSourceType:
            self.states[data_source] = {
                "status": "closed",  # closed, open, half_open
                "failure_count": 0,
                "last_failure_time": None,
                "last_success_time": None
            }
    
    def can_execute(self, data_source: DataSourceType) -> bool:
        """检查是否可以执行请求"""
        state = self.states[data_source]
        
        if state["status"] == "closed":
            return True
        elif state["status"] == "open":
            # 检查是否可以尝试恢复
            if (state["last_failure_time"] and 
                time.time() - state["last_failure_time"] > self.recovery_timeout):
                state["status"] = "half_open"
                return True
            return False
        elif state["status"] == "half_open":
            return True
        
        return False
    
    def record_success(self, data_source: DataSourceType):
        """记录成功请求"""
        state = self.states[data_source]
        state["failure_count"] = 0
        state["last_success_time"] = time.time()
        
        if state["status"] == "half_open":
            state["status"] = "closed"
    
    def record_failure(self, data_source: DataSourceType):
        """记录失败请求"""
        state = self.states[data_source]
        state["failure_count"] += 1
        state["last_failure_time"] = time.time()
        
        if state["failure_count"] >= self.failure_threshold:
            state["status"] = "open"
    
    def get_status(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        status = {}
        for data_source, state in self.states.items():
            status[data_source.value] = {
                "status": state["status"],
                "failure_count": state["failure_count"],
                "healthy": state["status"] == "closed"
            }
        
        return status


class FallbackDataProvider:
    """降级数据提供者"""
    
    def __init__(self):
        self.fallback_cache: Dict[str, Any] = {}
        self._initialize_fallback_data()
    
    def _initialize_fallback_data(self):
        """初始化降级数据"""
        self.fallback_cache = {
            "flight_search": {
                "message": "航班搜索服务暂时不可用，请稍后重试",
                "suggestions": ["使用其他时间搜索", "考虑高铁等替代交通方式"]
            },
            "hotel_search": {
                "message": "酒店搜索服务暂时不可用，请稍后重试", 
                "suggestions": ["尝试其他住宿平台", "联系当地旅行社"]
            },
            "weather": {
                "location": "未知",
                "current": {
                    "temperature": 20,
                    "condition": "数据暂不可用",
                    "message": "天气服务暂时不可用，请查看其他天气应用"
                }
            },
            "exchange_rates": {
                "base_currency": "USD",
                "rates": {"CNY": 7.2},
                "message": "汇率服务暂时不可用，显示近期平均汇率"
            }
        }
    
    async def get_fallback_data(self, data_source: DataSourceType, 
                               query: Dict[str, Any]) -> Any:
        """获取降级数据"""
        if data_source == DataSourceType.FLIGHT:
            return self.fallback_cache["flight_search"]
        elif data_source == DataSourceType.HOTEL:
            return self.fallback_cache["hotel_search"]
        elif data_source == DataSourceType.WEATHER:
            return self.fallback_cache["weather"]
        elif data_source == DataSourceType.EXCHANGE_RATE:
            return self.fallback_cache["exchange_rates"]
        else:
            return {"message": "服务暂时不可用，请稍后重试"}


# 全局数据集成器实例
_data_integrator: Optional[DataIntegrator] = None


def get_data_integrator() -> DataIntegrator:
    """获取数据集成器实例"""
    global _data_integrator
    if _data_integrator is None:
        _data_integrator = DataIntegrator()
    return _data_integrator 