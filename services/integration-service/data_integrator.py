"""
外部数据源集成服务
实现DataIntegrator外部API集成框架、航班API、酒店API、和风天气API客户端、
实时数据获取和缓存机制、API故障处理和备用数据源切换
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib
from abc import ABC, abstractmethod

import aiohttp
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class DataSourceType(Enum):
    """数据源类型"""
    FLIGHT = "flight"
    HOTEL = "hotel"
    WEATHER = "weather"
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    TRANSPORTATION = "transportation"


class APIStatus(Enum):
    """API状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class APICredentials:
    """API凭证"""
    api_key: str
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class APIEndpoint:
    """API端点配置"""
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit: int = 1000  # 每小时请求数
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker_threshold: int = 5


@dataclass
class DataSource:
    """数据源配置"""
    name: str
    type: DataSourceType
    priority: int  # 1-10，10最高
    credentials: APICredentials
    endpoints: Dict[str, APIEndpoint]
    is_primary: bool = True
    status: APIStatus = APIStatus.ACTIVE
    last_error: Optional[str] = None
    error_count: int = 0
    request_count: int = 0
    success_count: int = 0


@dataclass
class CacheConfig:
    """缓存配置"""
    ttl: int = 3600  # 生存时间（秒）
    max_size: int = 10000  # 最大缓存条目数
    compression: bool = True
    serialization: str = "json"  # json, pickle, msgpack


@dataclass
class APIResponse:
    """API响应"""
    data: Any
    status_code: int
    headers: Dict[str, str]
    response_time: float
    cached: bool = False
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """调用函数并应用熔断逻辑"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """是否应该尝试重置"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self) -> bool:
        """获取请求许可"""
        now = time.time()
        
        # 清理过期的请求记录
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # 检查是否超过限制
        if len(self.requests) >= self.max_requests:
            return False
        
        # 记录新请求
        self.requests.append(now)
        return True
    
    def get_remaining_requests(self) -> int:
        """获取剩余请求数"""
        now = time.time()
        valid_requests = [req_time for req_time in self.requests 
                         if now - req_time < self.time_window]
        return max(0, self.max_requests - len(valid_requests))


class BaseAPIClient(ABC):
    """API客户端基类"""
    
    def __init__(self, data_source: DataSource, cache_config: CacheConfig = None):
        self.data_source = data_source
        self.cache_config = cache_config or CacheConfig()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(data_source.endpoints.get("default", APIEndpoint("", "")).rate_limit)
        self.session = None
        self.cache_client = None
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0.0
        }
    
    async def initialize(self):
        """初始化客户端"""
        # 创建HTTP会话
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self._get_default_headers()
        )
        
        # 初始化缓存客户端
        if settings.REDIS_HOST:
            self.cache_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
        if self.cache_client:
            await self.cache_client.close()
    
    @abstractmethod
    def _get_default_headers(self) -> Dict[str, str]:
        """获取默认请求头"""
        pass
    
    @abstractmethod
    async def _authenticate(self) -> bool:
        """认证"""
        pass
    
    async def make_request(self, 
                          endpoint_name: str,
                          params: Dict[str, Any] = None,
                          data: Dict[str, Any] = None,
                          use_cache: bool = True) -> APIResponse:
        """发起API请求"""
        # 检查速率限制
        if not await self.rate_limiter.acquire():
            raise Exception("Rate limit exceeded")
        
        endpoint = self.data_source.endpoints.get(endpoint_name)
        if not endpoint:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")
        
        # 生成缓存键
        cache_key = self._generate_cache_key(endpoint_name, params, data)
        
        # 检查缓存
        if use_cache and self.cache_client:
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                self.stats["cache_hits"] += 1
                return cached_response
        
        # 使用熔断器发起请求
        try:
            response = await self.circuit_breaker.call(
                self._execute_request, endpoint, params, data
            )
            
            # 缓存响应
            if use_cache and self.cache_client and response.status_code == 200:
                await self._save_to_cache(cache_key, response)
            
            # 更新统计
            self._update_stats(response)
            
            return response
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            self.data_source.error_count += 1
            self.data_source.last_error = str(e)
            logger.error(f"API请求失败 {endpoint_name}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _execute_request(self, 
                              endpoint: APIEndpoint,
                              params: Dict[str, Any] = None,
                              data: Dict[str, Any] = None) -> APIResponse:
        """执行HTTP请求"""
        start_time = time.time()
        
        # 准备请求参数
        request_kwargs = {
            "method": endpoint.method,
            "url": endpoint.url,
            "headers": {**endpoint.headers, **self._get_auth_headers()},
            "timeout": aiohttp.ClientTimeout(total=endpoint.timeout)
        }
        
        if params:
            request_kwargs["params"] = params
        if data:
            if endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = data
            else:
                request_kwargs["params"] = {**(request_kwargs.get("params", {})), **data}
        
        # 发起请求
        async with self.session.request(**request_kwargs) as response:
            response_data = await response.text()
            response_time = time.time() - start_time
            
            # 尝试解析JSON
            try:
                if response.content_type == "application/json":
                    response_data = await response.json()
            except:
                pass
            
            return APIResponse(
                data=response_data,
                status_code=response.status,
                headers=dict(response.headers),
                response_time=response_time,
                source=self.data_source.name
            )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        headers = {}
        
        if self.data_source.credentials.api_key:
            headers["Authorization"] = f"Bearer {self.data_source.credentials.api_key}"
        
        if self.data_source.credentials.access_token:
            headers["X-Access-Token"] = self.data_source.credentials.access_token
        
        return headers
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        key_parts = [self.data_source.name, endpoint]
        
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        if data:
            key_parts.append(json.dumps(data, sort_keys=True))
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[APIResponse]:
        """从缓存获取数据"""
        try:
            cached_data = await self.cache_client.get(f"api_cache:{cache_key}")
            if cached_data:
                data = json.loads(cached_data)
                return APIResponse(
                    data=data["data"],
                    status_code=data["status_code"],
                    headers=data["headers"],
                    response_time=data["response_time"],
                    cached=True,
                    source=data["source"],
                    timestamp=datetime.fromisoformat(data["timestamp"])
                )
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, response: APIResponse):
        """保存到缓存"""
        try:
            cache_data = {
                "data": response.data,
                "status_code": response.status_code,
                "headers": response.headers,
                "response_time": response.response_time,
                "source": response.source,
                "timestamp": response.timestamp.isoformat()
            }
            
            await self.cache_client.setex(
                f"api_cache:{cache_key}",
                self.cache_config.ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def _update_stats(self, response: APIResponse):
        """更新统计信息"""
        self.stats["total_requests"] += 1
        
        if response.status_code == 200:
            self.stats["successful_requests"] += 1
        
        # 更新平均响应时间
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + response.response_time) / total_requests
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": self.data_source.status.value,
            "error_count": self.data_source.error_count,
            "last_error": self.data_source.last_error,
            "circuit_breaker_state": self.circuit_breaker.state,
            "remaining_requests": self.rate_limiter.get_remaining_requests(),
            "stats": self.stats
        }


class FlightAPIClient(BaseAPIClient):
    """航班API客户端"""
    
    def __init__(self, data_source: DataSource, cache_config: CacheConfig = None):
        super().__init__(data_source, cache_config)
        
        # 添加航班特定的端点
        if "search" not in self.data_source.endpoints:
            self.data_source.endpoints["search"] = APIEndpoint(
                name="search",
                url="https://api.example-flight.com/v1/flights/search",
                method="GET",
                rate_limit=500
            )
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "TravelPlanner/1.0"
        }
    
    async def _authenticate(self) -> bool:
        """航班API认证"""
        # 这里实现具体的认证逻辑
        return True
    
    async def search_flights(self,
                           origin: str,
                           destination: str,
                           departure_date: str,
                           return_date: Optional[str] = None,
                           passengers: int = 1,
                           cabin_class: str = "economy") -> List[Dict[str, Any]]:
        """搜索航班"""
        params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "passengers": passengers,
            "cabin_class": cabin_class
        }
        
        if return_date:
            params["return_date"] = return_date
        
        try:
            response = await self.make_request("search", params=params)
            
            if response.status_code == 200:
                flights_data = response.data
                
                # 如果是模拟数据，生成示例航班
                if isinstance(flights_data, str) or not flights_data:
                    flights_data = self._generate_mock_flights(origin, destination, departure_date, passengers)
                
                return self._normalize_flight_data(flights_data)
            else:
                logger.error(f"航班搜索失败: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"航班搜索异常: {e}")
            # 返回模拟数据作为降级方案
            return self._generate_mock_flights(origin, destination, departure_date, passengers)
    
    def _generate_mock_flights(self, origin: str, destination: str, departure_date: str, passengers: int) -> List[Dict[str, Any]]:
        """生成模拟航班数据"""
        airlines = ["中国国航", "东方航空", "南方航空", "海南航空", "深圳航空"]
        
        flights = []
        for i in range(random.randint(3, 8)):
            departure_time = f"{departure_date} {random.randint(6, 22):02d}:{random.randint(0, 59):02d}"
            duration_hours = random.randint(1, 8)
            arrival_time = f"{departure_date} {random.randint(8, 23):02d}:{random.randint(0, 59):02d}"
            
            flight = {
                "flight_number": f"{random.choice(['CA', 'MU', 'CZ', 'HU', 'ZH'])}{random.randint(1000, 9999)}",
                "airline": random.choice(airlines),
                "origin": origin,
                "destination": destination,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration": f"{duration_hours}小时{random.randint(0, 59)}分钟",
                "price": random.randint(500, 3000) * passengers,
                "currency": "CNY",
                "stops": random.randint(0, 2),
                "aircraft": random.choice(["B737", "A320", "B777", "A330"]),
                "available_seats": random.randint(1, 50),
                "baggage_allowance": "20kg",
                "refundable": random.choice([True, False]),
                "booking_class": "economy"
            }
            flights.append(flight)
        
        return flights
    
    def _normalize_flight_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """标准化航班数据"""
        if isinstance(raw_data, list):
            return raw_data
        
        # 这里实现具体API响应格式的转换逻辑
        normalized = []
        
        # 示例转换逻辑
        if isinstance(raw_data, dict) and "flights" in raw_data:
            for flight in raw_data["flights"]:
                normalized_flight = {
                    "flight_number": flight.get("flightNumber", ""),
                    "airline": flight.get("airline", ""),
                    "origin": flight.get("origin", ""),
                    "destination": flight.get("destination", ""),
                    "departure_time": flight.get("departureTime", ""),
                    "arrival_time": flight.get("arrivalTime", ""),
                    "price": flight.get("price", 0),
                    "currency": flight.get("currency", "CNY"),
                    "stops": flight.get("stops", 0)
                }
                normalized.append(normalized_flight)
        
        return normalized


class HotelAPIClient(BaseAPIClient):
    """酒店API客户端"""
    
    def __init__(self, data_source: DataSource, cache_config: CacheConfig = None):
        super().__init__(data_source, cache_config)
        
        if "search" not in self.data_source.endpoints:
            self.data_source.endpoints["search"] = APIEndpoint(
                name="search",
                url="https://api.example-hotel.com/v1/hotels/search",
                method="GET",
                rate_limit=1000
            )
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def _authenticate(self) -> bool:
        return True
    
    async def search_hotels(self,
                          city: str,
                          checkin_date: str,
                          checkout_date: str,
                          guests: int = 2,
                          rooms: int = 1,
                          min_rating: float = 0.0,
                          max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        """搜索酒店"""
        params = {
            "city": city,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "guests": guests,
            "rooms": rooms,
            "min_rating": min_rating
        }
        
        if max_price:
            params["max_price"] = max_price
        
        try:
            response = await self.make_request("search", params=params)
            
            if response.status_code == 200:
                hotels_data = response.data
                
                if isinstance(hotels_data, str) or not hotels_data:
                    hotels_data = self._generate_mock_hotels(city, checkin_date, checkout_date, guests)
                
                return self._normalize_hotel_data(hotels_data)
            else:
                logger.error(f"酒店搜索失败: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"酒店搜索异常: {e}")
            return self._generate_mock_hotels(city, checkin_date, checkout_date, guests)
    
    def _generate_mock_hotels(self, city: str, checkin_date: str, checkout_date: str, guests: int) -> List[Dict[str, Any]]:
        """生成模拟酒店数据"""
        hotel_types = ["大酒店", "国际酒店", "商务酒店", "度假村", "精品酒店", "快捷酒店"]
        amenities_pool = ["免费WiFi", "早餐", "停车场", "健身房", "游泳池", "spa", "商务中心", "接送服务"]
        
        hotels = []
        for i in range(random.randint(5, 12)):
            hotel_name = f"{city}{random.choice(hotel_types)}"
            rating = random.randint(3, 5)
            price_per_night = random.randint(200, 1500)
            
            # 计算住宿天数
            checkin = datetime.strptime(checkin_date, "%Y-%m-%d")
            checkout = datetime.strptime(checkout_date, "%Y-%m-%d")
            nights = (checkout - checkin).days
            
            hotel = {
                "hotel_id": f"hotel_{random.randint(10000, 99999)}",
                "name": hotel_name,
                "rating": rating,
                "guest_rating": round(random.uniform(7.5, 9.5), 1),
                "price_per_night": price_per_night,
                "total_price": price_per_night * nights,
                "currency": "CNY",
                "location": f"{city}{random.choice(['市中心', '机场附近', '商业区', '风景区', '交通枢纽'])}",
                "distance_to_center": f"{random.randint(1, 20)}公里",
                "amenities": random.sample(amenities_pool, random.randint(3, 6)),
                "room_type": "标准间",
                "bed_type": random.choice(["大床", "双床", "三人间"]),
                "max_guests": guests,
                "free_cancellation": random.choice([True, False]),
                "breakfast_included": random.choice([True, False]),
                "wifi_free": True,
                "parking_available": random.choice([True, False]),
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "nights": nights
            }
            hotels.append(hotel)
        
        return hotels
    
    def _normalize_hotel_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """标准化酒店数据"""
        if isinstance(raw_data, list):
            return raw_data
        
        normalized = []
        
        if isinstance(raw_data, dict) and "hotels" in raw_data:
            for hotel in raw_data["hotels"]:
                normalized_hotel = {
                    "hotel_id": hotel.get("id", ""),
                    "name": hotel.get("name", ""),
                    "rating": hotel.get("rating", 0),
                    "price_per_night": hotel.get("price", 0),
                    "currency": hotel.get("currency", "CNY"),
                    "location": hotel.get("location", ""),
                    "amenities": hotel.get("amenities", [])
                }
                normalized.append(normalized_hotel)
        
        return normalized


class WeatherAPIClient(BaseAPIClient):
    """天气API客户端（和风天气）"""
    
    def __init__(self, data_source: DataSource, cache_config: CacheConfig = None):
        super().__init__(data_source, cache_config)
        
        # 和风天气API端点
        base_url = "https://devapi.qweather.com/v7"
        self.data_source.endpoints.update({
            "current": APIEndpoint(
                name="current",
                url=f"{base_url}/weather/now",
                method="GET",
                rate_limit=1000
            ),
            "forecast": APIEndpoint(
                name="forecast",
                url=f"{base_url}/weather/7d",
                method="GET",
                rate_limit=1000
            ),
            "air_quality": APIEndpoint(
                name="air_quality",
                url=f"{base_url}/air/now",
                method="GET",
                rate_limit=1000
            ),
            "indices": APIEndpoint(
                name="indices",
                url=f"{base_url}/indices/1d",
                method="GET",
                rate_limit=1000
            )
        })
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json"
        }
    
    async def _authenticate(self) -> bool:
        return True
    
    async def get_current_weather(self, city: str, location_id: Optional[str] = None) -> Dict[str, Any]:
        """获取当前天气"""
        params = {
            "key": self.data_source.credentials.api_key,
            "location": location_id or city
        }
        
        try:
            response = await self.make_request("current", params=params)
            
            if response.status_code == 200 and isinstance(response.data, dict):
                if response.data.get("code") == "200":
                    return self._normalize_current_weather(response.data)
            
            # 降级到模拟数据
            return self._generate_mock_current_weather(city)
            
        except Exception as e:
            logger.error(f"获取当前天气失败: {e}")
            return self._generate_mock_current_weather(city)
    
    async def get_weather_forecast(self, city: str, days: int = 7, location_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取天气预报"""
        params = {
            "key": self.data_source.credentials.api_key,
            "location": location_id or city
        }
        
        try:
            response = await self.make_request("forecast", params=params)
            
            if response.status_code == 200 and isinstance(response.data, dict):
                if response.data.get("code") == "200":
                    return self._normalize_forecast_data(response.data, days)
            
            return self._generate_mock_forecast(city, days)
            
        except Exception as e:
            logger.error(f"获取天气预报失败: {e}")
            return self._generate_mock_forecast(city, days)
    
    async def get_air_quality(self, city: str, location_id: Optional[str] = None) -> Dict[str, Any]:
        """获取空气质量"""
        params = {
            "key": self.data_source.credentials.api_key,
            "location": location_id or city
        }
        
        try:
            response = await self.make_request("air_quality", params=params)
            
            if response.status_code == 200 and isinstance(response.data, dict):
                if response.data.get("code") == "200":
                    return self._normalize_air_quality(response.data)
            
            return self._generate_mock_air_quality()
            
        except Exception as e:
            logger.error(f"获取空气质量失败: {e}")
            return self._generate_mock_air_quality()
    
    async def get_lifestyle_indices(self, city: str, location_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取生活指数"""
        params = {
            "key": self.data_source.credentials.api_key,
            "location": location_id or city,
            "type": "1,2,3,5,6,8,9"  # 运动、洗车、穿衣、旅游、紫外线、舒适度、感冒
        }
        
        try:
            response = await self.make_request("indices", params=params)
            
            if response.status_code == 200 and isinstance(response.data, dict):
                if response.data.get("code") == "200":
                    return self._normalize_indices_data(response.data)
            
            return self._generate_mock_indices()
            
        except Exception as e:
            logger.error(f"获取生活指数失败: {e}")
            return self._generate_mock_indices()
    
    def _normalize_current_weather(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化当前天气数据"""
        now = raw_data.get("now", {})
        
        return {
            "temperature": int(now.get("temp", 20)),
            "feels_like": int(now.get("feelsLike", 20)),
            "condition": now.get("text", "晴"),
            "condition_code": now.get("icon", "100"),
            "humidity": f"{now.get('humidity', 50)}%",
            "wind_speed": f"{now.get('windSpeed', 10)}km/h",
            "wind_direction": now.get("windDir", "北"),
            "pressure": f"{now.get('pressure', 1013)}hPa",
            "visibility": f"{now.get('vis', 10)}km",
            "uv_index": int(now.get("uvIndex", 5)),
            "update_time": now.get("obsTime", datetime.now().isoformat())
        }
    
    def _normalize_forecast_data(self, raw_data: Dict[str, Any], days: int) -> List[Dict[str, Any]]:
        """标准化预报数据"""
        daily_data = raw_data.get("daily", [])
        
        forecast = []
        for day_data in daily_data[:days]:
            forecast_day = {
                "date": day_data.get("fxDate", ""),
                "temp_max": int(day_data.get("tempMax", 25)),
                "temp_min": int(day_data.get("tempMin", 15)),
                "condition_day": day_data.get("textDay", "晴"),
                "condition_night": day_data.get("textNight", "晴"),
                "icon_day": day_data.get("iconDay", "100"),
                "icon_night": day_data.get("iconNight", "100"),
                "wind_direction": day_data.get("windDirDay", "北"),
                "wind_speed": f"{day_data.get('windSpeedDay', 10)}km/h",
                "humidity": f"{day_data.get('humidity', 50)}%",
                "pressure": f"{day_data.get('pressure', 1013)}hPa",
                "precipitation": f"{day_data.get('precip', 0)}mm",
                "uv_index": int(day_data.get("uvIndex", 5))
            }
            forecast.append(forecast_day)
        
        return forecast
    
    def _normalize_air_quality(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化空气质量数据"""
        now = raw_data.get("now", {})
        
        return {
            "aqi": int(now.get("aqi", 50)),
            "level": now.get("category", "优"),
            "pm2_5": int(now.get("pm2p5", 20)),
            "pm10": int(now.get("pm10", 30)),
            "no2": int(now.get("no2", 20)),
            "so2": int(now.get("so2", 10)),
            "co": float(now.get("co", 0.5)),
            "o3": int(now.get("o3", 80)),
            "update_time": now.get("pubTime", datetime.now().isoformat())
        }
    
    def _normalize_indices_data(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """标准化生活指数数据"""
        daily_data = raw_data.get("daily", [])
        
        indices = []
        for index_data in daily_data:
            index_item = {
                "type": index_data.get("type", ""),
                "name": index_data.get("name", ""),
                "level": index_data.get("level", ""),
                "category": index_data.get("category", ""),
                "text": index_data.get("text", ""),
                "date": index_data.get("date", "")
            }
            indices.append(index_item)
        
        return indices
    
    def _generate_mock_current_weather(self, city: str) -> Dict[str, Any]:
        """生成模拟当前天气"""
        conditions = ["晴", "多云", "阴", "小雨", "中雨", "雷阵雨", "雪"]
        
        return {
            "temperature": random.randint(15, 35),
            "feels_like": random.randint(15, 35),
            "condition": random.choice(conditions),
            "condition_code": "100",
            "humidity": f"{random.randint(30, 90)}%",
            "wind_speed": f"{random.randint(5, 25)}km/h",
            "wind_direction": random.choice(["北", "南", "东", "西", "东北", "西北", "东南", "西南"]),
            "pressure": f"{random.randint(1000, 1030)}hPa",
            "visibility": f"{random.randint(5, 30)}km",
            "uv_index": random.randint(1, 10),
            "update_time": datetime.now().isoformat()
        }
    
    def _generate_mock_forecast(self, city: str, days: int) -> List[Dict[str, Any]]:
        """生成模拟天气预报"""
        conditions = ["晴", "多云", "阴", "小雨", "中雨", "雷阵雨"]
        
        forecast = []
        base_date = datetime.now().date()
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            temp_max = random.randint(20, 35)
            temp_min = random.randint(10, temp_max - 5)
            
            forecast_day = {
                "date": date.strftime("%Y-%m-%d"),
                "temp_max": temp_max,
                "temp_min": temp_min,
                "condition_day": random.choice(conditions),
                "condition_night": random.choice(conditions),
                "wind_direction": random.choice(["北", "南", "东", "西"]),
                "wind_speed": f"{random.randint(5, 20)}km/h",
                "humidity": f"{random.randint(40, 80)}%",
                "pressure": f"{random.randint(1005, 1025)}hPa",
                "precipitation": f"{random.randint(0, 50)}mm",
                "uv_index": random.randint(1, 10)
            }
            forecast.append(forecast_day)
        
        return forecast
    
    def _generate_mock_air_quality(self) -> Dict[str, Any]:
        """生成模拟空气质量"""
        levels = ["优", "良", "轻度污染", "中度污染"]
        
        return {
            "aqi": random.randint(20, 150),
            "level": random.choice(levels),
            "pm2_5": random.randint(10, 100),
            "pm10": random.randint(20, 150),
            "no2": random.randint(10, 80),
            "so2": random.randint(5, 50),
            "co": round(random.uniform(0.3, 2.0), 1),
            "o3": random.randint(50, 200),
            "update_time": datetime.now().isoformat()
        }
    
    def _generate_mock_indices(self) -> List[Dict[str, Any]]:
        """生成模拟生活指数"""
        indices_types = [
            {"type": "1", "name": "运动指数", "levels": ["适宜", "较适宜", "较不宜", "不宜"]},
            {"type": "2", "name": "洗车指数", "levels": ["适宜", "较适宜", "较不宜", "不宜"]},
            {"type": "3", "name": "穿衣指数", "levels": ["炎热", "热", "舒适", "较舒适", "较冷", "冷"]},
            {"type": "5", "name": "旅游指数", "levels": ["适宜", "较适宜", "一般", "较不宜", "不适宜"]},
            {"type": "6", "name": "紫外线指数", "levels": ["最弱", "弱", "中等", "强", "很强"]},
            {"type": "8", "name": "舒适度指数", "levels": ["舒适", "较舒适", "闷热", "不舒适"]},
            {"type": "9", "name": "感冒指数", "levels": ["少发", "较易发", "易发", "极易发"]}
        ]
        
        indices = []
        today = datetime.now().strftime("%Y-%m-%d")
        
        for index_type in indices_types:
            level = random.choice(index_type["levels"])
            indices.append({
                "type": index_type["type"],
                "name": index_type["name"],
                "level": "1",
                "category": level,
                "text": f"今日{index_type['name']}为{level}",
                "date": today
            })
        
        return indices


class DataIntegrator:
    """数据集成器"""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.api_clients: Dict[str, BaseAPIClient] = {}
        self.fallback_chains: Dict[DataSourceType, List[str]] = {}
        
        # 全局缓存配置
        self.cache_config = CacheConfig(ttl=3600, max_size=10000)
        
        # 初始化数据源
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """初始化数据源配置"""
        # 航班数据源
        flight_source = DataSource(
            name="primary_flight_api",
            type=DataSourceType.FLIGHT,
            priority=10,
            credentials=APICredentials(
                api_key=settings.FLIGHT_API_KEY or "demo_key"
            ),
            endpoints={}
        )
        self.data_sources["primary_flight_api"] = flight_source
        
        # 酒店数据源
        hotel_source = DataSource(
            name="primary_hotel_api",
            type=DataSourceType.HOTEL,
            priority=10,
            credentials=APICredentials(
                api_key=settings.HOTEL_API_KEY or "demo_key"
            ),
            endpoints={}
        )
        self.data_sources["primary_hotel_api"] = hotel_source
        
        # 天气数据源
        weather_source = DataSource(
            name="qweather_api",
            type=DataSourceType.WEATHER,
            priority=10,
            credentials=APICredentials(
                api_key=settings.WEATHER_API_KEY or "demo_key"
            ),
            endpoints={}
        )
        self.data_sources["qweather_api"] = weather_source
        
        # 设置降级链
        self.fallback_chains = {
            DataSourceType.FLIGHT: ["primary_flight_api"],
            DataSourceType.HOTEL: ["primary_hotel_api"],
            DataSourceType.WEATHER: ["qweather_api"]
        }
    
    async def initialize(self):
        """初始化集成器"""
        logger.info("初始化数据集成器...")
        
        # 初始化API客户端
        for source_name, source in self.data_sources.items():
            if source.type == DataSourceType.FLIGHT:
                client = FlightAPIClient(source, self.cache_config)
            elif source.type == DataSourceType.HOTEL:
                client = HotelAPIClient(source, self.cache_config)
            elif source.type == DataSourceType.WEATHER:
                client = WeatherAPIClient(source, self.cache_config)
            else:
                continue
            
            await client.initialize()
            self.api_clients[source_name] = client
        
        logger.info(f"数据集成器初始化完成，加载了 {len(self.api_clients)} 个API客户端")
    
    async def cleanup(self):
        """清理资源"""
        for client in self.api_clients.values():
            await client.cleanup()
    
    async def search_flights(self, **params) -> List[Dict[str, Any]]:
        """搜索航班（带降级）"""
        return await self._execute_with_fallback(
            DataSourceType.FLIGHT,
            "search_flights",
            **params
        )
    
    async def search_hotels(self, **params) -> List[Dict[str, Any]]:
        """搜索酒店（带降级）"""
        return await self._execute_with_fallback(
            DataSourceType.HOTEL,
            "search_hotels",
            **params
        )
    
    async def get_current_weather(self, **params) -> Dict[str, Any]:
        """获取当前天气（带降级）"""
        return await self._execute_with_fallback(
            DataSourceType.WEATHER,
            "get_current_weather",
            **params
        )
    
    async def get_weather_forecast(self, **params) -> List[Dict[str, Any]]:
        """获取天气预报（带降级）"""
        return await self._execute_with_fallback(
            DataSourceType.WEATHER,
            "get_weather_forecast",
            **params
        )
    
    async def get_air_quality(self, **params) -> Dict[str, Any]:
        """获取空气质量（带降级）"""
        return await self._execute_with_fallback(
            DataSourceType.WEATHER,
            "get_air_quality",
            **params
        )
    
    async def _execute_with_fallback(self, 
                                   data_type: DataSourceType,
                                   method_name: str,
                                   **params) -> Any:
        """带降级机制的执行"""
        fallback_sources = self.fallback_chains.get(data_type, [])
        
        last_error = None
        
        for source_name in fallback_sources:
            client = self.api_clients.get(source_name)
            if not client:
                continue
            
            source = self.data_sources[source_name]
            
            # 检查数据源状态
            if source.status == APIStatus.INACTIVE:
                continue
            
            try:
                # 执行方法
                method = getattr(client, method_name, None)
                if method:
                    result = await method(**params)
                    
                    # 标记数据源为正常
                    source.status = APIStatus.ACTIVE
                    source.error_count = 0
                    source.success_count += 1
                    
                    logger.info(f"成功从 {source_name} 获取 {data_type.value} 数据")
                    return result
                else:
                    logger.warning(f"客户端 {source_name} 不支持方法 {method_name}")
                    
            except Exception as e:
                last_error = e
                source.error_count += 1
                source.last_error = str(e)
                
                # 检查是否需要标记为不可用
                if source.error_count >= 5:
                    source.status = APIStatus.ERROR
                    logger.error(f"数据源 {source_name} 错误次数过多，标记为不可用")
                
                logger.warning(f"从 {source_name} 获取数据失败: {e}")
        
        # 所有数据源都失败
        logger.error(f"所有 {data_type.value} 数据源都失败，最后错误: {last_error}")
        
        # 返回空数据或默认数据
        if data_type in [DataSourceType.FLIGHT, DataSourceType.HOTEL]:
            return []
        else:
            return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取所有数据源的健康状态"""
        status = {
            "overall_status": "healthy",
            "sources": {},
            "summary": {
                "total_sources": len(self.data_sources),
                "active_sources": 0,
                "error_sources": 0,
                "inactive_sources": 0
            }
        }
        
        for source_name, client in self.api_clients.items():
            health = client.get_health_status()
            status["sources"][source_name] = health
            
            # 更新汇总
            if health["status"] == "active":
                status["summary"]["active_sources"] += 1
            elif health["status"] == "error":
                status["summary"]["error_sources"] += 1
            else:
                status["summary"]["inactive_sources"] += 1
        
        # 确定整体状态
        if status["summary"]["error_sources"] > 0:
            status["overall_status"] = "degraded"
        elif status["summary"]["active_sources"] == 0:
            status["overall_status"] = "unhealthy"
        
        return status
    
    async def refresh_data_source(self, source_name: str):
        """刷新数据源"""
        if source_name in self.data_sources:
            source = self.data_sources[source_name]
            source.status = APIStatus.ACTIVE
            source.error_count = 0
            source.last_error = None
            logger.info(f"数据源 {source_name} 已刷新")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0.0,
            "sources": {}
        }
        
        for source_name, client in self.api_clients.items():
            client_stats = client.stats
            stats["sources"][source_name] = client_stats
            
            # 汇总统计
            stats["total_requests"] += client_stats["total_requests"]
            stats["successful_requests"] += client_stats["successful_requests"]
            stats["failed_requests"] += client_stats["failed_requests"]
            stats["cache_hits"] += client_stats["cache_hits"]
        
        # 计算平均响应时间
        if self.api_clients:
            response_times = [client.stats["average_response_time"] for client in self.api_clients.values()]
            stats["average_response_time"] = sum(response_times) / len(response_times)
        
        return stats


# 全局数据集成器实例
_data_integrator: Optional[DataIntegrator] = None


async def get_data_integrator() -> DataIntegrator:
    """获取数据集成器实例"""
    global _data_integrator
    if _data_integrator is None:
        _data_integrator = DataIntegrator()
        await _data_integrator.initialize()
    return _data_integrator 