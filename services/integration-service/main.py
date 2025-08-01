"""
外部数据源集成服务
实现DataIntegrator外部API集成框架、航班API、酒店API、天气API客户端、实时数据获取和缓存机制、API故障处理和备用数据源切换
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import hashlib

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential

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


class APIStatus(Enum):
    """API状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


class CacheStrategy(Enum):
    """缓存策略"""
    NO_CACHE = "no_cache"
    SHORT_TERM = "short_term"      # 5分钟
    MEDIUM_TERM = "medium_term"    # 1小时
    LONG_TERM = "long_term"        # 24小时
    PERSISTENT = "persistent"       # 7天


@dataclass
class APIEndpoint:
    """API端点配置"""
    name: str
    base_url: str
    api_key: Optional[str]
    headers: Dict[str, str]
    rate_limit: int  # 每分钟请求数
    timeout: int     # 超时时间（秒）
    retry_attempts: int
    priority: int    # 优先级，数字越小优先级越高
    status: APIStatus = APIStatus.ACTIVE


@dataclass
class DataRequest:
    """数据请求"""
    request_id: str
    source_type: DataSourceType
    endpoint: str
    params: Dict[str, Any]
    cache_strategy: CacheStrategy
    timestamp: datetime
    user_id: Optional[str] = None


@dataclass
class DataResponse:
    """数据响应"""
    request_id: str
    success: bool
    data: Any
    source: str
    cached: bool
    timestamp: datetime
    cache_expires: Optional[datetime] = None
    error_message: Optional[str] = None


# Pydantic模型
class FlightSearchRequest(BaseModel):
    """航班搜索请求"""
    origin: str = Field(..., description="出发地机场代码")
    destination: str = Field(..., description="目的地机场代码")
    departure_date: str = Field(..., description="出发日期 YYYY-MM-DD")
    return_date: Optional[str] = Field(None, description="返回日期 YYYY-MM-DD")
    passengers: int = Field(1, ge=1, le=9, description="乘客数量")
    cabin_class: str = Field("economy", description="舱位等级")


class HotelSearchRequest(BaseModel):
    """酒店搜索请求"""
    city: str = Field(..., description="城市名称")
    check_in: str = Field(..., description="入住日期 YYYY-MM-DD")
    check_out: str = Field(..., description="退房日期 YYYY-MM-DD")
    guests: int = Field(2, ge=1, le=10, description="客人数量")
    rooms: int = Field(1, ge=1, le=5, description="房间数量")
    price_min: Optional[float] = Field(None, description="最低价格")
    price_max: Optional[float] = Field(None, description="最高价格")


class WeatherRequest(BaseModel):
    """天气查询请求"""
    city: str = Field(..., description="城市名称")
    days: int = Field(7, ge=1, le=14, description="预报天数")
    include_hourly: bool = Field(False, description="是否包含小时预报")


class ExchangeRateRequest(BaseModel):
    """汇率查询请求"""
    from_currency: str = Field(..., description="源货币代码")
    to_currency: str = Field(..., description="目标货币代码")
    amount: float = Field(1.0, description="金额")


class BaseAPIClient:
    """基础API客户端"""
    
    def __init__(self, endpoint: APIEndpoint):
        self.endpoint = endpoint
        self.session = None
        self.last_request_time = None
        self.request_count = 0
        self.request_window_start = datetime.now()
    
    async def initialize(self):
        """初始化客户端"""
        self.session = httpx.AsyncClient(
            base_url=self.endpoint.base_url,
            headers=self.endpoint.headers,
            timeout=self.endpoint.timeout
        )
    
    async def close(self):
        """关闭客户端"""
        if self.session:
            await self.session.aclose()
    
    async def _check_rate_limit(self):
        """检查速率限制"""
        now = datetime.now()
        
        # 重置计数器（每分钟）
        if (now - self.request_window_start).total_seconds() >= 60:
            self.request_count = 0
            self.request_window_start = now
        
        # 检查是否超过限制
        if self.request_count >= self.endpoint.rate_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {self.endpoint.name}"
            )
        
        self.request_count += 1
        self.last_request_time = now
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发起HTTP请求"""
        if not self.session:
            await self.initialize()
        
        await self._check_rate_limit()
        
        try:
            response = await self.session.request(
                method=method,
                url=path,
                params=params,
                json=data
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            if e.response.status_code == 429:
                self.endpoint.status = APIStatus.RATE_LIMITED
            else:
                self.endpoint.status = APIStatus.ERROR
            raise
        except Exception as e:
            logger.error(f"请求失败: {e}")
            self.endpoint.status = APIStatus.ERROR
            raise


class FlightAPIClient(BaseAPIClient):
    """航班API客户端"""
    
    async def search_flights(self, request: FlightSearchRequest) -> Dict[str, Any]:
        """搜索航班"""
        # 这里使用模拟的API调用，实际应该对接真实的航班API
        params = {
            "origin": request.origin,
            "destination": request.destination,
            "departure_date": request.departure_date,
            "return_date": request.return_date,
            "passengers": request.passengers,
            "cabin_class": request.cabin_class
        }
        
        # 模拟API响应
        await asyncio.sleep(1)  # 模拟网络延迟
        
        return {
            "flights": [
                {
                    "id": f"FL{i:03d}",
                    "airline": f"航空公司{i}",
                    "flight_number": f"CA{1000+i}",
                    "origin": request.origin,
                    "destination": request.destination,
                    "departure_time": f"{request.departure_date}T{8+i}:00:00",
                    "arrival_time": f"{request.departure_date}T{11+i}:30:00",
                    "duration": "3h 30m",
                    "price": 800 + i * 100,
                    "currency": "CNY",
                    "cabin_class": request.cabin_class,
                    "available_seats": 20 - i
                }
                for i in range(5)
            ],
            "search_params": params,
            "total_results": 5
        }


class HotelAPIClient(BaseAPIClient):
    """酒店API客户端"""
    
    async def search_hotels(self, request: HotelSearchRequest) -> Dict[str, Any]:
        """搜索酒店"""
        params = {
            "city": request.city,
            "check_in": request.check_in,
            "check_out": request.check_out,
            "guests": request.guests,
            "rooms": request.rooms
        }
        
        # 模拟API响应
        await asyncio.sleep(0.8)
        
        return {
            "hotels": [
                {
                    "id": f"HT{i:03d}",
                    "name": f"{request.city}酒店{i}",
                    "rating": 4.5 - i * 0.1,
                    "address": f"{request.city}市中心{i}号",
                    "price_per_night": 300 + i * 50,
                    "currency": "CNY",
                    "amenities": ["WiFi", "早餐", "停车场", "健身房"],
                    "availability": True,
                    "room_type": "标准间",
                    "images": [f"https://example.com/hotel{i}_1.jpg"],
                    "description": f"位于{request.city}市中心的优质酒店",
                    "distance_to_center": f"{i * 0.5}km"
                }
                for i in range(8)
            ],
            "search_params": params,
            "total_results": 8
        }


class WeatherAPIClient(BaseAPIClient):
    """天气API客户端"""
    
    async def get_weather(self, request: WeatherRequest) -> Dict[str, Any]:
        """获取天气信息"""
        # 模拟和风天气API
        await asyncio.sleep(0.5)
        
        base_date = datetime.now().date()
        forecasts = []
        
        for i in range(request.days):
            date = base_date + timedelta(days=i)
            forecasts.append({
                "date": date.isoformat(),
                "temperature_max": 25 + i,
                "temperature_min": 15 + i,
                "humidity": 60 + i * 2,
                "weather": "晴" if i % 3 == 0 else "多云" if i % 3 == 1 else "小雨",
                "wind_speed": 5 + i,
                "wind_direction": "东南风",
                "uv_index": 7 - i,
                "sunrise": "06:30",
                "sunset": "18:45"
            })
        
        return {
            "city": request.city,
            "current_weather": {
                "temperature": 22,
                "humidity": 65,
                "weather": "晴",
                "wind_speed": 8,
                "last_updated": datetime.now().isoformat()
            },
            "forecast": forecasts,
            "days": request.days
        }


class ExchangeRateClient(BaseAPIClient):
    """汇率API客户端"""
    
    async def get_exchange_rate(self, request: ExchangeRateRequest) -> Dict[str, Any]:
        """获取汇率"""
        # 模拟汇率API
        await asyncio.sleep(0.3)
        
        # 简单的汇率映射
        rates = {
            ("USD", "CNY"): 7.2,
            ("CNY", "USD"): 0.139,
            ("EUR", "CNY"): 7.8,
            ("CNY", "EUR"): 0.128,
            ("JPY", "CNY"): 0.048,
            ("CNY", "JPY"): 20.8
        }
        
        rate = rates.get((request.from_currency, request.to_currency), 1.0)
        converted_amount = request.amount * rate
        
        return {
            "from_currency": request.from_currency,
            "to_currency": request.to_currency,
            "exchange_rate": rate,
            "original_amount": request.amount,
            "converted_amount": converted_amount,
            "last_updated": datetime.now().isoformat()
        }


class DataIntegrator:
    """数据集成器"""
    
    def __init__(self):
        self.clients: Dict[str, BaseAPIClient] = {}
        self.endpoints: Dict[DataSourceType, List[APIEndpoint]] = {}
        self.cache_client = None
        self.initialized = False
    
    async def initialize(self):
        """初始化数据集成器"""
        logger.info("初始化数据集成器...")
        
        # 初始化Redis缓存
        self.cache_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=1,  # 使用不同的数据库
            decode_responses=True
        )
        
        # 配置API端点
        await self._setup_api_endpoints()
        
        # 初始化客户端
        await self._initialize_clients()
        
        self.initialized = True
        logger.info("数据集成器初始化完成")
    
    async def _setup_api_endpoints(self):
        """设置API端点"""
        # 航班API配置
        self.endpoints[DataSourceType.FLIGHT] = [
            APIEndpoint(
                name="primary_flight_api",
                base_url="https://api.flight-provider.com/v1",
                api_key="your_flight_api_key",
                headers={"Content-Type": "application/json"},
                rate_limit=100,
                timeout=30,
                retry_attempts=3,
                priority=1
            ),
            APIEndpoint(
                name="backup_flight_api",
                base_url="https://backup.flight-api.com/v2",
                api_key="backup_flight_key",
                headers={"Content-Type": "application/json"},
                rate_limit=50,
                timeout=30,
                retry_attempts=3,
                priority=2
            )
        ]
        
        # 酒店API配置
        self.endpoints[DataSourceType.HOTEL] = [
            APIEndpoint(
                name="primary_hotel_api",
                base_url="https://api.hotel-provider.com/v1",
                api_key="your_hotel_api_key",
                headers={"Content-Type": "application/json"},
                rate_limit=80,
                timeout=25,
                retry_attempts=3,
                priority=1
            )
        ]
        
        # 天气API配置
        self.endpoints[DataSourceType.WEATHER] = [
            APIEndpoint(
                name="weather_api",
                base_url="https://api.qweather.com/v7",
                api_key="your_weather_api_key",
                headers={"Content-Type": "application/json"},
                rate_limit=200,
                timeout=15,
                retry_attempts=2,
                priority=1
            )
        ]
        
        # 汇率API配置
        self.endpoints[DataSourceType.EXCHANGE_RATE] = [
            APIEndpoint(
                name="exchange_rate_api",
                base_url="https://api.exchangerate-api.com/v4",
                api_key=None,
                headers={"Content-Type": "application/json"},
                rate_limit=300,
                timeout=10,
                retry_attempts=2,
                priority=1
            )
        ]
    
    async def _initialize_clients(self):
        """初始化API客户端"""
        # 为每个端点创建客户端
        for source_type, endpoints in self.endpoints.items():
            for endpoint in endpoints:
                if source_type == DataSourceType.FLIGHT:
                    client = FlightAPIClient(endpoint)
                elif source_type == DataSourceType.HOTEL:
                    client = HotelAPIClient(endpoint)
                elif source_type == DataSourceType.WEATHER:
                    client = WeatherAPIClient(endpoint)
                elif source_type == DataSourceType.EXCHANGE_RATE:
                    client = ExchangeRateClient(endpoint)
                else:
                    client = BaseAPIClient(endpoint)
                
                await client.initialize()
                self.clients[endpoint.name] = client
    
    async def get_data(
        self,
        source_type: DataSourceType,
        request_data: Dict[str, Any],
        cache_strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM,
        force_refresh: bool = False
    ) -> DataResponse:
        """获取数据"""
        if not self.initialized:
            await self.initialize()
        
        request_id = str(uuid.uuid4())
        cache_key = self._generate_cache_key(source_type, request_data)
        
        # 检查缓存
        if not force_refresh and cache_strategy != CacheStrategy.NO_CACHE:
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return DataResponse(
                    request_id=request_id,
                    success=True,
                    data=cached_data["data"],
                    source=cached_data["source"],
                    cached=True,
                    timestamp=datetime.fromisoformat(cached_data["timestamp"]),
                    cache_expires=datetime.fromisoformat(cached_data["expires"]) if cached_data.get("expires") else None
                )
        
        # 获取可用的端点
        endpoints = self.endpoints.get(source_type, [])
        if not endpoints:
            raise HTTPException(
                status_code=404,
                detail=f"No endpoints configured for {source_type.value}"
            )
        
        # 按优先级排序并尝试获取数据
        sorted_endpoints = sorted(endpoints, key=lambda x: x.priority)
        last_error = None
        
        for endpoint in sorted_endpoints:
            if endpoint.status == APIStatus.ACTIVE:
                try:
                    client = self.clients[endpoint.name]
                    data = await self._fetch_data_from_client(client, source_type, request_data)
                    
                    # 缓存数据
                    if cache_strategy != CacheStrategy.NO_CACHE:
                        await self._cache_data(cache_key, data, endpoint.name, cache_strategy)
                    
                    return DataResponse(
                        request_id=request_id,
                        success=True,
                        data=data,
                        source=endpoint.name,
                        cached=False,
                        timestamp=datetime.now()
                    )
                    
                except Exception as e:
                    logger.error(f"从 {endpoint.name} 获取数据失败: {e}")
                    last_error = e
                    endpoint.status = APIStatus.ERROR
                    continue
        
        # 所有端点都失败
        return DataResponse(
            request_id=request_id,
            success=False,
            data=None,
            source="none",
            cached=False,
            timestamp=datetime.now(),
            error_message=str(last_error) if last_error else "All endpoints failed"
        )
    
    async def _fetch_data_from_client(
        self,
        client: BaseAPIClient,
        source_type: DataSourceType,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从客户端获取数据"""
        if source_type == DataSourceType.FLIGHT:
            request = FlightSearchRequest(**request_data)
            return await client.search_flights(request)
        elif source_type == DataSourceType.HOTEL:
            request = HotelSearchRequest(**request_data)
            return await client.search_hotels(request)
        elif source_type == DataSourceType.WEATHER:
            request = WeatherRequest(**request_data)
            return await client.get_weather(request)
        elif source_type == DataSourceType.EXCHANGE_RATE:
            request = ExchangeRateRequest(**request_data)
            return await client.get_exchange_rate(request)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _generate_cache_key(self, source_type: DataSourceType, request_data: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 对请求数据进行排序和序列化，确保一致性
        sorted_data = json.dumps(request_data, sort_keys=True, ensure_ascii=False)
        data_hash = hashlib.md5(sorted_data.encode()).hexdigest()
        return f"data_cache:{source_type.value}:{data_hash}"
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存数据"""
        try:
            cached_str = await self.cache_client.get(cache_key)
            if cached_str:
                cached_data = json.loads(cached_str)
                
                # 检查是否过期
                if cached_data.get("expires"):
                    expires = datetime.fromisoformat(cached_data["expires"])
                    if datetime.now() > expires:
                        await self.cache_client.delete(cache_key)
                        return None
                
                return cached_data
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        
        return None
    
    async def _cache_data(
        self,
        cache_key: str,
        data: Dict[str, Any],
        source: str,
        cache_strategy: CacheStrategy
    ):
        """缓存数据"""
        try:
            # 设置过期时间
            expires_map = {
                CacheStrategy.SHORT_TERM: timedelta(minutes=5),
                CacheStrategy.MEDIUM_TERM: timedelta(hours=1),
                CacheStrategy.LONG_TERM: timedelta(hours=24),
                CacheStrategy.PERSISTENT: timedelta(days=7)
            }
            
            expires_delta = expires_map.get(cache_strategy)
            if not expires_delta:
                return
            
            expires = datetime.now() + expires_delta
            
            cache_data = {
                "data": data,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "expires": expires.isoformat()
            }
            
            # 设置TTL
            ttl_seconds = int(expires_delta.total_seconds())
            
            await self.cache_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
        except Exception as e:
            logger.error(f"缓存数据失败: {e}")
    
    async def get_endpoint_status(self) -> Dict[str, Any]:
        """获取所有端点状态"""
        status = {}
        
        for source_type, endpoints in self.endpoints.items():
            status[source_type.value] = []
            for endpoint in endpoints:
                status[source_type.value].append({
                    "name": endpoint.name,
                    "status": endpoint.status.value,
                    "priority": endpoint.priority,
                    "rate_limit": endpoint.rate_limit,
                    "last_request": self.clients[endpoint.name].last_request_time.isoformat() if self.clients[endpoint.name].last_request_time else None
                })
        
        return status
    
    async def reset_endpoint_status(self, endpoint_name: str):
        """重置端点状态"""
        for endpoints in self.endpoints.values():
            for endpoint in endpoints:
                if endpoint.name == endpoint_name:
                    endpoint.status = APIStatus.ACTIVE
                    logger.info(f"重置端点状态: {endpoint_name}")
                    return True
        
        return False
    
    async def close(self):
        """关闭数据集成器"""
        for client in self.clients.values():
            await client.close()
        
        if self.cache_client:
            await self.cache_client.close()


# 依赖注入
async def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=0,
        decode_responses=True
    )


# 全局数据集成器实例
data_integrator = DataIntegrator()


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动外部数据源集成服务...")
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 初始化数据集成器
    await data_integrator.initialize()
    
    # 存储到应用状态
    app.state.redis_client = redis_client
    app.state.data_integrator = data_integrator
    
    logger.info("外部数据源集成服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭外部数据源集成服务...")
    await data_integrator.close()
    await redis_client.close()
    logger.info("外部数据源集成服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner Integration Service",
    description="外部数据源集成服务，提供航班、酒店、天气等数据接入",
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


# 航班搜索端点
@app.post("/api/v1/flights/search")
async def search_flights(request: FlightSearchRequest, background_tasks: BackgroundTasks):
    """搜索航班"""
    try:
        integrator = app.state.data_integrator
        response = await integrator.get_data(
            DataSourceType.FLIGHT,
            request.dict(),
            CacheStrategy.SHORT_TERM
        )
        
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "source": response.source,
                "cached": response.cached,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail=response.error_message)
            
    except Exception as e:
        logger.error(f"航班搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 酒店搜索端点
@app.post("/api/v1/hotels/search")
async def search_hotels(request: HotelSearchRequest):
    """搜索酒店"""
    try:
        integrator = app.state.data_integrator
        response = await integrator.get_data(
            DataSourceType.HOTEL,
            request.dict(),
            CacheStrategy.MEDIUM_TERM
        )
        
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "source": response.source,
                "cached": response.cached,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail=response.error_message)
            
    except Exception as e:
        logger.error(f"酒店搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 天气查询端点
@app.post("/api/v1/weather")
async def get_weather(request: WeatherRequest):
    """获取天气信息"""
    try:
        integrator = app.state.data_integrator
        response = await integrator.get_data(
            DataSourceType.WEATHER,
            request.dict(),
            CacheStrategy.SHORT_TERM
        )
        
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "source": response.source,
                "cached": response.cached,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail=response.error_message)
            
    except Exception as e:
        logger.error(f"天气查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 汇率查询端点
@app.post("/api/v1/exchange-rate")
async def get_exchange_rate(request: ExchangeRateRequest):
    """获取汇率"""
    try:
        integrator = app.state.data_integrator
        response = await integrator.get_data(
            DataSourceType.EXCHANGE_RATE,
            request.dict(),
            CacheStrategy.LONG_TERM
        )
        
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "source": response.source,
                "cached": response.cached,
                "timestamp": response.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail=response.error_message)
            
    except Exception as e:
        logger.error(f"汇率查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 系统状态端点
@app.get("/api/v1/status")
async def get_system_status():
    """获取系统状态"""
    try:
        integrator = app.state.data_integrator
        endpoint_status = await integrator.get_endpoint_status()
        
        return {
            "success": True,
            "endpoints": endpoint_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 端点管理
@app.post("/api/v1/endpoints/{endpoint_name}/reset")
async def reset_endpoint(endpoint_name: str):
    """重置端点状态"""
    try:
        integrator = app.state.data_integrator
        success = await integrator.reset_endpoint_status(endpoint_name)
        
        if success:
            return {
                "success": True,
                "message": f"端点 {endpoint_name} 状态已重置"
            }
        else:
            raise HTTPException(status_code=404, detail="端点不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重置端点状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 缓存管理
@app.delete("/api/v1/cache")
async def clear_cache():
    """清除缓存"""
    try:
        redis_client = app.state.redis_client
        
        # 删除所有数据缓存
        keys = await redis_client.keys("data_cache:*")
        if keys:
            await redis_client.delete(*keys)
        
        return {
            "success": True,
            "message": f"已清除 {len(keys)} 个缓存项",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
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
            "service": "integration-service",
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
        port=8005,
        reload=True,
        log_level="info"
    ) 