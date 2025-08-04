"""
API网关服务主程序
实现FastAPI网关服务、路由配置、认证中间件、负载均衡、限流控制、API版本管理和请求响应日志记录
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import hashlib
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
import uvicorn
from pydantic import BaseModel, Field
import redis.asyncio as redis
import aiohttp
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ==================== 数据模型 ====================

class ServiceEndpoint(BaseModel):
    """服务端点配置"""
    name: str
    host: str
    port: int
    path: str = ""
    health_check_path: str = "/health"
    timeout: int = 30
    max_retries: int = 3
    weight: int = 1
    is_active: bool = True


class RouteConfig(BaseModel):
    """路由配置"""
    path: str
    method: str = "GET"
    service_name: str
    target_path: str = ""
    auth_required: bool = True
    rate_limit: str = "100/minute"
    timeout: int = 30
    cache_ttl: int = 0
    version: str = "v1"


class AuthConfig(BaseModel):
    """认证配置"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    token_expire_hours: int = 24
    refresh_expire_days: int = 7


class GatewayConfig(BaseModel):
    """网关配置"""
    services: Dict[str, List[ServiceEndpoint]]
    routes: List[RouteConfig]
    auth: AuthConfig
    cors_origins: List[str] = ["*"]
    trusted_hosts: List[str] = ["*"]
    max_request_size: int = 10 * 1024 * 1024  # 10MB


# ==================== 服务发现和负载均衡 ====================

class ServiceRegistry:
    """服务注册表"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.services: Dict[str, List[ServiceEndpoint]] = {}
        self.health_check_interval = 30
        self.health_check_task = None
    
    async def register_service(self, service_name: str, endpoints: List[ServiceEndpoint]):
        """注册服务"""
        self.services[service_name] = endpoints
        
        # 保存到Redis
        service_data = [endpoint.dict() for endpoint in endpoints]
        await self.redis_client.set(
            f"service:{service_name}",
            json.dumps(service_data),
            ex=self.health_check_interval * 2
        )
        
        logger.info(f"服务 {service_name} 已注册，端点数量: {len(endpoints)}")
    
    async def get_service_endpoints(self, service_name: str) -> List[ServiceEndpoint]:
        """获取服务端点"""
        # 先从内存获取
        if service_name in self.services:
            return [ep for ep in self.services[service_name] if ep.is_active]
        
        # 从Redis获取
        service_data = await self.redis_client.get(f"service:{service_name}")
        if service_data:
            endpoints_data = json.loads(service_data)
            endpoints = [ServiceEndpoint(**ep) for ep in endpoints_data]
            self.services[service_name] = endpoints
            return [ep for ep in endpoints if ep.is_active]
        
        return []
    
    async def start_health_checks(self):
        """启动健康检查"""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop_health_checks(self):
        """停止健康检查"""
        if self.health_check_task:
            self.health_check_task.cancel()
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        async with aiohttp.ClientSession() as session:
            for service_name, endpoints in self.services.items():
                for endpoint in endpoints:
                    try:
                        health_url = f"http://{endpoint.host}:{endpoint.port}{endpoint.health_check_path}"
                        
                        async with session.get(health_url, timeout=5) as response:
                            is_healthy = response.status == 200
                            
                            if endpoint.is_active != is_healthy:
                                endpoint.is_active = is_healthy
                                status_text = "健康" if is_healthy else "不健康"
                                logger.info(f"服务 {service_name} 端点 {endpoint.host}:{endpoint.port} 状态变更: {status_text}")
                    
                    except Exception as e:
                        if endpoint.is_active:
                            endpoint.is_active = False
                            logger.warning(f"服务 {service_name} 端点 {endpoint.host}:{endpoint.port} 健康检查失败: {e}")


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.current_index: Dict[str, int] = {}
        self.request_counts: Dict[str, int] = {}
    
    def select_endpoint(self, 
                       service_name: str,
                       endpoints: List[ServiceEndpoint],
                       strategy: str = "round_robin") -> Optional[ServiceEndpoint]:
        """选择服务端点"""
        active_endpoints = [ep for ep in endpoints if ep.is_active]
        
        if not active_endpoints:
            return None
        
        if strategy == "round_robin":
            return self._round_robin_select(service_name, active_endpoints)
        elif strategy == "weighted_round_robin":
            return self._weighted_round_robin_select(service_name, active_endpoints)
        elif strategy == "least_connections":
            return self._least_connections_select(active_endpoints)
        else:
            return active_endpoints[0]
    
    def _round_robin_select(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """轮询选择"""
        if service_name not in self.current_index:
            self.current_index[service_name] = 0
        
        index = self.current_index[service_name] % len(endpoints)
        self.current_index[service_name] = (index + 1) % len(endpoints)
        
        return endpoints[index]
    
    def _weighted_round_robin_select(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """加权轮询选择"""
        # 构建加权列表
        weighted_endpoints = []
        for endpoint in endpoints:
            weighted_endpoints.extend([endpoint] * endpoint.weight)
        
        if not weighted_endpoints:
            return endpoints[0]
        
        return self._round_robin_select(service_name, weighted_endpoints)
    
    def _least_connections_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """最少连接选择"""
        # 简化实现：选择请求计数最少的端点
        min_requests = float('inf')
        selected_endpoint = endpoints[0]
        
        for endpoint in endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            request_count = self.request_counts.get(endpoint_key, 0)
            
            if request_count < min_requests:
                min_requests = request_count
                selected_endpoint = endpoint
        
        return selected_endpoint
    
    def record_request(self, endpoint: ServiceEndpoint):
        """记录请求"""
        endpoint_key = f"{endpoint.host}:{endpoint.port}"
        self.request_counts[endpoint_key] = self.request_counts.get(endpoint_key, 0) + 1
    
    def record_response(self, endpoint: ServiceEndpoint):
        """记录响应"""
        endpoint_key = f"{endpoint.host}:{endpoint.port}"
        if endpoint_key in self.request_counts:
            self.request_counts[endpoint_key] = max(0, self.request_counts[endpoint_key] - 1)


# ==================== 认证和授权 ====================

class AuthManager:
    """认证管理器"""
    
    def __init__(self, config: AuthConfig, redis_client: redis.Redis):
        self.config = config
        self.redis_client = redis_client
        self.security = HTTPBearer()
    
    async def authenticate(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """认证用户"""
        token = credentials.credentials
        
        try:
            # 检查token是否在黑名单中
            if await self.redis_client.get(f"blacklist:{token}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token已失效"
                )
            
            # 验证JWT token
            payload = self._verify_jwt_token(token)
            
            # 检查用户是否仍然有效
            user_id = payload.get("user_id")
            if user_id:
                user_info = await self._get_user_info(user_id)
                if user_info:
                    return {
                        "user_id": user_id,
                        "username": user_info.get("username"),
                        "roles": user_info.get("roles", []),
                        "permissions": user_info.get("permissions", [])
                    }
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据"
            )
        
        except Exception as e:
            logger.warning(f"认证失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="认证失败"
            )
    
    def _verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """验证JWT token"""
        import jwt
        
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token已过期"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的Token"
            )
    
    async def _get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        # 从Redis获取用户信息
        user_data = await self.redis_client.get(f"user:{user_id}")
        if user_data:
            return json.loads(user_data)
        
        # 模拟用户数据
        return {
            "user_id": user_id,
            "username": f"user_{user_id}",
            "roles": ["user"],
            "permissions": ["read", "write"]
        }
    
    async def logout(self, token: str):
        """登出用户"""
        # 将token加入黑名单
        await self.redis_client.setex(
            f"blacklist:{token}",
            self.config.token_expire_hours * 3600,
            "1"
        )


# ==================== 请求代理 ====================

class RequestProxy:
    """请求代理"""
    
    def __init__(self, service_registry: ServiceRegistry, load_balancer: LoadBalancer):
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.session = None
    
    async def initialize(self):
        """初始化代理"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
    
    async def proxy_request(self,
                           request: Request,
                           service_name: str,
                           target_path: str = "",
                           timeout: int = 30) -> Response:
        """代理请求"""
        # 获取服务端点
        endpoints = await self.service_registry.get_service_endpoints(service_name)
        if not endpoints:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"服务 {service_name} 不可用"
            )
        
        # 选择端点
        endpoint = self.load_balancer.select_endpoint(service_name, endpoints)
        if not endpoint:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"服务 {service_name} 所有端点都不可用"
            )
        
        # 记录请求
        self.load_balancer.record_request(endpoint)
        
        try:
            # 构建目标URL
            path = target_path or request.url.path
            target_url = f"http://{endpoint.host}:{endpoint.port}{path}"
            
            # 准备请求参数
            headers = dict(request.headers)
            headers.pop("host", None)  # 移除原始host头
            
            # 获取请求体
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
            
            # 发起代理请求
            async with self.session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=request.query_params,
                data=body,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                # 读取响应内容
                content = await response.read()
                
                # 准备响应头
                response_headers = dict(response.headers)
                
                # 移除不需要的头
                headers_to_remove = [
                    "transfer-encoding",
                    "connection",
                    "server",
                    "date"
                ]
                for header in headers_to_remove:
                    response_headers.pop(header, None)
                
                # 记录响应
                self.load_balancer.record_response(endpoint)
                
                # 返回响应
                return Response(
                    content=content,
                    status_code=response.status,
                    headers=response_headers
                )
        
        except asyncio.TimeoutError:
            logger.error(f"请求 {service_name} 超时")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="请求超时"
            )
        except Exception as e:
            logger.error(f"代理请求失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="代理请求失败"
            )


# ==================== 缓存管理 ====================

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存响应"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
        
        return None
    
    async def cache_response(self, 
                           cache_key: str,
                           response_data: Dict[str, Any],
                           ttl: int):
        """缓存响应"""
        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(response_data)
            )
        except Exception as e:
            logger.warning(f"缓存响应失败: {e}")
    
    def generate_cache_key(self, request: Request, route_config: RouteConfig) -> str:
        """生成缓存键"""
        key_parts = [
            route_config.service_name,
            route_config.path,
            request.method,
            str(request.query_params)
        ]
        
        key_string = "|".join(key_parts)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"


# ==================== 中间件 ====================

class RequestLoggingMiddleware:
    """请求日志中间件"""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        # 记录请求开始时间
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # 记录请求信息
        logger.info(
            f"请求开始 [{request_id}] {request.method} {request.url} "
            f"客户端: {request.client.host if request.client else 'unknown'}"
        )
        
        # 处理请求
        response = await call_next(request)
        
        # 记录响应信息
        process_time = time.time() - start_time
        logger.info(
            f"请求完成 [{request_id}] 状态码: {response.status_code} "
            f"处理时间: {process_time:.3f}s"
        )
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


# ==================== API网关应用 ====================

class APIGateway:
    """API网关"""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.app = FastAPI(
            title="AI Travel Planner API Gateway",
            description="智能旅行规划API网关",
            version="1.0.0"
        )
        
        # 初始化组件
        self.redis_client = None
        self.service_registry = None
        self.load_balancer = LoadBalancer()
        self.auth_manager = None
        self.request_proxy = None
        self.cache_manager = None
        
        # 速率限制器
        self.limiter = Limiter(key_func=get_remote_address)
        
        # 设置中间件和路由
        self._setup_middleware()
        self._setup_routes()
    
    async def initialize(self):
        """初始化网关"""
        # 初始化Redis客户端
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
        
        # 初始化组件
        self.service_registry = ServiceRegistry(self.redis_client)
        self.auth_manager = AuthManager(self.config.auth, self.redis_client)
        self.request_proxy = RequestProxy(self.service_registry, self.load_balancer)
        self.cache_manager = CacheManager(self.redis_client)
        
        # 初始化服务代理
        await self.request_proxy.initialize()
        
        # 注册服务
        for service_name, endpoints in self.config.services.items():
            await self.service_registry.register_service(service_name, endpoints)
        
        # 启动健康检查
        await self.service_registry.start_health_checks()
        
        logger.info("API网关初始化完成")
    
    async def cleanup(self):
        """清理资源"""
        if self.service_registry:
            await self.service_registry.stop_health_checks()
        
        if self.request_proxy:
            await self.request_proxy.cleanup()
        
        if self.redis_client:
            await self.redis_client.close()
    
    def _setup_middleware(self):
        """设置中间件"""
        # CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 受信任主机中间件
        if "*" not in self.config.trusted_hosts:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.trusted_hosts
            )
        
        # 速率限制中间件
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.app.add_middleware(SlowAPIMiddleware)
        
        # 请求日志中间件
        self.app.middleware("http")(RequestLoggingMiddleware(self.app))
    
    def _setup_routes(self):
        """设置路由"""
        # 健康检查端点
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # 网关状态端点
        @self.app.get("/gateway/status")
        async def gateway_status():
            return {
                "gateway": "running",
                "services": len(self.config.services),
                "routes": len(self.config.routes),
                "timestamp": datetime.now().isoformat()
            }
        
        # 服务健康状态端点
        @self.app.get("/gateway/services/health")
        async def services_health():
            health_status = {}
            
            for service_name in self.config.services.keys():
                endpoints = await self.service_registry.get_service_endpoints(service_name)
                health_status[service_name] = {
                    "total_endpoints": len(self.config.services[service_name]),
                    "active_endpoints": len(endpoints),
                    "endpoints": [
                        {
                            "host": ep.host,
                            "port": ep.port,
                            "active": ep.is_active,
                            "weight": ep.weight
                        }
                        for ep in self.config.services[service_name]
                    ]
                }
            
            return health_status
        
        # 动态添加配置的路由
        for route_config in self.config.routes:
            self._add_dynamic_route(route_config)
    
    def _add_dynamic_route(self, route_config: RouteConfig):
        """添加动态路由"""
        
        async def route_handler(request: Request):
            # 应用速率限制
            await self._apply_rate_limit(request, route_config.rate_limit)
            
            # 认证检查
            user_info = None
            if route_config.auth_required:
                credentials = await self.auth_manager.security(request)
                user_info = await self.auth_manager.authenticate(credentials)
                request.state.user = user_info
            
            # 检查缓存
            cached_response = None
            if route_config.cache_ttl > 0:
                cache_key = self.cache_manager.generate_cache_key(request, route_config)
                cached_response = await self.cache_manager.get_cached_response(cache_key)
                
                if cached_response:
                    return JSONResponse(
                        content=cached_response["content"],
                        status_code=cached_response["status_code"],
                        headers=cached_response.get("headers", {})
                    )
            
            # 代理请求
            response = await self.request_proxy.proxy_request(
                request=request,
                service_name=route_config.service_name,
                target_path=route_config.target_path,
                timeout=route_config.timeout
            )
            
            # 缓存响应
            if route_config.cache_ttl > 0 and response.status_code == 200:
                response_data = {
                    "content": response.body.decode() if response.body else "",
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                await self.cache_manager.cache_response(
                    cache_key, response_data, route_config.cache_ttl
                )
            
            return response
        
        # 添加路由
        self.app.add_api_route(
            path=f"/{route_config.version}{route_config.path}",
            endpoint=route_handler,
            methods=[route_config.method],
            tags=[route_config.service_name]
        )
        
        logger.info(f"添加路由: {route_config.method} /{route_config.version}{route_config.path} -> {route_config.service_name}")
    
    async def _apply_rate_limit(self, request: Request, rate_limit: str):
        """应用速率限制"""
        # 解析速率限制（例如 "100/minute"）
        parts = rate_limit.split("/")
        if len(parts) != 2:
            return
        
        limit_count = int(parts[0])
        limit_period = parts[1]
        
        # 应用速率限制
        await self.limiter.limit(f"{limit_count}/{limit_period}")(request)


# ==================== 配置加载 ====================

def load_gateway_config() -> GatewayConfig:
    """加载网关配置"""
    # 服务端点配置
    services = {
        "chat-service": [
            ServiceEndpoint(
                name="chat-primary",
                host="localhost",
                port=8001,
                health_check_path="/health",
                weight=1
            )
        ],
        "rag-service": [
            ServiceEndpoint(
                name="rag-primary",
                host="localhost",
                port=8002,
                health_check_path="/health",
                weight=1
            )
        ],
        "agent-service": [
            ServiceEndpoint(
                name="agent-primary",
                host="localhost",
                port=8003,
                health_check_path="/health",
                weight=1
            )
        ],
        "planning-service": [
            ServiceEndpoint(
                name="planning-primary",
                host="localhost",
                port=8004,
                health_check_path="/health",
                weight=1
            )
        ],
        "integration-service": [
            ServiceEndpoint(
                name="integration-primary",
                host="localhost",
                port=8005,
                health_check_path="/health",
                weight=1
            )
        ],
        "user-service": [
            ServiceEndpoint(
                name="user-primary",
                host="localhost",
                port=8006,
                health_check_path="/health",
                weight=1
            )
        ]
    }
    
    # 路由配置
    routes = [
        # 聊天服务路由
        RouteConfig(
            path="/chat/conversation",
            method="POST",
            service_name="chat-service",
            target_path="/conversation",
            auth_required=True,
            rate_limit="50/minute"
        ),
        RouteConfig(
            path="/chat/websocket",
            method="GET",
            service_name="chat-service",
            target_path="/ws",
            auth_required=True,
            rate_limit="10/minute"
        ),
        
        # RAG服务路由
        RouteConfig(
            path="/rag/search",
            method="POST",
            service_name="rag-service",
            target_path="/search",
            auth_required=True,
            rate_limit="100/minute",
            cache_ttl=300
        ),
        RouteConfig(
            path="/rag/knowledge",
            method="GET",
            service_name="rag-service",
            target_path="/knowledge",
            auth_required=True,
            rate_limit="200/minute",
            cache_ttl=600
        ),
        
        # 智能体服务路由
        RouteConfig(
            path="/agent/create",
            method="POST",
            service_name="agent-service",
            target_path="/agent/create",
            auth_required=True,
            rate_limit="20/minute"
        ),
        RouteConfig(
            path="/agent/status",
            method="GET",
            service_name="agent-service",
            target_path="/agent/status",
            auth_required=True,
            rate_limit="100/minute"
        ),
        
        # 规划服务路由
        RouteConfig(
            path="/planning/create",
            method="POST",
            service_name="planning-service",
            target_path="/plan/create",
            auth_required=True,
            rate_limit="10/minute"
        ),
        RouteConfig(
            path="/planning/optimize",
            method="POST",
            service_name="planning-service",
            target_path="/plan/optimize",
            auth_required=True,
            rate_limit="5/minute"
        ),
        
        # 集成服务路由
        RouteConfig(
            path="/data/flights",
            method="GET",
            service_name="integration-service",
            target_path="/flights/search",
            auth_required=True,
            rate_limit="30/minute",
            cache_ttl=1800
        ),
        RouteConfig(
            path="/data/hotels",
            method="GET",
            service_name="integration-service",
            target_path="/hotels/search",
            auth_required=True,
            rate_limit="30/minute",
            cache_ttl=1800
        ),
        RouteConfig(
            path="/data/weather",
            method="GET",
            service_name="integration-service",
            target_path="/weather/current",
            auth_required=True,
            rate_limit="100/minute",
            cache_ttl=600
        ),
        
        # 用户服务路由
        RouteConfig(
            path="/user/profile",
            method="GET",
            service_name="user-service",
            target_path="/profile",
            auth_required=True,
            rate_limit="50/minute"
        ),
        RouteConfig(
            path="/user/preferences",
            method="POST",
            service_name="user-service",
            target_path="/preferences",
            auth_required=True,
            rate_limit="20/minute"
        ),
    ]
    
    # 认证配置
    auth_config = AuthConfig(
        jwt_secret=settings.JWT_SECRET or "your-secret-key",
        jwt_algorithm="HS256",
        token_expire_hours=24,
        refresh_expire_days=7
    )
    
    return GatewayConfig(
        services=services,
        routes=routes,
        auth=auth_config,
        cors_origins=["*"],
        trusted_hosts=["*"]
    )


# ==================== 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await gateway.initialize()
    logger.info("API网关启动完成")
    
    yield
    
    # 关闭时清理
    await gateway.cleanup()
    logger.info("API网关已停止")


# ==================== 主程序 ====================

# 加载配置并创建网关
config = load_gateway_config()
gateway = APIGateway(config)
gateway.app.router.lifespan_context = lifespan

# 获取FastAPI应用实例
app = gateway.app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 