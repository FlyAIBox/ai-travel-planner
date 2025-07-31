"""
API网关服务
提供统一的API入口、路由、认证、限流、监控等功能
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from pydantic import BaseModel

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Prometheus指标
REQUEST_COUNT = Counter('gateway_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('gateway_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
SERVICE_REQUEST_COUNT = Counter('gateway_service_requests_total', 'Service requests', ['service', 'status'])


# Pydantic模型
class ServiceHealth(BaseModel):
    """服务健康状态"""
    service: str
    status: str
    response_time: float
    timestamp: str
    error: Optional[str] = None


class GatewayStats(BaseModel):
    """网关统计"""
    total_requests: int
    requests_per_service: Dict[str, int]
    average_response_time: float
    error_rate: float
    uptime_seconds: float


# 服务配置
SERVICES = {
    "chat": {
        "url": "http://chat-service:8000",
        "health_endpoint": "/api/v1/health",
        "timeout": 30
    },
    "rag": {
        "url": "http://rag-service:8001", 
        "health_endpoint": "/api/v1/health",
        "timeout": 30
    },
    "user": {
        "url": "http://user-service:8003",
        "health_endpoint": "/api/v1/health", 
        "timeout": 30
    },
    "agent": {
        "url": "http://agent-service:8002",
        "health_endpoint": "/api/v1/health",
        "timeout": 30
    }
}

# 路由映射
ROUTE_MAPPING = {
    "/api/v1/chat": "chat",
    "/api/v1/conversations": "chat",
    "/api/v1/mcp": "chat",
    "/api/v1/search": "rag",
    "/api/v1/documents": "rag",
    "/api/v1/collections": "rag",
    "/api/v1/knowledge-base": "rag",
    "/api/v1/users": "user",
    "/api/v1/auth": "user",
    "/api/v1/agents": "agent"
}


class ServiceRegistry:
    """服务注册表"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.services = SERVICES.copy()
        self.health_cache = {}
        self.cache_ttl = 30  # 健康状态缓存30秒
        
    async def get_service_url(self, service_name: str) -> Optional[str]:
        """获取服务URL"""
        service_info = self.services.get(service_name)
        if not service_info:
            return None
        
        # 检查服务健康状态
        if await self.is_service_healthy(service_name):
            return service_info["url"]
        
        return None
    
    async def is_service_healthy(self, service_name: str) -> bool:
        """检查服务健康状态"""
        cache_key = f"health:{service_name}"
        
        # 检查缓存
        if cache_key in self.health_cache:
            cached_time, is_healthy = self.health_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return is_healthy
        
        # 执行健康检查
        service_info = self.services.get(service_name)
        if not service_info:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{service_info['url']}{service_info['health_endpoint']}"
                )
                is_healthy = response.status_code == 200
                
                # 更新缓存
                self.health_cache[cache_key] = (time.time(), is_healthy)
                
                return is_healthy
                
        except Exception as e:
            logger.warning(f"服务 {service_name} 健康检查失败: {e}")
            self.health_cache[cache_key] = (time.time(), False)
            return False
    
    async def get_all_service_health(self) -> List[ServiceHealth]:
        """获取所有服务健康状态"""
        health_status = []
        
        for service_name, service_info in self.services.items():
            start_time = time.time()
            
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{service_info['url']}{service_info['health_endpoint']}"
                    )
                    
                    response_time = time.time() - start_time
                    
                    health_status.append(ServiceHealth(
                        service=service_name,
                        status="healthy" if response.status_code == 200 else "unhealthy",
                        response_time=response_time,
                        timestamp=datetime.now().isoformat()
                    ))
                    
            except Exception as e:
                response_time = time.time() - start_time
                health_status.append(ServiceHealth(
                    service=service_name,
                    status="unhealthy",
                    response_time=response_time,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                ))
        
        return health_status


class AuthManager:
    """认证管理器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证访问令牌"""
        # 这里应该实现真实的JWT验证逻辑
        # 为了演示，我们简化处理
        if not token or token == "invalid":
            return None
        
        # 模拟用户信息
        return {
            "user_id": "demo_user",
            "username": "demo",
            "roles": ["user"],
            "permissions": ["read", "write"]
        }
    
    async def check_permission(self, user_info: Dict[str, Any], endpoint: str) -> bool:
        """检查权限"""
        # 简化的权限检查
        permissions = user_info.get("permissions", [])
        
        # 读操作需要read权限
        if endpoint.startswith("GET"):
            return "read" in permissions
        
        # 写操作需要write权限
        return "write" in permissions


# 限流器
limiter = Limiter(key_func=get_remote_address)


# 依赖注入
async def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True
    )


async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """获取当前用户"""
    if not authorization:
        return None
    
    try:
        # 提取Bearer token
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            auth_manager = app.state.auth_manager
            return await auth_manager.validate_token(token)
    except Exception as e:
        logger.warning(f"令牌验证失败: {e}")
    
    return None


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动API网关...")
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 初始化组件
    service_registry = ServiceRegistry(redis_client)
    auth_manager = AuthManager(redis_client)
    
    # 存储到应用状态
    app.state.redis_client = redis_client
    app.state.service_registry = service_registry
    app.state.auth_manager = auth_manager
    app.state.start_time = time.time()
    app.state.request_count = 0
    app.state.error_count = 0
    
    logger.info("API网关启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭API网关...")
    await redis_client.close()
    logger.info("API网关已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner API Gateway",
    description="统一API网关，提供路由、认证、限流等功能",
    version="1.0.0",
    lifespan=lifespan
)

# 添加限流中间件
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 中间件
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """请求中间件"""
    start_time = time.time()
    
    # 增加请求计数
    app.state.request_count += 1
    
    # 记录Prometheus指标
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status="processing"
    ).inc()
    
    try:
        response = await call_next(request)
        
        # 记录响应时间
        process_time = time.time() - start_time
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)
        
        # 更新状态标签
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Gateway-Version"] = "1.0.0"
        
        return response
        
    except Exception as e:
        app.state.error_count += 1
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status="error"
        ).inc()
        
        logger.error(f"请求处理异常: {e}")
        raise


# 路由代理
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
@limiter.limit("100/minute")
async def proxy_request(request: Request, path: str, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """代理请求到后端服务"""
    full_path = f"/{path}"
    
    # 查找目标服务
    target_service = None
    for route_prefix, service_name in ROUTE_MAPPING.items():
        if full_path.startswith(route_prefix):
            target_service = service_name
            break
    
    if not target_service:
        raise HTTPException(status_code=404, detail="服务不存在")
    
    # 检查认证（某些端点需要认证）
    protected_endpoints = ["/api/v1/users", "/api/v1/documents", "/api/v1/agents"]
    if any(full_path.startswith(ep) for ep in protected_endpoints) and not user:
        raise HTTPException(status_code=401, detail="需要认证")
    
    # 检查权限
    if user:
        auth_manager = app.state.auth_manager
        if not await auth_manager.check_permission(user, f"{request.method} {full_path}"):
            raise HTTPException(status_code=403, detail="权限不足")
    
    # 获取服务URL
    service_registry = app.state.service_registry
    service_url = await service_registry.get_service_url(target_service)
    
    if not service_url:
        SERVICE_REQUEST_COUNT.labels(service=target_service, status="unavailable").inc()
        raise HTTPException(status_code=503, detail="服务不可用")
    
    # 构建目标URL
    target_url = f"{service_url}/{path}"
    if request.query_params:
        target_url += f"?{request.query_params}"
    
    # 准备请求
    headers = dict(request.headers)
    # 移除hop-by-hop headers
    hop_by_hop = ['connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade']
    for header in hop_by_hop:
        headers.pop(header, None)
    
    # 添加用户信息到请求头
    if user:
        headers["X-User-ID"] = user["user_id"]
        headers["X-User-Roles"] = ",".join(user["roles"])
    
    try:
        # 代理请求
        async with httpx.AsyncClient(timeout=30.0) as client:
            if request.method == "GET":
                response = await client.get(target_url, headers=headers)
            elif request.method == "POST":
                body = await request.body()
                response = await client.post(target_url, headers=headers, content=body)
            elif request.method == "PUT":
                body = await request.body()
                response = await client.put(target_url, headers=headers, content=body)
            elif request.method == "DELETE":
                response = await client.delete(target_url, headers=headers)
            elif request.method == "PATCH":
                body = await request.body()
                response = await client.patch(target_url, headers=headers, content=body)
            else:
                raise HTTPException(status_code=405, detail="方法不允许")
        
        SERVICE_REQUEST_COUNT.labels(service=target_service, status="success").inc()
        
        # 构建响应
        response_headers = dict(response.headers)
        # 移除hop-by-hop headers
        for header in hop_by_hop:
            response_headers.pop(header, None)
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type")
        )
        
    except httpx.TimeoutException:
        SERVICE_REQUEST_COUNT.labels(service=target_service, status="timeout").inc()
        raise HTTPException(status_code=504, detail="请求超时")
    
    except Exception as e:
        SERVICE_REQUEST_COUNT.labels(service=target_service, status="error").inc()
        logger.error(f"代理请求失败: {e}")
        raise HTTPException(status_code=502, detail="网关错误")


# WebSocket代理
@app.websocket("/ws/{path:path}")
async def websocket_proxy(websocket, path: str):
    """WebSocket代理"""
    # WebSocket代理实现
    # 这里需要根据路径判断目标服务
    target_service = "chat"  # 默认聊天服务
    
    service_registry = app.state.service_registry
    service_url = await service_registry.get_service_url(target_service)
    
    if not service_url:
        await websocket.close(code=1011, reason="服务不可用")
        return
    
    # 这里应该实现WebSocket代理逻辑
    # 由于复杂性，暂时直接拒绝连接
    await websocket.close(code=1011, reason="WebSocket代理暂未实现")


# 网关管理端点
@app.get("/gateway/health")
async def gateway_health():
    """网关健康检查"""
    try:
        redis_client = app.state.redis_client
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "api-gateway",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"网关健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/gateway/services/health")
async def services_health():
    """所有服务健康状态"""
    try:
        service_registry = app.state.service_registry
        health_status = await service_registry.get_all_service_health()
        
        return {
            "services": health_status,
            "total": len(health_status),
            "healthy": len([s for s in health_status if s.status == "healthy"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取服务健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gateway/stats", response_model=GatewayStats)
async def gateway_stats():
    """网关统计"""
    try:
        uptime = time.time() - app.state.start_time
        error_rate = app.state.error_count / max(app.state.request_count, 1)
        
        return GatewayStats(
            total_requests=app.state.request_count,
            requests_per_service={},  # 这里可以添加具体统计
            average_response_time=0.0,  # 这里可以添加响应时间统计
            error_rate=error_rate,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"获取网关统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gateway/metrics")
async def metrics():
    """Prometheus指标"""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/gateway/routes")
async def list_routes():
    """列出路由映射"""
    return {
        "routes": ROUTE_MAPPING,
        "services": SERVICES,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 