"""
AI Travel Planner - API Gateway
API网关服务，负责路由、认证、限流等功能
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

# 添加共享模块到路径
sys.path.append(str(Path(__file__).parent.parent.parent / "shared"))

from shared.config.settings import get_settings
from shared.monitoring.metrics import setup_metrics
from shared.auth.middleware import AuthMiddleware
from shared.cache.redis_client import get_redis_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("🚀 启动 AI Travel Planner API Gateway")
    
    # 初始化Redis连接
    redis_client = await get_redis_client()
    app.state.redis = redis_client
    
    # 设置监控指标
    setup_metrics(app)
    
    logger.info("✅ API Gateway 启动完成")
    
    yield
    
    # 关闭时清理
    logger.info("🔄 关闭 API Gateway")
    if hasattr(app.state, 'redis'):
        await app.state.redis.close()
    logger.info("✅ API Gateway 关闭完成")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner API",
    description="智能旅行规划助手API网关",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# 添加认证中间件
app.add_middleware(AuthMiddleware)

# 挂载Prometheus指标端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "AI Travel Planner API Gateway",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查Redis连接
        if hasattr(app.state, 'redis'):
            await app.state.redis.ping()
        
        return {
            "status": "healthy",
            "service": "api-gateway",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "api-gateway",
                "error": str(e)
            }
        )


@app.get("/api/v1/status")
async def api_status():
    """API状态端点"""
    return {
        "api_version": "v1",
        "status": "active",
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users",
            "conversations": "/api/v1/conversations",
            "travel_plans": "/api/v1/travel-plans",
            "search": "/api/v1/search",
            "agents": "/api/v1/agents"
        }
    }


# TODO: 添加路由模块
# from .routes import auth, users, conversations, travel_plans, search, agents
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["认证"])
# app.include_router(users.router, prefix="/api/v1/users", tags=["用户"])
# app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["对话"])
# app.include_router(travel_plans.router, prefix="/api/v1/travel-plans", tags=["旅行计划"])
# app.include_router(search.router, prefix="/api/v1/search", tags=["搜索"])
# app.include_router(agents.router, prefix="/api/v1/agents", tags=["智能体"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "服务器内部错误" if not settings.DEBUG else str(exc)
            },
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    ) 