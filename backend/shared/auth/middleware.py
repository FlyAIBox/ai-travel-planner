"""
认证中间件
处理JWT认证和授权
"""

import logging
from typing import Optional, List
import uuid

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 无需认证的路径
PUBLIC_PATHS: List[str] = [
    "/",
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/status",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
]


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    def __init__(self, app, skip_auth: bool = False):
        super().__init__(app)
        self.skip_auth = skip_auth or settings.DEBUG
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 检查是否需要认证
        if self.skip_auth or self._is_public_path(request.url.path):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        # 提取和验证token
        try:
            token = await self._extract_token(request)
            if not token:
                return self._unauthorized_response("缺少认证令牌")
            
            # 验证token (这里简化处理，实际应该验证JWT)
            user_info = await self._validate_token(token)
            if not user_info:
                return self._unauthorized_response("无效的认证令牌")
            
            # 将用户信息添加到请求状态
            request.state.user = user_info
            request.state.user_id = user_info.get("user_id")
            
        except Exception as e:
            logger.error(f"认证失败: {e}")
            return self._unauthorized_response("认证失败")
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    def _is_public_path(self, path: str) -> bool:
        """检查是否为公开路径"""
        return any(path.startswith(public_path) for public_path in PUBLIC_PATHS)
    
    async def _extract_token(self, request: Request) -> Optional[str]:
        """提取认证令牌"""
        # 从Authorization头提取
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:]  # 移除 "Bearer " 前缀
        
        # 从查询参数提取（用于WebSocket等）
        token = request.query_params.get("token")
        if token:
            return token
        
        return None
    
    async def _validate_token(self, token: str) -> Optional[dict]:
        """验证令牌"""
        # TODO: 实现JWT令牌验证
        # 这里简化处理，返回模拟用户信息
        if settings.DEBUG and token == "debug_token":
            return {
                "user_id": "debug_user",
                "username": "debug",
                "email": "debug@example.com",
                "roles": ["user"]
            }
        
        # 实际实现应该：
        # 1. 验证JWT签名
        # 2. 检查过期时间
        # 3. 从数据库获取用户信息
        # 4. 检查用户状态（是否被禁用等）
        
        return None
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """返回未授权响应"""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": message
                }
            }
        )


class JWTBearer(HTTPBearer):
    """JWT Bearer认证"""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="无效的认证方案"
                )
            if not await self._verify_jwt(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="无效的认证令牌"
                )
        return credentials
    
    async def _verify_jwt(self, token: str) -> bool:
        """验证JWT令牌"""
        # TODO: 实现JWT验证逻辑
        return True


# 便捷函数
def get_current_user(request: Request) -> Optional[dict]:
    """从请求中获取当前用户"""
    return getattr(request.state, "user", None)


def get_current_user_id(request: Request) -> Optional[str]:
    """从请求中获取当前用户ID"""
    return getattr(request.state, "user_id", None)


def require_auth(request: Request) -> dict:
    """要求认证，返回用户信息"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要认证"
        )
    return user 