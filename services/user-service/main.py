"""
用户服务
提供用户管理、认证、权限等功能
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
import mysql.connector.aio as mysql

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
SECRET_KEY = settings.JWT_SECRET_KEY if hasattr(settings, 'JWT_SECRET_KEY') else "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# HTTP Bearer认证
security = HTTPBearer()


# Pydantic模型
class UserCreate(BaseModel):
    """用户创建模型"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)


class UserLogin(BaseModel):
    """用户登录模型"""
    username: str
    password: str


class UserResponse(BaseModel):
    """用户响应模型"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    phone: Optional[str]
    is_active: bool
    created_at: str
    last_login: Optional[str]


class UserUpdate(BaseModel):
    """用户更新模型"""
    full_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None


class TokenResponse(BaseModel):
    """令牌响应模型"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserPreferences(BaseModel):
    """用户偏好模型"""
    preferred_language: str = "zh"
    preferred_currency: str = "CNY"
    travel_interests: List[str] = []
    budget_range: Optional[str] = None
    notification_settings: Dict[str, bool] = {}


# 数据库连接
class Database:
    """数据库连接管理"""
    
    def __init__(self):
        self.pool = None
    
    async def connect(self):
        """连接数据库"""
        try:
            self.pool = await mysql.create_pool(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE,
                minsize=5,
                maxsize=20
            )
            logger.info("数据库连接池创建成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开数据库连接"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("数据库连接池已关闭")
    
    async def get_connection(self):
        """获取数据库连接"""
        if not self.pool:
            await self.connect()
        return await self.pool.acquire()


# 全局数据库实例
db = Database()


# 认证相关函数
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict):
    """创建刷新令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # 从数据库获取用户
    user = await get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    
    return user


# 数据库操作函数
async def create_user_in_db(user_data: UserCreate) -> str:
    """在数据库中创建用户"""
    async with await db.get_connection() as conn:
        try:
            user_id = secrets.token_urlsafe(16)
            hashed_password = get_password_hash(user_data.password)
            
            query = """
            INSERT INTO users (user_id, username, email, password_hash, full_name, phone, created_at, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            await conn.execute(query, (
                user_id,
                user_data.username,
                user_data.email,
                hashed_password,
                user_data.full_name,
                user_data.phone,
                datetime.now(),
                True
            ))
            
            await conn.commit()
            logger.info(f"用户创建成功: {user_data.username}")
            return user_id
            
        except Exception as e:
            await conn.rollback()
            logger.error(f"创建用户失败: {e}")
            raise


async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """根据用户名获取用户"""
    async with await db.get_connection() as conn:
        try:
            query = "SELECT * FROM users WHERE username = %s AND is_active = TRUE"
            cursor = await conn.execute(query, (username,))
            result = await cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            
            return None
            
        except Exception as e:
            logger.error(f"查询用户失败: {e}")
            return None


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """根据用户ID获取用户"""
    async with await db.get_connection() as conn:
        try:
            query = "SELECT * FROM users WHERE user_id = %s AND is_active = TRUE"
            cursor = await conn.execute(query, (user_id,))
            result = await cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            
            return None
            
        except Exception as e:
            logger.error(f"查询用户失败: {e}")
            return None


async def update_user_in_db(user_id: str, update_data: UserUpdate) -> bool:
    """更新用户信息"""
    async with await db.get_connection() as conn:
        try:
            fields = []
            values = []
            
            if update_data.full_name is not None:
                fields.append("full_name = %s")
                values.append(update_data.full_name)
            
            if update_data.phone is not None:
                fields.append("phone = %s")
                values.append(update_data.phone)
            
            if update_data.email is not None:
                fields.append("email = %s")
                values.append(update_data.email)
            
            if not fields:
                return True
            
            fields.append("updated_at = %s")
            values.append(datetime.now())
            values.append(user_id)
            
            query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s"
            await conn.execute(query, values)
            await conn.commit()
            
            logger.info(f"用户信息更新成功: {user_id}")
            return True
            
        except Exception as e:
            await conn.rollback()
            logger.error(f"更新用户信息失败: {e}")
            return False


async def update_last_login(user_id: str):
    """更新最后登录时间"""
    async with await db.get_connection() as conn:
        try:
            query = "UPDATE users SET last_login = %s WHERE user_id = %s"
            await conn.execute(query, (datetime.now(), user_id))
            await conn.commit()
        except Exception as e:
            logger.error(f"更新登录时间失败: {e}")


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


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动用户服务...")
    
    # 连接数据库
    await db.connect()
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    app.state.redis_client = redis_client
    
    # 初始化数据库表
    await init_database()
    
    logger.info("用户服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭用户服务...")
    await db.disconnect()
    await redis_client.close()
    logger.info("用户服务已关闭")


async def init_database():
    """初始化数据库表"""
    async with await db.get_connection() as conn:
        try:
            # 创建用户表
            users_table = """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(32) PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(100),
                phone VARCHAR(20),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                INDEX idx_username (username),
                INDEX idx_email (email)
            )
            """
            
            # 创建用户偏好表
            preferences_table = """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id VARCHAR(32) PRIMARY KEY,
                preferred_language VARCHAR(10) DEFAULT 'zh',
                preferred_currency VARCHAR(10) DEFAULT 'CNY',
                travel_interests JSON,
                budget_range VARCHAR(50),
                notification_settings JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
            
            await conn.execute(users_table)
            await conn.execute(preferences_table)
            await conn.commit()
            
            logger.info("数据库表初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner User Service",
    description="用户管理服务，提供认证、用户管理等功能",
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


# 认证端点
@app.post("/api/v1/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """用户注册"""
    try:
        # 检查用户名是否已存在
        existing_user = await get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
        
        # 创建用户
        user_id = await create_user_in_db(user_data)
        
        # 获取创建的用户信息
        user = await get_user_by_id(user_id)
        
        return UserResponse(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            full_name=user["full_name"],
            phone=user["phone"],
            is_active=user["is_active"],
            created_at=user["created_at"].isoformat(),
            last_login=user["last_login"].isoformat() if user["last_login"] else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(status_code=500, detail="注册失败")


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(login_data: UserLogin):
    """用户登录"""
    try:
        # 验证用户
        user = await get_user_by_username(login_data.username)
        if not user or not verify_password(login_data.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 更新最后登录时间
        await update_last_login(user["user_id"])
        
        # 创建令牌
        access_token = create_access_token(data={"sub": user["user_id"]})
        refresh_token = create_refresh_token(data={"sub": user["user_id"]})
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"用户登录失败: {e}")
        raise HTTPException(status_code=500, detail="登录失败")


@app.post("/api/v1/auth/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """刷新访问令牌"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的刷新令牌"
            )
        
        # 验证用户是否存在
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在"
            )
        
        # 创建新的访问令牌
        access_token = create_access_token(data={"sub": user_id})
        new_refresh_token = create_refresh_token(data={"sub": user_id})
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌"
        )
    except Exception as e:
        logger.error(f"刷新令牌失败: {e}")
        raise HTTPException(status_code=500, detail="刷新令牌失败")


# 用户管理端点
@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        phone=current_user["phone"],
        is_active=current_user["is_active"],
        created_at=current_user["created_at"].isoformat(),
        last_login=current_user["last_login"].isoformat() if current_user["last_login"] else None
    )


@app.put("/api/v1/users/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: dict = Depends(get_current_user)
):
    """更新当前用户信息"""
    try:
        success = await update_user_in_db(current_user["user_id"], update_data)
        if not success:
            raise HTTPException(status_code=500, detail="更新失败")
        
        # 获取更新后的用户信息
        updated_user = await get_user_by_id(current_user["user_id"])
        
        return UserResponse(
            user_id=updated_user["user_id"],
            username=updated_user["username"],
            email=updated_user["email"],
            full_name=updated_user["full_name"],
            phone=updated_user["phone"],
            is_active=updated_user["is_active"],
            created_at=updated_user["created_at"].isoformat(),
            last_login=updated_user["last_login"].isoformat() if updated_user["last_login"] else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户信息失败: {e}")
        raise HTTPException(status_code=500, detail="更新失败")


@app.get("/api/v1/users/{user_id}", response_model=UserResponse)
async def get_user_by_id_endpoint(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    """根据ID获取用户信息（仅管理员或本人）"""
    # 简单权限检查：只能查看自己的信息
    if current_user["user_id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限访问"
        )
    
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return UserResponse(
        user_id=user["user_id"],
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        phone=user["phone"],
        is_active=user["is_active"],
        created_at=user["created_at"].isoformat(),
        last_login=user["last_login"].isoformat() if user["last_login"] else None
    )


# 用户偏好端点
@app.get("/api/v1/users/me/preferences")
async def get_user_preferences(current_user: dict = Depends(get_current_user)):
    """获取用户偏好"""
    async with await db.get_connection() as conn:
        try:
            query = "SELECT * FROM user_preferences WHERE user_id = %s"
            cursor = await conn.execute(query, (current_user["user_id"],))
            result = await cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                prefs = dict(zip(columns, result))
                
                return {
                    "preferred_language": prefs["preferred_language"],
                    "preferred_currency": prefs["preferred_currency"],
                    "travel_interests": prefs["travel_interests"] or [],
                    "budget_range": prefs["budget_range"],
                    "notification_settings": prefs["notification_settings"] or {}
                }
            else:
                # 返回默认偏好
                return {
                    "preferred_language": "zh",
                    "preferred_currency": "CNY",
                    "travel_interests": [],
                    "budget_range": None,
                    "notification_settings": {}
                }
                
        except Exception as e:
            logger.error(f"获取用户偏好失败: {e}")
            raise HTTPException(status_code=500, detail="获取偏好失败")


@app.put("/api/v1/users/me/preferences")
async def update_user_preferences(
    preferences: UserPreferences,
    current_user: dict = Depends(get_current_user)
):
    """更新用户偏好"""
    async with await db.get_connection() as conn:
        try:
            # 使用REPLACE INTO来插入或更新
            query = """
            REPLACE INTO user_preferences 
            (user_id, preferred_language, preferred_currency, travel_interests, budget_range, notification_settings)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            await conn.execute(query, (
                current_user["user_id"],
                preferences.preferred_language,
                preferences.preferred_currency,
                preferences.travel_interests,
                preferences.budget_range,
                preferences.notification_settings
            ))
            
            await conn.commit()
            
            return {"message": "偏好更新成功"}
            
        except Exception as e:
            await conn.rollback()
            logger.error(f"更新用户偏好失败: {e}")
            raise HTTPException(status_code=500, detail="更新偏好失败")


# 健康检查
@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    try:
        # 检查数据库连接
        async with await db.get_connection() as conn:
            await conn.execute("SELECT 1")
        
        # 检查Redis连接
        redis_client = app.state.redis_client
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "user-service",
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
        port=8003,
        reload=True,
        log_level="info"
    ) 