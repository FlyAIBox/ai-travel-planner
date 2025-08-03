"""
数据库连接管理
提供数据库连接池、会话管理和配置
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import logging

from sqlalchemy import create_engine, pool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 创建基础模型类
Base = declarative_base()


class Database:
    """数据库连接管理器"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """初始化数据库连接"""
        if self._initialized:
            return
        
        try:
            # 构建数据库URL
            database_url = self._build_database_url()
            async_database_url = self._build_async_database_url()
            
            # 创建同步引擎
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG,
                echo_pool=settings.DEBUG,
                future=True
            )
            
            # 创建异步引擎
            self.async_engine = create_async_engine(
                async_database_url,
                poolclass=pool.QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG,
                echo_pool=settings.DEBUG,
                future=True
            )
            
            # 创建会话工厂
            self.session_factory = sessionmaker(
                bind=self.engine,
                class_=Session,
                expire_on_commit=False
            )
            
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._initialized = True
            logger.info("数据库连接初始化成功")
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
    
    def _build_database_url(self) -> str:
        """构建数据库URL"""
        return (
            f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
            f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )
    
    def _build_async_database_url(self) -> str:
        """构建异步数据库URL"""
        return (
            f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
            f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )
    
    def get_session(self) -> Session:
        """获取同步数据库会话"""
        if not self._initialized:
            raise RuntimeError("数据库未初始化")
        return self.session_factory()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取异步数据库会话"""
        if not self._initialized:
            raise RuntimeError("数据库未初始化")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    async def close(self):
        """关闭数据库连接"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        
        self._initialized = False
        logger.info("数据库连接已关闭")


# 全局数据库实例
_database: Optional[Database] = None


async def get_database() -> Database:
    """获取数据库实例"""
    global _database
    
    if _database is None:
        _database = Database()
        await _database.initialize()
    
    return _database


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（依赖注入用）"""
    database = await get_database()
    async with database.get_async_session() as session:
        yield session


@asynccontextmanager
async def get_sync_session() -> AsyncGenerator[Session, None]:
    """获取同步会话"""
    database = await get_database()
    session = database.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class DatabaseHealthCheck:
    """数据库健康检查"""
    
    def __init__(self, database: Database):
        self.database = database
    
    async def check_health(self) -> dict:
        """检查数据库健康状态"""
        try:
            # 测试连接
            connection_ok = await self.database.test_connection()
            
            if not connection_ok:
                return {
                    "status": "unhealthy",
                    "message": "数据库连接失败",
                    "details": {}
                }
            
            # 获取连接池状态
            pool_status = {}
            if self.database.async_engine:
                pool = self.database.async_engine.pool
                pool_status = {
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                }
            
            return {
                "status": "healthy",
                "message": "数据库连接正常",
                "details": {
                    "pool_status": pool_status,
                    "engine_info": {
                        "driver": "mysql+aiomysql",
                        "database": settings.MYSQL_DATABASE,
                        "host": settings.MYSQL_HOST,
                        "port": settings.MYSQL_PORT
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"健康检查失败: {str(e)}",
                "details": {}
            } 