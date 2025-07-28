"""
数据库连接配置模块
管理异步数据库连接、会话和事务
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ==================== 基础模型类 ====================
class Base(DeclarativeBase):
    """SQLAlchemy基础模型类"""
    pass


# ==================== 数据库引擎配置 ====================
class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self._async_engine: Optional[AsyncEngine] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
    
    def get_async_engine(self) -> AsyncEngine:
        """获取异步数据库引擎"""
        if self._async_engine is None:
            # 创建异步引擎
            self._async_engine = create_async_engine(
                settings.DATABASE_URL,
                # 连接池配置
                poolclass=QueuePool,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1小时回收连接
                
                # 引擎配置
                echo=settings.ENABLE_QUERY_LOG,
                echo_pool=settings.DEBUG,
                future=True,
                
                # 连接参数
                connect_args={
                    "charset": "utf8mb4",
                    "autocommit": False,
                }
            )
            
            # 添加连接事件监听器
            self._setup_engine_events(self._async_engine)
        
        return self._async_engine
    
    def get_async_session_factory(self) -> async_sessionmaker:
        """获取异步会话工厂"""
        if self._async_session_factory is None:
            engine = self.get_async_engine()
            self._async_session_factory = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
        return self._async_session_factory
    
    def _setup_engine_events(self, engine: AsyncEngine):
        """设置引擎事件监听器"""
        
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """设置MySQL连接参数"""
            if "mysql" in str(engine.url):
                cursor = dbapi_connection.cursor()
                # 设置字符集
                cursor.execute("SET NAMES utf8mb4")
                # 设置时区
                cursor.execute("SET time_zone = '+00:00'")
                # 设置SQL模式
                cursor.execute("SET sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'")
                cursor.close()
        
        @event.listens_for(engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出事件"""
            logger.debug("数据库连接已检出")
        
        @event.listens_for(engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """连接检入事件"""
            logger.debug("数据库连接已检入")
    
    async def close(self):
        """关闭数据库连接"""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_session_factory = None
            logger.info("数据库连接已关闭")


# ==================== 全局数据库管理器 ====================
db_manager = DatabaseManager()


# ==================== 便捷函数 ====================
def get_async_engine() -> AsyncEngine:
    """获取异步数据库引擎"""
    return db_manager.get_async_engine()


def get_async_session_factory() -> async_sessionmaker:
    """获取异步会话工厂"""
    return db_manager.get_async_session_factory()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话（上下文管理器）"""
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"数据库会话错误: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_async_transaction() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库事务（自动提交/回滚）"""
    async with get_async_session() as session:
        try:
            yield session
            await session.commit()
            logger.debug("数据库事务已提交")
        except Exception as e:
            logger.error(f"数据库事务错误: {e}")
            await session.rollback()
            raise


# ==================== 数据库工具函数 ====================
async def check_database_connection() -> bool:
    """检查数据库连接"""
    try:
        async with get_async_session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"数据库连接检查失败: {e}")
        return False


async def get_database_info() -> dict:
    """获取数据库信息"""
    try:
        async with get_async_session() as session:
            # 获取数据库版本
            version_result = await session.execute(text("SELECT VERSION()"))
            version = version_result.scalar()
            
            # 获取当前数据库名
            db_result = await session.execute(text("SELECT DATABASE()"))
            database_name = db_result.scalar()
            
            # 获取字符集
            charset_result = await session.execute(text("SHOW VARIABLES LIKE 'character_set_database'"))
            charset_row = charset_result.fetchone()
            charset = charset_row[1] if charset_row else "unknown"
            
            return {
                "version": version,
                "database_name": database_name,
                "charset": charset,
                "url": str(get_async_engine().url).replace(get_async_engine().url.password or "", "***"),
            }
    except Exception as e:
        logger.error(f"获取数据库信息失败: {e}")
        return {"error": str(e)}


async def create_database_tables():
    """创建数据库表"""
    engine = get_async_engine()
    async with engine.begin() as conn:
        # 导入所有ORM模型以确保表被创建
        from shared.database import models  # noqa
        
        await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表创建完成")


async def drop_database_tables():
    """删除数据库表（危险操作）"""
    if not settings.DEBUG:
        raise RuntimeError("只能在调试模式下删除数据库表")
    
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logger.warning("数据库表已删除")


# ==================== 会话依赖注入 ====================
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI依赖注入：获取数据库会话"""
    async with get_async_session() as session:
        yield session


async def get_db_transaction() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI依赖注入：获取数据库事务"""
    async with get_async_transaction() as session:
        yield session 