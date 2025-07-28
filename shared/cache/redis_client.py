"""
Redis客户端模块
提供Redis连接和操作功能
"""

import logging
from typing import Optional, Any, Dict
import json
import asyncio
from functools import lru_cache

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisClient:
    """Redis客户端类"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def ping(self) -> bool:
        """检查Redis连接"""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping失败: {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """获取值"""
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get失败 {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """设置值"""
        try:
            if expire:
                return await self.redis.setex(key, expire, value)
            else:
                return await self.redis.set(key, value)
        except Exception as e:
            logger.error(f"Redis set失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除键"""
        try:
            return bool(await self.redis.delete(key))
        except Exception as e:
            logger.error(f"Redis delete失败 {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Redis exists失败 {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        try:
            return await self.redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Redis expire失败 {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """获取JSON值"""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get_json失败 {key}: {e}")
            return None
    
    async def set_json(self, key: str, value: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """设置JSON值"""
        try:
            json_str = json.dumps(value, ensure_ascii=False)
            return await self.set(key, json_str, expire)
        except Exception as e:
            logger.error(f"Redis set_json失败 {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """增加数值"""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis increment失败 {key}: {e}")
            return None
    
    async def hash_get(self, key: str, field: str) -> Optional[str]:
        """获取哈希字段值"""
        try:
            return await self.redis.hget(key, field)
        except Exception as e:
            logger.error(f"Redis hget失败 {key}.{field}: {e}")
            return None
    
    async def hash_set(self, key: str, field: str, value: str) -> bool:
        """设置哈希字段值"""
        try:
            return bool(await self.redis.hset(key, field, value))
        except Exception as e:
            logger.error(f"Redis hset失败 {key}.{field}: {e}")
            return False
    
    async def hash_get_all(self, key: str) -> Optional[Dict[str, str]]:
        """获取所有哈希字段"""
        try:
            return await self.redis.hgetall(key)
        except Exception as e:
            logger.error(f"Redis hgetall失败 {key}: {e}")
            return None
    
    async def list_push(self, key: str, value: str) -> Optional[int]:
        """向列表推入值"""
        try:
            return await self.redis.lpush(key, value)
        except Exception as e:
            logger.error(f"Redis lpush失败 {key}: {e}")
            return None
    
    async def list_pop(self, key: str) -> Optional[str]:
        """从列表弹出值"""
        try:
            return await self.redis.rpop(key)
        except Exception as e:
            logger.error(f"Redis rpop失败 {key}: {e}")
            return None
    
    async def list_length(self, key: str) -> int:
        """获取列表长度"""
        try:
            return await self.redis.llen(key)
        except Exception as e:
            logger.error(f"Redis llen失败 {key}: {e}")
            return 0
    
    async def close(self):
        """关闭连接"""
        try:
            await self.redis.close()
        except Exception as e:
            logger.error(f"关闭Redis连接失败: {e}")


# 全局Redis实例
_redis_pools: Dict[int, ConnectionPool] = {}
_redis_clients: Dict[int, RedisClient] = {}


async def get_redis_pool(db: int = 0) -> ConnectionPool:
    """获取Redis连接池"""
    if db not in _redis_pools:
        _redis_pools[db] = ConnectionPool.from_url(
            settings.REDIS_URL,
            db=db,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            retry_on_timeout=True
        )
    return _redis_pools[db]


async def get_redis_client(db: int = 0) -> RedisClient:
    """获取Redis客户端"""
    if db not in _redis_clients:
        pool = await get_redis_pool(db)
        redis_client = Redis(connection_pool=pool)
        _redis_clients[db] = RedisClient(redis_client)
    return _redis_clients[db]


async def close_all_redis_connections():
    """关闭所有Redis连接"""
    for client in _redis_clients.values():
        await client.close()
    
    for pool in _redis_pools.values():
        await pool.disconnect()
    
    _redis_clients.clear()
    _redis_pools.clear()


# 便捷函数
async def get_session_redis() -> RedisClient:
    """获取会话Redis客户端"""
    return await get_redis_client(settings.REDIS_DB_SESSION)


async def get_cache_redis() -> RedisClient:
    """获取缓存Redis客户端"""
    return await get_redis_client(settings.REDIS_DB_CACHE)


async def get_queue_redis() -> RedisClient:
    """获取队列Redis客户端"""
    return await get_redis_client(settings.REDIS_DB_QUEUE)


async def get_agent_redis() -> RedisClient:
    """获取智能体Redis客户端"""
    return await get_redis_client(settings.REDIS_DB_AGENT) 