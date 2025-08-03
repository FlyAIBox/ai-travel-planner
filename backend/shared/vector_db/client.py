"""
Qdrant向量数据库客户端管理
提供连接池、健康检查和操作接口
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import Distance, VectorParams, CollectionInfo

from shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QdrantManager:
    """Qdrant向量数据库管理器"""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self._initialized = False
        self._collections_cache: Dict[str, CollectionInfo] = {}
    
    async def initialize(self):
        """初始化Qdrant客户端"""
        if self._initialized:
            return
        
        try:
            # 创建Qdrant客户端
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                grpc_port=settings.QDRANT_GRPC_PORT,
                prefer_grpc=True,
                timeout=30,
                # API密钥（如果配置）
                api_key=getattr(settings, 'QDRANT_API_KEY', None)
            )
            
            # 测试连接
            health_check = await self.health_check()
            if not health_check["healthy"]:
                raise ConnectionError(f"Qdrant连接失败: {health_check['message']}")
            
            self._initialized = True
            logger.info("Qdrant向量数据库初始化成功")
            
        except Exception as e:
            logger.error(f"Qdrant初始化失败: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self.client:
                return {"healthy": False, "message": "客户端未初始化"}
            
            # 检查Qdrant服务状态
            health = self.client.get_cluster_info()
            
            return {
                "healthy": True,
                "message": "Qdrant服务正常",
                "details": {
                    "status": health.status,
                    "peer_count": health.peer_count,
                    "pending_operations": health.pending_operations,
                    "host": settings.QDRANT_HOST,
                    "port": settings.QDRANT_PORT
                }
            }
            
        except Exception as e:
            logger.error(f"Qdrant健康检查失败: {e}")
            return {
                "healthy": False,
                "message": f"健康检查失败: {str(e)}",
                "details": {}
            }
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        **kwargs
    ) -> bool:
        """创建向量集合"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            # 检查集合是否已存在
            if await self.collection_exists(collection_name):
                logger.info(f"集合 {collection_name} 已存在")
                return True
            
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                ),
                **kwargs
            )
            
            # 清除缓存
            self._collections_cache.pop(collection_name, None)
            
            logger.info(f"成功创建集合: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败 {collection_name}: {e}")
            return False
    
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            if not self.client:
                return False
            
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
            
        except Exception as e:
            logger.error(f"检查集合存在性失败 {collection_name}: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """获取集合信息"""
        try:
            if not self.client:
                return None
            
            # 使用缓存
            if collection_name in self._collections_cache:
                return self._collections_cache[collection_name]
            
            collection_info = self.client.get_collection(collection_name)
            self._collections_cache[collection_name] = collection_info
            
            return collection_info
            
        except Exception as e:
            logger.error(f"获取集合信息失败 {collection_name}: {e}")
            return None
    
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Union[int, str]]] = None
    ) -> bool:
        """插入向量"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            # 准备数据
            points = []
            for i, vector in enumerate(vectors):
                point_id = ids[i] if ids else i
                payload = payloads[i] if payloads else {}
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            # 批量插入
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            if operation_info.status == models.UpdateStatus.COMPLETED:
                logger.info(f"成功插入 {len(vectors)} 个向量到集合 {collection_name}")
                return True
            else:
                logger.error(f"向量插入未完成: {operation_info.status}")
                return False
                
        except Exception as e:
            logger.error(f"插入向量失败: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        payload_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索向量"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            # 构建过滤条件
            filter_conditions = None
            if payload_filter:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in payload_filter.items()
                    ]
                )
            
            # 执行搜索
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # 格式化结果
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload or {}
                })
            
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[Union[int, str]]
    ) -> bool:
        """删除向量"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            
            if operation_info.status == models.UpdateStatus.COMPLETED:
                logger.info(f"成功删除 {len(ids)} 个向量")
                return True
            else:
                logger.error(f"向量删除未完成: {operation_info.status}")
                return False
                
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            self.client.delete_collection(collection_name)
            
            # 清除缓存
            self._collections_cache.pop(collection_name, None)
            
            logger.info(f"成功删除集合: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败 {collection_name}: {e}")
            return False
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.client:
                return {}
            
            collection_info = await self.get_collection_info(collection_name)
            if not collection_info:
                return {}
            
            return {
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count
            }
            
        except Exception as e:
            logger.error(f"获取集合统计失败 {collection_name}: {e}")
            return {}
    
    async def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """备份集合"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            # 创建快照
            snapshot_info = self.client.create_snapshot(collection_name)
            
            logger.info(f"集合 {collection_name} 备份成功: {snapshot_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"备份集合失败 {collection_name}: {e}")
            return False
    
    async def optimize_collection(self, collection_name: str) -> bool:
        """优化集合"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant客户端未初始化")
            
            # 手动触发优化
            operation_info = self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            
            if operation_info.status == models.UpdateStatus.COMPLETED:
                logger.info(f"集合 {collection_name} 优化成功")
                return True
            else:
                logger.warning(f"集合优化未完成: {operation_info.status}")
                return False
                
        except Exception as e:
            logger.error(f"优化集合失败 {collection_name}: {e}")
            return False
    
    async def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            self.client = None
            self._initialized = False
            self._collections_cache.clear()
            logger.info("Qdrant连接已关闭")


# 全局Qdrant管理器实例
_qdrant_manager: Optional[QdrantManager] = None


async def get_qdrant_client() -> QdrantManager:
    """获取Qdrant管理器实例"""
    global _qdrant_manager
    
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager()
        await _qdrant_manager.initialize()
    
    return _qdrant_manager


@asynccontextmanager
async def get_qdrant_session():
    """获取Qdrant会话（上下文管理器）"""
    qdrant = await get_qdrant_client()
    try:
        yield qdrant
    except Exception as e:
        logger.error(f"Qdrant操作错误: {e}")
        raise
    finally:
        # 这里可以添加清理逻辑
        pass


class QdrantHealthCheck:
    """Qdrant健康检查"""
    
    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant_manager = qdrant_manager
    
    async def check_health(self) -> Dict[str, Any]:
        """检查Qdrant健康状态"""
        return await self.qdrant_manager.health_check()
    
    async def check_collections_health(self, collection_names: List[str]) -> Dict[str, Any]:
        """检查集合健康状态"""
        results = {}
        
        for collection_name in collection_names:
            try:
                stats = await self.qdrant_manager.get_collection_stats(collection_name)
                exists = await self.qdrant_manager.collection_exists(collection_name)
                
                results[collection_name] = {
                    "exists": exists,
                    "healthy": exists and stats.get("status") == "green",
                    "stats": stats
                }
                
            except Exception as e:
                results[collection_name] = {
                    "exists": False,
                    "healthy": False,
                    "error": str(e)
                }
        
        return results 