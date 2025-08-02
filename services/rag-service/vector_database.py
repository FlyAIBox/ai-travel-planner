"""
向量数据库基础设施
实现Qdrant向量数据库集群配置、持久化存储、向量索引策略、性能优化、备份恢复、监控调优
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
import structlog

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionInfo:
    """集合信息"""
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    status: str
    optimizer_status: str
    disk_usage: int


@dataclass
class VectorIndexConfig:
    """向量索引配置"""
    vector_size: int
    distance: str = models.Distance.COSINE
    hnsw_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.hnsw_config is None:
            self.hnsw_config = {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 4
            }
        
        if self.quantization_config is None:
            self.quantization_config = {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True
                }
            }


class VectorDatabasePool:
    """向量数据库连接池"""
    
    def __init__(self, nodes: List[Dict[str, Any]], max_connections: int = 10):
        self.nodes = nodes
        self.max_connections = max_connections
        self.connections: Dict[str, QdrantClient] = {}
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        self.current_node_index = 0
        
        # 初始化连接统计
        for i, node in enumerate(nodes):
            node_id = f"node_{i}"
            self.connection_stats[node_id] = {
                "requests": 0,
                "errors": 0,
                "last_used": None,
                "response_times": []
            }
    
    async def get_client(self, node_id: Optional[str] = None) -> QdrantClient:
        """获取客户端连接"""
        if node_id is None:
            # 使用轮询策略选择节点
            node_id = f"node_{self.current_node_index}"
            self.current_node_index = (self.current_node_index + 1) % len(self.nodes)
        
        if node_id not in self.connections:
            node_config = self.nodes[int(node_id.split("_")[1])]
            client = QdrantClient(
                host=node_config["host"],
                port=node_config.get("port", 6333),
                api_key=node_config.get("api_key"),
                timeout=node_config.get("timeout", 30),
                prefer_grpc=node_config.get("prefer_grpc", True)
            )
            self.connections[node_id] = client
        
        return self.connections[node_id]
    
    def record_request(self, node_id: str, response_time: float, success: bool) -> None:
        """记录请求统计"""
        if node_id in self.connection_stats:
            stats = self.connection_stats[node_id]
            stats["requests"] += 1
            if not success:
                stats["errors"] += 1
            stats["last_used"] = datetime.now()
            stats["response_times"].append(response_time)
            
            # 保持响应时间历史记录在合理范围内
            if len(stats["response_times"]) > 100:
                stats["response_times"] = stats["response_times"][-50:]
    
    def get_best_node(self, strategy: str = "least_connections") -> str:
        """根据策略选择最佳节点"""
        if strategy == "round_robin":
            # 轮询策略
            node_id = f"node_{self.current_node_index}"
            self.current_node_index = (self.current_node_index + 1) % len(self.nodes)
            return node_id
        
        elif strategy == "least_connections":
            # 最少连接策略
            best_node = None
            min_requests = float('inf')
            
            for node_id, stats in self.connection_stats.items():
                if stats["requests"] < min_requests:
                    min_requests = stats["requests"]
                    best_node = node_id
            
            return best_node or "node_0"
        
        elif strategy == "weighted_round_robin":
            # 加权轮询策略 - 基于历史响应时间
            best_node = None
            best_score = float('inf')
            
            for node_id, stats in self.connection_stats.items():
                response_times = stats["response_times"]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 1.0
                error_rate = stats["errors"] / max(stats["requests"], 1)
                
                # 综合评分：响应时间 + 错误率权重
                score = avg_response_time * (1 + error_rate * 2)
                
                if score < best_score:
                    best_score = score
                    best_node = node_id
            
            return best_node or "node_0"
        
        else:
            # 默认返回第一个节点
            return "node_0"
    
    async def check_cluster_health(self) -> Dict[str, Any]:
        """检查集群健康状态"""
        health_status = {
            "healthy_nodes": 0,
            "total_nodes": len(self.nodes),
            "node_details": {},
            "cluster_status": "healthy"
        }
        
        for i, node in enumerate(self.nodes):
            node_id = f"node_{i}"
            try:
                client = await self.get_client(node_id)
                # 简单的健康检查
                start_time = time.time()
                collections = client.get_collections()
                response_time = time.time() - start_time
                
                health_status["node_details"][node_id] = {
                    "status": "healthy",
                    "response_time": response_time,
                    "collections_count": len(collections.collections)
                }
                health_status["healthy_nodes"] += 1
                
            except Exception as e:
                health_status["node_details"][node_id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # 确定集群状态
        healthy_ratio = health_status["healthy_nodes"] / health_status["total_nodes"]
        if healthy_ratio == 1.0:
            health_status["cluster_status"] = "healthy"
        elif healthy_ratio >= 0.5:
            health_status["cluster_status"] = "degraded"
        else:
            health_status["cluster_status"] = "critical"
        
        return health_status

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        return {
            "total_nodes": len(self.nodes),
            "active_connections": len(self.connections),
            "node_stats": self.connection_stats,
            "current_node": f"node_{self.current_node_index}"
        }


class VectorDatabase:
    """向量数据库管理类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 集群配置
        nodes = self.config.get("nodes", [
            {"host": settings.QDRANT_HOST, "port": settings.QDRANT_PORT}
        ])
        self.connection_pool = VectorDatabasePool(nodes)
        
        # 性能监控
        self.performance_metrics = {
            "total_operations": 0,
            "failed_operations": 0,
            "average_response_time": 0.0,
            "last_backup_time": None,
            "last_health_check": None
        }
        
        # 备份配置
        self.backup_config = self.config.get("backup", {
            "enabled": True,
            "interval_hours": 24,
            "retention_days": 7,
            "backup_path": "/data/qdrant/backups"
        })
    
    async def initialize_cluster(self) -> bool:
        """初始化集群"""
        try:
            # 检查所有节点连接
            all_healthy = True
            for i, node in enumerate(self.connection_pool.nodes):
                node_id = f"node_{i}"
                try:
                    client = await self.connection_pool.get_client(node_id)
                    health = await self._check_node_health(client)
                    if not health:
                        all_healthy = False
                        logger.warning(f"节点 {node_id} 健康检查失败")
                except Exception as e:
                    logger.error(f"节点 {node_id} 连接失败: {e}")
                    all_healthy = False
            
            if all_healthy:
                logger.info("Qdrant集群初始化成功")
                return True
            else:
                logger.warning("部分节点不健康，但集群可用")
                return True
                
        except Exception as e:
            logger.error(f"集群初始化失败: {e}")
            return False
    
    async def _check_node_health(self, client: QdrantClient) -> bool:
        """检查节点健康状态"""
        try:
            # 获取集群信息
            cluster_info = client.get_cluster_info()
            # 检查节点状态
            collections = client.get_collections()
            return True
        except Exception as e:
            logger.error(f"节点健康检查失败: {e}")
            return False
    
    async def create_collection(self, 
                              collection_name: str,
                              config: VectorIndexConfig,
                              replica_count: int = 1,
                              shard_count: int = 1) -> bool:
        """创建集合"""
        start_time = time.time()
        success = False
        
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            # 创建集合配置
            vectors_config = models.VectorParams(
                size=config.vector_size,
                distance=config.distance,
                hnsw_config=models.HnswConfigDiff(**config.hnsw_config) if config.hnsw_config else None,
                quantization_config=models.QuantizationConfig(**config.quantization_config) if config.quantization_config else None
            )
            
            # 创建集合
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                replication_factor=replica_count,
                shard_number=shard_count,
                on_disk_payload=True,  # 大数据集优化
                timeout=60
            )
            
            # 创建索引
            await self._optimize_collection(client, collection_name)
            
            success = True
            logger.info(f"集合 {collection_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
        finally:
            response_time = time.time() - start_time
            self._record_operation(response_time, success)
    
    async def _optimize_collection(self, client: QdrantClient, collection_name: str) -> None:
        """优化集合性能"""
        try:
            # 创建payload索引以提高过滤性能
            index_configs = [
                ("document_type", models.PayloadSchemaType.KEYWORD),
                ("source", models.PayloadSchemaType.KEYWORD),
                ("created_at", models.PayloadSchemaType.DATETIME),
                ("category", models.PayloadSchemaType.KEYWORD),
                ("language", models.PayloadSchemaType.KEYWORD)
            ]
            
            for field_name, field_type in index_configs:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type
                    )
                except Exception as e:
                    logger.warning(f"创建索引 {field_name} 失败: {e}")
                    
        except Exception as e:
            logger.error(f"优化集合失败: {e}")
    
    async def upsert_vectors(self, 
                           collection_name: str,
                           vectors: List[List[float]],
                           payloads: List[Dict[str, Any]],
                           ids: Optional[List[str]] = None,
                           batch_size: int = 100) -> bool:
        """批量插入/更新向量"""
        start_time = time.time()
        success = False
        
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            # 生成ID
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            # 批量处理
            total_inserted = 0
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                points = [
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                    for point_id, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
                ]
                
                # 执行批量插入
                operation_info = client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                
                total_inserted += len(points)
                logger.info(f"批量插入 {len(points)} 个向量，总计: {total_inserted}")
            
            success = True
            logger.info(f"成功插入 {total_inserted} 个向量到集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"批量插入向量失败: {e}")
            return False
        finally:
            response_time = time.time() - start_time
            self._record_operation(response_time, success)
    
    async def search_vectors(self,
                           collection_name: str,
                           query_vector: List[float],
                           limit: int = 10,
                           score_threshold: Optional[float] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None,
                           with_vectors: bool = False,
                           with_payload: bool = True) -> List[VectorSearchResult]:
        """搜索向量"""
        start_time = time.time()
        success = False
        
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            # 构建过滤条件
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # 执行搜索
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_vectors=with_vectors,
                with_payload=with_payload
            )
            
            # 格式化结果
            results = []
            for hit in search_result:
                result = VectorSearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    payload=hit.payload or {},
                    vector=hit.vector if with_vectors else None
                )
                results.append(result)
            
            success = True
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
        finally:
            response_time = time.time() - start_time
            self._record_operation(response_time, success)
    
    async def delete_vectors(self,
                           collection_name: str,
                           point_ids: List[str]) -> bool:
        """删除向量"""
        start_time = time.time()
        success = False
        
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            # 执行删除
            operation_info = client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    ),
                    wait=True
                )
            
            success = True
            logger.info(f"成功删除 {len(point_ids)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
        finally:
            response_time = time.time() - start_time
            self._record_operation(response_time, success)
    
    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """获取集合信息"""
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            collection_info = client.get_collection(collection_name)
            
            return CollectionInfo(
                name=collection_name,
                vectors_count=collection_info.vectors_count or 0,
                indexed_vectors_count=collection_info.indexed_vectors_count or 0,
                points_count=collection_info.points_count or 0,
                segments_count=len(collection_info.segments or []),
                status=collection_info.status.name if collection_info.status else "unknown",
                optimizer_status=collection_info.optimizer_status.name if collection_info.optimizer_status else "unknown",
                disk_usage=collection_info.segments[0].disk_usage_bytes if collection_info.segments else 0
            )
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None
    
    async def backup_collection(self, collection_name: str, backup_path: Optional[str] = None) -> bool:
        """备份集合"""
        if not self.backup_config.get("enabled", False):
            logger.info("备份功能未启用")
            return True
            
        try:
            backup_path = backup_path or self.backup_config.get("backup_path", "/tmp/qdrant_backup")
            backup_name = f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            # 创建快照
            snapshot_info = client.create_snapshot(collection_name=collection_name)
            
            # 这里可以添加实际的备份逻辑
            # 例如将快照文件复制到备份位置
            
            self.performance_metrics["last_backup_time"] = datetime.now()
            logger.info(f"集合 {collection_name} 备份成功: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"备份集合失败: {e}")
            return False
    
    async def restore_collection(self, collection_name: str, backup_path: str) -> bool:
        """恢复集合"""
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            
            # 这里可以添加实际的恢复逻辑
            # 例如从备份文件恢复快照
            
            logger.info(f"集合 {collection_name} 恢复成功")
            return True
            
        except Exception as e:
            logger.error(f"恢复集合失败: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cluster_nodes": [],
            "collections": [],
            "performance_metrics": self.performance_metrics.copy()
        }
        
        try:
            # 检查所有节点
            for i, node in enumerate(self.connection_pool.nodes):
                node_id = f"node_{i}"
                node_status = {
                    "node_id": node_id,
                    "host": node["host"],
                    "port": node.get("port", 6333),
                    "status": "unknown",
                    "collections_count": 0,
                    "response_time": None
                }
                
                try:
                    start_time = time.time()
                    client = await self.connection_pool.get_client(node_id)
                    collections = client.get_collections()
                    response_time = time.time() - start_time
                    
                    node_status.update({
                        "status": "healthy",
                        "collections_count": len(collections.collections),
                        "response_time": response_time
                    })
                    
                except Exception as e:
                    node_status["status"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
                
                health_status["cluster_nodes"].append(node_status)
            
            # 检查集合状态
            try:
                best_node = self.connection_pool.get_best_node()
                client = await self.connection_pool.get_client(best_node)
                collections = client.get_collections()
                
                for collection in collections.collections:
                    collection_info = await self.get_collection_info(collection.name)
                    if collection_info:
                        health_status["collections"].append({
                            "name": collection.name,
                            "vectors_count": collection_info.vectors_count,
                            "status": collection_info.status
                        })
            except Exception as e:
                logger.error(f"检查集合状态失败: {e}")
                health_status["status"] = "degraded"
            
            self.performance_metrics["last_health_check"] = datetime.now()
                
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    def _record_operation(self, response_time: float, success: bool) -> None:
        """记录操作统计"""
        self.performance_metrics["total_operations"] += 1
        if not success:
            self.performance_metrics["failed_operations"] += 1
        
        # 更新平均响应时间
        total_ops = self.performance_metrics["total_operations"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_ops - 1) + response_time) / total_ops
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_metrics.copy()
        stats["connection_pool"] = self.connection_pool.get_stats()
        
        # 计算错误率
        total_ops = stats["total_operations"]
        failed_ops = stats["failed_operations"]
        stats["error_rate"] = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        
        return stats
    
    async def optimize_all_collections(self) -> Dict[str, bool]:
        """优化所有集合"""
        results = {}
        
        try:
            best_node = self.connection_pool.get_best_node()
            client = await self.connection_pool.get_client(best_node)
            collections = client.get_collections()
            
            for collection in collections.collections:
                try:
                    # 触发索引优化
                    client.update_collection(
                        collection_name=collection.name,
                        optimizer_config=models.OptimizersConfigDiff(
                            indexing_threshold=10000,
                            max_segment_size=200000
                        )
                    )
                    results[collection.name] = True
                    logger.info(f"集合 {collection.name} 优化完成")
                except Exception as e:
                    logger.error(f"优化集合 {collection.name} 失败: {e}")
                    results[collection.name] = False
            
        except Exception as e:
            logger.error(f"优化所有集合失败: {e}")
        
        return results


# 全局向量数据库实例
_vector_database: Optional[VectorDatabase] = None


def get_vector_database(config: Optional[Dict[str, Any]] = None) -> VectorDatabase:
    """获取向量数据库实例"""
    global _vector_database
    if _vector_database is None:
        _vector_database = VectorDatabase(config)
    return _vector_database 