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
        stats = self.connection_stats[node_id]
        stats["requests"] += 1
        stats["last_used"] = datetime.now()
        stats["response_times"].append(response_time)
        
        if not success:
            stats["errors"] += 1
        
        # 保持最近100次响应时间
        if len(stats["response_times"]) > 100:
            stats["response_times"] = stats["response_times"][-100:]
    
    def get_best_node(self) -> str:
        """获取最佳节点"""
        best_node = None
        best_score = float('inf')
        
        for node_id, stats in self.connection_stats.items():
            if stats["requests"] == 0:
                return node_id  # 优先使用未使用的节点
            
            # 计算综合分数（错误率 + 平均响应时间）
            error_rate = stats["errors"] / stats["requests"]
            avg_response_time = np.mean(stats["response_times"]) if stats["response_times"] else 0
            score = error_rate * 1000 + avg_response_time  # 错误率权重更高
            
            if score < best_score:
                best_score = score
                best_node = node_id
        
        return best_node or "node_0"
    
    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        health_status = {}
        
        for i, node in enumerate(self.nodes):
            node_id = f"node_{i}"
            try:
                client = await self.get_client(node_id)
                await client.get_collections()
                health_status[node_id] = True
            except Exception as e:
                logger.error(f"节点 {node_id} 健康检查失败: {e}")
                health_status[node_id] = False
        
        return health_status


class VectorDatabase:
    """向量数据库管理器"""
    
    def __init__(self, 
                 nodes: List[Dict[str, Any]] = None,
                 default_collection: str = None):
        # 默认配置
        if nodes is None:
            nodes = [{
                "host": settings.QDRANT_HOST,
                "port": settings.QDRANT_PORT,
                "api_key": settings.QDRANT_API_KEY,
                "timeout": 30,
                "prefer_grpc": True
            }]
        
        self.pool = VectorDatabasePool(nodes)
        self.default_collection = default_collection or settings.QDRANT_COLLECTION_NAME
        
        # 缓存
        self.collection_cache: Dict[str, CollectionInfo] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # 性能监控
        self.performance_stats = {
            "total_operations": 0,
            "total_errors": 0,
            "operation_times": [],
            "start_time": datetime.now()
        }
        
        # 备份配置
        self.backup_config = {
            "enabled": True,
            "backup_dir": Path("data/backups/qdrant"),
            "retention_days": 30,
            "auto_backup_interval": timedelta(hours=6)
        }
        
        # 确保备份目录存在
        self.backup_config["backup_dir"].mkdir(parents=True, exist_ok=True)
    
    async def create_collection(self, 
                              collection_name: str,
                              vector_config: VectorIndexConfig,
                              shard_number: int = 1,
                              replication_factor: int = 1,
                              write_consistency_factor: int = 1) -> bool:
        """创建集合"""
        start_time = time.time()
        
        try:
            client = await self.pool.get_client()
            
            # 配置向量参数
            vectors_config = models.VectorParams(
                size=vector_config.vector_size,
                distance=models.Distance(vector_config.distance)
            )
            
            # 配置HNSW索引
            hnsw_config = models.HnswConfigDiff(
                m=vector_config.hnsw_config.get("m", 16),
                ef_construct=vector_config.hnsw_config.get("ef_construct", 100),
                full_scan_threshold=vector_config.hnsw_config.get("full_scan_threshold", 10000),
                max_indexing_threads=vector_config.hnsw_config.get("max_indexing_threads", 4)
            )
            
            # 配置量化
            quantization_config = None
            if vector_config.quantization_config:
                quantization_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=vector_config.quantization_config["scalar"].get("quantile", 0.99),
                        always_ram=vector_config.quantization_config["scalar"].get("always_ram", True)
                    )
                )
            
            # 配置优化器
            optimizers_config = models.OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=shard_number,
                max_segment_size=None,
                memmap_threshold=None,
                indexing_threshold=20000,
                flush_interval_sec=5,
                max_optimization_threads=4
            )
            
            # 创建集合
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                hnsw_config=hnsw_config,
                quantization_config=quantization_config,
                optimizers_config=optimizers_config
            )
            
            # 创建索引
            await self._create_payload_indexes(client, collection_name)
            
            execution_time = time.time() - start_time
            self._record_operation("create_collection", execution_time, True)
            
            logger.info(f"成功创建集合: {collection_name}")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_operation("create_collection", execution_time, False)
            logger.error(f"创建集合失败: {e}")
            return False
    
    async def _create_payload_indexes(self, client: QdrantClient, collection_name: str) -> None:
        """创建载荷索引"""
        # 常用字段索引
        indexes = [
            ("document_type", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
            ("timestamp", models.PayloadSchemaType.DATETIME),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("language", models.PayloadSchemaType.KEYWORD),
            ("metadata.priority", models.PayloadSchemaType.INTEGER),
        ]
        
        for field_name, field_type in indexes:
            try:
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"创建索引: {collection_name}.{field_name}")
            except Exception as e:
                logger.warning(f"创建索引失败 {field_name}: {e}")
    
    async def insert_vectors(self, 
                           collection_name: str,
                           vectors: List[List[float]],
                           payloads: List[Dict[str, Any]],
                           ids: Optional[List[str]] = None,
                           batch_size: int = 100) -> bool:
        """插入向量"""
        start_time = time.time()
        
        try:
            client = await self.pool.get_client()
            
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            # 分批插入
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
                
                await client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                
                logger.info(f"插入批次 {i//batch_size + 1}: {len(points)} 个向量")
            
            execution_time = time.time() - start_time
            self._record_operation("insert_vectors", execution_time, True)
            
            logger.info(f"成功插入 {len(vectors)} 个向量到集合 {collection_name}")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_operation("insert_vectors", execution_time, False)
            logger.error(f"插入向量失败: {e}")
            return False
    
    async def search_vectors(self,
                           collection_name: str,
                           query_vector: List[float],
                           limit: int = 10,
                           score_threshold: Optional[float] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None,
                           with_payload: bool = True,
                           with_vectors: bool = False) -> List[VectorSearchResult]:
        """搜索向量"""
        start_time = time.time()
        
        try:
            client = await self.pool.get_client(self.pool.get_best_node())
            
            # 构建过滤条件
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            # 执行搜索
            search_result = await client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            # 转换结果
            results = []
            for scored_point in search_result:
                result = VectorSearchResult(
                    id=str(scored_point.id),
                    score=scored_point.score,
                    payload=scored_point.payload or {},
                    vector=scored_point.vector if with_vectors else None
                )
                results.append(result)
            
            execution_time = time.time() - start_time
            self._record_operation("search_vectors", execution_time, True)
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_operation("search_vectors", execution_time, False)
            logger.error(f"搜索向量失败: {e}")
            return []
    
    def _build_filter(self, conditions: Dict[str, Any]) -> models.Filter:
        """构建查询过滤器"""
        must_conditions = []
        should_conditions = []
        must_not_conditions = []
        
        for key, value in conditions.items():
            if key.startswith("must_"):
                field_name = key[5:]  # 移除 "must_" 前缀
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchValue(value=value)
                        )
                    )
            elif key.startswith("should_"):
                field_name = key[7:]  # 移除 "should_" 前缀
                if isinstance(value, list):
                    should_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    should_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchValue(value=value)
                        )
                    )
            elif key.startswith("not_"):
                field_name = key[4:]  # 移除 "not_" 前缀
                if isinstance(value, list):
                    must_not_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    must_not_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchValue(value=value)
                        )
                    )
            elif key.startswith("range_"):
                field_name = key[6:]  # 移除 "range_" 前缀
                if isinstance(value, dict) and ("gte" in value or "lte" in value or "gt" in value or "lt" in value):
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                                gt=value.get("gt"),
                                lt=value.get("lt")
                            )
                        )
                    )
        
        return models.Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )
    
    async def delete_vectors(self,
                           collection_name: str,
                           point_ids: List[str] = None,
                           filter_conditions: Dict[str, Any] = None) -> bool:
        """删除向量"""
        start_time = time.time()
        
        try:
            client = await self.pool.get_client()
            
            if point_ids:
                # 按ID删除
                await client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    ),
                    wait=True
                )
                logger.info(f"删除了 {len(point_ids)} 个向量")
            elif filter_conditions:
                # 按条件删除
                query_filter = self._build_filter(filter_conditions)
                await client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=query_filter
                    ),
                    wait=True
                )
                logger.info(f"按条件删除向量: {filter_conditions}")
            
            execution_time = time.time() - start_time
            self._record_operation("delete_vectors", execution_time, True)
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_operation("delete_vectors", execution_time, False)
            logger.error(f"删除向量失败: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str, use_cache: bool = True) -> Optional[CollectionInfo]:
        """获取集合信息"""
        # 检查缓存
        if use_cache and collection_name in self.collection_cache:
            cached_time = self.cache_timestamps.get(collection_name)
            if cached_time and datetime.now() - cached_time < self.cache_ttl:
                return self.collection_cache[collection_name]
        
        try:
            client = await self.pool.get_client()
            collection_info = await client.get_collection(collection_name)
            
            info = CollectionInfo(
                name=collection_name,
                vectors_count=collection_info.vectors_count,
                indexed_vectors_count=collection_info.indexed_vectors_count,
                points_count=collection_info.points_count,
                segments_count=collection_info.segments_count,
                status=collection_info.status.value,
                optimizer_status=collection_info.optimizer_status.status.value,
                disk_usage=collection_info.disk_usage
            )
            
            # 更新缓存
            self.collection_cache[collection_name] = info
            self.cache_timestamps[collection_name] = datetime.now()
            
            return info
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None
    
    async def optimize_collection(self, collection_name: str) -> bool:
        """优化集合"""
        try:
            client = await self.pool.get_client()
            
            # 执行优化
            await client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,
                    max_optimization_threads=4
                )
            )
            
            logger.info(f"集合优化已启动: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"集合优化失败: {e}")
            return False
    
    async def create_snapshot(self, collection_name: str) -> Optional[str]:
        """创建快照"""
        try:
            client = await self.pool.get_client()
            
            # 创建快照
            snapshot_info = await client.create_snapshot(collection_name=collection_name)
            snapshot_name = snapshot_info.name
            
            # 下载快照到本地
            backup_path = self.backup_config["backup_dir"] / f"{collection_name}_{snapshot_name}.snapshot"
            await client.download_snapshot(
                collection_name=collection_name,
                snapshot_name=snapshot_name,
                output_path=str(backup_path)
            )
            
            logger.info(f"快照已创建并下载: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"创建快照失败: {e}")
            return None
    
    async def restore_snapshot(self, collection_name: str, snapshot_path: str) -> bool:
        """恢复快照"""
        try:
            client = await self.pool.get_client()
            
            # 上传并恢复快照
            await client.restore_snapshot(
                collection_name=collection_name,
                snapshot_path=snapshot_path
            )
            
            logger.info(f"快照已恢复: {snapshot_path}")
            return True
            
        except Exception as e:
            logger.error(f"恢复快照失败: {e}")
            return False
    
    async def cleanup_old_backups(self) -> None:
        """清理旧备份"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_config["retention_days"])
            backup_dir = self.backup_config["backup_dir"]
            
            removed_count = 0
            for backup_file in backup_dir.glob("*.snapshot"):
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"清理了 {removed_count} 个旧备份文件")
                
        except Exception as e:
            logger.error(f"清理备份失败: {e}")
    
    def _record_operation(self, operation: str, execution_time: float, success: bool) -> None:
        """记录操作统计"""
        self.performance_stats["total_operations"] += 1
        if not success:
            self.performance_stats["total_errors"] += 1
        
        self.performance_stats["operation_times"].append(execution_time)
        
        # 保持最近1000次操作时间
        if len(self.performance_stats["operation_times"]) > 1000:
            self.performance_stats["operation_times"] = self.performance_stats["operation_times"][-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        operation_times = self.performance_stats["operation_times"]
        
        stats = {
            "total_operations": self.performance_stats["total_operations"],
            "total_errors": self.performance_stats["total_errors"],
            "error_rate": self.performance_stats["total_errors"] / max(self.performance_stats["total_operations"], 1),
            "uptime_seconds": (datetime.now() - self.performance_stats["start_time"]).total_seconds()
        }
        
        if operation_times:
            stats.update({
                "avg_operation_time": np.mean(operation_times),
                "min_operation_time": np.min(operation_times),
                "max_operation_time": np.max(operation_times),
                "p95_operation_time": np.percentile(operation_times, 95),
                "p99_operation_time": np.percentile(operation_times, 99)
            })
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "nodes": {},
            "collections": {},
            "performance": self.get_performance_stats()
        }
        
        try:
            # 检查节点健康状态
            node_health = await self.pool.health_check()
            health_info["nodes"] = node_health
            
            # 检查集合状态
            client = await self.pool.get_client()
            collections = await client.get_collections()
            
            for collection in collections.collections:
                collection_info = await self.get_collection_info(collection.name, use_cache=False)
                if collection_info:
                    health_info["collections"][collection.name] = {
                        "status": collection_info.status,
                        "points_count": collection_info.points_count,
                        "vectors_count": collection_info.vectors_count
                    }
            
            # 判断整体健康状态
            unhealthy_nodes = [node for node, healthy in node_health.items() if not healthy]
            if unhealthy_nodes:
                health_info["status"] = "degraded"
                health_info["unhealthy_nodes"] = unhealthy_nodes
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.error(f"健康检查失败: {e}")
        
        return health_info


# 全局实例
vector_database = None

def get_vector_database() -> VectorDatabase:
    """获取向量数据库实例"""
    global vector_database
    if vector_database is None:
        vector_database = VectorDatabase()
    return vector_database 