"""
Qdrant集合管理器
定义和管理旅行相关的向量集合
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
from shared.vector_db.client import QdrantManager

logger = logging.getLogger(__name__)


class TravelCollection(str, Enum):
    """旅行相关集合"""
    DESTINATIONS = "destinations"           # 目的地
    ACTIVITIES = "activities"              # 活动
    ACCOMMODATIONS = "accommodations"       # 住宿
    RESTAURANTS = "restaurants"            # 餐厅
    ATTRACTIONS = "attractions"            # 景点
    TRAVEL_GUIDES = "travel_guides"        # 旅行指南
    REVIEWS = "reviews"                    # 评价
    KNOWLEDGE_BASE = "knowledge_base"      # 知识库
    USER_PREFERENCES = "user_preferences"  # 用户偏好
    CONVERSATION_CONTEXT = "conversation_context"  # 对话上下文


class CollectionConfig:
    """集合配置"""
    
    COLLECTION_CONFIGS = {
        TravelCollection.DESTINATIONS: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "目的地向量集合",
            "payload_schema": {
                "destination_id": "string",
                "name": "string",
                "country": "string",
                "region": "string",
                "category": "string",
                "popularity_score": "float",
                "best_months": "integer[]",
                "tags": "string[]",
                "description": "string"
            }
        },
        
        TravelCollection.ACTIVITIES: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "活动向量集合",
            "payload_schema": {
                "activity_id": "string",
                "name": "string",
                "destination_id": "string",
                "category": "string",
                "duration_hours": "float",
                "price": "float",
                "difficulty_level": "integer",
                "rating": "float",
                "tags": "string[]",
                "description": "string"
            }
        },
        
        TravelCollection.ACCOMMODATIONS: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "住宿向量集合",
            "payload_schema": {
                "accommodation_id": "string",
                "name": "string",
                "destination_id": "string",
                "type": "string",
                "rating": "float",
                "price_range": "string",
                "amenities": "string[]",
                "location": "geo",
                "description": "string"
            }
        },
        
        TravelCollection.RESTAURANTS: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "餐厅向量集合",
            "payload_schema": {
                "restaurant_id": "string",
                "name": "string",
                "destination_id": "string",
                "cuisine_type": "string",
                "price_range": "string",
                "rating": "float",
                "features": "string[]",
                "location": "geo",
                "description": "string"
            }
        },
        
        TravelCollection.ATTRACTIONS: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "景点向量集合",
            "payload_schema": {
                "attraction_id": "string",
                "name": "string",
                "destination_id": "string",
                "category": "string",
                "opening_hours": "string",
                "entrance_fee": "float",
                "rating": "float",
                "location": "geo",
                "tags": "string[]",
                "description": "string"
            }
        },
        
        TravelCollection.TRAVEL_GUIDES: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "旅行指南向量集合",
            "payload_schema": {
                "guide_id": "string",
                "title": "string",
                "destination_id": "string",
                "guide_type": "string",
                "author": "string",
                "publish_date": "string",
                "tags": "string[]",
                "content": "string"
            }
        },
        
        TravelCollection.REVIEWS: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "评价向量集合",
            "payload_schema": {
                "review_id": "string",
                "target_id": "string",
                "target_type": "string",
                "user_id": "string",
                "rating": "float",
                "sentiment": "string",
                "helpful_count": "integer",
                "review_date": "string",
                "content": "string"
            }
        },
        
        TravelCollection.KNOWLEDGE_BASE: {
            "vector_size": 768,
            "distance": Distance.COSINE,
            "description": "知识库向量集合",
            "payload_schema": {
                "document_id": "string",
                "title": "string",
                "category": "string",
                "subcategory": "string",
                "source": "string",
                "last_updated": "string",
                "tags": "string[]",
                "content": "string",
                "chunk_index": "integer",
                "total_chunks": "integer"
            }
        },
        
        TravelCollection.USER_PREFERENCES: {
            "vector_size": 512,
            "distance": Distance.COSINE,
            "description": "用户偏好向量集合",
            "payload_schema": {
                "user_id": "string",
                "preference_type": "string",
                "travel_style": "string",
                "budget_range": "string",
                "preferred_destinations": "string[]",
                "activity_preferences": "string[]",
                "last_updated": "string"
            }
        },
        
        TravelCollection.CONVERSATION_CONTEXT: {
            "vector_size": 512,
            "distance": Distance.COSINE,
            "description": "对话上下文向量集合",
            "payload_schema": {
                "conversation_id": "string",
                "user_id": "string",
                "message_id": "string",
                "intent": "string",
                "entities": "string[]",
                "context_type": "string",
                "timestamp": "string",
                "content": "string"
            }
        }
    }


class CollectionManager:
    """集合管理器"""
    
    def __init__(self, qdrant_manager: QdrantManager):
        self.qdrant_manager = qdrant_manager
    
    async def create_all_collections(self) -> Dict[str, bool]:
        """创建所有旅行相关集合"""
        results = {}
        
        for collection_name, config in CollectionConfig.COLLECTION_CONFIGS.items():
            try:
                success = await self.qdrant_manager.create_collection(
                    collection_name=collection_name.value,
                    vector_size=config["vector_size"],
                    distance=config["distance"]
                )
                
                if success:
                    # 应用优化配置
                    await self._apply_collection_optimizations(collection_name.value)
                
                results[collection_name.value] = success
                
            except Exception as e:
                logger.error(f"创建集合失败 {collection_name.value}: {e}")
                results[collection_name.value] = False
        
        return results
    
    async def _apply_collection_optimizations(self, collection_name: str) -> bool:
        """应用集合优化配置"""
        try:
            # 根据集合类型应用不同的优化策略
            if collection_name in [TravelCollection.KNOWLEDGE_BASE.value, TravelCollection.TRAVEL_GUIDES.value]:
                # 知识库和指南集合使用较高的索引阈值
                indexing_threshold = 50000
            elif collection_name in [TravelCollection.USER_PREFERENCES.value, TravelCollection.CONVERSATION_CONTEXT.value]:
                # 用户相关集合使用较低的索引阈值
                indexing_threshold = 1000
            else:
                # 其他集合使用中等阈值
                indexing_threshold = 10000
            
            await self.qdrant_manager.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=indexing_threshold,
                    memmap_threshold=20000
                )
            )
            
            logger.info(f"集合 {collection_name} 优化配置已应用")
            return True
            
        except Exception as e:
            logger.error(f"应用集合优化失败 {collection_name}: {e}")
            return False
    
    async def get_collection_schema(self, collection_name: str) -> Dict[str, Any]:
        """获取集合架构"""
        for coll_enum, config in CollectionConfig.COLLECTION_CONFIGS.items():
            if coll_enum.value == collection_name:
                return {
                    "collection_name": collection_name,
                    "vector_size": config["vector_size"],
                    "distance": config["distance"].value,
                    "description": config["description"],
                    "payload_schema": config["payload_schema"]
                }
        
        return {}
    
    async def validate_collection_health(self, collection_name: str) -> Dict[str, Any]:
        """验证集合健康状态"""
        try:
            # 检查集合是否存在
            exists = await self.qdrant_manager.collection_exists(collection_name)
            if not exists:
                return {
                    "healthy": False,
                    "message": f"集合 {collection_name} 不存在"
                }
            
            # 获取集合统计信息
            stats = await self.qdrant_manager.get_collection_stats(collection_name)
            
            # 检查集合状态
            status = stats.get("status", "unknown")
            if status != "green":
                return {
                    "healthy": False,
                    "message": f"集合状态异常: {status}",
                    "stats": stats
                }
            
            # 检查向量数量
            points_count = stats.get("points_count", 0)
            if points_count == 0:
                return {
                    "healthy": True,
                    "message": "集合为空但状态正常",
                    "stats": stats
                }
            
            return {
                "healthy": True,
                "message": "集合状态正常",
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"验证集合健康状态失败 {collection_name}: {e}")
            return {
                "healthy": False,
                "message": f"健康检查失败: {str(e)}"
            }
    
    async def backup_all_collections(self) -> Dict[str, bool]:
        """备份所有集合"""
        results = {}
        
        for collection_name in CollectionConfig.COLLECTION_CONFIGS.keys():
            try:
                success = await self.qdrant_manager.backup_collection(
                    collection_name.value,
                    f"/backup/{collection_name.value}"
                )
                results[collection_name.value] = success
                
            except Exception as e:
                logger.error(f"备份集合失败 {collection_name.value}: {e}")
                results[collection_name.value] = False
        
        return results
    
    async def get_all_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有集合统计信息"""
        stats = {}
        
        for collection_name in CollectionConfig.COLLECTION_CONFIGS.keys():
            try:
                collection_stats = await self.qdrant_manager.get_collection_stats(collection_name.value)
                stats[collection_name.value] = collection_stats
                
            except Exception as e:
                logger.error(f"获取集合统计失败 {collection_name.value}: {e}")
                stats[collection_name.value] = {"error": str(e)}
        
        return stats
    
    async def reset_collection(self, collection_name: str) -> bool:
        """重置集合（清空数据）"""
        try:
            # 获取集合配置
            config = None
            for coll_enum, coll_config in CollectionConfig.COLLECTION_CONFIGS.items():
                if coll_enum.value == collection_name:
                    config = coll_config
                    break
            
            if not config:
                logger.error(f"未找到集合配置: {collection_name}")
                return False
            
            # 删除集合
            await self.qdrant_manager.delete_collection(collection_name)
            
            # 重新创建集合
            success = await self.qdrant_manager.create_collection(
                collection_name=collection_name,
                vector_size=config["vector_size"],
                distance=config["distance"]
            )
            
            if success:
                await self._apply_collection_optimizations(collection_name)
            
            logger.info(f"集合 {collection_name} 重置成功")
            return success
            
        except Exception as e:
            logger.error(f"重置集合失败 {collection_name}: {e}")
            return False


async def create_travel_collections(qdrant_manager: QdrantManager) -> Dict[str, bool]:
    """创建旅行相关的所有向量集合"""
    collection_manager = CollectionManager(qdrant_manager)
    return await collection_manager.create_all_collections()


async def get_collection_manager(qdrant_manager: QdrantManager) -> CollectionManager:
    """获取集合管理器实例"""
    return CollectionManager(qdrant_manager) 