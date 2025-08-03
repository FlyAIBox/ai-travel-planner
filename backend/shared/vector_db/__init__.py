"""
向量数据库模块
提供Qdrant向量数据库的配置、连接管理和操作接口
"""

from .client import QdrantManager, get_qdrant_client
from .collections import CollectionManager, create_travel_collections
from .models import VectorSearchQuery, SearchResult, DocumentVector

__all__ = [
    "QdrantManager",
    "get_qdrant_client", 
    "CollectionManager",
    "create_travel_collections",
    "VectorSearchQuery",
    "SearchResult",
    "DocumentVector"
] 