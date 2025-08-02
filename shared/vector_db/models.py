"""
向量数据库模型定义
定义向量搜索、结果和文档相关的数据模型
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class SearchType(str, Enum):
    """搜索类型"""
    SEMANTIC = "semantic"           # 语义搜索
    KEYWORD = "keyword"            # 关键词搜索
    HYBRID = "hybrid"              # 混合搜索
    SIMILARITY = "similarity"      # 相似度搜索


@dataclass
class VectorSearchQuery:
    """向量搜索查询"""
    query_text: str                                    # 查询文本
    query_vector: Optional[List[float]] = None         # 查询向量
    collection_name: str = "default"                   # 集合名称
    limit: int = 10                                    # 返回结果数量
    score_threshold: Optional[float] = None            # 相似度阈值
    search_type: SearchType = SearchType.SEMANTIC     # 搜索类型
    filters: Optional[Dict[str, Any]] = None           # 过滤条件
    with_vectors: bool = False                         # 是否返回向量
    with_payload: bool = True                          # 是否返回载荷
    offset: int = 0                                    # 偏移量
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class SearchResult:
    """搜索结果"""
    id: Union[str, int]                               # 文档ID
    score: float                                      # 相似度分数
    payload: Dict[str, Any]                          # 载荷数据
    vector: Optional[List[float]] = None             # 向量数据
    rank: int = 0                                    # 排名
    collection_name: str = ""                        # 集合名称
    search_method: str = "vector"                    # 搜索方法
    timestamp: Optional[datetime] = None             # 时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def title(self) -> str:
        """获取标题"""
        return self.payload.get("title", "")
    
    @property
    def content(self) -> str:
        """获取内容"""
        return self.payload.get("content", "")
    
    @property
    def category(self) -> str:
        """获取分类"""
        return self.payload.get("category", "")
    
    @property
    def tags(self) -> List[str]:
        """获取标签"""
        return self.payload.get("tags", [])


@dataclass
class DocumentVector:
    """文档向量"""
    document_id: str                                  # 文档ID
    content: str                                      # 文档内容
    vector: List[float]                              # 向量表示
    metadata: Dict[str, Any]                         # 元数据
    chunk_id: Optional[str] = None                   # 分块ID
    chunk_index: int = 0                             # 分块索引
    total_chunks: int = 1                            # 总分块数
    embedding_model: str = "default"                 # 嵌入模型
    vector_dimension: int = 768                      # 向量维度
    created_at: Optional[datetime] = None            # 创建时间
    updated_at: Optional[datetime] = None            # 更新时间
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.chunk_id is None:
            self.chunk_id = f"{self.document_id}_chunk_{self.chunk_index}"
        if self.vector:
            self.vector_dimension = len(self.vector)
    
    @property
    def title(self) -> str:
        """获取标题"""
        return self.metadata.get("title", "")
    
    @property
    def source(self) -> str:
        """获取来源"""
        return self.metadata.get("source", "")
    
    @property
    def category(self) -> str:
        """获取分类"""
        return self.metadata.get("category", "")
    
    @property
    def tags(self) -> List[str]:
        """获取标签"""
        return self.metadata.get("tags", [])


@dataclass
class SearchStats:
    """搜索统计"""
    total_results: int                               # 总结果数
    search_time: float                               # 搜索时间(秒)
    collection_name: str                             # 集合名称
    query_type: SearchType                           # 查询类型
    filters_applied: bool = False                    # 是否应用过滤器
    cache_hit: bool = False                          # 是否命中缓存
    timestamp: Optional[datetime] = None             # 时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BatchSearchQuery:
    """批量搜索查询"""
    queries: List[VectorSearchQuery]                 # 查询列表
    collection_name: str = "default"                 # 集合名称
    parallel: bool = True                            # 是否并行执行
    max_workers: int = 4                             # 最大工作线程数
    
    def __post_init__(self):
        # 确保所有查询使用相同的集合名称
        for query in self.queries:
            if not query.collection_name or query.collection_name == "default":
                query.collection_name = self.collection_name


@dataclass
class BatchSearchResult:
    """批量搜索结果"""
    results: List[List[SearchResult]]                # 结果列表
    stats: List[SearchStats]                         # 统计信息列表
    total_time: float                                # 总时间
    success_count: int                               # 成功数量
    error_count: int                                 # 错误数量
    errors: List[str]                                # 错误信息列表
    timestamp: Optional[datetime] = None             # 时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if not hasattr(self, 'errors'):
            self.errors = []


# 类型别名
VectorList = List[float]
PayloadDict = Dict[str, Any]
FilterDict = Dict[str, Any]
MetadataDict = Dict[str, Any]

# 常用常量
DEFAULT_VECTOR_SIZE = 768
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_SCORE_THRESHOLD = 0.7
MAX_SEARCH_LIMIT = 1000
