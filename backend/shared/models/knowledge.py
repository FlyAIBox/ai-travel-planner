"""
知识库和向量数据模型
包含知识文档、向量搜索、RAG相关模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .user import User
from .common import IDMixin, TimestampMixin


# ==================== 枚举类型 ====================
class DocumentType(str, Enum):
    """文档类型"""
    TEXT = "text"           # 纯文本
    PDF = "pdf"             # PDF文档
    WORD = "word"           # Word文档
    MARKDOWN = "markdown"   # Markdown文档
    HTML = "html"           # HTML文档
    JSON = "json"           # JSON数据
    CSV = "csv"             # CSV数据
    IMAGE = "image"         # 图像
    AUDIO = "audio"         # 音频
    VIDEO = "video"         # 视频
    WEB_PAGE = "web_page"   # 网页
    API_DOC = "api_doc"     # API文档
    FAQ = "faq"             # 常见问题
    POLICY = "policy"       # 政策文档
    OTHER = "other"         # 其他


class DocumentStatus(str, Enum):
    """文档状态"""
    DRAFT = "draft"             # 草稿
    PROCESSING = "processing"   # 处理中
    INDEXED = "indexed"         # 已索引
    PUBLISHED = "published"     # 已发布
    ARCHIVED = "archived"       # 已归档
    DELETED = "deleted"         # 已删除
    ERROR = "error"             # 错误


class ChunkStrategy(str, Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"       # 固定大小
    SENTENCE = "sentence"           # 按句子
    PARAGRAPH = "paragraph"         # 按段落
    SECTION = "section"             # 按章节
    SEMANTIC = "semantic"           # 语义分块
    OVERLAP = "overlap"             # 重叠分块


class EmbeddingModel(str, Enum):
    """嵌入模型"""
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMERS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    BGE_LARGE_ZH = "BAAI/bge-large-zh-v1.5"
    M3E_BASE = "moka-ai/m3e-base"


class SearchType(str, Enum):
    """搜索类型"""
    VECTOR = "vector"           # 向量搜索
    KEYWORD = "keyword"         # 关键词搜索
    HYBRID = "hybrid"           # 混合搜索
    SEMANTIC = "semantic"       # 语义搜索


# ==================== 知识文档模型 ====================
class KnowledgeDocument(IDMixin, TimestampMixin):
    """知识文档"""
    
    id: UUID = Field(default_factory=uuid4, description="文档ID")
    
    # 基本信息
    title: str = Field(..., min_length=1, max_length=500, description="文档标题")
    content: str = Field(..., min_length=1, description="文档内容")
    summary: Optional[str] = Field(None, max_length=2000, description="文档摘要")
    
    # 文档属性
    document_type: DocumentType = Field(..., description="文档类型")
    language: str = Field(default="zh-cn", description="文档语言")
    status: DocumentStatus = Field(default=DocumentStatus.DRAFT, description="文档状态")
    
    # 来源信息
    source_url: Optional[str] = Field(None, description="来源URL")
    source_path: Optional[str] = Field(None, description="来源路径")
    author: Optional[str] = Field(None, description="作者")
    
    # 分类信息
    category: Optional[str] = Field(None, description="分类")
    tags: List[str] = Field(default=[], description="标签")
    keywords: List[str] = Field(default=[], description="关键词")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    # 统计信息
    word_count: int = Field(default=0, ge=0, description="字数")
    character_count: int = Field(default=0, ge=0, description="字符数")
    chunk_count: int = Field(default=0, ge=0, description="分块数量")
    
    # 版本信息
    version: int = Field(default=1, ge=1, description="版本号")
    parent_id: Optional[UUID] = Field(None, description="父文档ID")
    
    # 质量评分
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="质量评分")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="相关性评分")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    published_at: Optional[datetime] = Field(None, description="发布时间")
    indexed_at: Optional[datetime] = Field(None, description="索引时间")
    
    @validator('word_count', always=True)
    def calculate_word_count(cls, v, values):
        """计算字数"""
        content = values.get('content', '')
        if content:
            # 简单的中英文字数统计
            import re
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            english_words = len(re.findall(r'\b[a-zA-Z]+\b', content))
            return chinese_chars + english_words
        return v
    
    @validator('character_count', always=True)
    def calculate_character_count(cls, v, values):
        """计算字符数"""
        content = values.get('content', '')
        return len(content) if content else v


class DocumentChunk(IDMixin, TimestampMixin):
    """文档分块"""
    
    id: UUID = Field(default_factory=uuid4, description="分块ID")
    document_id: UUID = Field(..., description="文档ID")
    
    # 内容信息
    content: str = Field(..., min_length=1, description="分块内容")
    title: Optional[str] = Field(None, description="分块标题")
    
    # 位置信息
    chunk_index: int = Field(..., ge=0, description="分块索引")
    start_position: int = Field(..., ge=0, description="开始位置")
    end_position: int = Field(..., ge=0, description="结束位置")
    
    # 分块属性
    chunk_size: int = Field(..., ge=1, description="分块大小")
    overlap_size: int = Field(default=0, ge=0, description="重叠大小")
    strategy: ChunkStrategy = Field(..., description="分块策略")
    
    # 向量信息
    embedding: Optional[List[float]] = Field(None, description="向量嵌入")
    embedding_model: Optional[EmbeddingModel] = Field(None, description="嵌入模型")
    vector_dimension: Optional[int] = Field(None, ge=1, description="向量维度")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    keywords: List[str] = Field(default=[], description="关键词")
    
    # 质量评分
    semantic_density: Optional[float] = Field(None, ge=0, le=1, description="语义密度")
    coherence_score: Optional[float] = Field(None, ge=0, le=1, description="连贯性评分")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    
    @validator('end_position')
    def validate_position_range(cls, v, values):
        """验证位置范围"""
        if 'start_position' in values and v <= values['start_position']:
            raise ValueError('结束位置必须大于开始位置')
        return v


# ==================== 向量搜索模型 ====================
class VectorSearchQuery(IDMixin, TimestampMixin):
    """向量搜索查询"""
    
    # 查询信息
    query_text: str = Field(..., min_length=1, description="查询文本")
    query_embedding: Optional[List[float]] = Field(None, description="查询向量")
    
    # 搜索参数
    search_type: SearchType = Field(default=SearchType.VECTOR, description="搜索类型")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    similarity_threshold: float = Field(default=0.7, ge=0, le=1, description="相似度阈值")
    
    # 过滤条件
    document_types: Optional[List[DocumentType]] = Field(None, description="文档类型过滤")
    categories: Optional[List[str]] = Field(None, description="分类过滤")
    tags: Optional[List[str]] = Field(None, description="标签过滤")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="时间范围过滤")
    
    # 重排序参数
    enable_rerank: bool = Field(default=False, description="是否启用重排序")
    rerank_model: Optional[str] = Field(None, description="重排序模型")
    
    # 元数据
    user_id: Optional[UUID] = Field(None, description="用户ID")
    session_id: Optional[UUID] = Field(None, description="会话ID")
    request_id: Optional[str] = Field(None, description="请求ID")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="查询时间")


class VectorSearchResult(IDMixin, TimestampMixin):
    """向量搜索结果"""
    
    # 文档信息
    document_id: UUID = Field(..., description="文档ID")
    chunk_id: UUID = Field(..., description="分块ID")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="内容")
    
    # 相关性评分
    similarity_score: float = Field(..., ge=0, le=1, description="相似度分数")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="相关性分数")
    rank: int = Field(..., ge=1, description="排名")
    
    # 文档属性
    document_type: DocumentType = Field(..., description="文档类型")
    category: Optional[str] = Field(None, description="分类")
    tags: List[str] = Field(default=[], description="标签")
    
    # 高亮信息
    highlights: List[str] = Field(default=[], description="高亮片段")
    matched_keywords: List[str] = Field(default=[], description="匹配关键词")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    # 来源信息
    source_url: Optional[str] = Field(None, description="来源URL")
    author: Optional[str] = Field(None, description="作者")
    
    # 时间信息
    published_at: Optional[datetime] = Field(None, description="发布时间")


class SearchResponse(IDMixin, TimestampMixin):
    """搜索响应"""
    
    # 查询信息
    query: str = Field(..., description="查询文本")
    search_type: SearchType = Field(..., description="搜索类型")
    
    # 结果信息
    results: List[VectorSearchResult] = Field(..., description="搜索结果")
    total_results: int = Field(..., ge=0, description="总结果数")
    search_time_ms: float = Field(..., ge=0, description="搜索耗时（毫秒）")
    
    # 分页信息
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=10, ge=1, le=100, description="页面大小")
    has_more: bool = Field(..., description="是否有更多结果")
    
    # 聚合信息
    facets: Dict[str, List[Dict[str, Any]]] = Field(default={}, description="聚合信息")
    suggestions: List[str] = Field(default=[], description="搜索建议")
    
    # 元数据
    request_id: Optional[str] = Field(None, description="请求ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间")


# ==================== RAG相关模型 ====================
class RAGContext(IDMixin, TimestampMixin):
    """RAG上下文"""
    
    # 查询信息
    query: str = Field(..., description="原始查询")
    refined_query: Optional[str] = Field(None, description="优化后查询")
    
    # 检索结果
    retrieved_documents: List[VectorSearchResult] = Field(..., description="检索文档")
    context_window: str = Field(..., description="上下文窗口")
    
    # 相关性评估
    relevance_scores: List[float] = Field(..., description="相关性评分")
    context_quality: Optional[float] = Field(None, ge=0, le=1, description="上下文质量")
    
    # 多样性信息
    diversity_score: Optional[float] = Field(None, ge=0, le=1, description="多样性评分")
    coverage_score: Optional[float] = Field(None, ge=0, le=1, description="覆盖度评分")
    
    # 元数据
    retrieval_strategy: str = Field(..., description="检索策略")
    total_tokens: int = Field(..., ge=0, description="总令牌数")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class RAGResponse(IDMixin, TimestampMixin):
    """RAG响应"""
    
    # 生成内容
    answer: str = Field(..., description="生成答案")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    
    # 上下文信息
    context: RAGContext = Field(..., description="RAG上下文")
    sources: List[VectorSearchResult] = Field(..., description="信息源")
    
    # 质量评估
    factuality_score: Optional[float] = Field(None, ge=0, le=1, description="事实性评分")
    completeness_score: Optional[float] = Field(None, ge=0, le=1, description="完整性评分")
    coherence_score: Optional[float] = Field(None, ge=0, le=1, description="连贯性评分")
    
    # 生成信息
    model_used: str = Field(..., description="使用模型")
    generation_time_ms: float = Field(..., ge=0, description="生成耗时（毫秒）")
    total_time_ms: float = Field(..., ge=0, description="总耗时（毫秒）")
    
    # 令牌使用
    prompt_tokens: int = Field(..., ge=0, description="提示令牌数")
    completion_tokens: int = Field(..., ge=0, description="完成令牌数")
    total_tokens: int = Field(..., ge=0, description="总令牌数")


# ==================== 用户向量模型 ====================
class UserVector(IDMixin, TimestampMixin):
    """用户向量画像"""
    
    id: UUID = Field(default_factory=uuid4, description="用户向量ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 兴趣向量
    interest_vector: List[float] = Field(..., description="兴趣向量")
    preference_vector: List[float] = Field(..., description="偏好向量")
    behavior_vector: List[float] = Field(..., description="行为向量")
    
    # 向量属性
    vector_dimension: int = Field(..., ge=1, description="向量维度")
    embedding_model: EmbeddingModel = Field(..., description="嵌入模型")
    
    # 计算信息
    data_points: int = Field(..., ge=0, description="数据点数量")
    last_activity_count: int = Field(..., ge=0, description="最近活动数量")
    
    # 质量指标
    vector_quality: Optional[float] = Field(None, ge=0, le=1, description="向量质量")
    stability_score: Optional[float] = Field(None, ge=0, le=1, description="稳定性评分")
    
    # 时间信息
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")


# ==================== 知识图谱模型 ====================
class KnowledgeEntity(IDMixin, TimestampMixin):
    """知识实体"""
    
    id: UUID = Field(default_factory=uuid4, description="实体ID")
    
    # 基本信息
    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型")
    description: Optional[str] = Field(None, description="实体描述")
    
    # 属性信息
    properties: Dict[str, Any] = Field(default={}, description="实体属性")
    aliases: List[str] = Field(default=[], description="别名")
    
    # 关联信息
    document_ids: List[UUID] = Field(default=[], description="关联文档ID")
    mention_count: int = Field(default=0, ge=0, description="提及次数")
    
    # 向量表示
    embedding: Optional[List[float]] = Field(None, description="实体向量")
    
    # 质量评分
    confidence_score: float = Field(..., ge=0, le=1, description="置信度")
    importance_score: Optional[float] = Field(None, ge=0, le=1, description="重要性评分")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")


class KnowledgeRelation(IDMixin, TimestampMixin):
    """知识关系"""
    
    id: UUID = Field(default_factory=uuid4, description="关系ID")
    
    # 关系三元组
    subject_id: UUID = Field(..., description="主体实体ID")
    predicate: str = Field(..., description="关系谓词")
    object_id: UUID = Field(..., description="客体实体ID")
    
    # 关系属性
    relation_type: str = Field(..., description="关系类型")
    confidence_score: float = Field(..., ge=0, le=1, description="置信度")
    weight: float = Field(default=1.0, ge=0, description="关系权重")
    
    # 来源信息
    source_document_ids: List[UUID] = Field(default=[], description="来源文档ID")
    evidence_text: Optional[str] = Field(None, description="证据文本")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


# ==================== 请求/响应模型 ====================
class DocumentCreate(IDMixin, TimestampMixin):
    """创建文档请求"""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    document_type: DocumentType
    language: str = Field(default="zh-cn")
    category: Optional[str] = None
    tags: List[str] = Field(default=[])
    keywords: List[str] = Field(default=[])
    source_url: Optional[str] = None
    author: Optional[str] = None
    metadata: Dict[str, Any] = Field(default={})


class DocumentUpdate(IDMixin, TimestampMixin):
    """更新文档请求"""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    summary: Optional[str] = Field(None, max_length=2000)
    status: Optional[DocumentStatus] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(IDMixin, TimestampMixin):
    """文档响应"""
    id: UUID
    title: str
    summary: Optional[str]
    document_type: DocumentType
    language: str
    status: DocumentStatus
    category: Optional[str]
    tags: List[str]
    word_count: int
    chunk_count: int
    quality_score: Optional[float]
    created_at: datetime
    updated_at: datetime
    


class SearchRequest(IDMixin, TimestampMixin):
    """搜索请求"""
    query: str = Field(..., min_length=1)
    search_type: SearchType = Field(default=SearchType.VECTOR)
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    document_types: Optional[List[DocumentType]] = None
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    enable_rerank: bool = Field(default=False)


class RAGRequest(IDMixin, TimestampMixin):
    """RAG请求"""
    query: str = Field(..., min_length=1)
    context_size: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    include_sources: bool = Field(default=True)
    search_params: Optional[SearchRequest] = None 