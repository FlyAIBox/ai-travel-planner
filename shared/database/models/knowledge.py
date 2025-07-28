"""
知识库域 SQLAlchemy ORM 模型
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Boolean, DateTime, String, Text, Integer, Numeric, JSON, 
    Enum as SQLEnum, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.dialects.mysql import CHAR, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from shared.database.connection import Base
from shared.models.knowledge import (
    DocumentType, DocumentStatus, ChunkStrategy, EmbeddingModel, SearchType
)


# ==================== 知识文档模型 ====================
class KnowledgeDocumentORM(Base):
    """知识文档表"""
    __tablename__ = "knowledge_documents"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="文档ID")
    
    # 基本信息
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="文档标题")
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="文档内容")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="文档摘要")
    
    # 文档属性
    document_type: Mapped[DocumentType] = mapped_column(
        SQLEnum(DocumentType), 
        nullable=False,
        comment="文档类型"
    )
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="zh-cn", comment="文档语言")
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus), 
        nullable=False, 
        default=DocumentStatus.DRAFT,
        comment="文档状态"
    )
    
    # 来源信息
    source_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True, comment="来源URL")
    source_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="来源路径")
    author: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="作者")
    
    # 分类信息
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="分类")
    tags: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="标签JSON")
    keywords: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="关键词JSON")
    
    # 元数据
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    
    # 统计信息
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="字数")
    character_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="字符数")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="分块数量")
    
    # 版本信息
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="版本号")
    parent_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36), 
        ForeignKey("knowledge_documents.id", ondelete="SET NULL"), 
        nullable=True,
        comment="父文档ID"
    )
    
    # 质量评分
    quality_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="质量评分"
    )
    relevance_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="相关性评分"
    )
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="发布时间")
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="索引时间")
    
    # 关联关系
    parent: Mapped[Optional["KnowledgeDocumentORM"]] = relationship("KnowledgeDocumentORM", remote_side=[id])
    children: Mapped[List["KnowledgeDocumentORM"]] = relationship("KnowledgeDocumentORM", back_populates="parent")
    chunks: Mapped[List["DocumentChunkORM"]] = relationship(
        "DocumentChunkORM", 
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunkORM.chunk_index"
    )
    entities: Mapped[List["KnowledgeEntityORM"]] = relationship(
        "KnowledgeEntityORM", 
        secondary="document_entity_associations",
        back_populates="documents"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_knowledge_documents_type", "document_type"),
        Index("idx_knowledge_documents_status", "status"),
        Index("idx_knowledge_documents_language", "language"),
        Index("idx_knowledge_documents_category", "category"),
        Index("idx_knowledge_documents_author", "author"),
        Index("idx_knowledge_documents_created_at", "created_at"),
        Index("idx_knowledge_documents_published_at", "published_at"),
        Index("idx_knowledge_documents_quality", "quality_score"),
        Index("idx_knowledge_documents_parent_id", "parent_id"),
        CheckConstraint("word_count >= 0", name="check_word_count_positive"),
        CheckConstraint("character_count >= 0", name="check_character_count_positive"),
        CheckConstraint("chunk_count >= 0", name="check_chunk_count_positive"),
        CheckConstraint("version >= 1", name="check_version_positive"),
        CheckConstraint("quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)", name="check_quality_score_range"),
        CheckConstraint("relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1)", name="check_relevance_score_range"),
    )


class DocumentChunkORM(Base):
    """文档分块表"""
    __tablename__ = "document_chunks"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="分块ID")
    document_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), 
        nullable=False,
        comment="文档ID"
    )
    
    # 内容信息
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="分块内容")
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="分块标题")
    
    # 位置信息
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="分块索引")
    start_position: Mapped[int] = mapped_column(Integer, nullable=False, comment="开始位置")
    end_position: Mapped[int] = mapped_column(Integer, nullable=False, comment="结束位置")
    
    # 分块属性
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, comment="分块大小")
    overlap_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="重叠大小")
    strategy: Mapped[ChunkStrategy] = mapped_column(
        SQLEnum(ChunkStrategy), 
        nullable=False,
        comment="分块策略"
    )
    
    # 向量信息
    embedding: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="向量嵌入JSON")
    embedding_model: Mapped[Optional[EmbeddingModel]] = mapped_column(
        SQLEnum(EmbeddingModel), 
        nullable=True,
        comment="嵌入模型"
    )
    vector_dimension: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="向量维度")
    
    # 元数据
    metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="元数据JSON")
    keywords: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="关键词JSON")
    
    # 质量评分
    semantic_density: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="语义密度"
    )
    coherence_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="连贯性评分"
    )
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    document: Mapped["KnowledgeDocumentORM"] = relationship("KnowledgeDocumentORM", back_populates="chunks")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_document_chunks_document_id", "document_id"),
        Index("idx_document_chunks_index", "document_id", "chunk_index"),
        Index("idx_document_chunks_strategy", "strategy"),
        Index("idx_document_chunks_embedding_model", "embedding_model"),
        Index("idx_document_chunks_created_at", "created_at"),
        CheckConstraint("chunk_index >= 0", name="check_chunk_index_positive"),
        CheckConstraint("start_position >= 0", name="check_start_position_positive"),
        CheckConstraint("end_position > start_position", name="check_position_range"),
        CheckConstraint("chunk_size >= 1", name="check_chunk_size_positive"),
        CheckConstraint("overlap_size >= 0", name="check_overlap_size_positive"),
        CheckConstraint("vector_dimension IS NULL OR vector_dimension >= 1", name="check_vector_dimension_positive"),
        CheckConstraint("semantic_density IS NULL OR (semantic_density >= 0 AND semantic_density <= 1)", name="check_semantic_density_range"),
        CheckConstraint("coherence_score IS NULL OR (coherence_score >= 0 AND coherence_score <= 1)", name="check_coherence_score_range"),
    )


# ==================== 用户向量模型 ====================
class UserVectorORM(Base):
    """用户向量画像表"""
    __tablename__ = "user_vectors"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="用户向量ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True,
        comment="用户ID"
    )
    
    # 兴趣向量
    interest_vector: Mapped[str] = mapped_column(JSON, nullable=False, comment="兴趣向量JSON")
    preference_vector: Mapped[str] = mapped_column(JSON, nullable=False, comment="偏好向量JSON")
    behavior_vector: Mapped[str] = mapped_column(JSON, nullable=False, comment="行为向量JSON")
    
    # 向量属性
    vector_dimension: Mapped[int] = mapped_column(Integer, nullable=False, comment="向量维度")
    embedding_model: Mapped[EmbeddingModel] = mapped_column(
        SQLEnum(EmbeddingModel), 
        nullable=False,
        comment="嵌入模型"
    )
    
    # 计算信息
    data_points: Mapped[int] = mapped_column(Integer, nullable=False, comment="数据点数量")
    last_activity_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="最近活动数量")
    
    # 质量指标
    vector_quality: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="向量质量"
    )
    stability_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="稳定性评分"
    )
    
    # 时间信息
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="过期时间")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_user_vectors_user_id", "user_id"),
        Index("idx_user_vectors_embedding_model", "embedding_model"),
        Index("idx_user_vectors_updated_at", "updated_at"),
        Index("idx_user_vectors_expires_at", "expires_at"),
        CheckConstraint("vector_dimension >= 1", name="check_vector_dimension_positive"),
        CheckConstraint("data_points >= 0", name="check_data_points_positive"),
        CheckConstraint("last_activity_count >= 0", name="check_activity_count_positive"),
        CheckConstraint("vector_quality IS NULL OR (vector_quality >= 0 AND vector_quality <= 1)", name="check_vector_quality_range"),
        CheckConstraint("stability_score IS NULL OR (stability_score >= 0 AND stability_score <= 1)", name="check_stability_score_range"),
    )


# ==================== 知识图谱模型 ====================
class KnowledgeEntityORM(Base):
    """知识实体表"""
    __tablename__ = "knowledge_entities"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="实体ID")
    
    # 基本信息
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="实体名称")
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="实体类型")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="实体描述")
    
    # 属性信息
    properties: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="实体属性JSON")
    aliases: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="别名JSON")
    
    # 关联信息
    document_ids: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="关联文档ID JSON")
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="提及次数")
    
    # 向量表示
    embedding: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="实体向量JSON")
    
    # 质量评分
    confidence_score: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="置信度")
    importance_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), 
        nullable=True,
        comment="重要性评分"
    )
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    
    # 关联关系
    documents: Mapped[List["KnowledgeDocumentORM"]] = relationship(
        "KnowledgeDocumentORM", 
        secondary="document_entity_associations",
        back_populates="entities"
    )
    subject_relations: Mapped[List["KnowledgeRelationORM"]] = relationship(
        "KnowledgeRelationORM", 
        foreign_keys="KnowledgeRelationORM.subject_id",
        back_populates="subject"
    )
    object_relations: Mapped[List["KnowledgeRelationORM"]] = relationship(
        "KnowledgeRelationORM", 
        foreign_keys="KnowledgeRelationORM.object_id",
        back_populates="object"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_knowledge_entities_name", "name"),
        Index("idx_knowledge_entities_type", "entity_type"),
        Index("idx_knowledge_entities_mention_count", "mention_count"),
        Index("idx_knowledge_entities_confidence", "confidence_score"),
        Index("idx_knowledge_entities_importance", "importance_score"),
        Index("idx_knowledge_entities_created_at", "created_at"),
        CheckConstraint("mention_count >= 0", name="check_mention_count_positive"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_score_range"),
        CheckConstraint("importance_score IS NULL OR (importance_score >= 0 AND importance_score <= 1)", name="check_importance_score_range"),
    )


class KnowledgeRelationORM(Base):
    """知识关系表"""
    __tablename__ = "knowledge_relations"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="关系ID")
    
    # 关系三元组
    subject_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"), 
        nullable=False,
        comment="主体实体ID"
    )
    predicate: Mapped[str] = mapped_column(String(100), nullable=False, comment="关系谓词")
    object_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"), 
        nullable=False,
        comment="客体实体ID"
    )
    
    # 关系属性
    relation_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="关系类型")
    confidence_score: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False, comment="置信度")
    weight: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, default=1.0, comment="关系权重")
    
    # 来源信息
    source_document_ids: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="来源文档ID JSON")
    evidence_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="证据文本")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    subject: Mapped["KnowledgeEntityORM"] = relationship(
        "KnowledgeEntityORM", 
        foreign_keys=[subject_id],
        back_populates="subject_relations"
    )
    object: Mapped["KnowledgeEntityORM"] = relationship(
        "KnowledgeEntityORM", 
        foreign_keys=[object_id],
        back_populates="object_relations"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_knowledge_relations_subject_id", "subject_id"),
        Index("idx_knowledge_relations_object_id", "object_id"),
        Index("idx_knowledge_relations_predicate", "predicate"),
        Index("idx_knowledge_relations_type", "relation_type"),
        Index("idx_knowledge_relations_confidence", "confidence_score"),
        Index("idx_knowledge_relations_created_at", "created_at"),
        Index("idx_knowledge_relations_triple", "subject_id", "predicate", "object_id"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_score_range"),
        CheckConstraint("weight >= 0", name="check_weight_positive"),
        CheckConstraint("subject_id != object_id", name="check_different_entities"),
    )


# ==================== 关联表 ====================
from sqlalchemy import Table, Column

# 文档-实体关联表
document_entity_associations = Table(
    'document_entity_associations',
    Base.metadata,
    Column('document_id', CHAR(36), ForeignKey('knowledge_documents.id', ondelete='CASCADE'), primary_key=True),
    Column('entity_id', CHAR(36), ForeignKey('knowledge_entities.id', ondelete='CASCADE'), primary_key=True),
    Column('mentioned_count', Integer, nullable=False, default=1, comment='提及次数'),
    Column('first_mentioned_at', DateTime, nullable=False, server_default=func.now(), comment='首次提及时间'),
    Column('last_mentioned_at', DateTime, nullable=False, server_default=func.now(), comment='最后提及时间'),
    
    # 索引
    Index('idx_document_entity_document_id', 'document_id'),
    Index('idx_document_entity_entity_id', 'entity_id'),
    Index('idx_document_entity_mentioned_count', 'mentioned_count'),
) 