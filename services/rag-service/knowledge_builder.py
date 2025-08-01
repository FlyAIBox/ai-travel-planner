"""
知识库构建系统
实现文档向量化、知识库构建、ETL流水线、质量评估和增量更新
"""

import asyncio
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from urllib.parse import urlparse
import mimetypes

import aiofiles
import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import jieba
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, JSONLoader
)
import tiktoken

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .vector_database import get_vector_database, VectorIndexConfig

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class DocumentMetadata:
    """文档元数据"""
    document_id: str
    title: str
    source: str
    document_type: str
    language: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class DocumentChunk:
    """文档块"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    char_count: int
    metadata: DocumentMetadata
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "metadata": self.metadata.to_dict()
        }


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    document_id: str
    chunks_count: int
    total_tokens: int
    processing_time: float
    error_message: Optional[str] = None
    quality_score: Optional[float] = None


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        # 初始化NLP工具
        try:
            self.nlp_zh = spacy.load("zh_core_web_sm")
        except OSError:
            logger.warning("中文spaCy模型未安装")
            self.nlp_zh = None
        
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("英文spaCy模型未安装")
            self.nlp_en = None
        
        # 初始化分词器
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 文本分割器
        self.text_splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
            ),
            "token": TokenTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                encoding_name="cl100k_base"
            )
        }
    
    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return "unknown"
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3:
            return "zh"
        elif chinese_ratio < 0.1:
            return "en"
        else:
            return "mixed"
    
    def extract_content_from_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """从文件提取内容"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        metadata = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_extension
        }
        
        try:
            if file_extension == ".txt":
                loader = TextLoader(str(file_path))
            elif file_extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_extension in [".html", ".htm"]:
                loader = UnstructuredHTMLLoader(str(file_path))
            elif file_extension in [".md", ".markdown"]:
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif file_extension == ".json":
                loader = JSONLoader(str(file_path), jq_schema=".", text_content=False)
            else:
                raise ValueError(f"不支持的文件类型: {file_extension}")
            
            documents = loader.load()
            content = "\n\n".join([doc.page_content for doc in documents])
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"提取文件内容失败 {file_path}: {e}")
            raise
    
    async def extract_content_from_url(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """从URL提取内容"""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        
                        if 'text/html' in content_type:
                            html_content = await response.text()
                            # 简单的HTML内容提取
                            content = re.sub(r'<[^>]+>', '', html_content)
                            content = re.sub(r'\s+', ' ', content).strip()
                        else:
                            content = await response.text()
                        
                        metadata.update({
                            "content_type": content_type,
                            "content_length": len(content)
                        })
                        
                        return content, metadata
                    else:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                        
        except Exception as e:
            logger.error(f"提取URL内容失败 {url}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 标准化空白字符
        text = re.sub(r'\n+', '\n', text)  # 标准化换行符
        text = text.strip()
        
        # 移除过短的段落
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines)
    
    def split_text(self, 
                   text: str, 
                   splitter_type: str = "recursive",
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> List[str]:
        """分割文本"""
        splitter = self.text_splitters.get(splitter_type, self.text_splitters["recursive"])
        
        # 动态调整参数
        if chunk_size or chunk_overlap:
            if splitter_type == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size or 1000,
                    chunk_overlap=chunk_overlap or 200,
                    length_function=len
                )
            elif splitter_type == "token":
                splitter = TokenTextSplitter(
                    chunk_size=chunk_size or 500,
                    chunk_overlap=chunk_overlap or 50
                )
        
        chunks = splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
    
    def calculate_content_quality(self, text: str) -> float:
        """计算内容质量分数"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # 长度检查 (10%)
        if len(text) > 100:
            score += 0.1
        
        # 语言多样性 (20%)
        words = text.split()
        unique_words = set(words)
        if len(words) > 0:
            lexical_diversity = len(unique_words) / len(words)
            score += min(lexical_diversity * 0.4, 0.2)
        
        # 句子结构 (20%)
        sentences = re.split(r'[.!?。！？]', text)
        if len(sentences) > 1:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            if 5 <= avg_sentence_length <= 30:  # 合理的句子长度
                score += 0.2
        
        # 信息密度 (30%)
        # 检查数字、日期、专有名词等
        info_patterns = [
            r'\d+',  # 数字
            r'\d{4}年|\d{1,2}月|\d{1,2}日',  # 日期
            r'[A-Z][a-z]+',  # 专有名词
            r'https?://\S+',  # URL
        ]
        
        info_count = sum(len(re.findall(pattern, text)) for pattern in info_patterns)
        info_density = min(info_count / len(words) * 10, 0.3) if words else 0
        score += info_density
        
        # 连贯性 (20%)
        # 简单检查连接词和代词的使用
        coherence_words = ['因为', '所以', '但是', '然而', '此外', '另外', '因此', '这样', '这些', '那些']
        coherence_count = sum(1 for word in coherence_words if word in text)
        coherence_score = min(coherence_count / len(sentences) * 0.5, 0.2) if sentences else 0
        score += coherence_score
        
        return min(score, 1.0)
    
    def extract_entities(self, text: str, language: str = "zh") -> Dict[str, List[str]]:
        """提取命名实体"""
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # 地理政治实体
            "DATE": [],
            "MONEY": [],
            "PRODUCT": []
        }
        
        try:
            if language == "zh" and self.nlp_zh:
                doc = self.nlp_zh(text)
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
            elif language == "en" and self.nlp_en:
                doc = self.nlp_en(text)
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text)
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def create_document_chunks(self, 
                             document_id: str,
                             content: str, 
                             metadata: DocumentMetadata,
                             splitter_type: str = "recursive") -> List[DocumentChunk]:
        """创建文档块"""
        # 预处理文本
        processed_content = self.preprocess_text(content)
        
        # 分割文本
        text_chunks = self.split_text(processed_content, splitter_type)
        
        # 创建文档块
        document_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{document_id}_{i:04d}"
            
            # 计算token数量
            token_count = len(self.tokenizer.encode(chunk_text))
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                token_count=token_count,
                char_count=len(chunk_text),
                metadata=metadata
            )
            
            document_chunks.append(chunk)
        
        return document_chunks


class EmbeddingGenerator:
    """向量生成器"""
    
    def __init__(self, model_configs: Optional[Dict[str, Any]] = None):
        self.model_configs = model_configs or {
            "default": {
                "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }
        
        self.models = {}
        self.current_model = "default"
    
    async def load_model(self, model_name: str = "default") -> None:
        """加载向量模型"""
        if model_name not in self.models:
            config = self.model_configs.get(model_name, self.model_configs["default"])
            
            try:
                model = SentenceTransformer(config["model_name"])
                if torch.cuda.is_available():
                    model = model.to(config["device"])
                
                self.models[model_name] = model
                logger.info(f"向量模型 {config['model_name']} 加载成功")
                
            except Exception as e:
                logger.error(f"加载向量模型失败: {e}")
                raise
    
    async def generate_embeddings(self, 
                                texts: List[str], 
                                model_name: str = "default",
                                batch_size: int = 32) -> List[List[float]]:
        """生成向量"""
        if model_name not in self.models:
            await self.load_model(model_name)
        
        model = self.models[model_name]
        
        try:
            # 批量生成向量
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            raise
    
    def get_vector_dimension(self, model_name: str = "default") -> int:
        """获取向量维度"""
        if model_name not in self.models:
            # 返回默认维度
            return 384
        
        model = self.models[model_name]
        return model.get_sentence_embedding_dimension()


class KnowledgeBuilder:
    """知识库构建器"""
    
    def __init__(self, 
                 vector_db: Optional[Any] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        self.vector_db = vector_db or get_vector_database()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.document_processor = DocumentProcessor()
        
        # 处理统计
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "total_vectors": 0,
            "average_quality_score": 0.0
        }
    
    async def initialize(self) -> bool:
        """初始化知识库构建器"""
        try:
            # 初始化向量数据库
            if hasattr(self.vector_db, 'initialize_cluster'):
                await self.vector_db.initialize_cluster()
            
            # 加载默认向量模型
            await self.embedding_generator.load_model()
            
            # 创建默认集合
            await self._ensure_default_collection()
            
            logger.info("知识库构建器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"知识库构建器初始化失败: {e}")
            return False
    
    async def _ensure_default_collection(self) -> None:
        """确保默认集合存在"""
        collection_name = settings.QDRANT_COLLECTION_NAME
        
        try:
            # 检查集合是否存在
            collection_info = await self.vector_db.get_collection_info(collection_name)
            
            if collection_info is None:
                # 创建默认集合
                vector_config = VectorIndexConfig(
                    vector_size=self.embedding_generator.get_vector_dimension(),
                    distance="Cosine"
                )
                
                success = await self.vector_db.create_collection(
                    collection_name=collection_name,
                    config=vector_config,
                    replica_count=1,
                    shard_count=1
                )
                
                if success:
                    logger.info(f"默认集合 {collection_name} 创建成功")
                else:
                    raise Exception("创建默认集合失败")
                    
        except Exception as e:
            logger.error(f"确保默认集合失败: {e}")
            raise
    
    async def process_document_from_file(self, 
                                       file_path: str,
                                       metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """处理文件文档"""
        start_time = datetime.now()
        document_id = str(uuid.uuid4())
        
        try:
            # 提取文件内容
            content, file_metadata = self.document_processor.extract_content_from_file(file_path)
            
            # 创建文档元数据
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                title=metadata.get("title", Path(file_path).stem),
                source="file",
                document_type=file_metadata.get("file_extension", "unknown"),
                language=self.document_processor.detect_language(content),
                created_at=datetime.now(),
                file_path=file_path,
                file_size=file_metadata.get("file_size"),
                content_hash=hashlib.md5(content.encode()).hexdigest()
            )
            
            if metadata:
                for key, value in metadata.items():
                    if hasattr(doc_metadata, key):
                        setattr(doc_metadata, key, value)
            
            # 处理文档
            result = await self._process_document_content(document_id, content, doc_metadata)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"处理文件文档失败 {file_path}: {e}")
            return ProcessingResult(
                success=False,
                document_id=document_id,
                chunks_count=0,
                total_tokens=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )
    
    async def process_document_from_url(self, 
                                      url: str,
                                      metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """处理URL文档"""
        start_time = datetime.now()
        document_id = str(uuid.uuid4())
        
        try:
            # 提取URL内容
            content, url_metadata = await self.document_processor.extract_content_from_url(url)
            
            # 创建文档元数据
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                title=metadata.get("title", urlparse(url).path.split("/")[-1] or url),
                source="url",
                document_type="web",
                language=self.document_processor.detect_language(content),
                created_at=datetime.now(),
                url=url,
                content_hash=hashlib.md5(content.encode()).hexdigest()
            )
            
            if metadata:
                for key, value in metadata.items():
                    if hasattr(doc_metadata, key):
                        setattr(doc_metadata, key, value)
            
            # 处理文档
            result = await self._process_document_content(document_id, content, doc_metadata)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"处理URL文档失败 {url}: {e}")
            return ProcessingResult(
                success=False,
                document_id=document_id,
                chunks_count=0,
                total_tokens=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )
    
    async def process_document_from_text(self,
                                       text: str,
                                       metadata: Dict[str, Any]) -> ProcessingResult:
        """处理文本文档"""
        start_time = datetime.now()
        document_id = str(uuid.uuid4())
        
        try:
            # 创建文档元数据
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                title=metadata.get("title", "文本文档"),
                source="text",
                document_type="text",
                language=self.document_processor.detect_language(text),
                created_at=datetime.now(),
                content_hash=hashlib.md5(text.encode()).hexdigest()
            )
            
            for key, value in metadata.items():
                if hasattr(doc_metadata, key):
                    setattr(doc_metadata, key, value)
            
            # 处理文档
            result = await self._process_document_content(document_id, text, doc_metadata)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"处理文本文档失败: {e}")
            return ProcessingResult(
                success=False,
                document_id=document_id,
                chunks_count=0,
                total_tokens=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )
    
    async def _process_document_content(self,
                                      document_id: str,
                                      content: str,
                                      metadata: DocumentMetadata) -> ProcessingResult:
        """处理文档内容"""
        try:
            # 计算内容质量
            quality_score = self.document_processor.calculate_content_quality(content)
            
            # 质量过滤
            if quality_score < 0.3:
                logger.warning(f"文档质量分数过低: {quality_score}")
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunks_count=0,
                    total_tokens=0,
                    processing_time=0,
                    error_message="文档质量分数过低",
                    quality_score=quality_score
                )
            
            # 创建文档块
            chunks = self.document_processor.create_document_chunks(
                document_id=document_id,
                content=content,
                metadata=metadata
            )
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunks_count=0,
                    total_tokens=0,
                    processing_time=0,
                    error_message="未能创建文档块"
                )
            
            # 生成向量
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_generator.generate_embeddings(chunk_texts)
            
            # 添加向量到块
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embeddings = embedding
            
            # 存储到向量数据库
            vectors = [chunk.embeddings for chunk in chunks]
            payloads = [self._create_chunk_payload(chunk) for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            
            collection_name = settings.QDRANT_COLLECTION_NAME
            success = await self.vector_db.upsert_vectors(
                collection_name=collection_name,
                vectors=vectors,
                payloads=payloads,
                ids=ids
            )
            
            if success:
                # 更新统计信息
                self._update_stats(len(chunks), sum(chunk.token_count for chunk in chunks), quality_score, True)
                
                return ProcessingResult(
                    success=True,
                    document_id=document_id,
                    chunks_count=len(chunks),
                    total_tokens=sum(chunk.token_count for chunk in chunks),
                    processing_time=0,
                    quality_score=quality_score
                )
            else:
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunks_count=len(chunks),
                    total_tokens=sum(chunk.token_count for chunk in chunks),
                    processing_time=0,
                    error_message="向量存储失败"
                )
                
        except Exception as e:
            logger.error(f"处理文档内容失败: {e}")
            self._update_stats(0, 0, 0, False)
            raise
    
    def _create_chunk_payload(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """创建块的载荷数据"""
        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "document_type": chunk.metadata.document_type,
            "source": chunk.metadata.source,
            "language": chunk.metadata.language,
            "title": chunk.metadata.title,
            "created_at": chunk.metadata.created_at.isoformat() if chunk.metadata.created_at else None
        }
        
        # 添加可选字段
        optional_fields = ["category", "tags", "author", "url", "file_path"]
        for field in optional_fields:
            value = getattr(chunk.metadata, field, None)
            if value is not None:
                payload[field] = value
        
        return payload
    
    def _update_stats(self, chunks_count: int, tokens_count: int, quality_score: float, success: bool) -> None:
        """更新处理统计"""
        self.processing_stats["total_documents"] += 1
        
        if success:
            self.processing_stats["successful_documents"] += 1
            self.processing_stats["total_chunks"] += chunks_count
            self.processing_stats["total_vectors"] += chunks_count
            
            # 更新平均质量分数
            total_success = self.processing_stats["successful_documents"]
            current_avg = self.processing_stats["average_quality_score"]
            self.processing_stats["average_quality_score"] = (
                (current_avg * (total_success - 1) + quality_score) / total_success
            )
        else:
            self.processing_stats["failed_documents"] += 1
    
    async def batch_process_documents(self, 
                                    document_sources: List[Dict[str, Any]],
                                    max_concurrent: int = 5) -> List[ProcessingResult]:
        """批量处理文档"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_document(source: Dict[str, Any]) -> ProcessingResult:
            async with semaphore:
                source_type = source.get("type")
                
                if source_type == "file":
                    return await self.process_document_from_file(
                        source["path"], 
                        source.get("metadata", {})
                    )
                elif source_type == "url":
                    return await self.process_document_from_url(
                        source["url"], 
                        source.get("metadata", {})
                    )
                elif source_type == "text":
                    return await self.process_document_from_text(
                        source["content"], 
                        source.get("metadata", {})
                    )
                else:
                    return ProcessingResult(
                        success=False,
                        document_id="unknown",
                        chunks_count=0,
                        total_tokens=0,
                        processing_time=0,
                        error_message=f"不支持的文档类型: {source_type}"
                    )
        
        # 并发处理所有文档
        tasks = [process_single_document(source) for source in document_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    document_id="error",
                    chunks_count=0,
                    total_tokens=0,
                    processing_time=0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def update_document(self, document_id: str, new_content: str, metadata: Dict[str, Any]) -> bool:
        """更新文档"""
        try:
            # 删除旧的文档块
            await self.delete_document(document_id)
            
            # 处理新内容
            result = await self.process_document_from_text(new_content, metadata)
            
            return result.success
            
        except Exception as e:
            logger.error(f"更新文档失败 {document_id}: {e}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            # 查找所有相关的块ID
            # 这里需要实现查询功能，暂时使用简单的ID模式
            chunk_ids = []
            for i in range(1000):  # 假设最多1000个块
                chunk_id = f"{document_id}_{i:04d}"
                chunk_ids.append(chunk_id)
            
            # 删除向量
            collection_name = settings.QDRANT_COLLECTION_NAME
            success = await self.vector_db.delete_vectors(
                collection_name=collection_name,
                point_ids=chunk_ids
            )
            
            return success
            
        except Exception as e:
            logger.error(f"删除文档失败 {document_id}: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        
        # 计算成功率
        total_docs = stats["total_documents"]
        if total_docs > 0:
            stats["success_rate"] = stats["successful_documents"] / total_docs * 100
        else:
            stats["success_rate"] = 0
        
        return stats
    
    async def create_travel_knowledge_base(self) -> Dict[str, Any]:
        """创建旅行知识库"""
        logger.info("开始创建旅行知识库...")
        
        # 定义旅行相关的文档源
        travel_documents = [
            {
                "type": "text",
                "content": """
                旅行规划指南
                
                制定旅行计划时，需要考虑以下几个重要因素：
                
                1. 目的地选择：根据个人兴趣、预算和时间来选择合适的目的地。
                
                2. 最佳旅行时间：了解目的地的气候特点和旅游旺季，选择合适的出行时间。
                
                3. 预算规划：包括交通费、住宿费、餐饮费、景点门票和购物费用等。
                
                4. 交通安排：选择合适的交通方式，包括飞机、火车、汽车等。
                
                5. 住宿预订：根据预算和需求选择合适的住宿类型。
                
                6. 行程安排：合理安排每天的活动，留出适当的休息时间。
                
                7. 必备物品：准备护照、签证、保险、常用药品等必需品。
                """,
                "metadata": {
                    "title": "旅行规划指南",
                    "category": "travel_planning",
                    "tags": ["规划", "指南", "基础知识"]
                }
            },
            {
                "type": "text",
                "content": """
                酒店预订攻略
                
                选择和预订酒店的技巧：
                
                1. 位置选择：优先考虑交通便利、安全的区域。
                
                2. 设施服务：根据需求选择带有合适设施的酒店。
                
                3. 价格比较：使用多个预订平台比较价格。
                
                4. 用户评价：仔细阅读其他住客的真实评价。
                
                5. 取消政策：了解酒店的取消和修改政策。
                
                6. 预订时机：提前预订通常能获得更好的价格。
                
                7. 会员优惠：利用酒店会员计划获得优惠和升级。
                """,
                "metadata": {
                    "title": "酒店预订攻略",
                    "category": "accommodation",
                    "tags": ["酒店", "预订", "攻略"]
                }
            },
            {
                "type": "text",
                "content": """
                航班预订指南
                
                预订机票的最佳实践：
                
                1. 提前预订：通常提前2-3个月预订能获得较好的价格。
                
                2. 灵活日期：如果时间灵活，可以选择价格较低的日期。
                
                3. 比较价格：使用比价网站比较不同航空公司的价格。
                
                4. 考虑中转：有时中转航班比直飞更便宜。
                
                5. 里程积累：选择能累积里程的航空公司。
                
                6. 行李政策：了解不同航空公司的行李限制和费用。
                
                7. 退改签：关注票价的退改签条件。
                """,
                "metadata": {
                    "title": "航班预订指南",
                    "category": "transportation",
                    "tags": ["航班", "机票", "预订"]
                }
            }
        ]
        
        # 批量处理文档
        results = await self.batch_process_documents(travel_documents)
        
        # 统计结果
        success_count = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_count for r in results if r.success)
        
        summary = {
            "total_documents": len(travel_documents),
            "successful_documents": success_count,
            "total_chunks": total_chunks,
            "processing_results": [
                {
                    "document_id": r.document_id,
                    "success": r.success,
                    "chunks_count": r.chunks_count,
                    "error": r.error_message
                }
                for r in results
            ]
        }
        
        logger.info(f"旅行知识库创建完成: {success_count}/{len(travel_documents)} 文档处理成功")
        return summary


# 全局知识库构建器实例
_knowledge_builder: Optional[KnowledgeBuilder] = None


def get_knowledge_builder(vector_db: Optional[Any] = None, 
                         embedding_generator: Optional[EmbeddingGenerator] = None) -> KnowledgeBuilder:
    """获取知识库构建器实例"""
    global _knowledge_builder
    if _knowledge_builder is None:
        _knowledge_builder = KnowledgeBuilder(vector_db, embedding_generator)
    return _knowledge_builder 