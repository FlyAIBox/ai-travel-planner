"""
知识库构建器
实现文档向量化、知识库构建、文档管理、版本控制等功能
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLTextSplitter,
    PythonCodeTextSplitter
)
from langchain.document_loaders import (
    TextLoader,
    JSONLoader,
    CSVLoader
)

try:
    from langchain.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
    PDF_LOADERS_AVAILABLE = True
except ImportError:
    PDF_LOADERS_AVAILABLE = False

import structlog
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
    category: str
    language: str = "zh"
    tags: List[str] = None
    priority: int = 1
    created_at: datetime = None
    updated_at: datetime = None
    version: str = "1.0"
    author: str = ""
    file_size: int = 0
    checksum: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class DocumentChunk:
    """文档分块"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    token_count: int = 0
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    document_id: str
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None
    embedding_model: str = ""
    total_tokens: int = 0


class EmbeddingGenerator:
    """向量生成器"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.model_type = self._detect_model_type()
        self.max_length = 512
        self._load_model()
    
    def _detect_model_type(self) -> str:
        """检测模型类型"""
        if "sentence-transformers" in self.model_name.lower():
            return "sentence_transformers"
        elif "bge" in self.model_name.lower():
            return "bge"
        else:
            return "transformers"
    
    def _load_model(self):
        """加载模型"""
        try:
            if self.model_type == "sentence_transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(self.model_name)
                self.max_length = self.model.max_seq_length
                logger.info(f"已加载 Sentence Transformers 模型: {self.model_name}")
                
            elif self.model_type == "bge" and TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.max_length = self.tokenizer.model_max_length
                logger.info(f"已加载 BGE 模型: {self.model_name}")
                
            elif TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.max_length = self.tokenizer.model_max_length
                logger.info(f"已加载 Transformers 模型: {self.model_name}")
                
            else:
                raise RuntimeError("未安装必要的依赖包")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """生成向量"""
        if not texts:
            return []
        
        try:
            if self.model_type == "sentence_transformers":
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 100,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
                
            elif self.model_type == "bge":
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self._encode_bge_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
                return embeddings
                
            else:
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self._encode_transformers_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
                return embeddings
                
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return []
    
    def _encode_bge_batch(self, texts: List[str]) -> List[List[float]]:
        """BGE模型批量编码"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()
    
    def _encode_transformers_batch(self, texts: List[str]) -> List[List[float]]:
        """通用 Transformers 模型批量编码"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            # 使用平均池化
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.model_type == "sentence_transformers":
            return self.model.get_sentence_embedding_dimension()
        else:
            # 对于其他模型，需要通过测试文本来获取维度
            test_embedding = self.generate_embeddings(["test"])
            return len(test_embedding[0]) if test_embedding else 768


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.text_splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", "。", "!", "?", ";", " ", ""]
            ),
            "markdown": MarkdownTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            "html": HTMLTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            "python": PythonCodeTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        }
        
        self.document_loaders = {
            "txt": TextLoader,
            "json": JSONLoader,
            "csv": CSVLoader
        }
        
        if PDF_LOADERS_AVAILABLE:
            self.document_loaders.update({
                "pdf": PyMuPDFLoader,
                "pdf_unstructured": UnstructuredPDFLoader
            })
    
    def load_document(self, file_path: str, loader_type: str = "auto") -> List[str]:
        """加载文档"""
        file_path = Path(file_path)
        
        if loader_type == "auto":
            loader_type = file_path.suffix.lower().lstrip('.')
        
        if loader_type not in self.document_loaders:
            raise ValueError(f"不支持的文档类型: {loader_type}")
        
        try:
            loader_class = self.document_loaders[loader_type]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            return [doc.page_content for doc in documents]
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
    
    def split_text(self, text: str, splitter_type: str = "recursive", 
                   chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """分割文本"""
        if splitter_type not in self.text_splitters:
            splitter_type = "recursive"
        
        splitter = self.text_splitters[splitter_type]
        
        # 更新分割器参数
        if hasattr(splitter, '_chunk_size'):
            splitter._chunk_size = chunk_size
            splitter._chunk_overlap = chunk_overlap
        
        try:
            chunks = splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"文本分割失败: {e}")
            return [text]  # 返回原文本作为单个分块
    
    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取元数据"""
        metadata = {}
        
        # 基础统计
        metadata["character_count"] = len(text)
        metadata["word_count"] = len(text.split())
        metadata["line_count"] = len(text.split('\n'))
        
        # 语言检测（简单方法）
        chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_word_count = len(re.findall(r'[a-zA-Z]+', text))
        
        if chinese_char_count > english_word_count:
            metadata["language"] = "zh"
        else:
            metadata["language"] = "en"
        
        # 内容类型检测
        if re.search(r'```|def |class |import ', text):
            metadata["content_type"] = "code"
        elif re.search(r'#+|##|###', text):
            metadata["content_type"] = "markdown"
        elif re.search(r'<html>|<div>|<p>', text):
            metadata["content_type"] = "html"
        else:
            metadata["content_type"] = "text"
        
        # 关键词提取（简单方法）
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 2:  # 只考虑长度大于2的词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 提取最频繁的5个词作为关键词
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        metadata["keywords"] = [word for word, freq in keywords]
        
        return metadata
    
    def calculate_checksum(self, content: str) -> str:
        """计算内容校验和"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, collection_name: str = "travel_knowledge"):
        self.collection_name = collection_name
        self.vector_db = get_vector_database()
        self.embedding_generator = EmbeddingGenerator()
        self.document_processor = DocumentProcessor()
        
        # 文档缓存
        self.document_cache = {}
        self.processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "last_update": None,
            "errors": []
        }
    
    async def initialize_collection(self) -> bool:
        """初始化集合"""
        try:
            # 检查集合是否已存在
            collection_info = await self.vector_db.get_collection_info(self.collection_name)
            if collection_info:
                logger.info(f"集合 {self.collection_name} 已存在")
                return True
            
            # 创建集合配置
            config = VectorIndexConfig(
                vector_size=self.embedding_generator.get_dimension(),
                distance="Cosine"
            )
            
            # 创建集合
            success = await self.vector_db.create_collection(
                collection_name=self.collection_name,
                config=config,
                replica_count=1,
                shard_count=1
            )
            
            if success:
                logger.info(f"集合 {self.collection_name} 创建成功")
            else:
                logger.error(f"集合 {self.collection_name} 创建失败")
            
            return success
            
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            return False
    
    async def build_knowledge_base(self, documents: List[Dict[str, Any]], 
                                 batch_size: int = 32) -> List[ProcessingResult]:
        """构建知识库"""
        if not await self.initialize_collection():
            raise RuntimeError("集合初始化失败")
        
        results = []
        
        # 使用线程池处理文档
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for doc_data in documents:
                future = executor.submit(self._process_document, doc_data)
                futures.append(future)
            
            # 处理完成的任务
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 更新统计信息
                    self.processing_stats["total_documents"] += 1
                    self.processing_stats["total_chunks"] += result.chunks_created
                    self.processing_stats["total_processing_time"] += result.processing_time
                    
                except Exception as e:
                    logger.error(f"文档处理失败: {e}")
                    self.processing_stats["errors"].append(str(e))
        
        # 批量向量化和插入
        await self._batch_vectorize_and_insert(results, batch_size)
        
        self.processing_stats["last_update"] = datetime.now()
        logger.info(f"知识库构建完成，处理了 {len(results)} 个文档")
        
        return results
    
    def _process_document(self, doc_data: Dict[str, Any]) -> ProcessingResult:
        """处理单个文档"""
        start_time = time.time()
        
        try:
            # 提取文档信息
            content = doc_data.get("content", "")
            metadata = DocumentMetadata(
                document_id=doc_data.get("document_id", str(uuid.uuid4())),
                title=doc_data.get("title", "Untitled"),
                source=doc_data.get("source", "unknown"),
                category=doc_data.get("category", "general"),
                language=doc_data.get("language", "zh"),
                tags=doc_data.get("tags", []),
                priority=doc_data.get("priority", 1),
                author=doc_data.get("author", ""),
                checksum=self.document_processor.calculate_checksum(content)
            )
            
            # 分割文档
            splitter_type = doc_data.get("splitter_type", "recursive")
            chunk_size = doc_data.get("chunk_size", 1000)
            chunk_overlap = doc_data.get("chunk_overlap", 200)
            
            chunks = self.document_processor.split_text(
                content, splitter_type, chunk_size, chunk_overlap
            )
            
            # 创建文档分块
            document_chunks = []
            for i, chunk_content in enumerate(chunks):
                chunk_metadata = self.document_processor.extract_metadata_from_text(chunk_content)
                chunk_metadata.update(asdict(metadata))
                
                chunk = DocumentChunk(
                    chunk_id=f"{metadata.document_id}_chunk_{i}",
                    document_id=metadata.document_id,
                    content=chunk_content,
                    chunk_index=i,
                    metadata=chunk_metadata,
                    token_count=len(chunk_content.split())
                )
                document_chunks.append(chunk)
            
            # 缓存文档
            self.document_cache[metadata.document_id] = {
                "metadata": metadata,
                "chunks": document_chunks
            }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                document_id=metadata.document_id,
                chunks_created=len(document_chunks),
                processing_time=processing_time,
                embedding_model=self.embedding_generator.model_name,
                total_tokens=sum(chunk.token_count for chunk in document_chunks)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"文档处理失败: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=doc_data.get("document_id", "unknown"),
                chunks_created=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _batch_vectorize_and_insert(self, results: List[ProcessingResult], 
                                        batch_size: int = 32) -> None:
        """批量向量化和插入"""
        all_chunks = []
        
        # 收集所有成功处理的文档分块
        for result in results:
            if result.success and result.document_id in self.document_cache:
                chunks = self.document_cache[result.document_id]["chunks"]
                all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("没有可插入的文档分块")
            return
        
        # 批量生成向量
        texts = [chunk.content for chunk in all_chunks]
        logger.info(f"开始生成 {len(texts)} 个文档分块的向量...")
        
        embeddings = self.embedding_generator.generate_embeddings(texts, batch_size)
        
        if len(embeddings) != len(all_chunks):
            logger.error(f"向量数量 ({len(embeddings)}) 与分块数量 ({len(all_chunks)}) 不匹配")
            return
        
        # 准备插入数据
        vectors = []
        payloads = []
        ids = []
        
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
            vectors.append(embedding)
            payloads.append(chunk.metadata)
            ids.append(chunk.chunk_id)
        
        # 批量插入向量数据库
        logger.info(f"开始插入 {len(vectors)} 个向量到数据库...")
        success = await self.vector_db.upsert_vectors(
            collection_name=self.collection_name,
            vectors=vectors,
            payloads=payloads,
            ids=ids,
            batch_size=batch_size
        )
        
        if success:
            logger.info("向量批量插入成功")
        else:
            logger.error("向量批量插入失败")
    
    async def update_knowledge_base(self, document_id: str, doc_data: Dict[str, Any]) -> ProcessingResult:
        """更新知识库中的单个文档"""
        try:
            # 先删除旧版本
            await self.delete_document(document_id)
            
            # 处理新版本
            result = self._process_document(doc_data)
            
            if result.success:
                # 向量化和插入
                await self._batch_vectorize_and_insert([result])
                
                self.processing_stats["last_update"] = datetime.now()
                logger.info(f"文档 {document_id} 更新成功")
            
            return result
            
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return ProcessingResult(
                success=False,
                document_id=document_id,
                chunks_created=0,
                processing_time=0,
                error_message=str(e)
            )
    
    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            if document_id not in self.document_cache:
                logger.warning(f"文档 {document_id} 不在缓存中")
                return True
            
            # 获取文档分块IDs
            chunks = self.document_cache[document_id]["chunks"]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # 从向量数据库删除
            success = await self.vector_db.delete_vectors(
                collection_name=self.collection_name,
                point_ids=chunk_ids
            )
            
            if success:
                # 从缓存删除
                del self.document_cache[document_id]
                logger.info(f"文档 {document_id} 删除成功")
            
            return success
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        if document_id in self.document_cache:
            return self.document_cache[document_id]
        
        # 如果缓存中没有，可以从向量数据库查询
        # 这里简化处理，实际可以实现更复杂的查询逻辑
        return None
    
    async def get_documents_by_metadata(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据元数据过滤获取文档"""
        matching_docs = []
        
        for doc_id, doc_data in self.document_cache.items():
            metadata = doc_data["metadata"]
            
            # 简单的过滤逻辑
            match = True
            for key, value in filters.items():
                if hasattr(metadata, key):
                    doc_value = getattr(metadata, key)
                    if doc_value != value:
                        match = False
                        break
            
            if match:
                matching_docs.append(doc_data)
        
        return matching_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = self.processing_stats.copy()
        stats["cached_documents"] = len(self.document_cache)
        stats["embedding_model"] = self.embedding_generator.model_name
        stats["collection_name"] = self.collection_name
        
        return stats


# 全局知识库构建器实例
_knowledge_builder: Optional[KnowledgeBaseBuilder] = None


def get_knowledge_builder(collection_name: str = "travel_knowledge") -> KnowledgeBaseBuilder:
    """获取知识库构建器实例"""
    global _knowledge_builder
    if _knowledge_builder is None:
        _knowledge_builder = KnowledgeBaseBuilder(collection_name)
    return _knowledge_builder 