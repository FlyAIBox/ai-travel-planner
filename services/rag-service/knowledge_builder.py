"""
知识库构建系统
实现文档向量化、预处理分块、ETL流水线、版本管理、增量更新、质量评估和自动筛选
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import re
from urllib.parse import urlparse
import mimetypes

import aiofiles
import aiohttp
import numpy as np
from bs4 import BeautifulSoup
import pypdf
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
import jieba

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .vector_database import get_vector_database, VectorIndexConfig

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class DocumentMetadata:
    """文档元数据"""
    source_id: str
    source_type: str  # 'file', 'url', 'api', 'manual'
    source_path: str
    title: Optional[str] = None
    author: Optional[str] = None
    language: str = 'zh'
    category: str = 'general'
    tags: List[str] = None
    priority: int = 1  # 1-5, 5最高
    created_at: datetime = None
    updated_at: datetime = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class DocumentChunk:
    """文档块"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    quality_score: float = 0.0
    
    def to_vector_payload(self) -> Dict[str, Any]:
        """转换为向量数据库载荷"""
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "source_type": self.metadata.source_type,
            "source_path": self.metadata.source_path,
            "title": self.metadata.title,
            "author": self.metadata.author,
            "language": self.metadata.language,
            "category": self.metadata.category,
            "tags": self.metadata.tags,
            "priority": self.metadata.priority,
            "quality_score": self.quality_score,
            "created_at": self.metadata.created_at.isoformat(),
            "updated_at": self.metadata.updated_at.isoformat()
        }


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    document_id: str
    chunks_count: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    quality_stats: Dict[str, Any] = None


class DocumentPreprocessor:
    """文档预处理器"""
    
    def __init__(self):
        self.text_cleaners = {
            'html': self._clean_html,
            'pdf': self._clean_pdf_text,
            'text': self._clean_plain_text,
            'markdown': self._clean_markdown
        }
        
        # 初始化中文分词
        jieba.initialize()
    
    def preprocess_text(self, text: str, content_type: str = 'text') -> str:
        """预处理文本"""
        # 选择清理器
        cleaner = self.text_cleaners.get(content_type, self._clean_plain_text)
        
        # 清理文本
        cleaned_text = cleaner(text)
        
        # 通用清理
        cleaned_text = self._apply_common_cleaning(cleaned_text)
        
        return cleaned_text
    
    def _clean_html(self, html_content: str) -> str:
        """清理HTML内容"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 提取文本
        text = soup.get_text()
        
        # 清理空白行
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)
    
    def _clean_pdf_text(self, pdf_text: str) -> str:
        """清理PDF文本"""
        # 移除页眉页脚模式
        lines = pdf_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过可能的页码
            if re.match(r'^\d+$', line):
                continue
            
            # 跳过短行（可能是页眉页脚）
            if len(line) < 10:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_plain_text(self, text: str) -> str:
        """清理纯文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()[\]{}""''—–-]', '', text)
        
        return text.strip()
    
    def _clean_markdown(self, markdown_text: str) -> str:
        """清理Markdown文本"""
        # 移除Markdown语法
        text = re.sub(r'#{1,6}\s*', '', markdown_text)  # 标题
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 粗体
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # 斜体
        text = re.sub(r'`(.*?)`', r'\1', text)  # 代码
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 链接
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # 图片
        
        return self._clean_plain_text(text)
    
    def _apply_common_cleaning(self, text: str) -> str:
        """应用通用清理规则"""
        # 规范化空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除过长的重复字符
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
        
        # 移除空行
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)


class DocumentSplitter:
    """文档分块器"""
    
    def __init__(self):
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=['\n\n', '\n', '。', '！', '？', '.', '!', '?', ' ', '']
            ),
            'character': CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separator='\n'
            ),
            'sentence': self._sentence_splitter,
            'semantic': self._semantic_splitter
        }
    
    def split_document(self, 
                      text: str, 
                      method: str = 'recursive',
                      chunk_size: int = 500,
                      chunk_overlap: int = 50) -> List[str]:
        """分块文档"""
        if method in ['recursive', 'character']:
            splitter = self.splitters[method]
            splitter.chunk_size = chunk_size
            splitter.chunk_overlap = chunk_overlap
            return splitter.split_text(text)
        
        elif method == 'sentence':
            return self._sentence_splitter(text, chunk_size)
        
        elif method == 'semantic':
            return self._semantic_splitter(text, chunk_size)
        
        else:
            raise ValueError(f"未知的分块方法: {method}")
    
    def _sentence_splitter(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """按句子分块"""
        # 中英文句子分割
        sentences = re.split(r'[。！？.!?]\s*', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否需要开始新块
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _semantic_splitter(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """语义分块（简化版）"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果段落太长，按句子分割
            if len(paragraph) > max_chunk_size:
                sentences = self._sentence_splitter(paragraph, max_chunk_size)
                chunks.extend(sentences)
            else:
                # 检查是否需要开始新块
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class EmbeddingGenerator:
    """向量生成器"""
    
    def __init__(self):
        self.models = {}
        self.default_model = settings.EMBEDDING_MODEL
        
        # 支持的模型配置
        self.model_configs = {
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': {
                'dimension': 384,
                'language': 'multilingual'
            },
            'BAAI/bge-small-zh-v1.5': {
                'dimension': 512,
                'language': 'zh'
            },
            'BAAI/bge-large-zh-v1.5': {
                'dimension': 1024,
                'language': 'zh'
            },
            'text-embedding-ada-002': {
                'dimension': 1536,
                'language': 'multilingual',
                'api_based': True
            }
        }
    
    def load_model(self, model_name: str) -> SentenceTransformer:
        """加载模型"""
        if model_name not in self.models:
            if self.model_configs.get(model_name, {}).get('api_based', False):
                # API based model (如OpenAI)
                self.models[model_name] = None  # 将在generate_embeddings中处理
            else:
                self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"加载向量模型: {model_name}")
        
        return self.models[model_name]
    
    async def generate_embeddings(self, 
                                texts: List[str],
                                model_name: str = None,
                                batch_size: int = 32) -> List[List[float]]:
        """生成向量"""
        if model_name is None:
            model_name = self.default_model
        
        model = self.load_model(model_name)
        
        if self.model_configs.get(model_name, {}).get('api_based', False):
            # API based embedding
            return await self._generate_api_embeddings(texts, model_name)
        else:
            # Local model embedding
            return await self._generate_local_embeddings(texts, model, batch_size)
    
    async def _generate_local_embeddings(self, 
                                       texts: List[str],
                                       model: SentenceTransformer,
                                       batch_size: int) -> List[List[float]]:
        """生成本地模型向量"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
            embeddings.extend(batch_embeddings.tolist())
            
            # 添加延迟避免过载
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    async def _generate_api_embeddings(self, 
                                     texts: List[str],
                                     model_name: str) -> List[List[float]]:
        """生成API模型向量"""
        # 这里应该实现OpenAI API调用
        # 为了演示，返回随机向量
        dimension = self.model_configs[model_name]['dimension']
        embeddings = []
        
        for text in texts:
            # 生成随机向量（实际应该调用API）
            embedding = np.random.rand(dimension).tolist()
            embeddings.append(embedding)
            await asyncio.sleep(0.01)  # 模拟API延迟
        
        return embeddings
    
    def get_model_dimension(self, model_name: str) -> int:
        """获取模型维度"""
        return self.model_configs.get(model_name, {}).get('dimension', 384)


class QualityAssessment:
    """质量评估器"""
    
    def __init__(self):
        self.min_content_length = 50
        self.max_content_length = 5000
        self.min_word_count = 10
        self.stopwords_ratio_threshold = 0.8
        
        # 质量评估权重
        self.weights = {
            'length': 0.2,
            'diversity': 0.3,
            'coherence': 0.2,
            'informativeness': 0.3
        }
    
    def assess_chunk_quality(self, chunk: DocumentChunk) -> float:
        """评估块质量"""
        content = chunk.content
        
        # 长度评分
        length_score = self._assess_length(content)
        
        # 多样性评分
        diversity_score = self._assess_diversity(content)
        
        # 连贯性评分
        coherence_score = self._assess_coherence(content)
        
        # 信息量评分
        informativeness_score = self._assess_informativeness(content)
        
        # 综合评分
        total_score = (
            length_score * self.weights['length'] +
            diversity_score * self.weights['diversity'] +
            coherence_score * self.weights['coherence'] +
            informativeness_score * self.weights['informativeness']
        )
        
        return min(max(total_score, 0.0), 1.0)
    
    def _assess_length(self, content: str) -> float:
        """评估长度适当性"""
        length = len(content)
        
        if length < self.min_content_length:
            return length / self.min_content_length
        elif length > self.max_content_length:
            return max(0.5, 1.0 - (length - self.max_content_length) / self.max_content_length)
        else:
            return 1.0
    
    def _assess_diversity(self, content: str) -> float:
        """评估词汇多样性"""
        words = jieba.lcut(content)
        if len(words) == 0:
            return 0.0
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        return min(diversity_ratio * 2, 1.0)  # 归一化到[0, 1]
    
    def _assess_coherence(self, content: str) -> float:
        """评估连贯性（简化版）"""
        sentences = re.split(r'[。！？.!?]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # 计算句子长度的一致性
        lengths = [len(s) for s in sentences]
        if len(lengths) == 0:
            return 0.0
        
        length_variance = np.var(lengths)
        mean_length = np.mean(lengths)
        
        if mean_length == 0:
            return 0.0
        
        coherence_score = 1.0 / (1.0 + length_variance / (mean_length ** 2))
        
        return min(max(coherence_score, 0.0), 1.0)
    
    def _assess_informativeness(self, content: str) -> float:
        """评估信息量"""
        words = jieba.lcut(content)
        if len(words) == 0:
            return 0.0
        
        # 计算信息密度（非停用词比例）
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', 
                    '这', '那', '这个', '那个', '什么', '怎么', '为什么', '可以', '能够', '应该'}
        
        non_stopwords = [w for w in words if w not in stopwords and len(w) > 1]
        informativeness = len(non_stopwords) / len(words)
        
        return min(informativeness * 1.5, 1.0)  # 放大信息密度的影响


class KnowledgeVersionManager:
    """知识库版本管理器"""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path("data/knowledge_versions")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.current_version = self._get_latest_version()
        self.version_info = {}
    
    def _get_latest_version(self) -> str:
        """获取最新版本"""
        version_files = list(self.base_path.glob("v*.json"))
        if not version_files:
            return "v1.0.0"
        
        versions = []
        for file in version_files:
            version_str = file.stem
            try:
                # 解析版本号 (v1.2.3)
                parts = version_str[1:].split('.')
                version_tuple = tuple(int(p) for p in parts)
                versions.append((version_tuple, version_str))
            except ValueError:
                continue
        
        if versions:
            latest = max(versions, key=lambda x: x[0])
            return latest[1]
        
        return "v1.0.0"
    
    def create_new_version(self, changes: List[str]) -> str:
        """创建新版本"""
        # 解析当前版本号
        current_parts = self.current_version[1:].split('.')
        major, minor, patch = int(current_parts[0]), int(current_parts[1]), int(current_parts[2])
        
        # 递增版本号
        patch += 1
        new_version = f"v{major}.{minor}.{patch}"
        
        # 保存版本信息
        version_info = {
            "version": new_version,
            "created_at": datetime.now().isoformat(),
            "changes": changes,
            "parent_version": self.current_version,
            "status": "active"
        }
        
        version_file = self.base_path / f"{new_version}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, ensure_ascii=False, indent=2)
        
        self.current_version = new_version
        self.version_info[new_version] = version_info
        
        logger.info(f"创建新版本: {new_version}")
        return new_version
    
    def get_version_info(self, version: str = None) -> Dict[str, Any]:
        """获取版本信息"""
        if version is None:
            version = self.current_version
        
        if version in self.version_info:
            return self.version_info[version]
        
        version_file = self.base_path / f"{version}.json"
        if version_file.exists():
            with open(version_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self.version_info[version] = info
                return info
        
        return {}
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        versions = []
        for version_file in sorted(self.base_path.glob("v*.json")):
            version = version_file.stem
            info = self.get_version_info(version)
            if info:
                versions.append(info)
        
        return versions


class KnowledgeBuilder:
    """知识库构建器"""
    
    def __init__(self):
        self.vector_db = get_vector_database()
        self.preprocessor = DocumentPreprocessor()
        self.splitter = DocumentSplitter()
        self.embedding_generator = EmbeddingGenerator()
        self.quality_assessor = QualityAssessment()
        self.version_manager = KnowledgeVersionManager()
        
        # 处理统计
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_chunks": 0,
            "high_quality_chunks": 0,
            "processing_times": []
        }
        
        # 默认配置
        self.default_chunk_size = 500
        self.default_overlap = 50
        self.quality_threshold = 0.6
        self.collection_name = settings.QDRANT_COLLECTION_NAME
    
    async def process_document(self, 
                             content: str,
                             metadata: DocumentMetadata,
                             chunk_method: str = 'recursive',
                             embedding_model: str = None) -> ProcessingResult:
        """处理单个文档"""
        start_time = time.time()
        document_id = metadata.source_id
        
        try:
            # 预处理文档
            cleaned_content = self.preprocessor.preprocess_text(
                content, 
                metadata.mime_type or 'text'
            )
            
            # 分块
            chunks_text = self.splitter.split_document(
                cleaned_content,
                method=chunk_method,
                chunk_size=self.default_chunk_size,
                chunk_overlap=self.default_overlap
            )
            
            # 创建文档块
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{i}",
                    document_id=document_id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata=metadata
                )
                
                # 质量评估
                chunk.quality_score = self.quality_assessor.assess_chunk_quality(chunk)
                
                # 只保留高质量块
                if chunk.quality_score >= self.quality_threshold:
                    chunks.append(chunk)
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    error="没有通过质量检查的块"
                )
            
            # 生成向量
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_generator.generate_embeddings(
                chunk_contents,
                model_name=embedding_model
            )
            
            # 添加向量到块
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # 存储到向量数据库
            await self._store_chunks(chunks)
            
            processing_time = time.time() - start_time
            
            # 更新统计
            self.processing_stats["total_documents"] += 1
            self.processing_stats["successful_documents"] += 1
            self.processing_stats["total_chunks"] += len(chunks)
            self.processing_stats["high_quality_chunks"] += len(chunks)
            self.processing_stats["processing_times"].append(processing_time)
            
            logger.info(f"成功处理文档 {document_id}: {len(chunks)} 个块")
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                chunks_count=len(chunks),
                processing_time=processing_time,
                quality_stats={
                    "total_chunks": len(chunks_text),
                    "quality_chunks": len(chunks),
                    "avg_quality_score": np.mean([c.quality_score for c in chunks])
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.processing_stats["total_documents"] += 1
            self.processing_stats["failed_documents"] += 1
            self.processing_stats["processing_times"].append(processing_time)
            
            logger.error(f"处理文档失败 {document_id}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """存储块到向量数据库"""
        vectors = [chunk.embedding for chunk in chunks]
        payloads = [chunk.to_vector_payload() for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        await self.vector_db.insert_vectors(
            collection_name=self.collection_name,
            vectors=vectors,
            payloads=payloads,
            ids=ids
        )
    
    async def process_batch(self, 
                          documents: List[Tuple[str, DocumentMetadata]],
                          max_concurrent: int = 5) -> List[ProcessingResult]:
        """批量处理文档"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(content, metadata):
            async with semaphore:
                return await self.process_document(content, metadata)
        
        tasks = [
            process_with_semaphore(content, metadata)
            for content, metadata in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    document_id=documents[i][1].source_id,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def update_document(self, 
                            content: str,
                            metadata: DocumentMetadata) -> ProcessingResult:
        """更新文档"""
        document_id = metadata.source_id
        
        # 删除旧版本
        await self.vector_db.delete_vectors(
            collection_name=self.collection_name,
            filter_conditions={"must_document_id": document_id}
        )
        
        # 处理新版本
        metadata.updated_at = datetime.now()
        return await self.process_document(content, metadata)
    
    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            await self.vector_db.delete_vectors(
                collection_name=self.collection_name,
                filter_conditions={"must_document_id": document_id}
            )
            logger.info(f"删除文档: {document_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    async def build_travel_knowledge_base(self) -> str:
        """构建旅行知识库"""
        logger.info("开始构建旅行知识库")
        
        # 收集旅行相关文档
        travel_documents = await self._collect_travel_documents()
        
        if not travel_documents:
            logger.warning("没有找到旅行文档")
            return ""
        
        # 批量处理
        results = await self.process_batch(travel_documents, max_concurrent=3)
        
        # 统计结果
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        changes = [
            f"处理了 {len(travel_documents)} 个旅行文档",
            f"成功: {len(successful)}, 失败: {len(failed)}",
            f"总计生成 {sum(r.chunks_count for r in successful)} 个高质量块"
        ]
        
        # 创建新版本
        version = self.version_manager.create_new_version(changes)
        
        logger.info(f"旅行知识库构建完成，版本: {version}")
        return version
    
    async def _collect_travel_documents(self) -> List[Tuple[str, DocumentMetadata]]:
        """收集旅行文档"""
        documents = []
        
        # 示例：添加一些旅行相关文档
        travel_data = [
            {
                "content": """
                北京是中国的首都，拥有悠久的历史和丰富的文化遗产。主要景点包括故宫、天安门广场、长城、颐和园等。
                最佳旅游时间是春季（4-5月）和秋季（9-10月），气候宜人。
                交通便利，有多个机场和火车站。住宿选择丰富，从经济型酒店到豪华酒店应有尽有。
                美食推荐：北京烤鸭、炸酱面、豆汁等。
                """,
                "metadata": DocumentMetadata(
                    source_id="beijing_travel_guide",
                    source_type="manual",
                    source_path="travel_guides/beijing.txt",
                    title="北京旅游指南",
                    category="destination_guide",
                    tags=["北京", "首都", "历史", "文化"],
                    priority=5
                )
            },
            {
                "content": """
                上海是中国的经济中心，现代化程度很高。著名景点有外滩、东方明珠塔、豫园、南京路等。
                上海的夜景非常美丽，特别是黄浦江两岸的建筑群。
                交通发达，地铁网络覆盖全市。购物和美食选择众多。
                推荐住宿：外滩附近的酒店可以欣赏江景。
                """,
                "metadata": DocumentMetadata(
                    source_id="shanghai_travel_guide",
                    source_type="manual",
                    source_path="travel_guides/shanghai.txt",
                    title="上海旅游指南",
                    category="destination_guide",
                    tags=["上海", "现代化", "外滩", "购物"],
                    priority=5
                )
            },
            {
                "content": """
                西安是著名的历史文化名城，有3000多年的历史。兵马俑是世界八大奇迹之一，不可错过。
                其他重要景点：大雁塔、华清池、城墙、钟楼等。
                西安美食丰富：肉夹馍、凉皮、羊肉泡馍、胡辣汤等。
                建议游玩时间：3-4天。交通便利，有机场和高铁站。
                """,
                "metadata": DocumentMetadata(
                    source_id="xian_travel_guide",
                    source_type="manual",
                    source_path="travel_guides/xian.txt",
                    title="西安旅游指南",
                    category="destination_guide",
                    tags=["西安", "历史", "兵马俑", "美食"],
                    priority=5
                )
            },
            {
                "content": """
                预订机票的最佳时间通常是出发前2-8周。周二和周三的机票通常比较便宜。
                比较不同航空公司和预订网站的价格。考虑使用里程积分或信用卡积分。
                注意行李规定和退改签政策。建议购买旅行保险。
                特价机票注意事项：时间限制、退改限制、座位选择等。
                """,
                "metadata": DocumentMetadata(
                    source_id="flight_booking_tips",
                    source_type="manual",
                    source_path="tips/flight_booking.txt",
                    title="机票预订攻略",
                    category="booking_tips",
                    tags=["机票", "预订", "省钱", "攻略"],
                    priority=4
                )
            },
            {
                "content": """
                选择酒店的关键因素：位置、价格、设施、评价、安全性。
                预订平台比较：Booking.com、Agoda、携程、去哪儿等。
                住宿类型：酒店、民宿、青旅、公寓等各有优缺点。
                预订技巧：提前预订享优惠、关注促销活动、会员积分等。
                入住注意事项：确认预订信息、了解取消政策、检查房间设施等。
                """,
                "metadata": DocumentMetadata(
                    source_id="hotel_booking_guide",
                    source_type="manual",
                    source_path="tips/hotel_booking.txt",
                    title="酒店预订指南",
                    category="booking_tips",
                    tags=["酒店", "住宿", "预订", "攻略"],
                    priority=4
                )
            }
        ]
        
        for data in travel_data:
            documents.append((data["content"], data["metadata"]))
        
        return documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        processing_times = self.processing_stats["processing_times"]
        
        stats = {
            "total_documents": self.processing_stats["total_documents"],
            "successful_documents": self.processing_stats["successful_documents"],
            "failed_documents": self.processing_stats["failed_documents"],
            "success_rate": self.processing_stats["successful_documents"] / max(self.processing_stats["total_documents"], 1),
            "total_chunks": self.processing_stats["total_chunks"],
            "high_quality_chunks": self.processing_stats["high_quality_chunks"],
            "quality_rate": self.processing_stats["high_quality_chunks"] / max(self.processing_stats["total_chunks"], 1)
        }
        
        if processing_times:
            stats.update({
                "avg_processing_time": np.mean(processing_times),
                "min_processing_time": np.min(processing_times),
                "max_processing_time": np.max(processing_times)
            })
        
        return stats


# 全局实例
knowledge_builder = None

def get_knowledge_builder() -> KnowledgeBuilder:
    """获取知识库构建器实例"""
    global knowledge_builder
    if knowledge_builder is None:
        knowledge_builder = KnowledgeBuilder()
    return knowledge_builder 