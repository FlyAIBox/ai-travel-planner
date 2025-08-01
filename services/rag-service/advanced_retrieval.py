"""
高级RAG检索策略
实现混合检索系统（向量检索 + BM25 + 图检索）、查询理解和意图分析、动态检索策略、结果重排序
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import networkx as nx
from rank_bm25 import BM25Okapi
import jieba

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .vector_database import get_vector_database, VectorSearchResult
from .knowledge_builder import get_knowledge_builder

logger = get_logger(__name__)
settings = get_settings()


class QueryType(Enum):
    """查询类型枚举"""
    FACTUAL = "factual"              # 事实性查询
    PROCEDURAL = "procedural"        # 程序性查询
    NAVIGATIONAL = "navigational"    # 导航性查询
    COMPARISON = "comparison"        # 比较性查询
    RECOMMENDATION = "recommendation" # 推荐性查询


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR_ONLY = "vector_only"      # 仅向量检索
    BM25_ONLY = "bm25_only"          # 仅BM25检索
    HYBRID = "hybrid"                # 混合检索
    GRAPH = "graph"                  # 图检索
    ADAPTIVE = "adaptive"            # 自适应检索


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    processed_query: str
    query_type: QueryType
    intent: str
    entities: List[Dict[str, Any]]
    keywords: List[str]
    language: str
    complexity_score: float
    semantic_vector: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_method: str
    rank: int
    

@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    query_analysis: QueryAnalysis
    results: List[RetrievalResult]
    retrieval_stats: Dict[str, Any]
    total_time: float


class QueryProcessor:
    """查询处理器"""
    
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
        
        # 初始化意图分类器
        try:
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            logger.warning("意图分类器初始化失败")
            self.intent_classifier = None
        
        # 查询类型模式
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'什么是|什么叫|是什么|定义|介绍',
                r'what is|what are|define|definition'
            ],
            QueryType.PROCEDURAL: [
                r'如何|怎么|怎样|步骤|方法',
                r'how to|how do|steps|method|process'
            ],
            QueryType.NAVIGATIONAL: [
                r'在哪里|哪里有|位置|地址',
                r'where is|where are|location|address'
            ],
            QueryType.COMPARISON: [
                r'比较|对比|区别|差异|哪个更好',
                r'compare|comparison|difference|better|versus'
            ],
            QueryType.RECOMMENDATION: [
                r'推荐|建议|最好的|推荐一下',
                r'recommend|suggestion|best|top|suggest'
            ]
        }
    
    def detect_language(self, text: str) -> str:
        """检测查询语言"""
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
    
    def classify_query_type(self, query: str) -> QueryType:
        """分类查询类型"""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.FACTUAL  # 默认为事实性查询
    
    def extract_entities(self, text: str, language: str) -> List[Dict[str, Any]]:
        """提取命名实体"""
        entities = []
        
        try:
            if language == "zh" and self.nlp_zh:
                doc = self.nlp_zh(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            elif language == "en" and self.nlp_en:
                doc = self.nlp_en(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")
        
        return entities
    
    def extract_keywords(self, text: str, language: str) -> List[str]:
        """提取关键词"""
        keywords = []
        
        try:
            if language == "zh":
                # 使用jieba分词
                words = jieba.cut(text)
                # 过滤停用词和短词
                stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '等'}
                keywords = [word for word in words if len(word) > 1 and word not in stop_words]
            else:
                # 英文关键词提取
                if self.nlp_en:
                    doc = self.nlp_en(text)
                    keywords = [token.lemma_ for token in doc 
                              if not token.is_stop and not token.is_punct and len(token.text) > 2]
                else:
                    # 简单的英文分词
                    words = re.findall(r'\b\w+\b', text.lower())
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            # 回退到简单分词
            keywords = text.split()
        
        return list(set(keywords))[:10]  # 最多返回10个关键词
    
    def calculate_complexity_score(self, text: str) -> float:
        """计算查询复杂度分数"""
        # 基于多个因素计算复杂度
        factors = {
            "length": len(text.split()) / 20,  # 基于长度
            "entities": 0,
            "operators": 0,
            "specificity": 0
        }
        
        # 检查逻辑操作符
        logical_operators = ['and', 'or', 'not', '与', '或', '非', '但是', '然而']
        factors["operators"] = sum(1 for op in logical_operators if op in text.lower()) / 5
        
        # 检查特异性指标（日期、数字、专有名词等）
        specificity_patterns = [
            r'\d{4}年|\d{1,2}月|\d{1,2}日',  # 日期
            r'\d+',  # 数字
            r'[A-Z][a-z]+',  # 专有名词
        ]
        
        specificity_count = sum(len(re.findall(pattern, text)) for pattern in specificity_patterns)
        factors["specificity"] = min(specificity_count / 5, 1.0)
        
        # 计算综合复杂度
        complexity = np.mean(list(factors.values()))
        return min(complexity, 1.0)
    
    def extract_intent(self, text: str) -> str:
        """提取查询意图"""
        # 旅行相关意图模式
        travel_intents = {
            "planning": ["计划", "规划", "安排", "plan", "schedule"],
            "booking": ["预订", "预定", "订票", "订房", "book", "reserve"],
            "information": ["信息", "介绍", "了解", "知道", "info", "about"],
            "recommendation": ["推荐", "建议", "suggest", "recommend"],
            "comparison": ["比较", "对比", "compare", "versus"],
            "navigation": ["怎么去", "路线", "交通", "how to get", "route"]
        }
        
        text_lower = text.lower()
        for intent, keywords in travel_intents.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent
        
        return "general"
    
    async def process_query(self, query: str) -> QueryAnalysis:
        """处理查询"""
        # 检测语言
        language = self.detect_language(query)
        
        # 预处理查询
        processed_query = re.sub(r'\s+', ' ', query.strip())
        
        # 分类查询类型
        query_type = self.classify_query_type(processed_query)
        
        # 提取意图
        intent = self.extract_intent(processed_query)
        
        # 提取实体
        entities = self.extract_entities(processed_query, language)
        
        # 提取关键词
        keywords = self.extract_keywords(processed_query, language)
        
        # 计算复杂度
        complexity_score = self.calculate_complexity_score(processed_query)
        
        return QueryAnalysis(
            original_query=query,
            processed_query=processed_query,
            query_type=query_type,
            intent=intent,
            entities=entities,
            keywords=keywords,
            language=language,
            complexity_score=complexity_score
        )


class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.document_ids = []
        self.is_fitted = False
    
    async def fit(self, documents: List[Dict[str, Any]]) -> None:
        """训练BM25模型"""
        try:
            self.documents = documents
            self.document_ids = [doc["id"] for doc in documents]
            
            # 预处理文档
            corpus = []
            for doc in documents:
                content = doc.get("content", "")
                # 简单的中英文分词
                if any('\u4e00' <= char <= '\u9fff' for char in content):
                    # 中文分词
                    tokens = list(jieba.cut(content))
                else:
                    # 英文分词
                    tokens = re.findall(r'\b\w+\b', content.lower())
                corpus.append(tokens)
            
            # 训练BM25
            self.bm25 = BM25Okapi(corpus)
            self.is_fitted = True
            
            logger.info(f"BM25模型训练完成，文档数量: {len(documents)}")
            
        except Exception as e:
            logger.error(f"BM25训练失败: {e}")
            raise
    
    async def search(self, query: str, limit: int = 10) -> List[RetrievalResult]:
        """BM25检索"""
        if not self.is_fitted:
            logger.warning("BM25模型未训练")
            return []
        
        try:
            # 查询分词
            if any('\u4e00' <= char <= '\u9fff' for char in query):
                query_tokens = list(jieba.cut(query))
            else:
                query_tokens = re.findall(r'\b\w+\b', query.lower())
            
            # 计算分数
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = np.argsort(scores)[::-1][:limit]
            
            results = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] > 0:  # 只返回有分数的结果
                    doc = self.documents[idx]
                    result = RetrievalResult(
                        id=doc["id"],
                        content=doc.get("content", ""),
                        score=float(scores[idx]),
                        source=doc.get("source", "unknown"),
                        metadata=doc.get("metadata", {}),
                        retrieval_method="bm25",
                        rank=rank
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []


class GraphRetriever:
    """图检索器"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_embeddings = {}
        self.is_built = False
    
    async def build_graph(self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> None:
        """构建文档图"""
        try:
            # 清空现有图
            self.graph.clear()
            self.node_embeddings.clear()
            
            # 添加节点
            for doc in documents:
                doc_id = doc["id"]
                self.graph.add_node(doc_id, **doc)
            
            # 计算文档相似度并添加边
            from .knowledge_builder import get_knowledge_builder
            knowledge_builder = get_knowledge_builder()
            
            # 生成文档向量
            contents = [doc.get("content", "") for doc in documents]
            embeddings = await knowledge_builder.embedding_generator.generate_embeddings(contents)
            
            # 存储向量
            for doc, embedding in zip(documents, embeddings):
                self.node_embeddings[doc["id"]] = embedding
            
            # 添加相似性边
            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents[i+1:], i+1):
                    similarity = cosine_similarity(
                        [embeddings[i]], 
                        [embeddings[j]]
                    )[0][0]
                    
                    if similarity > similarity_threshold:
                        self.graph.add_edge(
                            doc1["id"], 
                            doc2["id"], 
                            weight=similarity
                        )
            
            self.is_built = True
            logger.info(f"文档图构建完成，节点: {self.graph.number_of_nodes()}, 边: {self.graph.number_of_edges()}")
            
        except Exception as e:
            logger.error(f"构建文档图失败: {e}")
            raise
    
    async def search(self, query: str, limit: int = 10, max_hops: int = 2) -> List[RetrievalResult]:
        """图检索"""
        if not self.is_built:
            logger.warning("文档图未构建")
            return []
        
        try:
            # 生成查询向量
            from .knowledge_builder import get_knowledge_builder
            knowledge_builder = get_knowledge_builder()
            query_embedding = (await knowledge_builder.embedding_generator.generate_embeddings([query]))[0]
            
            # 计算与所有节点的相似度
            node_similarities = {}
            for node_id, node_embedding in self.node_embeddings.items():
                similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                node_similarities[node_id] = similarity
            
            # 找到最相似的起始节点
            start_nodes = sorted(node_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # 使用随机游走或广度优先搜索扩展结果
            expanded_nodes = set()
            for start_node, start_similarity in start_nodes:
                # 添加起始节点
                expanded_nodes.add(start_node)
                
                # BFS扩展
                visited = {start_node}
                queue = [(start_node, 0)]
                
                while queue and len(expanded_nodes) < limit * 2:
                    current_node, hop = queue.pop(0)
                    
                    if hop >= max_hops:
                        continue
                    
                    # 访问邻居节点
                    for neighbor in self.graph.neighbors(current_node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            expanded_nodes.add(neighbor)
                            queue.append((neighbor, hop + 1))
            
            # 构建结果
            results = []
            for rank, node_id in enumerate(list(expanded_nodes)[:limit]):
                node_data = self.graph.nodes[node_id]
                similarity = node_similarities.get(node_id, 0)
                
                result = RetrievalResult(
                    id=node_id,
                    content=node_data.get("content", ""),
                    score=similarity,
                    source=node_data.get("source", "graph"),
                    metadata=node_data.get("metadata", {}),
                    retrieval_method="graph",
                    rank=rank
                )
                results.append(result)
            
            # 按相似度排序
            results.sort(key=lambda x: x.score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"图检索失败: {e}")
            return []


class CrossEncoder:
    """交叉编码器重排序"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    async def load_model(self) -> None:
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder as SentenceCrossEncoder
            self.model = SentenceCrossEncoder(self.model_name)
            logger.info(f"重排序模型 {self.model_name} 加载成功")
        except Exception as e:
            logger.error(f"加载重排序模型失败: {e}")
            self.model = None
    
    async def rerank(self, query: str, results: List[RetrievalResult], top_k: int = None) -> List[RetrievalResult]:
        """重排序结果"""
        if not self.model:
            await self.load_model()
        
        if not self.model or len(results) <= 1:
            return results
        
        try:
            # 准备查询-文档对
            query_doc_pairs = [(query, result.content) for result in results]
            
            # 计算重排序分数
            rerank_scores = self.model.predict(query_doc_pairs)
            
            # 更新结果分数并重新排序
            for i, (result, score) in enumerate(zip(results, rerank_scores)):
                result.score = float(score)
                result.rank = i
            
            # 按新分数排序
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # 更新排名
            for rank, result in enumerate(reranked_results):
                result.rank = rank
            
            # 返回top-k结果
            if top_k:
                reranked_results = reranked_results[:top_k]
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self):
        self.vector_db = get_vector_database()
        self.knowledge_builder = get_knowledge_builder()
        self.query_processor = QueryProcessor()
        self.bm25_retriever = BM25Retriever()
        self.graph_retriever = GraphRetriever()
        self.cross_encoder = CrossEncoder()
        
        # 检索权重配置
        self.retrieval_weights = {
            "vector": 0.5,
            "bm25": 0.3,
            "graph": 0.2
        }
        
        # 缓存
        self.document_cache = {}
        self.last_cache_update = None
        
    async def initialize(self) -> None:
        """初始化混合检索器"""
        try:
            # 加载文档到缓存
            await self._refresh_document_cache()
            
            # 训练BM25模型
            if self.document_cache:
                documents_list = list(self.document_cache.values())
                await self.bm25_retriever.fit(documents_list)
                await self.graph_retriever.build_graph(documents_list)
            
            # 加载重排序模型
            await self.cross_encoder.load_model()
            
            logger.info("混合检索器初始化完成")
            
        except Exception as e:
            logger.error(f"混合检索器初始化失败: {e}")
            raise
    
    async def _refresh_document_cache(self) -> None:
        """刷新文档缓存"""
        try:
            # 这里应该从向量数据库加载所有文档
            # 由于Qdrant API限制，我们创建一些示例文档
            sample_documents = [
                {
                    "id": "doc_001",
                    "content": "旅行规划需要考虑预算、时间、目的地选择等多个因素。合理的规划能让旅行更加愉快。",
                    "source": "travel_guide",
                    "metadata": {"category": "planning", "type": "guide"}
                },
                {
                    "id": "doc_002", 
                    "content": "酒店预订时要注意位置、设施、价格和用户评价。提前预订通常能获得更好的价格。",
                    "source": "booking_tips",
                    "metadata": {"category": "accommodation", "type": "tips"}
                },
                {
                    "id": "doc_003",
                    "content": "航班选择要考虑价格、时间、航空公司服务。中转航班有时比直飞更经济。",
                    "source": "flight_guide", 
                    "metadata": {"category": "transportation", "type": "guide"}
                }
            ]
            
            self.document_cache = {doc["id"]: doc for doc in sample_documents}
            self.last_cache_update = datetime.now()
            
            logger.info(f"文档缓存刷新完成，共 {len(self.document_cache)} 个文档")
            
        except Exception as e:
            logger.error(f"刷新文档缓存失败: {e}")
            raise
    
    async def search(self, 
                    query: str,
                    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                    limit: int = 10,
                    rerank: bool = True) -> HybridSearchResult:
        """执行混合检索"""
        start_time = datetime.now()
        
        # 查询分析
        query_analysis = await self.query_processor.process_query(query)
        
        # 生成查询向量
        query_embeddings = await self.knowledge_builder.embedding_generator.generate_embeddings([query])
        query_analysis.semantic_vector = query_embeddings[0]
        
        # 执行不同检索策略
        all_results = []
        retrieval_stats = {
            "vector_results": 0,
            "bm25_results": 0, 
            "graph_results": 0,
            "total_candidates": 0
        }
        
        if strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID, RetrievalStrategy.ADAPTIVE]:
            # 向量检索
            vector_results = await self._vector_search(query_analysis.semantic_vector, limit)
            for result in vector_results:
                result.retrieval_method = "vector"
            all_results.extend(vector_results)
            retrieval_stats["vector_results"] = len(vector_results)
        
        if strategy in [RetrievalStrategy.BM25_ONLY, RetrievalStrategy.HYBRID, RetrievalStrategy.ADAPTIVE]:
            # BM25检索
            bm25_results = await self.bm25_retriever.search(query, limit)
            all_results.extend(bm25_results)
            retrieval_stats["bm25_results"] = len(bm25_results)
        
        if strategy in [RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID, RetrievalStrategy.ADAPTIVE]:
            # 图检索
            graph_results = await self.graph_retriever.search(query, limit)
            all_results.extend(graph_results)
            retrieval_stats["graph_results"] = len(graph_results)
        
        # 合并和去重结果
        merged_results = self._merge_results(all_results, strategy)
        retrieval_stats["total_candidates"] = len(merged_results)
        
        # 重排序
        if rerank and len(merged_results) > 1:
            merged_results = await self.cross_encoder.rerank(query, merged_results, limit)
        
        # 限制结果数量
        final_results = merged_results[:limit]
        
        # 计算总时间
        total_time = (datetime.now() - start_time).total_seconds()
        
        return HybridSearchResult(
            query_analysis=query_analysis,
            results=final_results,
            retrieval_stats=retrieval_stats,
            total_time=total_time
        )
    
    async def _vector_search(self, query_vector: List[float], limit: int) -> List[RetrievalResult]:
        """向量检索"""
        try:
            collection_name = settings.QDRANT_COLLECTION_NAME
            search_results = await self.vector_db.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for rank, search_result in enumerate(search_results):
                result = RetrievalResult(
                    id=search_result.id,
                    content=search_result.payload.get("content", ""),
                    score=search_result.score,
                    source=search_result.payload.get("source", "vector_db"),
                    metadata=search_result.payload,
                    retrieval_method="vector",
                    rank=rank
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _merge_results(self, 
                      all_results: List[RetrievalResult], 
                      strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """合并检索结果"""
        # 按ID去重
        results_by_id = {}
        
        for result in all_results:
            result_id = result.id
            
            if result_id not in results_by_id:
                results_by_id[result_id] = result
            else:
                # 合并分数
                existing_result = results_by_id[result_id]
                
                if strategy == RetrievalStrategy.HYBRID:
                    # 加权合并
                    weight = self.retrieval_weights.get(result.retrieval_method, 0.1)
                    existing_weight = self.retrieval_weights.get(existing_result.retrieval_method, 0.1)
                    
                    combined_score = (existing_result.score * existing_weight + result.score * weight) / (existing_weight + weight)
                    existing_result.score = combined_score
                    existing_result.retrieval_method += f"+{result.retrieval_method}"
                else:
                    # 取最高分
                    if result.score > existing_result.score:
                        results_by_id[result_id] = result
        
        # 按分数排序
        merged_results = list(results_by_id.values())
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        # 更新排名
        for rank, result in enumerate(merged_results):
            result.rank = rank
        
        return merged_results
    
    async def get_adaptive_strategy(self, query_analysis: QueryAnalysis) -> RetrievalStrategy:
        """根据查询特征选择自适应策略"""
        # 根据查询类型和复杂度选择策略
        if query_analysis.query_type == QueryType.FACTUAL:
            if query_analysis.complexity_score < 0.3:
                return RetrievalStrategy.VECTOR_ONLY
            else:
                return RetrievalStrategy.HYBRID
        
        elif query_analysis.query_type == QueryType.PROCEDURAL:
            return RetrievalStrategy.BM25_ONLY
        
        elif query_analysis.query_type == QueryType.COMPARISON:
            return RetrievalStrategy.HYBRID
        
        elif query_analysis.query_type == QueryType.RECOMMENDATION:
            return RetrievalStrategy.GRAPH
        
        else:
            return RetrievalStrategy.HYBRID
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            "document_cache_size": len(self.document_cache),
            "last_cache_update": self.last_cache_update.isoformat() if self.last_cache_update else None,
            "bm25_fitted": self.bm25_retriever.is_fitted,
            "graph_built": self.graph_retriever.is_built,
            "cross_encoder_loaded": self.cross_encoder.model is not None
        }


# 全局混合检索器实例
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器实例"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever 