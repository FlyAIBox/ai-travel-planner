"""
高级RAG检索策略
实现混合检索、智能查询处理、结果重排、自适应策略等功能
"""

import asyncio
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

import structlog
from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .vector_database import get_vector_database, VectorSearchResult
from .knowledge_builder import get_knowledge_builder

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    language: str
    query_type: str
    entities: List[str]
    keywords: List[str]
    intent: str
    complexity_score: float
    expanded_queries: List[str]
    
    
@dataclass 
class RetrievalResult:
    """检索结果"""
    document_id: str
    chunk_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    retrieval_method: str


class QueryProcessor:
    """查询处理器"""
    
    def __init__(self):
        self.entity_patterns = {
            "地点": [
                r'[\u4e00-\u9fff]+(?:市|省|县|区|镇|村|街|路|号)',
                r'[\u4e00-\u9fff]{2,4}(?:机场|车站|港口|景点|酒店)',
            ],
            "时间": [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{1,2}月\d{1,2}日',
                r'明天|今天|昨天|下周|上周|下月|上月',
                r'\d+天|\d+小时|\d+分钟'
            ],
            "价格": [
                r'\d+元|\d+块|\d+万',
                r'\$\d+|USD\d+|￥\d+',
                r'预算.*?\d+'
            ]
        }
        
        # 查询意图模式
        self.intent_patterns = {
            "搜索": ["找", "搜索", "查找", "寻找", "推荐"],
            "规划": ["规划", "安排", "计划", "设计", "路线"],
            "预订": ["预订", "订购", "购买", "预约", "买"],
            "询问": ["怎么", "如何", "什么", "哪里", "什么时候"],
            "比较": ["比较", "对比", "区别", "差异", "哪个好"]
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """分析查询"""
        analysis = QueryAnalysis(
            original_query=query,
            language=self.detect_language(query),
            query_type=self.classify_query_type(query),
            entities=self.extract_entities(query),
            keywords=self.extract_keywords(query),
            intent=self.extract_intent(query),
            complexity_score=self.calculate_complexity_score(query),
            expanded_queries=self.expand_query(query)
        )
        
        return analysis
    
    def detect_language(self, query: str) -> str:
        """检测查询语言"""
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', query))
        total_chars = len(query.replace(' ', ''))
        
        if chinese_count / max(total_chars, 1) > 0.3:
            return "zh"
        else:
            return "en"
    
    def classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["旅行", "旅游", "游览", "景点"]):
            return "travel"
        elif any(word in query_lower for word in ["酒店", "住宿", "宾馆"]):
            return "accommodation"
        elif any(word in query_lower for word in ["餐厅", "美食", "小吃", "饭店"]):
            return "food"
        elif any(word in query_lower for word in ["交通", "出行", "路线", "车票"]):
            return "transportation"
        else:
            return "general"
    
    def extract_entities(self, query: str) -> List[str]:
        """提取实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query)
                for match in matches:
                    entities.append(f"{entity_type}:{match}")
        
        return entities
    
    def extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        if JIEBA_AVAILABLE:
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(query, topK=10)
            return keywords
        else:
            # 简单的关键词提取
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query)
            # 过滤停用词
            stop_words = {"的", "是", "在", "有", "和", "或", "但", "与", "及", "等"}
            keywords = [word for word in words if len(word) > 1 and word not in stop_words]
            return keywords[:10]
    
    def extract_intent(self, query: str) -> str:
        """提取查询意图"""
        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query for pattern in patterns):
                return intent
        return "搜索"  # 默认意图
    
    def calculate_complexity_score(self, query: str) -> float:
        """计算查询复杂度分数"""
        score = 0.0
        
        # 长度因子
        score += len(query) * 0.01
        
        # 实体数量因子
        entities = self.extract_entities(query)
        score += len(entities) * 0.2
        
        # 关键词数量因子
        keywords = self.extract_keywords(query)
        score += len(keywords) * 0.1
        
        # 特殊字符因子
        special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', query))
        score += special_chars * 0.05
        
        return min(score, 10.0)  # 限制最大分数为10
    
    def expand_query(self, query: str) -> List[str]:
        """查询扩展"""
        expanded = []
        
        # 同义词扩展（简化版）
        synonym_map = {
            "旅行": ["旅游", "游览", "出行"],
            "酒店": ["宾馆", "住宿", "旅店"],
            "美食": ["小吃", "餐厅", "饭店"],
            "景点": ["名胜", "风景", "旅游点"],
            "便宜": ["经济", "实惠", "低价"],
            "好": ["不错", "优秀", "棒"],
        }
        
        expanded_query = query
        for word, synonyms in synonym_map.items():
            if word in query:
                for synonym in synonyms:
                    if synonym not in query:
                        expanded.append(query.replace(word, synonym))
        
        # 添加相关词扩展
        keywords = self.extract_keywords(query)
        if keywords:
            expanded.append(" ".join(keywords))
        
        return expanded[:5]  # 限制扩展查询数量


class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.corpus_ids = []
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.fitted = False
    
    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """训练BM25模型"""
        self.corpus = [doc["content"] for doc in documents]
        self.corpus_ids = [doc["id"] for doc in documents]
        
        # 分词
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        
        # 计算文档长度
        self.doc_len = [len(tokens) for tokens in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # 计算词频和逆文档频率
        df = defaultdict(int)
        for tokens in tokenized_corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        # 计算IDF
        N = len(self.corpus)
        for term, freq in df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5))
        
        self.fitted = True
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text))
        else:
            # 简单分词
            return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """BM25搜索"""
        if not self.fitted:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_tokens in enumerate([self._tokenize(doc) for doc in self.corpus]):
            score = 0.0
            doc_len = self.doc_len[i]
            
            for term in query_tokens:
                if term in self.idf:
                    # 计算词频
                    tf = doc_tokens.count(term)
                    
                    # BM25得分
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += self.idf[term] * numerator / denominator
            
            scores.append({
                "id": self.corpus_ids[i],
                "score": score,
                "content": self.corpus[i]
            })
        
        # 排序
        scores.sort(key=lambda x: x["score"], reverse=True)
        
        # 转换为结果格式
        results = []
        for rank, item in enumerate(scores[:top_k]):
            result = RetrievalResult(
                document_id=item["id"].split("_")[0] if "_" in item["id"] else item["id"],
                chunk_id=item["id"],
                content=item["content"],
                score=item["score"],
                rank=rank + 1,
                metadata={"method": "bm25"},
                retrieval_method="bm25"
            )
            results.append(result)
        
        return results


class GraphRetriever:
    """基于图的检索器"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_content = {}
        self.similarity_threshold = 0.5
    
    def build_graph(self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.5) -> None:
        """构建文档图"""
        self.similarity_threshold = similarity_threshold
        
        # 添加节点
        for doc in documents:
            doc_id = doc["id"]
            self.graph.add_node(doc_id)
            self.node_content[doc_id] = doc["content"]
        
        # 计算文档相似度并添加边
        vectorizer = TfidfVectorizer(max_features=1000)
        contents = [doc["content"] for doc in documents]
        tfidf_matrix = vectorizer.fit_transform(contents)
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents):
                if i < j and similarity_matrix[i][j] > self.similarity_threshold:
                    self.graph.add_edge(
                        doc1["id"], 
                        doc2["id"], 
                        weight=similarity_matrix[i][j]
                    )
    
    def search(self, query: str, start_nodes: List[str], top_k: int = 10) -> List[RetrievalResult]:
        """基于图的搜索"""
        if not self.graph.nodes:
            return []
        
        # 使用PageRank算法计算节点重要性
        personalization = {node: 1.0 if node in start_nodes else 0.0 
                          for node in self.graph.nodes}
        
        try:
            pagerank_scores = nx.pagerank(
                self.graph, 
                personalization=personalization,
                alpha=0.85,
                max_iter=100
            )
        except:
            # 如果PageRank失败，使用度中心性
            pagerank_scores = nx.degree_centrality(self.graph)
        
        # 排序和转换结果
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (node_id, score) in enumerate(sorted_nodes[:top_k]):
            result = RetrievalResult(
                document_id=node_id.split("_")[0] if "_" in node_id else node_id,
                chunk_id=node_id,
                content=self.node_content.get(node_id, ""),
                score=score,
                rank=rank + 1,
                metadata={"method": "graph", "centrality": score},
                retrieval_method="graph"
            )
            results.append(result)
        
        return results


class CrossEncoder:
    """交叉编码器重排序"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(self.model_name)
                logger.info(f"已加载交叉编码器: {self.model_name}")
            except Exception as e:
                logger.warning(f"加载交叉编码器失败: {e}")
                self.model = None
        else:
            logger.warning("sentence-transformers未安装，无法使用交叉编码器")
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """重排序结果"""
        if not self.model or not results:
            return results[:top_k]
        
        try:
            # 准备输入对
            pairs = [(query, result.content) for result in results]
            
            # 计算相关性分数
            scores = self.model.predict(pairs)
            
            # 更新结果分数并重排序
            for i, result in enumerate(results):
                result.score = float(scores[i])
            
            # 重排序
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            
            # 更新排名
            for rank, result in enumerate(reranked_results):
                result.rank = rank + 1
                result.metadata["reranked"] = True
                result.metadata["cross_encoder_score"] = result.score
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results[:top_k]


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, collection_name: str = "travel_knowledge"):
        self.collection_name = collection_name
        self.vector_db = get_vector_database()
        self.knowledge_builder = get_knowledge_builder(collection_name)
        self.query_processor = QueryProcessor()
        self.bm25_retriever = BM25Retriever()
        self.graph_retriever = GraphRetriever()
        self.cross_encoder = CrossEncoder()
        
        # 配置权重
        self.weights = {
            "vector": 0.5,
            "bm25": 0.3,
            "graph": 0.2
        }
        
        # 缓存
        self.document_cache = {}
        self.last_update = None
        
        # 初始化
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """初始化检索器"""
        try:
            # 构建文档缓存
            await self._build_document_cache()
            
            # 训练BM25
            if self.document_cache:
                documents = list(self.document_cache.values())
                self.bm25_retriever.fit(documents)
                
                # 构建图
                self.graph_retriever.build_graph(documents)
                
                logger.info("混合检索器初始化完成")
            
        except Exception as e:
            logger.error(f"混合检索器初始化失败: {e}")
    
    async def _build_document_cache(self):
        """构建文档缓存"""
        try:
            # 从知识库获取所有文档
            stats = self.knowledge_builder.get_statistics()
            
            # 这里简化处理，实际应该从向量数据库查询
            # 使用一个示例文档来测试
            if not self.document_cache:
                self.document_cache = {
                    "doc1": {
                        "id": "doc1_chunk_0",
                        "content": "北京是中国的首都，有很多著名景点如故宫、长城等。",
                        "metadata": {"category": "travel", "location": "北京"}
                    },
                    "doc2": {
                        "id": "doc2_chunk_0", 
                        "content": "上海是中国的经济中心，外滩和东方明珠是著名景点。",
                        "metadata": {"category": "travel", "location": "上海"}
                    }
                }
            
        except Exception as e:
            logger.error(f"构建文档缓存失败: {e}")
    
    async def search(self, query: str, top_k: int = 10, strategy: str = "auto") -> List[RetrievalResult]:
        """混合搜索"""
        try:
            # 查询分析
            analysis = self.query_processor.analyze_query(query)
            logger.info(f"查询分析: {analysis.intent}, 复杂度: {analysis.complexity_score}")
            
            # 获取自适应策略
            if strategy == "auto":
                strategy = self._get_adaptive_strategy(analysis)
            
            # 多路检索
            all_results = []
            
            # 向量检索
            if "vector" in strategy:
                vector_results = await self._vector_search(query, top_k * 2)
                all_results.extend(vector_results)
            
            # BM25检索  
            if "bm25" in strategy:
                bm25_results = self._bm25_search(query, top_k * 2)
                all_results.extend(bm25_results)
            
            # 图检索
            if "graph" in strategy:
                graph_results = self._graph_search(query, analysis, top_k * 2)
                all_results.extend(graph_results)
            
            # 结果融合
            fused_results = self._fuse_results(all_results, strategy)
            
            # 重排序
            if len(fused_results) > top_k:
                reranked_results = self.cross_encoder.rerank(query, fused_results, top_k)
            else:
                reranked_results = fused_results
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    async def _vector_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """向量搜索"""
        try:
            # 这里需要实际的向量搜索实现
            # 由于没有实际的向量数据，返回模拟结果
            results = []
            for i, (doc_id, doc_data) in enumerate(list(self.document_cache.items())[:top_k]):
                if query in doc_data["content"]:
                    result = RetrievalResult(
                        document_id=doc_id,
                        chunk_id=doc_data["id"],
                        content=doc_data["content"],
                        score=0.9 - i * 0.1,
                        rank=i + 1,
                        metadata=doc_data["metadata"],
                        retrieval_method="vector"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25搜索"""
        try:
            return self.bm25_retriever.search(query, top_k)
        except Exception as e:
            logger.error(f"BM25搜索失败: {e}")
            return []
    
    def _graph_search(self, query: str, analysis: QueryAnalysis, top_k: int) -> List[RetrievalResult]:
        """图搜索"""
        try:
            # 根据查询分析确定起始节点
            start_nodes = []
            for entity in analysis.entities:
                if "地点" in entity:
                    location = entity.split(":")[1]
                    # 找到包含该地点的文档
                    for doc_id, doc_data in self.document_cache.items():
                        if location in doc_data["content"]:
                            start_nodes.append(doc_data["id"])
            
            if not start_nodes:
                start_nodes = list(self.document_cache.keys())[:1]
            
            return self.graph_retriever.search(query, start_nodes, top_k)
            
        except Exception as e:
            logger.error(f"图搜索失败: {e}")
            return []
    
    def _fuse_results(self, results: List[RetrievalResult], strategy: str) -> List[RetrievalResult]:
        """融合多路检索结果"""
        # 按chunk_id分组
        result_groups = defaultdict(list)
        for result in results:
            result_groups[result.chunk_id].append(result)
        
        # 融合分数
        fused_results = []
        for chunk_id, group_results in result_groups.items():
            # 计算融合分数
            vector_score = next((r.score for r in group_results if r.retrieval_method == "vector"), 0.0)
            bm25_score = next((r.score for r in group_results if r.retrieval_method == "bm25"), 0.0)
            graph_score = next((r.score for r in group_results if r.retrieval_method == "graph"), 0.0)
            
            # 归一化分数（简化处理）
            max_vector = max([r.score for r in results if r.retrieval_method == "vector"], default=1.0)
            max_bm25 = max([r.score for r in results if r.retrieval_method == "bm25"], default=1.0)
            max_graph = max([r.score for r in results if r.retrieval_method == "graph"], default=1.0)
            
            normalized_vector = vector_score / max_vector if max_vector > 0 else 0
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            normalized_graph = graph_score / max_graph if max_graph > 0 else 0
            
            # 加权融合
            fused_score = (
                self.weights["vector"] * normalized_vector +
                self.weights["bm25"] * normalized_bm25 + 
                self.weights["graph"] * normalized_graph
            )
            
            # 创建融合结果
            base_result = group_results[0]
            fused_result = RetrievalResult(
                document_id=base_result.document_id,
                chunk_id=base_result.chunk_id,
                content=base_result.content,
                score=fused_score,
                rank=0,  # 稍后重新排序
                metadata={
                    **base_result.metadata,
                    "fusion_method": strategy,
                    "original_scores": {
                        "vector": vector_score,
                        "bm25": bm25_score,
                        "graph": graph_score
                    }
                },
                retrieval_method="hybrid"
            )
            fused_results.append(fused_result)
        
        # 按融合分数排序
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # 更新排名
        for rank, result in enumerate(fused_results):
            result.rank = rank + 1
        
        return fused_results
    
    def _get_adaptive_strategy(self, analysis: QueryAnalysis) -> str:
        """获取自适应策略"""
        strategy_parts = []
        
        # 根据查询复杂度
        if analysis.complexity_score > 5.0:
            strategy_parts.extend(["vector", "bm25", "graph"])
        elif analysis.complexity_score > 2.0:
            strategy_parts.extend(["vector", "bm25"])
        else:
            strategy_parts.append("vector")
        
        # 根据查询类型
        if analysis.query_type in ["travel", "accommodation"]:
            if "graph" not in strategy_parts:
                strategy_parts.append("graph")
        
        # 根据查询意图
        if analysis.intent in ["比较", "规划"]:
            if "bm25" not in strategy_parts:
                strategy_parts.append("bm25")
        
        return "_".join(strategy_parts)


# 全局混合检索器实例
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever(collection_name: str = "travel_knowledge") -> HybridRetriever:
    """获取混合检索器实例"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(collection_name)
    return _hybrid_retriever 