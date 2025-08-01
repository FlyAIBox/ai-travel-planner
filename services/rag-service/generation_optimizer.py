"""
RAG生成优化系统
实现检索结果与生成内容的智能融合、上下文压缩和信息提取算法、多源信息一致性检查机制、RAG结果质量评估和自动优化
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import math
import statistics

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import spacy
from sentence_transformers import SentenceTransformer
import jieba
from rouge_score import rouge_scorer
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    logger.warning("NLTK not available, BLEU score will not be computed")
    sentence_bleu = None

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .advanced_retrieval import RetrievalResult, QueryAnalysis

logger = get_logger(__name__)
settings = get_settings()


class FusionStrategy(Enum):
    """融合策略"""
    CONCATENATION = "concatenation"        # 简单拼接
    WEIGHTED_FUSION = "weighted_fusion"    # 加权融合
    HIERARCHICAL = "hierarchical"          # 层次化融合
    ATTENTION_BASED = "attention_based"    # 注意力机制
    SEMANTIC_MERGE = "semantic_merge"      # 语义合并


class CompressionMethod(Enum):
    """压缩方法"""
    EXTRACTIVE = "extractive"              # 抽取式
    ABSTRACTIVE = "abstractive"            # 生成式
    HYBRID = "hybrid"                      # 混合式
    KEYWORD_BASED = "keyword_based"        # 关键词式
    SEMANTIC_CLUSTERING = "semantic_clustering"  # 语义聚类


class QualityMetric(Enum):
    """质量指标"""
    RELEVANCE = "relevance"                # 相关性
    COHERENCE = "coherence"                # 连贯性
    CONSISTENCY = "consistency"            # 一致性
    COMPLETENESS = "completeness"          # 完整性
    FACTUALITY = "factuality"              # 事实性
    READABILITY = "readability"            # 可读性


@dataclass
class FusedContext:
    """融合后的上下文"""
    content: str
    sources: List[str]
    confidence: float
    compression_ratio: float
    fusion_method: str
    metadata: Dict[str, Any]


@dataclass
class QualityAssessment:
    """质量评估结果"""
    overall_score: float
    metrics: Dict[str, float]
    issues: List[str]
    suggestions: List[str]
    confidence: float


@dataclass
class GenerationResult:
    """生成结果"""
    content: str
    fused_context: FusedContext
    quality_assessment: QualityAssessment
    generation_time: float
    metadata: Dict[str, Any]


class ContextFusion:
    """上下文融合器"""
    
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
        
        # 向量化工具
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # 注意力权重
        self.attention_weights = {}
    
    async def fuse_contexts(self, 
                          retrieval_results: List[RetrievalResult],
                          query_analysis: QueryAnalysis,
                          strategy: FusionStrategy = FusionStrategy.SEMANTIC_MERGE,
                          max_length: int = 2000) -> FusedContext:
        """融合多个检索结果"""
        if not retrieval_results:
            return FusedContext(
                content="",
                sources=[],
                confidence=0.0,
                compression_ratio=0.0,
                fusion_method=strategy.value,
                metadata={}
            )
        
        if strategy == FusionStrategy.CONCATENATION:
            return await self._concatenation_fusion(retrieval_results, max_length)
        elif strategy == FusionStrategy.WEIGHTED_FUSION:
            return await self._weighted_fusion(retrieval_results, query_analysis, max_length)
        elif strategy == FusionStrategy.HIERARCHICAL:
            return await self._hierarchical_fusion(retrieval_results, query_analysis, max_length)
        elif strategy == FusionStrategy.ATTENTION_BASED:
            return await self._attention_fusion(retrieval_results, query_analysis, max_length)
        elif strategy == FusionStrategy.SEMANTIC_MERGE:
            return await self._semantic_fusion(retrieval_results, query_analysis, max_length)
        else:
            return await self._semantic_fusion(retrieval_results, query_analysis, max_length)
    
    async def _concatenation_fusion(self, 
                                   results: List[RetrievalResult], 
                                   max_length: int) -> FusedContext:
        """简单拼接融合"""
        contents = []
        sources = []
        total_score = 0.0
        
        current_length = 0
        for result in results:
            if current_length + len(result.content) > max_length:
                # 截断内容
                remaining_length = max_length - current_length
                if remaining_length > 100:  # 至少保留100字符
                    contents.append(result.content[:remaining_length] + "...")
                    sources.append(result.source)
                break
            
            contents.append(result.content)
            sources.append(result.source)
            total_score += result.score
            current_length += len(result.content)
        
        fused_content = "\n\n".join(contents)
        avg_confidence = total_score / len(results) if results else 0.0
        compression_ratio = len(fused_content) / sum(len(r.content) for r in results) if results else 0.0
        
        return FusedContext(
            content=fused_content,
            sources=sources,
            confidence=avg_confidence,
            compression_ratio=compression_ratio,
            fusion_method="concatenation",
            metadata={"original_count": len(results)}
        )
    
    async def _weighted_fusion(self, 
                              results: List[RetrievalResult],
                              query_analysis: QueryAnalysis, 
                              max_length: int) -> FusedContext:
        """加权融合"""
        if not results:
            return FusedContext("", [], 0.0, 0.0, "weighted", {})
        
        # 计算权重
        weights = []
        for result in results:
            weight = self._calculate_relevance_weight(result, query_analysis)
            weights.append(weight)
        
        # 标准化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(results)] * len(results)
        
        # 按权重排序并选择内容
        weighted_results = list(zip(results, weights))
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        contents = []
        sources = []
        total_confidence = 0.0
        current_length = 0
        
        for result, weight in weighted_results:
            if current_length >= max_length:
                break
            
            # 根据权重决定内容长度
            content_length = min(len(result.content), max_length - current_length)
            if weight < 0.1:  # 低权重内容截短
                content_length = min(content_length, 200)
            
            content = result.content[:content_length]
            if content_length < len(result.content):
                content += "..."
            
            contents.append(f"[权重:{weight:.2f}] {content}")
            sources.append(result.source)
            total_confidence += result.score * weight
            current_length += len(content)
        
        fused_content = "\n\n".join(contents)
        compression_ratio = len(fused_content) / sum(len(r.content) for r in results)
        
        return FusedContext(
            content=fused_content,
            sources=sources,
            confidence=total_confidence,
            compression_ratio=compression_ratio,
            fusion_method="weighted",
            metadata={"weights": weights[:len(contents)]}
        )
    
    async def _hierarchical_fusion(self, 
                                  results: List[RetrievalResult],
                                  query_analysis: QueryAnalysis, 
                                  max_length: int) -> FusedContext:
        """层次化融合"""
        if not results:
            return FusedContext("", [], 0.0, 0.0, "hierarchical", {})
        
        # 按来源分组
        source_groups = {}
        for result in results:
            source = result.source
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)
        
        # 为每个来源组生成摘要
        group_summaries = []
        all_sources = []
        
        for source, group_results in source_groups.items():
            # 合并同源内容
            group_content = "\n".join([r.content for r in group_results])
            
            # 生成摘要（简化版，实际可以使用更复杂的摘要算法）
            summary = await self._extract_key_sentences(group_content, max_sentences=3)
            
            avg_score = np.mean([r.score for r in group_results])
            group_summaries.append({
                "source": source,
                "summary": summary,
                "score": avg_score,
                "count": len(group_results)
            })
            all_sources.extend([source] * len(group_results))
        
        # 按分数排序组
        group_summaries.sort(key=lambda x: x["score"], reverse=True)
        
        # 构建层次化内容
        contents = []
        current_length = 0
        
        for group in group_summaries:
            if current_length >= max_length:
                break
            
            header = f"## {group['source']} (相关度: {group['score']:.2f})"
            content = f"{header}\n{group['summary']}\n"
            
            if current_length + len(content) <= max_length:
                contents.append(content)
                current_length += len(content)
            else:
                # 截断最后一个组的内容
                remaining = max_length - current_length - len(header) - 2
                if remaining > 50:
                    truncated_summary = group['summary'][:remaining] + "..."
                    contents.append(f"{header}\n{truncated_summary}\n")
                break
        
        fused_content = "\n".join(contents)
        avg_confidence = np.mean([g["score"] for g in group_summaries]) if group_summaries else 0.0
        compression_ratio = len(fused_content) / sum(len(r.content) for r in results)
        
        return FusedContext(
            content=fused_content,
            sources=list(set(all_sources)),
            confidence=avg_confidence,
            compression_ratio=compression_ratio,
            fusion_method="hierarchical",
            metadata={"groups": len(source_groups)}
        )
    
    async def _attention_fusion(self, 
                               results: List[RetrievalResult],
                               query_analysis: QueryAnalysis, 
                               max_length: int) -> FusedContext:
        """基于注意力机制的融合"""
        if not results:
            return FusedContext("", [], 0.0, 0.0, "attention", {})
        
        # 计算查询与每个结果的注意力权重
        query_keywords = set(query_analysis.keywords)
        attention_weights = []
        
        for result in results:
            # 基于关键词重叠计算注意力
            result_words = set(self._tokenize_text(result.content, query_analysis.language))
            keyword_overlap = len(query_keywords.intersection(result_words))
            
            # 基于相似度计算注意力
            similarity_score = result.score
            
            # 综合注意力权重
            attention = (keyword_overlap * 0.3 + similarity_score * 0.7)
            attention_weights.append(attention)
        
        # 标准化注意力权重
        if sum(attention_weights) > 0:
            attention_weights = [w / sum(attention_weights) for w in attention_weights]
        else:
            attention_weights = [1.0 / len(results)] * len(results)
        
        # 基于注意力权重重新排序和选择内容
        attended_results = list(zip(results, attention_weights))
        attended_results.sort(key=lambda x: x[1], reverse=True)
        
        contents = []
        sources = []
        current_length = 0
        total_confidence = 0.0
        
        for result, attention in attended_results:
            if current_length >= max_length:
                break
            
            # 根据注意力权重调整内容长度
            max_content_length = int(max_length * attention * 2)  # 注意力高的内容可以更长
            max_content_length = min(max_content_length, max_length - current_length)
            
            if max_content_length < 50:  # 最小内容长度
                break
            
            content = result.content[:max_content_length]
            if len(content) < len(result.content):
                content += "..."
            
            # 添加注意力权重标识
            contents.append(f"[注意力:{attention:.3f}] {content}")
            sources.append(result.source)
            total_confidence += result.score * attention
            current_length += len(content)
        
        fused_content = "\n\n".join(contents)
        compression_ratio = len(fused_content) / sum(len(r.content) for r in results)
        
        return FusedContext(
            content=fused_content,
            sources=sources,
            confidence=total_confidence,
            compression_ratio=compression_ratio,
            fusion_method="attention",
            metadata={"attention_weights": attention_weights[:len(contents)]}
        )
    
    async def _semantic_fusion(self, 
                              results: List[RetrievalResult],
                              query_analysis: QueryAnalysis, 
                              max_length: int) -> FusedContext:
        """语义融合"""
        if not results:
            return FusedContext("", [], 0.0, 0.0, "semantic", {})
        
        # 1. 语义去重
        unique_results = await self._semantic_deduplication(results)
        
        # 2. 语义聚类
        clusters = await self._semantic_clustering(unique_results)
        
        # 3. 每个聚类生成代表性内容
        cluster_contents = []
        all_sources = []
        total_confidence = 0.0
        
        for cluster in clusters:
            cluster_summary = await self._generate_cluster_summary(cluster, query_analysis)
            cluster_contents.append(cluster_summary["content"])
            all_sources.extend([r.source for r in cluster])
            total_confidence += cluster_summary["confidence"]
        
        # 4. 按重要性排序和长度控制
        fused_content = await self._merge_cluster_contents(cluster_contents, max_length)
        
        avg_confidence = total_confidence / len(clusters) if clusters else 0.0
        compression_ratio = len(fused_content) / sum(len(r.content) for r in results)
        
        return FusedContext(
            content=fused_content,
            sources=list(set(all_sources)),
            confidence=avg_confidence,
            compression_ratio=compression_ratio,
            fusion_method="semantic",
            metadata={"clusters": len(clusters), "original_results": len(results)}
        )
    
    async def _semantic_deduplication(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """语义去重"""
        if len(results) <= 1:
            return results
        
        # 简单的相似度去重
        unique_results = []
        
        for result in results:
            is_duplicate = False
            
            for existing in unique_results:
                # 计算内容相似度
                similarity = self._calculate_text_similarity(result.content, existing.content)
                if similarity > 0.8:  # 高相似度阈值
                    is_duplicate = True
                    # 保留分数更高的结果
                    if result.score > existing.score:
                        unique_results.remove(existing)
                        unique_results.append(result)
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    async def _semantic_clustering(self, results: List[RetrievalResult]) -> List[List[RetrievalResult]]:
        """语义聚类"""
        if len(results) <= 2:
            return [results]
        
        # 简化的聚类算法
        clusters = []
        remaining_results = results.copy()
        
        while remaining_results:
            # 选择第一个结果作为聚类中心
            center = remaining_results.pop(0)
            cluster = [center]
            
            # 找到与中心相似的结果
            to_remove = []
            for i, result in enumerate(remaining_results):
                similarity = self._calculate_text_similarity(center.content, result.content)
                if similarity > 0.5:  # 聚类阈值
                    cluster.append(result)
                    to_remove.append(i)
            
            # 移除已聚类的结果
            for i in reversed(to_remove):
                remaining_results.pop(i)
            
            clusters.append(cluster)
        
        return clusters
    
    async def _generate_cluster_summary(self, 
                                       cluster: List[RetrievalResult],
                                       query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """生成聚类摘要"""
        if len(cluster) == 1:
            return {
                "content": cluster[0].content,
                "confidence": cluster[0].score
            }
        
        # 合并聚类内容
        combined_content = "\n".join([r.content for r in cluster])
        
        # 提取关键句子
        key_sentences = await self._extract_key_sentences(combined_content, max_sentences=5)
        
        # 计算平均置信度
        avg_confidence = np.mean([r.score for r in cluster])
        
        return {
            "content": key_sentences,
            "confidence": avg_confidence
        }
    
    async def _merge_cluster_contents(self, contents: List[str], max_length: int) -> str:
        """合并聚类内容"""
        if not contents:
            return ""
        
        # 按重要性排序（这里简化为按长度）
        contents.sort(key=len, reverse=True)
        
        merged = []
        current_length = 0
        
        for content in contents:
            if current_length + len(content) <= max_length:
                merged.append(content)
                current_length += len(content)
            else:
                # 截断最后一个内容
                remaining = max_length - current_length
                if remaining > 100:
                    merged.append(content[:remaining] + "...")
                break
        
        return "\n\n".join(merged)
    
    async def _extract_key_sentences(self, text: str, max_sentences: int = 3) -> str:
        """提取关键句子"""
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return ". ".join(sentences)
        
        # 简单的关键句提取：选择最长的句子
        sentences.sort(key=len, reverse=True)
        key_sentences = sentences[:max_sentences]
        
        return ". ".join(key_sentences)
    
    def _calculate_relevance_weight(self, result: RetrievalResult, query_analysis: QueryAnalysis) -> float:
        """计算相关性权重"""
        # 基础权重来自检索分数
        base_weight = result.score
        
        # 关键词匹配加权
        result_words = set(self._tokenize_text(result.content, query_analysis.language))
        query_keywords = set(query_analysis.keywords)
        keyword_overlap = len(query_keywords.intersection(result_words))
        keyword_weight = keyword_overlap / max(len(query_keywords), 1) * 0.3
        
        # 实体匹配加权
        entity_weight = 0.0
        for entity in query_analysis.entities:
            if entity["text"].lower() in result.content.lower():
                entity_weight += 0.1
        
        return base_weight + keyword_weight + min(entity_weight, 0.3)
    
    def _tokenize_text(self, text: str, language: str) -> List[str]:
        """文本分词"""
        if language == "zh":
            return list(jieba.cut(text))
        else:
            return re.findall(r'\b\w+\b', text.lower())
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            # 使用TF-IDF向量计算相似度
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # 回退到简单的词重叠
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self):
        self.compression_stats = {
            "total_compressions": 0,
            "average_ratio": 0.0,
            "method_usage": {}
        }
    
    async def compress_context(self, 
                              context: str,
                              target_length: int,
                              method: CompressionMethod = CompressionMethod.HYBRID,
                              preserve_keywords: List[str] = None) -> Tuple[str, float]:
        """压缩上下文"""
        if len(context) <= target_length:
            return context, 1.0
        
        preserve_keywords = preserve_keywords or []
        
        if method == CompressionMethod.EXTRACTIVE:
            compressed, ratio = await self._extractive_compression(context, target_length, preserve_keywords)
        elif method == CompressionMethod.ABSTRACTIVE:
            compressed, ratio = await self._abstractive_compression(context, target_length)
        elif method == CompressionMethod.KEYWORD_BASED:
            compressed, ratio = await self._keyword_compression(context, target_length, preserve_keywords)
        elif method == CompressionMethod.SEMANTIC_CLUSTERING:
            compressed, ratio = await self._semantic_compression(context, target_length)
        else:  # HYBRID
            compressed, ratio = await self._hybrid_compression(context, target_length, preserve_keywords)
        
        # 更新统计信息
        self._update_compression_stats(method, ratio)
        
        return compressed, ratio
    
    async def _extractive_compression(self, 
                                     context: str, 
                                     target_length: int,
                                     preserve_keywords: List[str]) -> Tuple[str, float]:
        """抽取式压缩"""
        sentences = re.split(r'[.!?。！？]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return context, 1.0
        
        # 计算句子重要性分数
        sentence_scores = []
        for sentence in sentences:
            score = 0.0
            
            # 长度因子
            score += len(sentence) / 1000
            
            # 关键词因子
            for keyword in preserve_keywords:
                if keyword.lower() in sentence.lower():
                    score += 1.0
            
            # 位置因子（开头和结尾句子更重要）
            position_factor = 1.0
            if sentences.index(sentence) < len(sentences) * 0.2:
                position_factor = 1.2
            elif sentences.index(sentence) > len(sentences) * 0.8:
                position_factor = 1.1
            
            score *= position_factor
            sentence_scores.append((sentence, score))
        
        # 按分数排序
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择句子直到达到目标长度
        selected_sentences = []
        current_length = 0
        
        for sentence, score in sentence_scores:
            if current_length + len(sentence) <= target_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            elif current_length < target_length * 0.8:  # 至少达到目标长度的80%
                # 截断句子
                remaining = target_length - current_length
                if remaining > 50:
                    selected_sentences.append(sentence[:remaining] + "...")
                break
        
        compressed = ". ".join(selected_sentences)
        ratio = len(compressed) / len(context)
        
        return compressed, ratio
    
    async def _abstractive_compression(self, context: str, target_length: int) -> Tuple[str, float]:
        """生成式压缩（简化版）"""
        # 这里应该使用生成式模型进行摘要
        # 作为简化实现，我们使用关键句提取
        
        # 分段处理长文本
        paragraphs = context.split('\n\n')
        compressed_paragraphs = []
        
        for paragraph in paragraphs:
            if len(paragraph) > 200:
                # 提取段落关键句
                sentences = re.split(r'[.!?。！？]', paragraph)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                
                if len(sentences) > 1:
                    # 选择最重要的1-2句
                    key_sentences = sentences[:2] if len(sentences) > 3 else sentences[:1]
                    compressed_paragraphs.append(". ".join(key_sentences))
                else:
                    compressed_paragraphs.append(paragraph)
            else:
                compressed_paragraphs.append(paragraph)
        
        compressed = "\n\n".join(compressed_paragraphs)
        
        # 如果仍然太长，进一步压缩
        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."
        
        ratio = len(compressed) / len(context)
        return compressed, ratio
    
    async def _keyword_compression(self, 
                                  context: str, 
                                  target_length: int,
                                  preserve_keywords: List[str]) -> Tuple[str, float]:
        """基于关键词的压缩"""
        sentences = re.split(r'[.!?。！？]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # 优先保留包含关键词的句子
        keyword_sentences = []
        other_sentences = []
        
        for sentence in sentences:
            has_keyword = any(keyword.lower() in sentence.lower() for keyword in preserve_keywords)
            if has_keyword:
                keyword_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # 首先添加关键词句子
        selected = []
        current_length = 0
        
        for sentence in keyword_sentences:
            if current_length + len(sentence) <= target_length:
                selected.append(sentence)
                current_length += len(sentence)
        
        # 如果还有空间，添加其他句子
        for sentence in other_sentences:
            if current_length + len(sentence) <= target_length:
                selected.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        compressed = ". ".join(selected)
        ratio = len(compressed) / len(context)
        
        return compressed, ratio
    
    async def _semantic_compression(self, context: str, target_length: int) -> Tuple[str, float]:
        """基于语义聚类的压缩"""
        sentences = re.split(r'[.!?。！？]', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 3:
            return context, 1.0
        
        # 简单的语义聚类（基于词汇重叠）
        clusters = []
        remaining = sentences.copy()
        
        while remaining:
            center = remaining.pop(0)
            cluster = [center]
            
            to_remove = []
            for i, sentence in enumerate(remaining):
                similarity = self._calculate_sentence_similarity(center, sentence)
                if similarity > 0.3:
                    cluster.append(sentence)
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                remaining.pop(i)
            
            clusters.append(cluster)
        
        # 从每个聚类中选择代表句子
        representative_sentences = []
        for cluster in clusters:
            # 选择最长的句子作为代表
            representative = max(cluster, key=len)
            representative_sentences.append(representative)
        
        # 根据目标长度选择句子
        selected = []
        current_length = 0
        
        for sentence in representative_sentences:
            if current_length + len(sentence) <= target_length:
                selected.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        compressed = ". ".join(selected)
        ratio = len(compressed) / len(context)
        
        return compressed, ratio
    
    async def _hybrid_compression(self, 
                                 context: str, 
                                 target_length: int,
                                 preserve_keywords: List[str]) -> Tuple[str, float]:
        """混合压缩方法"""
        # 1. 首先使用关键词压缩
        keyword_compressed, _ = await self._keyword_compression(context, target_length * 1.2, preserve_keywords)
        
        # 2. 如果仍然太长，使用抽取式压缩
        if len(keyword_compressed) > target_length:
            final_compressed, ratio = await self._extractive_compression(keyword_compressed, target_length, preserve_keywords)
        else:
            final_compressed = keyword_compressed
            ratio = len(final_compressed) / len(context)
        
        return final_compressed, ratio
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """计算句子相似度"""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_compression_stats(self, method: CompressionMethod, ratio: float):
        """更新压缩统计"""
        self.compression_stats["total_compressions"] += 1
        
        # 更新平均压缩比
        total = self.compression_stats["total_compressions"]
        current_avg = self.compression_stats["average_ratio"]
        self.compression_stats["average_ratio"] = (current_avg * (total - 1) + ratio) / total
        
        # 更新方法使用统计
        method_name = method.value
        if method_name not in self.compression_stats["method_usage"]:
            self.compression_stats["method_usage"][method_name] = 0
        self.compression_stats["method_usage"][method_name] += 1
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return self.compression_stats.copy()


class ConsistencyChecker:
    """一致性检查器"""
    
    def __init__(self):
        self.inconsistency_patterns = [
            # 数字不一致
            r'(\d+)\s*[年岁]\s*.*?(\d+)\s*[年岁]',
            # 日期不一致
            r'(\d{4}年\d{1,2}月\d{1,2}日).*?(\d{4}年\d{1,2}月\d{1,2}日)',
            # 价格不一致
            r'(\d+)\s*[元].*?(\d+)\s*[元]',
        ]
    
    async def check_consistency(self, 
                               content: str,
                               sources: List[str] = None) -> Dict[str, Any]:
        """检查内容一致性"""
        issues = []
        confidence = 1.0
        
        # 1. 检查内部一致性
        internal_issues = await self._check_internal_consistency(content)
        issues.extend(internal_issues)
        
        # 2. 检查事实一致性
        factual_issues = await self._check_factual_consistency(content)
        issues.extend(factual_issues)
        
        # 3. 检查逻辑一致性
        logical_issues = await self._check_logical_consistency(content)
        issues.extend(logical_issues)
        
        # 计算一致性置信度
        if issues:
            confidence = max(0.0, 1.0 - len(issues) * 0.1)
        
        return {
            "consistent": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "issue_count": len(issues),
            "issue_types": list(set([issue["type"] for issue in issues]))
        }
    
    async def _check_internal_consistency(self, content: str) -> List[Dict[str, str]]:
        """检查内部一致性"""
        issues = []
        
        # 检查数字一致性
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        if len(numbers) > 1:
            # 简单检查：如果有相同的数字在不同上下文中，可能不一致
            from collections import Counter
            number_counts = Counter(numbers)
            frequent_numbers = [num for num, count in number_counts.items() if count > 2]
            
            for num in frequent_numbers:
                # 这里可以添加更复杂的上下文分析
                if len(frequent_numbers) > 3:  # 简化的不一致检测
                    issues.append({
                        "type": "numerical_inconsistency",
                        "description": f"数字 {num} 多次出现，可能存在不一致",
                        "severity": "low"
                    })
        
        # 检查日期一致性
        dates = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日', content)
        if len(dates) > 1:
            # 检查日期范围是否合理
            try:
                from datetime import datetime
                parsed_dates = []
                for date_str in dates:
                    # 简单的日期解析
                    year_match = re.search(r'(\d{4})年', date_str)
                    if year_match:
                        year = int(year_match.group(1))
                        if year < 1900 or year > 2030:
                            issues.append({
                                "type": "date_inconsistency", 
                                "description": f"日期 {date_str} 可能不合理",
                                "severity": "medium"
                            })
            except:
                pass
        
        return issues
    
    async def _check_factual_consistency(self, content: str) -> List[Dict[str, str]]:
        """检查事实一致性"""
        issues = []
        
        # 检查常见的事实性错误模式
        fact_patterns = [
            (r'价格.*?免费', "价格与免费矛盾"),
            (r'必须.*?可选', "必须与可选矛盾"),
            (r'24小时.*?营业时间', "24小时营业与特定营业时间矛盾"),
        ]
        
        for pattern, description in fact_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    "type": "factual_inconsistency",
                    "description": description,
                    "severity": "high"
                })
        
        return issues
    
    async def _check_logical_consistency(self, content: str) -> List[Dict[str, str]]:
        """检查逻辑一致性"""
        issues = []
        
        # 检查逻辑矛盾
        contradiction_patterns = [
            (r'不需要.*?必须', "不需要与必须的逻辑矛盾"),
            (r'免费.*?收费', "免费与收费的逻辑矛盾"),
            (r'简单.*?复杂', "简单与复杂的评价矛盾"),
        ]
        
        for pattern, description in contradiction_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.append({
                    "type": "logical_inconsistency",
                    "description": description,
                    "severity": "medium"
                })
        
        return issues


class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self):
        # 初始化ROUGE评分器
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except:
            logger.warning("ROUGE评分器初始化失败")
            self.rouge_scorer = None
        
        # 质量评估统计
        self.evaluation_stats = {
            "total_evaluations": 0,
            "average_scores": {},
            "score_distribution": {}
        }
    
    async def evaluate_quality(self, 
                              content: str,
                              reference: str = None,
                              query_analysis: QueryAnalysis = None,
                              retrieval_results: List[RetrievalResult] = None) -> QualityAssessment:
        """评估生成质量"""
        metrics = {}
        issues = []
        suggestions = []
        
        # 1. 相关性评估
        if query_analysis:
            relevance_score = await self._evaluate_relevance(content, query_analysis)
            metrics[QualityMetric.RELEVANCE.value] = relevance_score
            
            if relevance_score < 0.6:
                issues.append("内容与查询相关性较低")
                suggestions.append("增加更多与查询相关的信息")
        
        # 2. 连贯性评估
        coherence_score = await self._evaluate_coherence(content)
        metrics[QualityMetric.COHERENCE.value] = coherence_score
        
        if coherence_score < 0.7:
            issues.append("内容连贯性需要改善")
            suggestions.append("重新组织段落结构，添加过渡句")
        
        # 3. 一致性评估
        consistency_checker = ConsistencyChecker()
        consistency_result = await consistency_checker.check_consistency(content)
        metrics[QualityMetric.CONSISTENCY.value] = consistency_result["confidence"]
        
        if not consistency_result["consistent"]:
            issues.extend([issue["description"] for issue in consistency_result["issues"]])
            suggestions.append("检查并修正内容中的不一致之处")
        
        # 4. 完整性评估
        if query_analysis:
            completeness_score = await self._evaluate_completeness(content, query_analysis)
            metrics[QualityMetric.COMPLETENESS.value] = completeness_score
            
            if completeness_score < 0.8:
                issues.append("内容可能不够完整")
                suggestions.append("补充更多相关信息")
        
        # 5. 可读性评估
        readability_score = await self._evaluate_readability(content)
        metrics[QualityMetric.READABILITY.value] = readability_score
        
        if readability_score < 0.7:
            issues.append("内容可读性需要提升")
            suggestions.append("简化句子结构，使用更清晰的表达")
        
        # 6. 事实性评估
        factuality_score = await self._evaluate_factuality(content, retrieval_results)
        metrics[QualityMetric.FACTUALITY.value] = factuality_score
        
        if factuality_score < 0.8:
            issues.append("内容事实性有待验证")
            suggestions.append("验证关键事实信息的准确性")
        
        # 计算综合分数
        overall_score = np.mean(list(metrics.values()))
        confidence = min(overall_score, 1.0 - len(issues) * 0.05)
        
        # 更新统计信息
        self._update_evaluation_stats(metrics, overall_score)
        
        return QualityAssessment(
            overall_score=overall_score,
            metrics=metrics,
            issues=issues,
            suggestions=suggestions,
            confidence=confidence
        )
    
    async def _evaluate_relevance(self, content: str, query_analysis: QueryAnalysis) -> float:
        """评估相关性"""
        content_lower = content.lower()
        
        # 关键词匹配
        keyword_matches = sum(1 for kw in query_analysis.keywords if kw.lower() in content_lower)
        keyword_score = keyword_matches / max(len(query_analysis.keywords), 1)
        
        # 实体匹配
        entity_matches = sum(1 for entity in query_analysis.entities 
                           if entity["text"].lower() in content_lower)
        entity_score = entity_matches / max(len(query_analysis.entities), 1) if query_analysis.entities else 0
        
        # 意图匹配（简化）
        intent_keywords = {
            "planning": ["计划", "安排", "规划"],
            "booking": ["预订", "预定", "订票"],
            "information": ["信息", "介绍", "了解"],
            "recommendation": ["推荐", "建议"],
            "comparison": ["比较", "对比"],
            "navigation": ["路线", "交通", "怎么去"]
        }
        
        intent_score = 0.0
        if query_analysis.intent in intent_keywords:
            intent_words = intent_keywords[query_analysis.intent]
            intent_matches = sum(1 for word in intent_words if word in content_lower)
            intent_score = min(intent_matches / len(intent_words), 1.0)
        
        # 综合相关性分数
        relevance = (keyword_score * 0.4 + entity_score * 0.3 + intent_score * 0.3)
        return min(relevance, 1.0)
    
    async def _evaluate_coherence(self, content: str) -> float:
        """评估连贯性"""
        sentences = re.split(r'[.!?。！？]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return 1.0
        
        # 检查连接词的使用
        connectors = ['因为', '所以', '但是', '然而', '此外', '另外', '因此', '首先', '其次', '最后',
                     'because', 'therefore', 'however', 'moreover', 'furthermore', 'first', 'second', 'finally']
        
        connector_count = sum(1 for sentence in sentences 
                            for connector in connectors 
                            if connector in sentence.lower())
        connector_score = min(connector_count / len(sentences), 0.5)
        
        # 检查句子长度分布
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths) if sentence_lengths else 0
        length_score = max(0, 1.0 - length_variance / 100)  # 长度变化不宜过大
        
        # 检查重复词汇
        all_words = []
        for sentence in sentences:
            all_words.extend(sentence.lower().split())
        
        if all_words:
            unique_ratio = len(set(all_words)) / len(all_words)
            diversity_score = min(unique_ratio * 2, 1.0)  # 词汇多样性
        else:
            diversity_score = 0.0
        
        coherence = (connector_score * 0.3 + length_score * 0.3 + diversity_score * 0.4)
        return coherence
    
    async def _evaluate_completeness(self, content: str, query_analysis: QueryAnalysis) -> float:
        """评估完整性"""
        # 基于查询复杂度评估完整性
        expected_length = query_analysis.complexity_score * 1000  # 复杂查询期望更长的回答
        actual_length = len(content)
        
        length_score = min(actual_length / expected_length, 1.0) if expected_length > 0 else 1.0
        
        # 检查是否回答了查询的主要方面
        aspect_coverage = 0.0
        if query_analysis.query_type.value == "factual":
            # 事实性查询应该包含定义、特征等
            aspect_keywords = ["是", "定义", "特点", "特征"]
            aspect_coverage = sum(1 for kw in aspect_keywords if kw in content) / len(aspect_keywords)
        elif query_analysis.query_type.value == "procedural":
            # 程序性查询应该包含步骤、方法等
            aspect_keywords = ["步骤", "方法", "首先", "然后", "最后"]
            aspect_coverage = sum(1 for kw in aspect_keywords if kw in content) / len(aspect_keywords)
        elif query_analysis.query_type.value == "comparison":
            # 比较性查询应该包含对比信息
            aspect_keywords = ["相比", "区别", "优势", "缺点", "不同"]
            aspect_coverage = sum(1 for kw in aspect_keywords if kw in content) / len(aspect_keywords)
        
        completeness = (length_score * 0.6 + aspect_coverage * 0.4)
        return min(completeness, 1.0)
    
    async def _evaluate_readability(self, content: str) -> float:
        """评估可读性"""
        sentences = re.split(r'[.!?。！？]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return 0.0
        
        # 平均句子长度
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        length_score = 1.0 if 10 <= avg_sentence_length <= 25 else max(0, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
        
        # 段落结构
        paragraphs = content.split('\n\n')
        paragraph_score = min(len(paragraphs) / 5, 1.0)  # 适当的段落数
        
        # 标点符号使用
        punctuation_density = len(re.findall(r'[,，.。!！?？;；:：]', content)) / len(content)
        punctuation_score = 1.0 if 0.05 <= punctuation_density <= 0.15 else max(0, 1.0 - abs(punctuation_density - 0.1) / 0.1)
        
        readability = (length_score * 0.4 + paragraph_score * 0.3 + punctuation_score * 0.3)
        return readability
    
    async def _evaluate_factuality(self, content: str, retrieval_results: List[RetrievalResult] = None) -> float:
        """评估事实性"""
        if not retrieval_results:
            return 0.8  # 默认分数
        
        # 检查内容是否与检索结果一致
        content_lower = content.lower()
        source_contents = [result.content.lower() for result in retrieval_results]
        
        # 简单的事实验证：检查关键信息是否在源文档中
        content_sentences = re.split(r'[.!?。！？]', content_lower)
        verified_sentences = 0
        
        for sentence in content_sentences:
            if len(sentence.strip()) < 10:
                continue
            
            # 检查句子中的关键词是否在源文档中出现
            sentence_words = set(sentence.split())
            for source_content in source_contents:
                source_words = set(source_content.split())
                word_overlap = len(sentence_words.intersection(source_words))
                if word_overlap >= len(sentence_words) * 0.3:  # 30%词汇重叠
                    verified_sentences += 1
                    break
        
        if content_sentences:
            factuality = verified_sentences / len([s for s in content_sentences if len(s.strip()) >= 10])
        else:
            factuality = 0.0
        
        return min(factuality, 1.0)
    
    def _update_evaluation_stats(self, metrics: Dict[str, float], overall_score: float):
        """更新评估统计"""
        self.evaluation_stats["total_evaluations"] += 1
        
        # 更新平均分数
        for metric, score in metrics.items():
            if metric not in self.evaluation_stats["average_scores"]:
                self.evaluation_stats["average_scores"][metric] = 0.0
            
            total = self.evaluation_stats["total_evaluations"]
            current_avg = self.evaluation_stats["average_scores"][metric]
            self.evaluation_stats["average_scores"][metric] = (current_avg * (total - 1) + score) / total
        
        # 更新分数分布
        score_range = "high" if overall_score >= 0.8 else "medium" if overall_score >= 0.6 else "low"
        if score_range not in self.evaluation_stats["score_distribution"]:
            self.evaluation_stats["score_distribution"][score_range] = 0
        self.evaluation_stats["score_distribution"][score_range] += 1
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """获取评估统计"""
        return self.evaluation_stats.copy()


class GenerationOptimizer:
    """生成优化器"""
    
    def __init__(self):
        self.context_fusion = ContextFusion()
        self.context_compressor = ContextCompressor()
        self.consistency_checker = ConsistencyChecker()
        self.quality_evaluator = QualityEvaluator()
        
        # 优化策略配置
        self.optimization_config = {
            "max_context_length": 2000,
            "fusion_strategy": FusionStrategy.SEMANTIC_MERGE,
            "compression_method": CompressionMethod.HYBRID,
            "quality_threshold": 0.7,
            "max_iterations": 3
        }
    
    async def optimize_generation(self,
                                 retrieval_results: List[RetrievalResult],
                                 query_analysis: QueryAnalysis,
                                 target_quality: float = 0.8) -> GenerationResult:
        """优化生成过程"""
        start_time = datetime.now()
        
        # 1. 上下文融合
        fused_context = await self.context_fusion.fuse_contexts(
            retrieval_results=retrieval_results,
            query_analysis=query_analysis,
            strategy=self.optimization_config["fusion_strategy"],
            max_length=self.optimization_config["max_context_length"]
        )
        
        # 2. 上下文压缩（如果需要）
        if len(fused_context.content) > self.optimization_config["max_context_length"]:
            compressed_content, compression_ratio = await self.context_compressor.compress_context(
                context=fused_context.content,
                target_length=self.optimization_config["max_context_length"],
                method=self.optimization_config["compression_method"],
                preserve_keywords=query_analysis.keywords
            )
            
            fused_context.content = compressed_content
            fused_context.compression_ratio = compression_ratio
        
        # 3. 生成内容（这里模拟生成过程）
        generated_content = await self._generate_content(fused_context, query_analysis)
        
        # 4. 质量评估
        quality_assessment = await self.quality_evaluator.evaluate_quality(
            content=generated_content,
            query_analysis=query_analysis,
            retrieval_results=retrieval_results
        )
        
        # 5. 迭代优化
        iteration_count = 0
        while (quality_assessment.overall_score < target_quality and 
               iteration_count < self.optimization_config["max_iterations"]):
            
            logger.info(f"质量分数 {quality_assessment.overall_score:.3f} 低于目标 {target_quality}，进行优化迭代 {iteration_count + 1}")
            
            # 根据质量评估结果调整生成策略
            optimized_context = await self._optimize_context_based_on_feedback(
                fused_context, quality_assessment, query_analysis
            )
            
            # 重新生成
            generated_content = await self._generate_content(optimized_context, query_analysis)
            
            # 重新评估
            quality_assessment = await self.quality_evaluator.evaluate_quality(
                content=generated_content,
                query_analysis=query_analysis,
                retrieval_results=retrieval_results
            )
            
            iteration_count += 1
        
        # 6. 最终一致性检查
        consistency_result = await self.consistency_checker.check_consistency(
            content=generated_content,
            sources=fused_context.sources
        )
        
        if not consistency_result["consistent"]:
            quality_assessment.issues.extend([issue["description"] for issue in consistency_result["issues"]])
            quality_assessment.overall_score *= consistency_result["confidence"]
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return GenerationResult(
            content=generated_content,
            fused_context=fused_context,
            quality_assessment=quality_assessment,
            generation_time=generation_time,
            metadata={
                "iterations": iteration_count,
                "final_quality": quality_assessment.overall_score,
                "consistency_check": consistency_result,
                "optimization_config": self.optimization_config
            }
        )
    
    async def _generate_content(self, 
                               fused_context: FusedContext, 
                               query_analysis: QueryAnalysis) -> str:
        """生成内容（模拟实现）"""
        # 这里应该调用实际的生成模型
        # 作为演示，我们基于融合上下文生成简化的响应
        
        query = query_analysis.original_query
        context = fused_context.content
        
        # 简化的生成逻辑
        if query_analysis.query_type.value == "factual":
            generated = f"根据查询'{query}'，以下是相关信息：\n\n{context}\n\n总结：基于以上信息可以了解到相关的事实和详情。"
        elif query_analysis.query_type.value == "procedural":
            generated = f"关于'{query}'的方法和步骤：\n\n{context}\n\n按照以上步骤可以完成相关操作。"
        elif query_analysis.query_type.value == "recommendation":
            generated = f"针对'{query}'的推荐建议：\n\n{context}\n\n综合考虑以上因素，建议根据个人需求选择合适的选项。"
        elif query_analysis.query_type.value == "comparison":
            generated = f"关于'{query}'的比较分析：\n\n{context}\n\n通过比较可以看出各选项的优缺点，建议根据实际需求决策。"
        else:
            generated = f"关于'{query}'的相关信息：\n\n{context}\n\n希望这些信息对您有所帮助。"
        
        return generated
    
    async def _optimize_context_based_on_feedback(self,
                                                 context: FusedContext,
                                                 quality_assessment: QualityAssessment,
                                                 query_analysis: QueryAnalysis) -> FusedContext:
        """基于反馈优化上下文"""
        # 分析质量问题并调整上下文
        optimized_context = context
        
        # 如果相关性不足，增强关键词匹配
        if quality_assessment.metrics.get("relevance", 1.0) < 0.6:
            # 在上下文中突出显示关键词相关的内容
            content_lines = context.content.split('\n')
            keyword_enhanced_lines = []
            
            for line in content_lines:
                enhanced_line = line
                for keyword in query_analysis.keywords[:5]:  # 只处理前5个关键词
                    if keyword.lower() in line.lower():
                        enhanced_line = f"**{line}**"  # 标记重要行
                        break
                keyword_enhanced_lines.append(enhanced_line)
            
            optimized_context.content = '\n'.join(keyword_enhanced_lines)
        
        # 如果连贯性不足，添加过渡句
        if quality_assessment.metrics.get("coherence", 1.0) < 0.7:
            paragraphs = context.content.split('\n\n')
            connected_paragraphs = []
            
            for i, paragraph in enumerate(paragraphs):
                connected_paragraphs.append(paragraph)
                if i < len(paragraphs) - 1:
                    # 添加简单的过渡句
                    transitions = ["此外，", "另外，", "同时，", "进一步地，"]
                    if i < len(transitions):
                        connected_paragraphs.append(transitions[i % len(transitions)])
            
            optimized_context.content = '\n\n'.join(connected_paragraphs)
        
        # 如果完整性不足，尝试扩展内容
        if quality_assessment.metrics.get("completeness", 1.0) < 0.8:
            # 在内容末尾添加总结
            summary = "\n\n总结：以上信息提供了全面的答案，涵盖了查询的主要方面。"
            optimized_context.content += summary
        
        return optimized_context
    
    def update_optimization_config(self, new_config: Dict[str, Any]):
        """更新优化配置"""
        self.optimization_config.update(new_config)
        logger.info(f"优化配置已更新: {new_config}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        return {
            "fusion_stats": {},  # 可以从context_fusion获取
            "compression_stats": self.context_compressor.get_compression_stats(),
            "evaluation_stats": self.quality_evaluator.get_evaluation_stats(),
            "current_config": self.optimization_config
        }


# 全局生成优化器实例
_generation_optimizer: Optional[GenerationOptimizer] = None


def get_generation_optimizer() -> GenerationOptimizer:
    """获取生成优化器实例"""
    global _generation_optimizer
    if _generation_optimizer is None:
        _generation_optimizer = GenerationOptimizer()
    return _generation_optimizer 