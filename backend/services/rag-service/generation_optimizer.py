"""
RAG生成优化系统
实现内容融合、上下文压缩、一致性检查、质量评估、优化循环等功能
"""

import asyncio
import json
import math
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import structlog
from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .advanced_retrieval import RetrievalResult

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class GenerationContext:
    """生成上下文"""
    query: str
    retrieved_results: List[RetrievalResult]
    fused_content: str
    compressed_content: str
    generation_prompt: str
    metadata: Dict[str, Any]


@dataclass
class QualityMetrics:
    """质量评估指标"""
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    coherence_score: float
    faithfulness_score: float
    overall_score: float
    feedback: List[str]


@dataclass
class OptimizationResult:
    """优化结果"""
    original_content: str
    optimized_content: str
    optimization_steps: List[str]
    quality_improvement: float
    processing_time: float
    metadata: Dict[str, Any]


class ContentFusion:
    """内容融合器"""
    
    def __init__(self):
        self.fusion_strategies = {
            "concatenation": self._concatenation_fusion,
            "weighted": self._weighted_fusion,
            "hierarchical": self._hierarchical_fusion,
            "attention": self._attention_fusion,
            "semantic": self._semantic_fusion
        }
    
    def fuse_content(self, results: List[RetrievalResult], 
                    strategy: str = "semantic", 
                    max_length: int = 2000) -> str:
        """融合检索内容"""
        if not results:
            return ""
        
        if strategy not in self.fusion_strategies:
            strategy = "semantic"
        
        try:
            fusion_func = self.fusion_strategies[strategy]
            fused_content = fusion_func(results, max_length)
            
            logger.info(f"使用{strategy}策略融合了{len(results)}个结果，生成{len(fused_content)}字符的内容")
            return fused_content
            
        except Exception as e:
            logger.error(f"内容融合失败: {e}")
            # 降级到简单连接
            return self._concatenation_fusion(results, max_length)
    
    def _concatenation_fusion(self, results: List[RetrievalResult], max_length: int) -> str:
        """简单连接融合"""
        contents = []
        current_length = 0
        
        # 按分数排序
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        for result in sorted_results:
            content = result.content.strip()
            if current_length + len(content) <= max_length:
                contents.append(content)
                current_length += len(content)
            else:
                # 截断最后一个文档
                remaining = max_length - current_length
                if remaining > 100:  # 最少保留100字符
                    contents.append(content[:remaining])
                break
        
        return "\n\n".join(contents)
    
    def _weighted_fusion(self, results: List[RetrievalResult], max_length: int) -> str:
        """加权融合"""
        # 计算权重
        total_score = sum(result.score for result in results)
        weights = [result.score / total_score for result in results]
        
        # 根据权重分配长度
        contents = []
        for result, weight in zip(results, weights):
            allocated_length = int(max_length * weight)
            content = result.content.strip()
            
            if len(content) > allocated_length:
                content = content[:allocated_length]
            
            contents.append(f"[权重: {weight:.2f}] {content}")
        
        return "\n\n".join(contents)
    
    def _hierarchical_fusion(self, results: List[RetrievalResult], max_length: int) -> str:
        """分层融合"""
        # 按来源分组
        source_groups = defaultdict(list)
        for result in results:
            source = result.metadata.get("source", "unknown")
            source_groups[source].append(result)
        
        # 为每个来源生成摘要
        fused_contents = []
        for source, group_results in source_groups.items():
            group_contents = [result.content for result in group_results]
            summary = self._summarize_content_group(group_contents)
            fused_contents.append(f"来源 [{source}]:\n{summary}")
        
        combined = "\n\n".join(fused_contents)
        
        # 如果超长，进行截断
        if len(combined) > max_length:
            combined = combined[:max_length]
        
        return combined
    
    def _attention_fusion(self, results: List[RetrievalResult], max_length: int) -> str:
        """注意力机制融合"""
        if not results:
            return ""
        
        # 计算注意力权重
        contents = [result.content for result in results]
        scores = [result.score for result in results]
        
        # 基于TF-IDF计算内容重要性
        vectorizer = TfidfVectorizer(max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            # 计算注意力权重（结合检索分数和TF-IDF）
            attention_weights = []
            for i, (score, tfidf_vector) in enumerate(zip(scores, tfidf_matrix)):
                # TF-IDF向量的L2范数表示内容重要性
                content_importance = np.linalg.norm(tfidf_vector.toarray())
                attention_weight = score * content_importance
                attention_weights.append(attention_weight)
            
            # 归一化权重
            total_weight = sum(attention_weights)
            if total_weight > 0:
                attention_weights = [w / total_weight for w in attention_weights]
            else:
                attention_weights = [1.0 / len(contents) for _ in contents]
            
            # 根据注意力权重组合内容
            fused_parts = []
            current_length = 0
            
            # 按注意力权重排序
            weighted_results = list(zip(results, attention_weights))
            weighted_results.sort(key=lambda x: x[1], reverse=True)
            
            for result, weight in weighted_results:
                if current_length >= max_length:
                    break
                
                content = result.content.strip()
                allocated_length = min(
                    len(content),
                    int(max_length * weight),
                    max_length - current_length
                )
                
                if allocated_length > 50:  # 最少保留50字符
                    selected_content = content[:allocated_length]
                    fused_parts.append(f"[权重: {weight:.3f}] {selected_content}")
                    current_length += allocated_length
            
            return "\n\n".join(fused_parts)
            
        except Exception as e:
            logger.error(f"注意力融合失败: {e}")
            return self._concatenation_fusion(results, max_length)
    
    def _semantic_fusion(self, results: List[RetrievalResult], max_length: int) -> str:
        """语义融合"""
        if not results:
            return ""
        
        try:
            contents = [result.content for result in results]
            
            # 使用聚类进行语义分组
            vectorizer = TfidfVectorizer(max_features=200)
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            # 确定聚类数量
            n_clusters = min(3, len(contents))
            if n_clusters < 2:
                return self._concatenation_fusion(results, max_length)
            
            # 聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 按聚类组织内容
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append((results[i], contents[i]))
            
            # 为每个聚类生成代表性内容
            fused_parts = []
            for cluster_id, cluster_items in clusters.items():
                # 选择聚类中分数最高的内容作为代表
                cluster_items.sort(key=lambda x: x[0].score, reverse=True)
                representative = cluster_items[0][1]
                
                # 如果聚类有多个项目，生成摘要
                if len(cluster_items) > 1:
                    cluster_contents = [item[1] for item in cluster_items]
                    summary = self._summarize_content_group(cluster_contents)
                    fused_parts.append(f"主题 {cluster_id + 1}:\n{summary}")
                else:
                    fused_parts.append(f"主题 {cluster_id + 1}:\n{representative}")
            
            combined = "\n\n".join(fused_parts)
            
            # 长度控制
            if len(combined) > max_length:
                combined = combined[:max_length]
            
            return combined
            
        except Exception as e:
            logger.error(f"语义融合失败: {e}")
            return self._concatenation_fusion(results, max_length)
    
    def _summarize_content_group(self, contents: List[str]) -> str:
        """为内容组生成摘要"""
        if not contents:
            return ""
        
        if len(contents) == 1:
            return contents[0]
        
        # 简单的摘要策略：选择最长的内容并提取关键句子
        longest_content = max(contents, key=len)
        
        # 提取关键句子（简化版）
        sentences = re.split(r'[。！？]', longest_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # 选择前3个句子作为摘要
        summary_sentences = sentences[:3]
        return "。".join(summary_sentences) + "。" if summary_sentences else longest_content[:200]


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self):
        self.compression_methods = {
            "extractive": self._extractive_compression,
            "abstractive": self._abstractive_compression,
            "hybrid": self._hybrid_compression,
            "keyword": self._keyword_compression,
            "clustering": self._clustering_compression
        }
    
    def compress_context(self, content: str, target_length: int, 
                        method: str = "hybrid") -> str:
        """压缩上下文"""
        if len(content) <= target_length:
            return content
        
        if method not in self.compression_methods:
            method = "hybrid"
        
        try:
            compression_func = self.compression_methods[method]
            compressed = compression_func(content, target_length)
            
            logger.info(f"使用{method}方法将{len(content)}字符压缩到{len(compressed)}字符")
            return compressed
            
        except Exception as e:
            logger.error(f"上下文压缩失败: {e}")
            # 降级到简单截断
            return content[:target_length]
    
    def _extractive_compression(self, content: str, target_length: int) -> str:
        """抽取式压缩"""
        sentences = re.split(r'[。！？\n]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return content[:target_length]
        
        # 计算句子重要性
        sentence_scores = []
        for sentence in sentences:
            score = 0.0
            
            # 长度分数（中等长度的句子得分更高）
            length_score = 1.0 - abs(len(sentence) - 50) / 100
            score += length_score * 0.3
            
            # 关键词分数
            if JIEBA_AVAILABLE:
                keywords = jieba.analyse.extract_tags(sentence, topK=5)
                score += len(keywords) * 0.4
            
            # 位置分数（开头和结尾的句子得分更高）
            position = sentences.index(sentence)
            if position < len(sentences) * 0.2 or position > len(sentences) * 0.8:
                score += 0.3
            
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
            else:
                break
        
        return "。".join(selected_sentences) + "。" if selected_sentences else content[:target_length]
    
    def _abstractive_compression(self, content: str, target_length: int) -> str:
        """生成式压缩（简化版）"""
        # 由于没有专门的摘要模型，使用关键信息提取
        
        # 提取关键信息
        key_info = []
        
        # 提取数字和日期
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        dates = re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日', content)
        
        # 提取专有名词
        proper_nouns = re.findall(r'[\u4e00-\u9fff]{2,6}(?:市|省|县|区|景点|酒店|餐厅)', content)
        
        # 提取关键句子（包含重要信息的句子）
        sentences = re.split(r'[。！？]', content)
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 包含数字、日期或专有名词的句子
            if (any(num in sentence for num in numbers) or
                any(date in sentence for date in dates) or 
                any(noun in sentence for noun in proper_nouns)):
                key_sentences.append(sentence)
        
        # 如果没有关键句子，选择最长的几个句子
        if not key_sentences:
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            sentences.sort(key=len, reverse=True)
            key_sentences = sentences[:3]
        
        # 组合关键信息
        compressed = "。".join(key_sentences[:5]) + "。"
        
        # 如果仍然太长，进行截断
        if len(compressed) > target_length:
            compressed = compressed[:target_length]
        
        return compressed
    
    def _hybrid_compression(self, content: str, target_length: int) -> str:
        """混合压缩"""
        # 先进行抽取式压缩到中间长度
        intermediate_length = min(target_length * 2, len(content))
        extractive_result = self._extractive_compression(content, intermediate_length)
        
        # 再进行生成式压缩到目标长度
        if len(extractive_result) > target_length:
            return self._abstractive_compression(extractive_result, target_length)
        else:
            return extractive_result
    
    def _keyword_compression(self, content: str, target_length: int) -> str:
        """基于关键词的压缩"""
        if JIEBA_AVAILABLE:
            # 提取关键词
            keywords = jieba.analyse.extract_tags(content, topK=20, withWeight=True)
            
            # 选择包含高权重关键词的句子
            sentences = re.split(r'[。！？]', content)
            sentence_scores = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                score = 0.0
                for keyword, weight in keywords:
                    if keyword in sentence:
                        score += weight
                
                sentence_scores.append((sentence, score))
            
            # 按分数排序并选择
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected = []
            current_length = 0
            
            for sentence, score in sentence_scores:
                if current_length + len(sentence) <= target_length:
                    selected.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            return "。".join(selected) + "。" if selected else content[:target_length]
        else:
            # 降级到简单截断
            return content[:target_length]
    
    def _clustering_compression(self, content: str, target_length: int) -> str:
        """基于聚类的压缩"""
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return content[:target_length]
        
        try:
            # 向量化句子
            vectorizer = TfidfVectorizer(max_features=50)
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # 聚类
            n_clusters = min(3, len(sentences) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(sentence_vectors)
            
            # 每个聚类选择一个代表句子
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(sentences[i])
            
            representatives = []
            for cluster_sentences in clusters.values():
                # 选择最长的句子作为代表
                representative = max(cluster_sentences, key=len)
                representatives.append(representative)
            
            compressed = "。".join(representatives) + "。"
            
            # 长度控制
            if len(compressed) > target_length:
                compressed = compressed[:target_length]
            
            return compressed
            
        except Exception as e:
            logger.error(f"聚类压缩失败: {e}")
            return self._extractive_compression(content, target_length)


class ConsistencyChecker:
    """一致性检查器"""
    
    def __init__(self):
        self.conflict_patterns = {
            "数字冲突": [
                r'(\d+)元.*?(\d+)元',
                r'(\d+)天.*?(\d+)天',
                r'(\d+)小时.*?(\d+)小时'
            ],
            "时间冲突": [
                r'(\d+)月(\d+)日.*?(\d+)月(\d+)日',
                r'(上午|下午|早上|晚上).*?(上午|下午|早上|晚上)'
            ],
            "地点冲突": [
                r'([\u4e00-\u9fff]+市).*?([\u4e00-\u9fff]+市)',
                r'([\u4e00-\u9fff]+省).*?([\u4e00-\u9fff]+省)'
            ]
        }
    
    def check_consistency(self, content: str) -> Dict[str, Any]:
        """检查内容一致性"""
        conflicts = []
        
        for conflict_type, patterns in self.conflict_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(set(match)) > 1:  # 如果有不同的值
                        conflicts.append({
                            "type": conflict_type,
                            "values": list(match),
                            "pattern": pattern
                        })
        
        consistency_score = 1.0 - min(len(conflicts) * 0.1, 0.5)
        
        return {
            "consistency_score": consistency_score,
            "conflicts": conflicts,
            "total_conflicts": len(conflicts)
        }
    
    def resolve_conflicts(self, content: str, conflicts: List[Dict[str, Any]]) -> str:
        """解决冲突"""
        resolved_content = content
        
        for conflict in conflicts:
            conflict_type = conflict["type"]
            values = conflict["values"]
            
            if conflict_type == "数字冲突":
                # 选择最大值
                max_value = max(values, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
                for value in values:
                    if value != max_value:
                        resolved_content = resolved_content.replace(value, max_value, 1)
            
            elif conflict_type == "时间冲突":
                # 保留第一个时间
                first_value = values[0]
                for value in values[1:]:
                    resolved_content = resolved_content.replace(value, first_value, 1)
            
            elif conflict_type == "地点冲突":
                # 保留第一个地点
                first_value = values[0]
                for value in values[1:]:
                    resolved_content = resolved_content.replace(value, first_value, 1)
        
        return resolved_content


class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self):
        self.relevance_keywords = {
            "travel": ["旅行", "旅游", "景点", "酒店", "交通", "路线"],
            "food": ["美食", "餐厅", "小吃", "菜品", "口味"],
            "accommodation": ["住宿", "酒店", "宾馆", "客房", "设施"]
        }
    
    def evaluate_quality(self, query: str, generated_content: str, 
                        source_content: str = "") -> QualityMetrics:
        """评估生成质量"""
        metrics = QualityMetrics(
            relevance_score=self._evaluate_relevance(query, generated_content),
            accuracy_score=self._evaluate_accuracy(generated_content, source_content),
            completeness_score=self._evaluate_completeness(query, generated_content),
            coherence_score=self._evaluate_coherence(generated_content),
            faithfulness_score=self._evaluate_faithfulness(generated_content, source_content),
            overall_score=0.0,
            feedback=[]
        )
        
        # 计算总分
        metrics.overall_score = (
            metrics.relevance_score * 0.25 +
            metrics.accuracy_score * 0.2 +
            metrics.completeness_score * 0.2 +
            metrics.coherence_score * 0.15 +
            metrics.faithfulness_score * 0.2
        )
        
        # 生成反馈
        metrics.feedback = self._generate_feedback(metrics)
        
        return metrics
    
    def _evaluate_relevance(self, query: str, content: str) -> float:
        """评估相关性"""
        if not query or not content:
            return 0.0
        
        # 提取查询关键词
        if JIEBA_AVAILABLE:
            query_keywords = set(jieba.analyse.extract_tags(query, topK=10))
            content_keywords = set(jieba.analyse.extract_tags(content, topK=20))
        else:
            query_keywords = set(re.findall(r'[\u4e00-\u9fff]+', query))
            content_keywords = set(re.findall(r'[\u4e00-\u9fff]+', content))
        
        if not query_keywords:
            return 0.5
        
        # 计算关键词重叠度
        overlap = len(query_keywords.intersection(content_keywords))
        relevance = overlap / len(query_keywords)
        
        return min(relevance, 1.0)
    
    def _evaluate_accuracy(self, content: str, source_content: str) -> float:
        """评估准确性"""
        if not source_content:
            return 0.7  # 没有源内容时给默认分数
        
        # 检查是否包含虚假信息（简化版）
        false_indicators = ["可能", "也许", "大概", "据说", "传说"]
        false_count = sum(1 for indicator in false_indicators if indicator in content)
        
        # 检查具体信息的准确性
        accuracy_score = 1.0 - (false_count * 0.1)
        
        return max(accuracy_score, 0.0)
    
    def _evaluate_completeness(self, query: str, content: str) -> float:
        """评估完整性"""
        # 检查是否回答了查询的主要方面
        completeness_factors = []
        
        # 检查基本信息
        has_specific_info = bool(re.search(r'\d+', content))  # 包含数字
        has_location_info = bool(re.search(r'[\u4e00-\u9fff]+(?:市|省|区|街|路)', content))
        has_time_info = bool(re.search(r'\d+(?:年|月|日|小时|分钟|天)', content))
        
        completeness_factors.extend([has_specific_info, has_location_info, has_time_info])
        
        # 检查内容长度（太短可能不完整）
        length_score = min(len(content) / 200, 1.0)
        completeness_factors.append(length_score > 0.5)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _evaluate_coherence(self, content: str) -> float:
        """评估连贯性"""
        if not content:
            return 0.0
        
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # 检查句子间的连接词
        coherence_indicators = ["首先", "然后", "接着", "最后", "另外", "此外", "因此", "所以"]
        coherence_count = sum(1 for indicator in coherence_indicators 
                            if any(indicator in sentence for sentence in sentences))
        
        # 检查重复和矛盾
        word_overlap_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(re.findall(r'[\u4e00-\u9fff]+', sentences[i]))
            words2 = set(re.findall(r'[\u4e00-\u9fff]+', sentences[i + 1]))
            
            if words1 and words2:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                word_overlap_scores.append(overlap)
        
        avg_overlap = sum(word_overlap_scores) / len(word_overlap_scores) if word_overlap_scores else 0
        
        # 综合评分
        coherence_score = (
            min(coherence_count / len(sentences), 0.5) * 0.4 +
            avg_overlap * 0.6
        )
        
        return min(coherence_score, 1.0)
    
    def _evaluate_faithfulness(self, content: str, source_content: str) -> float:
        """评估忠实性"""
        if not source_content:
            return 0.7
        
        # 检查生成内容是否忠实于源内容
        if JIEBA_AVAILABLE:
            content_keywords = set(jieba.analyse.extract_tags(content, topK=15))
            source_keywords = set(jieba.analyse.extract_tags(source_content, topK=20))
        else:
            content_keywords = set(re.findall(r'[\u4e00-\u9fff]+', content))
            source_keywords = set(re.findall(r'[\u4e00-\u9fff]+', source_content))
        
        if not content_keywords:
            return 0.0
        
        # 计算关键词覆盖度
        coverage = len(content_keywords.intersection(source_keywords)) / len(content_keywords)
        
        # 检查是否有明显的编造内容
        hallucination_indicators = ["据最新消息", "根据专家预测", "最新研究表明"]
        hallucination_count = sum(1 for indicator in hallucination_indicators if indicator in content)
        
        faithfulness_score = coverage - (hallucination_count * 0.2)
        
        return max(faithfulness_score, 0.0)
    
    def _generate_feedback(self, metrics: QualityMetrics) -> List[str]:
        """生成改进建议"""
        feedback = []
        
        if metrics.relevance_score < 0.6:
            feedback.append("内容与查询的相关性较低，建议增加更多相关信息")
        
        if metrics.accuracy_score < 0.7:
            feedback.append("内容准确性需要改进，避免使用不确定的表述")
        
        if metrics.completeness_score < 0.6:
            feedback.append("信息不够完整，建议补充时间、地点、价格等具体信息")
        
        if metrics.coherence_score < 0.6:
            feedback.append("内容连贯性需要改善，建议增加过渡词和逻辑连接")
        
        if metrics.faithfulness_score < 0.7:
            feedback.append("内容忠实性有待提高，请确保基于可靠来源")
        
        if metrics.overall_score > 0.8:
            feedback.append("整体质量良好，继续保持")
        
        return feedback


class RAGOptimizer:
    """RAG优化器"""
    
    def __init__(self):
        self.content_fusion = ContentFusion()
        self.context_compressor = ContextCompressor()
        self.consistency_checker = ConsistencyChecker()
        self.quality_evaluator = QualityEvaluator()
        
        # 优化配置
        self.optimization_config = {
            "max_iterations": 3,
            "quality_threshold": 0.8,
            "compression_ratio": 0.7,
            "fusion_strategy": "semantic"
        }
    
    def optimize_generation(self, query: str, retrieval_results: List[RetrievalResult],
                          generated_content: str = "") -> OptimizationResult:
        """优化RAG生成"""
        start_time = time.time()
        original_content = generated_content
        optimization_steps = []
        
        try:
            # 1. 内容融合
            if retrieval_results:
                fused_content = self.content_fusion.fuse_content(
                    retrieval_results,
                    strategy=self.optimization_config["fusion_strategy"]
                )
                optimization_steps.append("内容融合")
            else:
                fused_content = generated_content
            
            # 2. 上下文压缩
            if len(fused_content) > 2000:
                target_length = int(len(fused_content) * self.optimization_config["compression_ratio"])
                compressed_content = self.context_compressor.compress_context(
                    fused_content, target_length
                )
                optimization_steps.append("上下文压缩")
            else:
                compressed_content = fused_content
            
            # 3. 一致性检查和修复
            consistency_result = self.consistency_checker.check_consistency(compressed_content)
            if consistency_result["conflicts"]:
                compressed_content = self.consistency_checker.resolve_conflicts(
                    compressed_content, consistency_result["conflicts"]
                )
                optimization_steps.append("一致性修复")
            
            # 4. 质量评估
            source_content = "\n".join([result.content for result in retrieval_results])
            quality_metrics = self.quality_evaluator.evaluate_quality(
                query, compressed_content, source_content
            )
            
            # 5. 迭代优化（如果质量不达标）
            current_content = compressed_content
            iteration = 0
            
            while (quality_metrics.overall_score < self.optimization_config["quality_threshold"] and
                   iteration < self.optimization_config["max_iterations"]):
                
                iteration += 1
                optimization_steps.append(f"迭代优化 #{iteration}")
                
                # 基于质量反馈进行优化
                current_content = self._iterative_improve(
                    current_content, quality_metrics.feedback
                )
                
                # 重新评估
                quality_metrics = self.quality_evaluator.evaluate_quality(
                    query, current_content, source_content
                )
            
            processing_time = time.time() - start_time
            
            # 计算质量改进
            if original_content:
                original_quality = self.quality_evaluator.evaluate_quality(
                    query, original_content, source_content
                )
                quality_improvement = quality_metrics.overall_score - original_quality.overall_score
            else:
                quality_improvement = quality_metrics.overall_score
            
            return OptimizationResult(
                original_content=original_content,
                optimized_content=current_content,
                optimization_steps=optimization_steps,
                quality_improvement=quality_improvement,
                processing_time=processing_time,
                metadata={
                    "final_quality_score": quality_metrics.overall_score,
                    "iterations_used": iteration,
                    "compression_ratio": len(current_content) / max(len(fused_content), 1),
                    "consistency_score": consistency_result["consistency_score"],
                    "quality_metrics": asdict(quality_metrics)
                }
            )
            
        except Exception as e:
            logger.error(f"RAG优化失败: {e}")
            processing_time = time.time() - start_time
            
            return OptimizationResult(
                original_content=original_content,
                optimized_content=original_content or "",
                optimization_steps=["优化失败"],
                quality_improvement=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _iterative_improve(self, content: str, feedback: List[str]) -> str:
        """基于反馈迭代改进内容"""
        improved_content = content
        
        for suggestion in feedback:
            if "相关性较低" in suggestion:
                # 尝试保留更多关键信息
                improved_content = self._enhance_relevance(improved_content)
            
            elif "信息不够完整" in suggestion:
                # 尝试添加更多细节
                improved_content = self._enhance_completeness(improved_content)
            
            elif "连贯性需要改善" in suggestion:
                # 添加连接词
                improved_content = self._enhance_coherence(improved_content)
        
        return improved_content
    
    def _enhance_relevance(self, content: str) -> str:
        """增强相关性"""
        # 确保关键信息在前面
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 将包含数字、地点等重要信息的句子前置
        important_sentences = []
        other_sentences = []
        
        for sentence in sentences:
            if (re.search(r'\d+', sentence) or 
                re.search(r'[\u4e00-\u9fff]+(?:市|省|区|景点)', sentence)):
                important_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        reordered = important_sentences + other_sentences
        return "。".join(reordered) + "。" if reordered else content
    
    def _enhance_completeness(self, content: str) -> str:
        """增强完整性"""
        # 简单的完整性增强：确保有基本信息框架
        if "时间" not in content and "小时" not in content:
            content = content + " 建议预留充足的游览时间。"
        
        if "价格" not in content and "元" not in content:
            content = content + " 费用请提前了解。"
        
        return content
    
    def _enhance_coherence(self, content: str) -> str:
        """增强连贯性"""
        sentences = re.split(r'[。！？]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return content
        
        # 添加连接词
        coherence_words = ["首先", "然后", "接下来", "最后", "另外"]
        
        enhanced_sentences = []
        for i, sentence in enumerate(sentences):
            if i < len(coherence_words) and not any(word in sentence for word in coherence_words):
                enhanced_sentences.append(f"{coherence_words[i]}，{sentence}")
            else:
                enhanced_sentences.append(sentence)
        
        return "。".join(enhanced_sentences) + "。"


class OptimizationLoop:
    """优化循环"""
    
    def __init__(self):
        self.rag_optimizer = RAGOptimizer()
        self.feedback_history = []
        self.performance_metrics = {
            "total_optimizations": 0,
            "average_quality_improvement": 0.0,
            "average_processing_time": 0.0
        }
    
    async def run_optimization_cycle(self, query: str, retrieval_results: List[RetrievalResult],
                                   initial_content: str = "") -> OptimizationResult:
        """运行优化循环"""
        result = self.rag_optimizer.optimize_generation(
            query, retrieval_results, initial_content
        )
        
        # 更新性能指标
        self._update_metrics(result)
        
        # 记录反馈
        self.feedback_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "quality_improvement": result.quality_improvement,
            "optimization_steps": result.optimization_steps
        })
        
        return result
    
    def _update_metrics(self, result: OptimizationResult):
        """更新性能指标"""
        self.performance_metrics["total_optimizations"] += 1
        
        # 更新平均质量改进
        total = self.performance_metrics["total_optimizations"]
        current_avg_quality = self.performance_metrics["average_quality_improvement"]
        self.performance_metrics["average_quality_improvement"] = (
            (current_avg_quality * (total - 1) + result.quality_improvement) / total
        )
        
        # 更新平均处理时间
        current_avg_time = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (
            (current_avg_time * (total - 1) + result.processing_time) / total
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "metrics": self.performance_metrics,
            "recent_feedback": self.feedback_history[-10:],  # 最近10次反馈
            "optimization_trends": self._analyze_trends()
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """分析优化趋势"""
        if len(self.feedback_history) < 5:
            return {"trend": "insufficient_data"}
        
        recent_improvements = [fb["quality_improvement"] for fb in self.feedback_history[-10:]]
        avg_recent = sum(recent_improvements) / len(recent_improvements)
        
        earlier_improvements = [fb["quality_improvement"] for fb in self.feedback_history[-20:-10]]
        avg_earlier = sum(earlier_improvements) / len(earlier_improvements) if earlier_improvements else 0
        
        if avg_recent > avg_earlier + 0.05:
            trend = "improving"
        elif avg_recent < avg_earlier - 0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_average": avg_recent,
            "earlier_average": avg_earlier,
            "trend_strength": abs(avg_recent - avg_earlier)
        }


# 全局优化循环实例
_optimization_loop: Optional[OptimizationLoop] = None


def get_optimization_loop() -> OptimizationLoop:
    """获取优化循环实例"""
    global _optimization_loop
    if _optimization_loop is None:
        _optimization_loop = OptimizationLoop()
    return _optimization_loop 