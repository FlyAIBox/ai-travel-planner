"""
上下文工程系统
实现智能上下文管理、信息提取压缩、一致性检查和多轮对话上下文维护
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib

from pydantic import BaseModel, Field
import spacy
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ContextChunk:
    """上下文块"""
    content: str
    chunk_type: str  # 'user_input', 'ai_response', 'tool_result', 'knowledge'
    timestamp: datetime
    importance_score: float
    tokens_count: int
    entities: List[Dict[str, Any]]
    topic_tags: List[str]
    chunk_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextChunk':
        """从字典创建"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationState:
    """对话状态"""
    conversation_id: str
    user_id: str
    current_topic: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = None
    travel_context: Dict[str, Any] = None
    preferences: Dict[str, Any] = None
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.travel_context is None:
            self.travel_context = {}
        if self.preferences is None:
            self.preferences = {}
        if self.last_activity is None:
            self.last_activity = datetime.now()


class ContextWindow:
    """上下文窗口管理器"""
    
    def __init__(self, max_tokens: int = 8192, overlap_tokens: int = 512):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chunks: List[ContextChunk] = []
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        
    def add_chunk(self, chunk: ContextChunk) -> None:
        """添加上下文块"""
        self.chunks.append(chunk)
        self._optimize_window()
        
    def _optimize_window(self) -> None:
        """优化上下文窗口，移除低重要性的旧内容"""
        total_tokens = sum(chunk.tokens_count for chunk in self.chunks)
        
        if total_tokens <= self.max_tokens:
            return
            
        # 按重要性和时间排序
        sorted_chunks = sorted(
            self.chunks,
            key=lambda x: (x.importance_score, x.timestamp.timestamp()),
            reverse=True
        )
        
        # 保留最重要的内容
        kept_chunks = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            if current_tokens + chunk.tokens_count <= self.max_tokens - self.overlap_tokens:
                kept_chunks.append(chunk)
                current_tokens += chunk.tokens_count
            else:
                break
                
        # 按时间重新排序
        self.chunks = sorted(kept_chunks, key=lambda x: x.timestamp)
        
    def get_context_string(self, max_length: Optional[int] = None) -> str:
        """获取上下文字符串"""
        if max_length is None:
            max_length = self.max_tokens
            
        context_parts = []
        current_tokens = 0
        
        # 倒序遍历，优先包含最新内容
        for chunk in reversed(self.chunks):
            if current_tokens + chunk.tokens_count <= max_length:
                context_parts.insert(0, chunk.content)
                current_tokens += chunk.tokens_count
            else:
                break
                
        return "\n\n".join(context_parts)
    
    def get_relevant_chunks(self, query: str, max_chunks: int = 5) -> List[ContextChunk]:
        """获取与查询相关的上下文块"""
        query_tokens = set(word_tokenize(query.lower()))
        
        # 计算相关性分数
        chunk_scores = []
        for chunk in self.chunks:
            chunk_tokens = set(word_tokenize(chunk.content.lower()))
            
            # 基于词汇重叠的相关性
            overlap = len(query_tokens & chunk_tokens)
            relevance_score = overlap / len(query_tokens) if query_tokens else 0
            
            # 结合重要性分数
            final_score = relevance_score * 0.7 + chunk.importance_score * 0.3
            
            chunk_scores.append((chunk, final_score))
            
        # 按分数排序并返回top-k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in chunk_scores[:max_chunks]]


class InformationExtractor:
    """信息提取器"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            logger.warning("中文spaCy模型未找到，使用英文模型")
            self.nlp = spacy.load("en_core_web_sm")
            
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
            
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取命名实体"""
        entities = []
        
        # 使用spaCy提取实体
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0  # spaCy不直接提供置信度
            })
            
        # 使用NLTK作为补充
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tree = ne_chunk(pos_tags)
        
        return entities
    
    def extract_travel_entities(self, text: str) -> Dict[str, List[str]]:
        """提取旅行相关实体"""
        travel_entities = {
            "destinations": [],
            "dates": [],
            "activities": [],
            "accommodations": [],
            "transport": [],
            "budget": [],
            "preferences": []
        }
        
        # 目的地匹配模式
        destination_patterns = [
            r'(?:去|到|在|想去|计划去|打算去)\s*([^\s，。！？]+?(?:市|省|国|地区|景区|景点))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 英文地名
        ]
        
        # 时间匹配模式
        date_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{1,2}月\d{1,2}日)',
            r'(明天|后天|下周|下个月|春节|国庆|暑假)',
            r'(\d{1,2}天|\d{1,2}周|\d{1,2}个月)',
        ]
        
        # 预算匹配模式
        budget_patterns = [
            r'(预算|花费|费用|价格).*?(\d+).*?(?:元|块|万|千)',
            r'(\d+).*?(?:元|块|万|千).*?(?:预算|花费|费用|价格)',
        ]
        
        # 提取目的地
        for pattern in destination_patterns:
            matches = re.findall(pattern, text)
            travel_entities["destinations"].extend(matches)
            
        # 提取日期
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            travel_entities["dates"].extend(matches)
            
        # 提取预算
        for pattern in budget_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    travel_entities["budget"].extend(match)
                else:
                    travel_entities["budget"].append(match)
        
        # 去重
        for key in travel_entities:
            travel_entities[key] = list(set(travel_entities[key]))
            
        return travel_entities
    
    def extract_destinations(self, text: str) -> List[str]:
        """提取目的地信息"""
        destinations = []
        
        # 预定义的城市名单
        city_names = {
            '北京', '上海', '广州', '深圳', '杭州', '成都', '重庆', '西安', '南京', '武汉',
            '东京', '大阪', '京都', '首尔', '釜山', '曼谷', '普吉岛', '新加坡', '吉隆坡',
            '巴黎', '伦敦', '罗马', '巴塞罗那', '阿姆斯特丹', '柏林', '慕尼黑',
            '纽约', '洛杉矶', '旧金山', '芝加哥', '波士顿', '华盛顿', '拉斯维加斯',
            '悉尼', '墨尔本', '布里斯班', '黄金海岸'
        }
        
        # 检查文本中的城市名
        for city in city_names:
            if city in text:
                destinations.append(city)
        
        # 使用正则表达式匹配目的地模式
        patterns = [
            r'(?:去|到|在|想去|计划去|打算去)\s*([^\s，。！？]+?(?:市|省|国|地区|景区|景点))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 英文地名
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            destinations.extend(matches)
        
        return list(set(destinations))
    
    def extract_dates(self, text: str) -> List[str]:
        """提取日期信息"""
        dates = []
        
        # 日期模式
        patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日?',  # 2024年3月15日
            r'\d{1,2}月\d{1,2}日?',  # 3月15日
            r'\d{1,2}/\d{1,2}',  # 3/15
            r'\d{4}-\d{1,2}-\d{1,2}',  # 2024-03-15
            r'明天|后天|下周|下个月|春节|国庆|暑假|寒假',
            r'\d{1,2}天|\d{1,2}周|\d{1,2}个月'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return list(set(dates))
    
    def extract_budget_info(self, text: str) -> List[str]:
        """提取预算信息"""
        budget_info = []
        
        # 预算模式
        patterns = [
            r'\d+万?元',  # 5000元, 1万元
            r'\$\d+',  # $500
            r'€\d+',  # €500
            r'\d+美元',  # 500美元
            r'\d+欧元',  # 500欧元
            r'预算.*?(\d+).*?(?:元|块|万|千)',
            r'(\d+).*?(?:元|块|万|千).*?预算'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            budget_info.extend(matches)
        
        return list(set(budget_info))
    
    def extract_travel_preferences(self, text: str) -> List[str]:
        """提取旅行偏好"""
        preferences = []
        
        # 偏好关键词
        preference_keywords = {
            '住宿': ['豪华', '经济', '民宿', '酒店', '青旅', '五星', '四星'],
            '交通': ['飞机', '火车', '自驾', '公共交通', '包车'],
            '活动': ['观光', '美食', '购物', '休闲', '冒险', '文化', '自然'],
            '人群': ['情侣', '家庭', '朋友', '独自', '蜜月', '亲子'],
            '风格': ['奢华', '经济', '舒适', '快节奏', '慢生活', '深度游']
        }
        
        for category, keywords in preference_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    preferences.append(f"{category}:{keyword}")
        
        return preferences
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """提取关键短语"""
        doc = self.nlp(text)
        
        # 提取名词短语
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # 基于TF-IDF的关键词提取（简化版）
        words = [token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and token.is_alpha]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        key_words = [word for word, _ in sorted_words[:max_phrases//2]]
        
        # 合并关键词和名词短语
        key_phrases = noun_phrases + key_words
        
        return key_phrases[:max_phrases]
    
    def calculate_importance_score(self, text: str, chunk_type: str) -> float:
        """计算重要性分数"""
        base_scores = {
            'user_input': 0.8,
            'ai_response': 0.6,
            'tool_result': 0.9,
            'knowledge': 0.5
        }
        
        base_score = base_scores.get(chunk_type, 0.5)
        
        # 基于内容长度调整
        length_factor = min(len(text) / 500, 1.0)  # 归一化到[0,1]
        
        # 基于实体数量调整
        entities = self.extract_entities(text)
        entity_factor = min(len(entities) / 5, 1.0)  # 归一化到[0,1]
        
        # 基于旅行相关性调整
        travel_entities = self.extract_travel_entities(text)
        travel_factor = min(sum(len(v) for v in travel_entities.values()) / 5, 1.0)
        
        # 计算最终分数
        final_score = base_score * (0.4 + 0.2 * length_factor + 0.2 * entity_factor + 0.2 * travel_factor)
        
        return min(final_score, 1.0)


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.extractor = InformationExtractor()
        
    def compress_text(self, text: str, target_length: Optional[int] = None) -> str:
        """压缩文本，保留关键信息"""
        if target_length is None:
            target_length = int(len(text) * self.compression_ratio)
            
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text[:target_length] if len(text) > target_length else text
            
        # 计算每个句子的重要性
        sentence_scores = []
        for sentence in sentences:
            # 基于实体数量
            entities = self.extractor.extract_entities(sentence)
            entity_score = len(entities) / 5  # 归一化
            
            # 基于关键词
            key_phrases = self.extractor.extract_key_phrases(sentence, max_phrases=3)
            keyword_score = len(key_phrases) / 3  # 归一化
            
            # 基于句子长度
            length_score = min(len(sentence) / 100, 1.0)  # 归一化
            
            # 综合分数
            total_score = entity_score * 0.4 + keyword_score * 0.4 + length_score * 0.2
            sentence_scores.append((sentence, total_score))
            
        # 按分数排序
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最重要的句子，直到达到目标长度
        selected_sentences = []
        current_length = 0
        
        for sentence, score in sentence_scores:
            if current_length + len(sentence) <= target_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
                
        # 如果没有句子被选中，至少返回最重要的一句
        if not selected_sentences and sentence_scores:
            selected_sentences = [sentence_scores[0][0]]
            
        return " ".join(selected_sentences)
    
    def compress_chunks(self, chunks: List[ContextChunk], max_total_tokens: int) -> List[ContextChunk]:
        """压缩上下文块列表"""
        total_tokens = sum(chunk.tokens_count for chunk in chunks)
        
        if total_tokens <= max_total_tokens:
            return chunks
            
        compression_needed = total_tokens / max_total_tokens
        
        compressed_chunks = []
        for chunk in chunks:
            if chunk.importance_score >= 0.8:
                # 高重要性内容保持不变
                compressed_chunks.append(chunk)
            else:
                # 压缩低重要性内容
                target_length = int(len(chunk.content) / compression_needed)
                compressed_content = self.compress_text(chunk.content, target_length)
                
                # 创建新的压缩块
                compressed_chunk = ContextChunk(
                    content=compressed_content,
                    chunk_type=chunk.chunk_type,
                    timestamp=chunk.timestamp,
                    importance_score=chunk.importance_score,
                    tokens_count=len(compressed_content) // 4,  # 估算token数
                    entities=chunk.entities,
                    topic_tags=chunk.topic_tags,
                    chunk_id=chunk.chunk_id + "_compressed"
                )
                compressed_chunks.append(compressed_chunk)
                
        return compressed_chunks


class ContextConsistencyChecker:
    """上下文一致性检查器"""
    
    def __init__(self):
        self.extractor = InformationExtractor()
        
    def check_consistency(self, chunks: List[ContextChunk]) -> Dict[str, Any]:
        """检查上下文一致性"""
        consistency_report = {
            "conflicts": [],
            "contradictions": [],
            "missing_context": [],
            "consistency_score": 1.0
        }
        
        # 提取所有实体和事实
        all_entities = {}
        all_travel_info = {}
        
        for chunk in chunks:
            entities = self.extractor.extract_entities(chunk.content)
            travel_entities = self.extractor.extract_travel_entities(chunk.content)
            
            # 检查实体一致性
            for entity in entities:
                entity_text = entity["text"].lower()
                if entity_text in all_entities:
                    if all_entities[entity_text]["label"] != entity["label"]:
                        consistency_report["conflicts"].append({
                            "type": "entity_label_conflict",
                            "entity": entity_text,
                            "labels": [all_entities[entity_text]["label"], entity["label"]],
                            "chunks": [all_entities[entity_text]["chunk_id"], chunk.chunk_id]
                        })
                else:
                    all_entities[entity_text] = {
                        "label": entity["label"],
                        "chunk_id": chunk.chunk_id
                    }
                    
            # 检查旅行信息一致性
            for category, values in travel_entities.items():
                if category not in all_travel_info:
                    all_travel_info[category] = {}
                    
                for value in values:
                    value_lower = value.lower()
                    if value_lower in all_travel_info[category]:
                        # 检查是否有冲突信息
                        pass  # 这里可以添加更复杂的冲突检测逻辑
                    else:
                        all_travel_info[category][value_lower] = chunk.chunk_id
        
        # 检查必要信息是否缺失
        required_travel_info = ["destinations", "dates"]
        for required in required_travel_info:
            if required not in all_travel_info or not all_travel_info[required]:
                consistency_report["missing_context"].append({
                    "type": "missing_required_info",
                    "info": required,
                    "severity": "high"
                })
        
        # 计算一致性分数
        total_issues = (len(consistency_report["conflicts"]) + 
                       len(consistency_report["contradictions"]) + 
                       len(consistency_report["missing_context"]))
        
        if total_issues > 0:
            consistency_report["consistency_score"] = max(0.0, 1.0 - (total_issues * 0.1))
        
        return consistency_report
    
    def resolve_conflicts(self, chunks: List[ContextChunk], consistency_report: Dict[str, Any]) -> List[ContextChunk]:
        """解决冲突，返回修正后的上下文块"""
        resolved_chunks = chunks.copy()
        
        # 这里可以实现冲突解决逻辑
        # 例如：优先相信更新的信息、重要性更高的信息等
        
        for conflict in consistency_report["conflicts"]:
            logger.warning(f"检测到冲突: {conflict}")
            # 实现具体的冲突解决策略
            
        return resolved_chunks


class ContextMemoryManager:
    """上下文记忆管理器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.short_term_memory = {}  # 短期记忆（当前会话）
        self.long_term_memory = {}   # 长期记忆（跨会话）
        
    async def store_short_term(self, conversation_id: str, context_data: Dict[str, Any]) -> None:
        """存储短期记忆"""
        self.short_term_memory[conversation_id] = context_data
        
        if self.redis_client:
            try:
                await self.redis_client.set(
                    f"short_term:{conversation_id}",
                    json.dumps(context_data, default=str),
                    ex=3600  # 1小时过期
                )
            except Exception as e:
                logger.error(f"存储短期记忆失败: {e}")
    
    async def get_short_term(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取短期记忆"""
        # 优先从内存获取
        if conversation_id in self.short_term_memory:
            return self.short_term_memory[conversation_id]
            
        # 从Redis获取
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"short_term:{conversation_id}")
                if data:
                    context_data = json.loads(data)
                    self.short_term_memory[conversation_id] = context_data
                    return context_data
            except Exception as e:
                logger.error(f"获取短期记忆失败: {e}")
        
        return None
    
    async def store_long_term(self, user_id: str, context_data: Dict[str, Any]) -> None:
        """存储长期记忆"""
        self.long_term_memory[user_id] = context_data
        
        if self.redis_client:
            try:
                await self.redis_client.set(
                    f"long_term:{user_id}",
                    json.dumps(context_data, default=str),
                    ex=86400 * 30  # 30天过期
                )
            except Exception as e:
                logger.error(f"存储长期记忆失败: {e}")
    
    async def get_long_term(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取长期记忆"""
        # 优先从内存获取
        if user_id in self.long_term_memory:
            return self.long_term_memory[user_id]
            
        # 从Redis获取
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"long_term:{user_id}")
                if data:
                    context_data = json.loads(data)
                    self.long_term_memory[user_id] = context_data
                    return context_data
            except Exception as e:
                logger.error(f"获取长期记忆失败: {e}")
        
        return None


class ContextEngine:
    """上下文工程主引擎"""
    
    def __init__(self, redis_client=None, max_context_tokens: int = 8192):
        self.context_window = ContextWindow(max_tokens=max_context_tokens)
        self.information_extractor = InformationExtractor()
        self.context_compressor = ContextCompressor()
        self.consistency_checker = ContextConsistencyChecker()
        self.memory_manager = ContextMemoryManager(redis_client)
        
        # 对话状态管理
        self.conversation_states: Dict[str, ConversationState] = {}
        
    async def process_input(self, 
                          conversation_id: str,
                          user_id: str,
                          input_text: str,
                          input_type: str = "user_input") -> ContextChunk:
        """处理用户输入，创建上下文块"""
        
        # 提取信息
        entities = self.information_extractor.extract_entities(input_text)
        travel_entities = self.information_extractor.extract_travel_entities(input_text)
        key_phrases = self.information_extractor.extract_key_phrases(input_text)
        importance_score = self.information_extractor.calculate_importance_score(input_text, input_type)
        
        # 创建上下文块
        chunk = ContextChunk(
            content=input_text,
            chunk_type=input_type,
            timestamp=datetime.now(),
            importance_score=importance_score,
            tokens_count=len(input_text) // 4,  # 估算token数
            entities=entities,
            topic_tags=key_phrases,
            chunk_id=hashlib.md5(f"{conversation_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        )
        
        # 添加到上下文窗口
        self.context_window.add_chunk(chunk)
        
        # 更新对话状态
        await self._update_conversation_state(conversation_id, user_id, travel_entities)
        
        return chunk
    
    async def get_context_for_prompt(self, 
                                   conversation_id: str,
                                   query: str,
                                   max_tokens: int = 4096) -> str:
        """获取用于提示词的上下文"""
        
        # 获取相关的上下文块
        relevant_chunks = self.context_window.get_relevant_chunks(query, max_chunks=10)
        
        # 检查一致性
        consistency_report = self.consistency_checker.check_consistency(relevant_chunks)
        
        # 如果有冲突，尝试解决
        if consistency_report["conflicts"] or consistency_report["contradictions"]:
            relevant_chunks = self.consistency_checker.resolve_conflicts(relevant_chunks, consistency_report)
        
        # 压缩上下文
        if sum(chunk.tokens_count for chunk in relevant_chunks) > max_tokens:
            relevant_chunks = self.context_compressor.compress_chunks(relevant_chunks, max_tokens)
        
        # 构建上下文字符串
        context_parts = []
        
        # 添加对话状态信息
        conv_state = self.conversation_states.get(conversation_id)
        if conv_state:
            state_info = []
            if conv_state.current_topic:
                state_info.append(f"当前话题: {conv_state.current_topic}")
            if conv_state.intent:
                state_info.append(f"用户意图: {conv_state.intent}")
            if conv_state.travel_context:
                travel_info = []
                for key, value in conv_state.travel_context.items():
                    if value:
                        travel_info.append(f"{key}: {value}")
                if travel_info:
                    state_info.append("旅行信息: " + ", ".join(travel_info))
            
            if state_info:
                context_parts.append("对话状态信息:\n" + "\n".join(state_info))
        
        # 添加历史对话
        for chunk in sorted(relevant_chunks, key=lambda x: x.timestamp):
            role = "用户" if chunk.chunk_type == "user_input" else "助手"
            context_parts.append(f"{role}: {chunk.content}")
        
        return "\n\n".join(context_parts)
    
    async def optimize_context_window(self, conversation_id: str) -> None:
        """优化上下文窗口"""
        
        # 获取当前对话的所有块
        conv_chunks = [chunk for chunk in self.context_window.chunks 
                      if chunk.chunk_id.startswith(conversation_id[:8])]
        
        if len(conv_chunks) <= self.context_window.max_chunks:
            return
        
        # 按重要性和时间排序
        sorted_chunks = sorted(conv_chunks, 
                             key=lambda x: (x.importance_score, x.timestamp.timestamp()), 
                             reverse=True)
        
        # 保留最重要的块
        keep_count = self.context_window.max_chunks // 2
        important_chunks = sorted_chunks[:keep_count]
        
        # 保留最近的块
        recent_chunks = sorted(conv_chunks, key=lambda x: x.timestamp)[-keep_count:]
        
        # 合并并去重
        kept_chunks = list({chunk.chunk_id: chunk for chunk in important_chunks + recent_chunks}.values())
        
        # 更新上下文窗口
        self.context_window.chunks = [chunk for chunk in self.context_window.chunks 
                                    if chunk not in conv_chunks or chunk in kept_chunks]
        
        logger.info(f"优化对话 {conversation_id} 上下文窗口: {len(conv_chunks)} -> {len(kept_chunks)}")
    
    async def extract_key_information(self, chunks: List[ContextChunk]) -> Dict[str, Any]:
        """提取关键信息摘要"""
        
        if not chunks:
            return {}
        
        # 合并所有内容
        combined_text = " ".join([chunk.content for chunk in chunks])
        
        # 提取关键实体
        all_entities = []
        travel_info = {
            "destinations": set(),
            "dates": set(),
            "budget": set(),
            "preferences": set()
        }
        
        for chunk in chunks:
            all_entities.extend(chunk.entities)
            
            # 提取旅行信息
            destinations = self.information_extractor.extract_destinations(chunk.content)
            dates = self.information_extractor.extract_dates(chunk.content)
            budget = self.information_extractor.extract_budget_info(chunk.content)
            preferences = self.information_extractor.extract_travel_preferences(chunk.content)
            
            travel_info["destinations"].update(destinations)
            travel_info["dates"].update(dates)
            travel_info["budget"].update(budget)
            travel_info["preferences"].update(preferences)
        
        # 转换为列表
        for key in travel_info:
            travel_info[key] = list(travel_info[key])
        
        # 提取关键主题
        topics = self.information_extractor.extract_topics(combined_text)
        
        return {
            "entities": all_entities,
            "travel_info": travel_info,
            "topics": topics,
            "summary": combined_text[:500] + "..." if len(combined_text) > 500 else combined_text,
            "chunk_count": len(chunks),
            "total_tokens": sum(chunk.tokens_count for chunk in chunks)
        }
    
    async def get_contextual_recommendations(self, conversation_id: str) -> List[Dict[str, Any]]:
        """获取上下文相关的推荐"""
        
        conv_state = self.conversation_states.get(conversation_id)
        if not conv_state:
            return []
        
        recommendations = []
        
        # 基于旅行上下文的推荐
        travel_context = conv_state.travel_context
        
        if "destinations" in travel_context and travel_context["destinations"]:
            destinations = travel_context["destinations"]
            recommendations.append({
                "type": "destination_info",
                "title": f"关于{destinations[0]}的更多信息",
                "content": f"我可以为您提供{destinations[0]}的详细旅行指南、最佳旅行时间、必游景点等信息。",
                "action": f"tell me about {destinations[0]}"
            })
        
        if "dates" in travel_context and travel_context["dates"]:
            recommendations.append({
                "type": "weather_info",
                "title": "天气信息查询",
                "content": "我可以为您查询目的地的天气预报，帮助您准备合适的衣物。",
                "action": "check weather"
            })
        
        if "budget" in travel_context and travel_context["budget"]:
            recommendations.append({
                "type": "budget_planning",
                "title": "预算规划建议",
                "content": "我可以根据您的预算为您推荐性价比最高的旅行方案。",
                "action": "budget planning"
            })
        
        # 基于对话历史的推荐
        if conv_state.current_topic:
            if "flight" in conv_state.current_topic.lower():
                recommendations.append({
                    "type": "flight_booking",
                    "title": "航班预订服务",
                    "content": "我可以帮您搜索并比较不同航班的价格和时间。",
                    "action": "search flights"
                })
            
            elif "hotel" in conv_state.current_topic.lower():
                recommendations.append({
                    "type": "hotel_booking",
                    "title": "酒店预订服务",
                    "content": "我可以为您推荐符合预算和偏好的酒店。",
                    "action": "search hotels"
                })
        
        return recommendations[:3]  # 最多返回3个推荐
    
    async def analyze_conversation_progress(self, conversation_id: str) -> Dict[str, Any]:
        """分析对话进度"""
        
        conv_state = self.conversation_states.get(conversation_id)
        if not conv_state:
            return {}
        
        # 获取对话块
        conv_chunks = [chunk for chunk in self.context_window.chunks 
                      if chunk.chunk_id.startswith(conversation_id[:8])]
        
        # 分析完整性
        required_info = ["destination", "dates", "budget", "preferences"]
        collected_info = []
        missing_info = []
        
        for info_type in required_info:
            if info_type in conv_state.travel_context and conv_state.travel_context[info_type]:
                collected_info.append(info_type)
            else:
                missing_info.append(info_type)
        
        # 计算进度
        progress_percentage = len(collected_info) / len(required_info) * 100
        
        # 分析对话阶段
        stage = "information_gathering"
        if progress_percentage >= 75:
            stage = "planning_ready"
        elif progress_percentage >= 50:
            stage = "partial_info"
        elif progress_percentage >= 25:
            stage = "initial_info"
        
        # 生成建议
        next_steps = []
        if "destination" in missing_info:
            next_steps.append("请告诉我您想去哪里旅行？")
        if "dates" in missing_info:
            next_steps.append("您计划什么时候出行？")
        if "budget" in missing_info:
            next_steps.append("您的预算大概是多少？")
        if "preferences" in missing_info:
            next_steps.append("您有什么特别的偏好或要求吗？")
        
        return {
            "conversation_id": conversation_id,
            "progress_percentage": progress_percentage,
            "stage": stage,
            "collected_info": collected_info,
            "missing_info": missing_info,
            "next_steps": next_steps,
            "total_interactions": len(conv_chunks),
            "conversation_duration": (datetime.now() - conv_state.last_activity).total_seconds() / 60
        }
    
    async def _update_conversation_state(self, 
                                       conversation_id: str,
                                       user_id: str,
                                       travel_entities: Dict[str, List[str]]) -> None:
        """更新对话状态"""
        
        if conversation_id not in self.conversation_states:
            self.conversation_states[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                user_id=user_id
            )
        
        conv_state = self.conversation_states[conversation_id]
        conv_state.last_activity = datetime.now()
        
        # 更新旅行上下文
        for category, values in travel_entities.items():
            if values:
                if category not in conv_state.travel_context:
                    conv_state.travel_context[category] = []
                
                # 添加新的值，避免重复
                existing_values = conv_state.travel_context[category]
                for value in values:
                    if value not in existing_values:
                        existing_values.append(value)
        
        # 存储到记忆系统
        await self.memory_manager.store_short_term(conversation_id, {
            "conversation_state": conv_state.__dict__,
            "context_chunks": [chunk.to_dict() for chunk in self.context_window.chunks[-5:]]  # 最近5个块
        })
    
    async def load_conversation_context(self, conversation_id: str, user_id: str) -> None:
        """加载对话上下文"""
        
        # 从短期记忆加载
        short_term_data = await self.memory_manager.get_short_term(conversation_id)
        if short_term_data:
            if "conversation_state" in short_term_data:
                state_data = short_term_data["conversation_state"]
                state_data["last_activity"] = datetime.fromisoformat(state_data["last_activity"])
                self.conversation_states[conversation_id] = ConversationState(**state_data)
            
            if "context_chunks" in short_term_data:
                for chunk_data in short_term_data["context_chunks"]:
                    chunk = ContextChunk.from_dict(chunk_data)
                    self.context_window.add_chunk(chunk)
        
        # 从长期记忆加载用户偏好
        long_term_data = await self.memory_manager.get_long_term(user_id)
        if long_term_data and conversation_id in self.conversation_states:
            conv_state = self.conversation_states[conversation_id]
            if "preferences" in long_term_data:
                conv_state.preferences.update(long_term_data["preferences"])
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取对话摘要"""
        conv_state = self.conversation_states.get(conversation_id)
        if not conv_state:
            return None
        
        # 统计信息
        chunks = [chunk for chunk in self.context_window.chunks 
                 if chunk.chunk_id.startswith(conversation_id)]
        
        summary = {
            "conversation_id": conversation_id,
            "user_id": conv_state.user_id,
            "current_topic": conv_state.current_topic,
            "intent": conv_state.intent,
            "travel_context": conv_state.travel_context,
            "total_chunks": len(chunks),
            "total_tokens": sum(chunk.tokens_count for chunk in chunks),
            "last_activity": conv_state.last_activity,
            "duration": (datetime.now() - conv_state.last_activity).total_seconds() / 60,  # 分钟
        }
        
        return summary
    
    async def cleanup_expired_conversations(self, max_age_hours: int = 24) -> None:
        """清理过期的对话"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        expired_conversations = []
        for conv_id, conv_state in self.conversation_states.items():
            if conv_state.last_activity < cutoff_time:
                expired_conversations.append(conv_id)
        
        for conv_id in expired_conversations:
            del self.conversation_states[conv_id]
            logger.info(f"清理过期对话: {conv_id}")
        
        # 清理上下文窗口中的旧块
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours * 2)
        self.context_window.chunks = [
            chunk for chunk in self.context_window.chunks
            if chunk.timestamp > cutoff_time
        ]


# 全局实例
context_engine = None

def get_context_engine(redis_client=None) -> ContextEngine:
    """获取上下文引擎实例"""
    global context_engine
    if context_engine is None:
        context_engine = ContextEngine(redis_client=redis_client)
    return context_engine 