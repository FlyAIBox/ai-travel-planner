"""
对话管理器
实现对话状态管理、会话存储、意图识别、实体提取和多轮对话上下文维护
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re

import redis.asyncio as redis
from pydantic import BaseModel, Field
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from context_engine import ContextEngine, get_context_engine

logger = get_logger(__name__)
settings = get_settings()


class ConversationStatus(Enum):
    """对话状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class MessageType(Enum):
    """消息类型枚举"""
    USER_TEXT = "user_text"
    USER_VOICE = "user_voice"
    USER_IMAGE = "user_image"
    AI_RESPONSE = "ai_response"
    SYSTEM_NOTIFICATION = "system_notification"
    TOOL_RESULT = "tool_result"


class IntentType(Enum):
    """意图类型枚举"""
    TRAVEL_PLANNING = "travel_planning"
    FLIGHT_SEARCH = "flight_search"
    HOTEL_SEARCH = "hotel_search"
    ITINERARY_CREATION = "itinerary_creation"
    BUDGET_PLANNING = "budget_planning"
    DESTINATION_INFO = "destination_info"
    WEATHER_INQUIRY = "weather_inquiry"
    BOOKING_ASSISTANCE = "booking_assistance"
    CANCELLATION = "cancellation"
    MODIFICATION = "modification"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """消息数据类"""
    message_id: str
    conversation_id: str
    user_id: str
    content: str
    message_type: MessageType
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建"""
        data['message_type'] = MessageType(data['message_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationSession:
    """对话会话数据类"""
    conversation_id: str
    user_id: str
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = None
    message_count: int = 0
    current_intent: Optional[IntentType] = None
    context_summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.context_summary is None:
            self.context_summary = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.current_intent:
            data['current_intent'] = self.current_intent.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """从字典创建"""
        data['status'] = ConversationStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('current_intent'):
            data['current_intent'] = IntentType(data['current_intent'])
        return cls(**data)


class NLUProcessor:
    """自然语言理解处理器"""
    
    def __init__(self):
        # 加载spaCy模型
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            logger.warning("中文spaCy模型未找到，使用英文模型")
            self.nlp = spacy.load("en_core_web_sm")
        
        # 意图识别规则
        self.intent_patterns = {
            IntentType.TRAVEL_PLANNING: [
                r'(?:计划|规划|安排).*?(?:旅行|旅游|出行|度假)',
                r'(?:想去|要去|准备去|打算去)',
                r'(?:制定|设计).*?(?:行程|路线)',
                r'(?:旅行|旅游).*?(?:计划|方案|建议)'
            ],
            IntentType.FLIGHT_SEARCH: [
                r'(?:机票|航班|飞机票)',
                r'(?:订|买|预定).*?(?:机票|航班)',
                r'(?:查询|搜索|找).*?(?:航班|机票)',
                r'(?:飞|坐飞机).*?(?:去|到)'
            ],
            IntentType.HOTEL_SEARCH: [
                r'(?:酒店|宾馆|住宿|民宿)',
                r'(?:订|预定|预订).*?(?:酒店|房间)',
                r'(?:查询|搜索|找).*?(?:酒店|住宿)',
                r'(?:住|入住).*?(?:哪里|哪个)'
            ],
            IntentType.ITINERARY_CREATION: [
                r'(?:行程|路线|日程).*?(?:安排|规划|制定)',
                r'(?:景点|地方).*?(?:推荐|建议)',
                r'(?:怎么|如何).*?(?:安排|规划).*?(?:行程|时间)',
                r'(?:第.*?天|.*?日).*?(?:去|游览|参观)'
            ],
            IntentType.BUDGET_PLANNING: [
                r'(?:预算|花费|费用|价格|钱)',
                r'(?:多少钱|成本|开销)',
                r'(?:便宜|省钱|经济)',
                r'(?:大概|大约).*?(?:需要|花费|费用)'
            ],
            IntentType.DESTINATION_INFO: [
                r'(?:介绍|了解).*?(?:地方|城市|景点)',
                r'(?:有什么|什么).*?(?:好玩|有趣|值得)',
                r'(?:天气|气候|温度)',
                r'(?:文化|历史|特色|美食)'
            ],
            IntentType.WEATHER_INQUIRY: [
                r'(?:天气|气温|温度|下雨|晴天|阴天)',
                r'(?:气候|气象|天气情况)',
                r'(?:冷|热|暖和|凉快)',
                r'(?:雨季|旱季|季节)'
            ],
            IntentType.BOOKING_ASSISTANCE: [
                r'(?:预定|预订|订购|购买)',
                r'(?:帮助|协助).*?(?:预定|预订)',
                r'(?:如何|怎么).*?(?:预定|预订|购买)',
                r'(?:联系|电话|网站).*?(?:预定|预订)'
            ],
            IntentType.CANCELLATION: [
                r'(?:取消|退订|退票|退款)',
                r'(?:不去了|不要了|改变主意)',
                r'(?:如何|怎么).*?(?:取消|退订)',
                r'(?:退|返回).*?(?:钱|费用)'
            ],
            IntentType.MODIFICATION: [
                r'(?:修改|更改|调整|变更)',
                r'(?:换|改).*?(?:时间|日期|地点)',
                r'(?:重新|再次).*?(?:安排|规划)',
                r'(?:更新|升级).*?(?:计划|行程)'
            ]
        }
        
        # 实体提取模式
        self.entity_patterns = {
            "date": [
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{1,2}月\d{1,2}日)',
                r'(明天|后天|大后天)',
                r'(今天|昨天|前天)',
                r'(下周|下个月|下季度)',
                r'(春节|国庆|中秋|端午|清明)',
                r'(\d{1,2}号|\d{1,2}日)',
                r'(周一|周二|周三|周四|周五|周六|周日|星期一|星期二|星期三|星期四|星期五|星期六|星期日)'
            ],
            "location": [
                r'([^\s，。！？]+?(?:市|省|县|区|国|地区|景区|景点|机场|车站))',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 英文地名
                r'([^\s，。！？]*?(?:古城|老街|广场|公园|寺庙|山|海|湖|江|河))',
            ],
            "person_count": [
                r'(\d+)(?:人|个人|位)',
                r'(一个人|两个人|三个人|四个人|五个人|六个人|七个人|八个人|九个人|十个人)',
                r'(单人|双人|三人|四人|多人|家庭|情侣|朋友)'
            ],
            "duration": [
                r'(\d+)(?:天|日|周|星期|个月|月)',
                r'(一天|两天|三天|四天|五天|六天|七天|一周|两周|一个月|两个月)',
                r'(短期|长期|周末|假期|长假|短假)'
            ],
            "budget": [
                r'(\d+)(?:元|块|万|千|百)',
                r'(便宜|经济|中等|高端|豪华)',
                r'(预算.*?(\d+))',
                r'(大概.*?(\d+).*?(?:元|块|万|千))'
            ],
            "transport": [
                r'(飞机|火车|高铁|汽车|大巴|自驾|地铁|公交|出租车|网约车)',
                r'(坐.*?(飞机|火车|高铁|汽车|大巴))',
                r'(开车|自驾游|租车)'
            ],
            "accommodation": [
                r'(酒店|宾馆|民宿|青旅|客栈|度假村|公寓)',
                r'(住.*?(酒店|宾馆|民宿|青旅|客栈))',
                r'(五星|四星|三星|经济型|豪华|标准)'
            ]
        }
    
    def extract_intent(self, text: str) -> Tuple[IntentType, float]:
        """提取用户意图"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text):
                    matches += 1
                    score += 1
            
            if matches > 0:
                # 计算置信度分数
                confidence = min(score / len(patterns), 1.0)
                intent_scores[intent] = confidence
        
        if intent_scores:
            # 返回置信度最高的意图
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0], best_intent[1]
        else:
            return IntentType.UNKNOWN, 0.0
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取命名实体"""
        entities = {}
        
        # 使用规则提取实体
        for entity_type, patterns in self.entity_patterns.items():
            entity_values = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    if isinstance(matches[0], tuple):
                        # 处理带分组的匹配
                        for match in matches:
                            entity_values.extend([m for m in match if m])
                    else:
                        entity_values.extend(matches)
            
            if entity_values:
                entities[entity_type] = list(set(entity_values))  # 去重
        
        # 使用spaCy提取补充实体
        doc = self.nlp(text)
        spacy_entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_.lower()
            if entity_type not in spacy_entities:
                spacy_entities[entity_type] = []
            spacy_entities[entity_type].append(ent.text)
        
        # 合并实体
        for entity_type, values in spacy_entities.items():
            if entity_type in entities:
                entities[entity_type].extend(values)
                entities[entity_type] = list(set(entities[entity_type]))  # 去重
            else:
                entities[entity_type] = values
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """分析情感"""
        # 简单的规则基础情感分析
        positive_words = ['喜欢', '好', '棒', '不错', '满意', '开心', '高兴', '期待', '想要']
        negative_words = ['不喜欢', '不好', '差', '不满意', '失望', '担心', '不想', '烦']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive", min(positive_count / (positive_count + negative_count + 1), 1.0)
        elif negative_count > positive_count:
            return "negative", min(negative_count / (positive_count + negative_count + 1), 1.0)
        else:
            return "neutral", 0.5
    
    def process_message(self, text: str) -> Dict[str, Any]:
        """处理消息，返回NLU结果"""
        intent, intent_confidence = self.extract_intent(text)
        entities = self.extract_entities(text)
        sentiment, sentiment_confidence = self.analyze_sentiment(text)
        
        return {
            "intent": {
                "type": intent,
                "confidence": intent_confidence
            },
            "entities": entities,
            "sentiment": {
                "label": sentiment,
                "confidence": sentiment_confidence
            },
            "text_length": len(text),
            "word_count": len(text.split())
        }


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or self._create_redis_client()
        self.nlu_processor = NLUProcessor()
        self.context_engine = get_context_engine(self.redis_client)
        
        # 内存缓存
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.session_timeout = timedelta(hours=24)  # 会话超时时间
        
    def _create_redis_client(self) -> redis.Redis:
        """创建Redis客户端"""
        try:
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            return client
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            return None
    
    async def create_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """创建新对话"""
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ConversationSession(
            conversation_id=conversation_id,
            user_id=user_id,
            status=ConversationStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        # 存储到内存缓存
        self.active_sessions[conversation_id] = session
        
        # 存储到Redis
        await self._save_session_to_redis(session)
        
        # 加载用户上下文
        await self.context_engine.load_conversation_context(conversation_id, user_id)
        
        logger.info(f"创建新对话: {conversation_id}, 用户: {user_id}")
        return conversation_id
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationSession]:
        """获取对话会话"""
        # 优先从内存缓存获取
        if conversation_id in self.active_sessions:
            session = self.active_sessions[conversation_id]
            
            # 检查是否过期
            if datetime.now() - session.updated_at > self.session_timeout:
                await self.end_conversation(conversation_id)
                return None
                
            return session
        
        # 从Redis获取
        session = await self._load_session_from_redis(conversation_id)
        if session:
            # 检查是否过期
            if datetime.now() - session.updated_at > self.session_timeout:
                await self.end_conversation(conversation_id)
                return None
            
            # 加载到内存缓存
            self.active_sessions[conversation_id] = session
            return session
        
        return None
    
    async def add_message(self, 
                         conversation_id: str,
                         content: str,
                         message_type: MessageType,
                         user_id: str,
                         metadata: Dict[str, Any] = None) -> Message:
        """添加消息到对话"""
        
        # 获取或创建对话会话
        session = await self.get_conversation(conversation_id)
        if not session:
            # 如果会话不存在，创建新会话
            await self.create_conversation(user_id)
            session = await self.get_conversation(conversation_id)
            if not session:
                raise ValueError(f"无法创建或获取对话会话: {conversation_id}")
        
        # 创建消息
        message = Message(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_id=user_id,
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # 如果是用户消息，进行NLU处理
        if message_type == MessageType.USER_TEXT:
            nlu_result = self.nlu_processor.process_message(content)
            message.metadata.update(nlu_result)
            
            # 更新会话状态
            if nlu_result["intent"]["confidence"] > 0.5:
                session.current_intent = nlu_result["intent"]["type"]
        
        # 更新会话
        session.message_count += 1
        session.updated_at = datetime.now()
        
        # 保存会话状态
        await self._save_session_to_redis(session)
        
        # 保存消息
        await self._save_message_to_redis(message)
        
        # 添加到上下文引擎
        chunk_type = "user_input" if message_type == MessageType.USER_TEXT else "ai_response"
        await self.context_engine.process_input(
            conversation_id=conversation_id,
            user_id=user_id,
            input_text=content,
            input_type=chunk_type
        )
        
        logger.info(f"添加消息到对话 {conversation_id}: {message.message_id}")
        return message
    
    async def get_conversation_history(self, 
                                     conversation_id: str,
                                     limit: int = 50,
                                     offset: int = 0) -> List[Message]:
        """获取对话历史"""
        try:
            # 从Redis获取消息列表
            messages_data = await self.redis_client.lrange(
                f"conversation_messages:{conversation_id}",
                offset,
                offset + limit - 1
            )
            
            messages = []
            for message_data in messages_data:
                message_dict = json.loads(message_data)
                message = Message.from_dict(message_dict)
                messages.append(message)
            
            # 按时间戳排序
            messages.sort(key=lambda x: x.timestamp)
            return messages
            
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    async def get_context_for_ai(self, 
                                conversation_id: str,
                                query: str,
                                max_tokens: int = 4096) -> str:
        """获取用于AI的上下文"""
        return await self.context_engine.get_context_for_prompt(
            conversation_id=conversation_id,
            query=query,
            max_tokens=max_tokens
        )
    
    async def update_conversation_metadata(self, 
                                         conversation_id: str,
                                         metadata: Dict[str, Any]) -> bool:
        """更新对话元数据"""
        session = await self.get_conversation(conversation_id)
        if not session:
            return False
        
        session.metadata.update(metadata)
        session.updated_at = datetime.now()
        
        await self._save_session_to_redis(session)
        return True
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """结束对话"""
        session = await self.get_conversation(conversation_id)
        if not session:
            return False
        
        session.status = ConversationStatus.ENDED
        session.updated_at = datetime.now()
        
        # 保存最终状态
        await self._save_session_to_redis(session)
        
        # 从内存缓存中移除
        if conversation_id in self.active_sessions:
            del self.active_sessions[conversation_id]
        
        logger.info(f"结束对话: {conversation_id}")
        return True
    
    async def pause_conversation(self, conversation_id: str) -> bool:
        """暂停对话"""
        session = await self.get_conversation(conversation_id)
        if not session:
            return False
        
        session.status = ConversationStatus.PAUSED
        session.updated_at = datetime.now()
        
        await self._save_session_to_redis(session)
        return True
    
    async def resume_conversation(self, conversation_id: str) -> bool:
        """恢复对话"""
        session = await self.get_conversation(conversation_id)
        if not session:
            return False
        
        session.status = ConversationStatus.ACTIVE
        session.updated_at = datetime.now()
        
        await self._save_session_to_redis(session)
        return True
    
    async def get_user_conversations(self, 
                                   user_id: str,
                                   status: Optional[ConversationStatus] = None,
                                   limit: int = 20) -> List[ConversationSession]:
        """获取用户的对话列表"""
        try:
            # 从Redis获取用户对话ID列表
            pattern = f"session:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            sessions = []
            for key in keys[:limit]:
                session_data = await self.redis_client.get(key)
                if session_data:
                    session_dict = json.loads(session_data)
                    session = ConversationSession.from_dict(session_dict)
                    
                    if status is None or session.status == status:
                        sessions.append(session)
            
            # 按更新时间排序
            sessions.sort(key=lambda x: x.updated_at, reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"获取用户对话列表失败: {e}")
            return []
    
    async def cleanup_expired_conversations(self) -> None:
        """清理过期对话"""
        expired_conversations = []
        cutoff_time = datetime.now() - self.session_timeout
        
        for conversation_id, session in self.active_sessions.items():
            if session.updated_at < cutoff_time:
                expired_conversations.append(conversation_id)
        
        for conversation_id in expired_conversations:
            await self.end_conversation(conversation_id)
        
        # 清理上下文引擎中的过期对话
        await self.context_engine.cleanup_expired_conversations(max_age_hours=24)
        
        logger.info(f"清理了 {len(expired_conversations)} 个过期对话")
    
    async def get_conversation_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话分析数据"""
        session = await self.get_conversation(conversation_id)
        if not session:
            return {}
        
        messages = await self.get_conversation_history(conversation_id, limit=1000)
        
        # 统计分析
        user_messages = [m for m in messages if m.message_type == MessageType.USER_TEXT]
        ai_messages = [m for m in messages if m.message_type == MessageType.AI_RESPONSE]
        
        intent_distribution = {}
        entity_distribution = {}
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        
        for message in user_messages:
            if "intent" in message.metadata:
                intent = message.metadata["intent"]["type"].value
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
            
            if "entities" in message.metadata:
                for entity_type, values in message.metadata["entities"].items():
                    entity_distribution[entity_type] = entity_distribution.get(entity_type, 0) + len(values)
            
            if "sentiment" in message.metadata:
                sentiment = message.metadata["sentiment"]["label"]
                sentiment_distribution[sentiment] += 1
        
        # 获取上下文摘要
        context_summary = self.context_engine.get_conversation_summary(conversation_id)
        
        analytics = {
            "conversation_id": conversation_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "duration_minutes": (session.updated_at - session.created_at).total_seconds() / 60,
            "current_intent": session.current_intent.value if session.current_intent else None,
            "intent_distribution": intent_distribution,
            "entity_distribution": entity_distribution,
            "sentiment_distribution": sentiment_distribution,
            "context_summary": context_summary,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
        
        return analytics
    
    # Redis操作方法
    async def _save_session_to_redis(self, session: ConversationSession) -> None:
        """保存会话到Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"session:{session.user_id}:{session.conversation_id}"
            data = json.dumps(session.to_dict())
            await self.redis_client.set(key, data, ex=86400 * 7)  # 7天过期
        except Exception as e:
            logger.error(f"保存会话到Redis失败: {e}")
    
    async def _load_session_from_redis(self, conversation_id: str) -> Optional[ConversationSession]:
        """从Redis加载会话"""
        if not self.redis_client:
            return None
        
        try:
            # 查找会话键
            pattern = f"session:*:{conversation_id}"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                session_data = await self.redis_client.get(keys[0])
                if session_data:
                    session_dict = json.loads(session_data)
                    return ConversationSession.from_dict(session_dict)
        except Exception as e:
            logger.error(f"从Redis加载会话失败: {e}")
        
        return None
    
    async def _save_message_to_redis(self, message: Message) -> None:
        """保存消息到Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"conversation_messages:{message.conversation_id}"
            data = json.dumps(message.to_dict())
            await self.redis_client.lpush(key, data)
            await self.redis_client.expire(key, 86400 * 7)  # 7天过期
        except Exception as e:
            logger.error(f"保存消息到Redis失败: {e}")


# 全局实例
conversation_manager = None

def get_conversation_manager(redis_client=None) -> ConversationManager:
    """获取对话管理器实例"""
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager(redis_client=redis_client)
    return conversation_manager 