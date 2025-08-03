"""
WebSocket连接管理器
实现WebSocket连接管理、消息路由、实时流式响应和连接状态监控
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket
import redis.asyncio as redis
from pydantic import BaseModel, Field

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from conversation_manager import ConversationManager, MessageType

logger = get_logger(__name__)
settings = get_settings()


class ConnectionStatus(Enum):
    """连接状态枚举"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageTypeWS(Enum):
    """WebSocket消息类型"""
    TEXT = "text"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    TYPING = "typing"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketConnection:
    """WebSocket连接信息"""
    connection_id: str
    user_id: str
    conversation_id: Optional[str]
    websocket: WebSocket
    connected_at: datetime
    last_activity: datetime
    status: ConnectionStatus
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['connected_at'] = self.connected_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        data['status'] = self.status.value
        # 移除websocket对象，因为不能序列化
        del data['websocket']
        return data


class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    type: str = Field(..., description="消息类型")
    content: Any = Field(None, description="消息内容")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="时间戳")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self, conversation_manager: ConversationManager = None):
        self.conversation_manager = conversation_manager
        self.redis_client = self._create_redis_client()
        
        # 连接管理
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.conversation_connections: Dict[str, Set[str]] = {}  # conversation_id -> connection_ids
        
        # 消息队列和处理
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_processing = False
        
        # 心跳和清理任务
        self._heartbeat_task = None
        self._cleanup_task = None
    
    def _create_redis_client(self) -> redis.Redis:
        """创建Redis客户端"""
        try:
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                db=settings.REDIS_DB_SESSION,
                decode_responses=True
            )
            # 测试连接
            client.ping()
            return client
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            return None

    def start(self):
        """启动WebSocket管理器"""
        logger.info("WebSocket管理器已启动")
        # 这里可以添加启动时的初始化逻辑
        pass

    def stop(self):
        """停止WebSocket管理器"""
        logger.info("WebSocket管理器已停止")
        # 这里可以添加停止时的清理逻辑
        pass

    async def start_background_tasks(self):
        """启动后台任务"""
        if not self._heartbeat_task:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if not self.is_processing:
            asyncio.create_task(self._process_message_queue())
    
    async def connect(self, 
                     websocket: WebSocket, 
                     user_id: str, 
                     conversation_id: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """建立WebSocket连接"""
        
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        now = datetime.now()

        connection = WebSocketConnection(
            connection_id=connection_id,
            user_id=user_id,
            conversation_id=conversation_id,
            websocket=websocket,
            connected_at=now,
            last_activity=now,
                status=ConnectionStatus.CONNECTED,
            metadata=metadata or {}
        )
        
        # 存储连接
        self.active_connections[connection_id] = connection
        
        # 用户连接映射
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
            
        # 对话连接映射
        if conversation_id:
            if conversation_id not in self.conversation_connections:
                self.conversation_connections[conversation_id] = set()
            self.conversation_connections[conversation_id].add(connection_id)
            
        # 记录连接信息到Redis
        await self._save_connection_to_redis(connection)
            
            # 发送连接成功消息
        await self.send_to_connection(connection_id, {
            "type": MessageTypeWS.NOTIFICATION.value,
            "content": {
                "message": "连接成功",
                "connection_id": connection_id,
                "user_id": user_id,
                "conversation_id": conversation_id
            }
        })

        logger.info(f"WebSocket连接建立: {connection_id}, 用户: {user_id}")
        return connection_id
            
    async def disconnect(self, connection_id: str, reason: str = "unknown"):
        """断开WebSocket连接"""
        
        connection = self.active_connections.get(connection_id)
        if not connection:
            return
        
        try:
            # 发送断开通知
            await self.send_to_connection(connection_id, {
                "type": MessageTypeWS.NOTIFICATION.value,
                "content": {
                    "message": "连接断开",
                    "reason": reason
                }
            })
        except:
            pass  # 连接可能已经断开
        
        # 更新连接状态
        connection.status = ConnectionStatus.DISCONNECTED
        connection.last_activity = datetime.now()
        
        # 从映射中移除
        if connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        if connection.conversation_id and connection.conversation_id in self.conversation_connections:
            self.conversation_connections[connection.conversation_id].discard(connection_id)
            if not self.conversation_connections[connection.conversation_id]:
                del self.conversation_connections[connection.conversation_id]
        
        # 从活跃连接中移除
        del self.active_connections[connection_id]
        
        # 更新Redis状态
        await self._save_connection_to_redis(connection)
        
        logger.info(f"WebSocket连接断开: {connection_id}, 原因: {reason}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """向指定连接发送消息"""
        
        connection = self.active_connections.get(connection_id)
        if not connection or connection.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            # 添加时间戳
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            
            await connection.websocket.send_text(json.dumps(message, ensure_ascii=False))
            connection.last_activity = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"发送消息失败 {connection_id}: {e}")
            # 标记连接为错误状态
            connection.status = ConnectionStatus.ERROR
            await self.disconnect(connection_id, f"send_error: {str(e)}")
            return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """向用户的所有连接发送消息"""
        
        connection_ids = self.user_connections.get(user_id, set())
        success_count = 0
        
        for connection_id in connection_ids.copy():
            if await self.send_to_connection(connection_id, message):
                success_count += 1
        
        return success_count
    
    async def send_to_conversation(self, conversation_id: str, message: Dict[str, Any]) -> int:
        """向对话的所有连接发送消息"""
        
        connection_ids = self.conversation_connections.get(conversation_id, set())
        success_count = 0
        
        for connection_id in connection_ids.copy():
            if await self.send_to_connection(connection_id, message):
                success_count += 1
        
        return success_count
    
    async def broadcast(self, message: Dict[str, Any], exclude_connections: Set[str] = None) -> int:
        """广播消息给所有连接"""
        
        exclude_connections = exclude_connections or set()
        success_count = 0
        
        for connection_id in list(self.active_connections.keys()):
            if connection_id not in exclude_connections:
                if await self.send_to_connection(connection_id, message):
                    success_count += 1
        
        return success_count
    
    async def send_stream_message(self, 
                                connection_id: str, 
                                stream_id: str,
                                chunk: str, 
                                is_final: bool = False) -> bool:
        """发送流式消息块"""
        
        if is_final:
            message = {
                "type": MessageTypeWS.STREAM_END.value,
                "content": {
                    "stream_id": stream_id,
                    "final_chunk": chunk
                }
            }
        else:
            message = {
                "type": MessageTypeWS.STREAM_CHUNK.value,
                "content": {
            "stream_id": stream_id,
                    "chunk": chunk
                }
            }
        
        return await self.send_to_connection(connection_id, message)
    
    async def start_stream(self, connection_id: str, stream_id: str, metadata: Dict[str, Any] = None) -> bool:
        """开始流式传输"""
        
        message = {
            "type": MessageTypeWS.STREAM_START.value,
            "content": {
                "stream_id": stream_id,
                "metadata": metadata or {}
            }
        }
        
        return await self.send_to_connection(connection_id, message)
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """处理接收到的消息"""
        
        connection = self.active_connections.get(connection_id)
        if not connection:
            logger.warning(f"收到未知连接的消息: {connection_id}")
            return
        
        try:
            message_data = json.loads(raw_message)
            message = WebSocketMessage(**message_data)
            
            # 更新活动时间
            connection.last_activity = datetime.now()
            
            # 处理不同类型的消息
            if message.type == MessageTypeWS.TEXT.value:
                await self._handle_text_message(connection, message)
            elif message.type == MessageTypeWS.HEARTBEAT.value:
                await self._handle_heartbeat(connection, message)
            elif message.type == MessageTypeWS.TYPING.value:
                await self._handle_typing_message(connection, message)
            else:
                logger.warning(f"未知消息类型: {message.type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"消息JSON解析错误 {connection_id}: {e}")
            await self.send_to_connection(connection_id, {
                "type": MessageTypeWS.ERROR.value,
                "content": {"error": "消息格式错误"}
            })
        except Exception as e:
            logger.error(f"处理消息错误 {connection_id}: {e}")
            await self.send_to_connection(connection_id, {
                "type": MessageTypeWS.ERROR.value,
                "content": {"error": f"消息处理错误: {str(e)}"}
            })
    
    async def _handle_text_message(self, connection: WebSocketConnection, message: WebSocketMessage):
        """处理文本消息"""
        
        if not self.conversation_manager:
            logger.error("ConversationManager未初始化")
            return
            
            # 添加消息到对话管理器
        if message.conversation_id:
            try:
                saved_message = await self.conversation_manager.add_message(
                    conversation_id=message.conversation_id,
                    content=message.content,
                    message_type=MessageType.USER_TEXT,
                    user_id=connection.user_id,
                    metadata=message.metadata
                )
                
                # 通知其他连接用户发送了消息
                await self.send_to_conversation(message.conversation_id, {
                    "type": MessageTypeWS.NOTIFICATION.value,
                    "content": {
                        "user_id": connection.user_id,
                        "message": "用户发送了消息",
                        "message_id": saved_message.message_id
                    }
                }, exclude_connections={connection.connection_id})
                
            except Exception as e:
                logger.error(f"保存消息失败: {e}")
                await self.send_to_connection(connection.connection_id, {
                    "type": MessageTypeWS.ERROR.value,
                    "content": {"error": "消息保存失败"}
                })
    
    async def _handle_heartbeat(self, connection: WebSocketConnection, message: WebSocketMessage):
        """处理心跳消息"""
        _ = message  # 占位符，避免未使用参数警告

        await self.send_to_connection(connection.connection_id, {
            "type": MessageTypeWS.HEARTBEAT.value,
            "content": {"status": "alive", "server_time": datetime.now().isoformat()}
        })
    
    async def _handle_typing_message(self, connection: WebSocketConnection, message: WebSocketMessage):
        """处理打字状态消息"""
        
        if message.conversation_id:
            # 转发打字状态给对话中的其他用户
            await self.send_to_conversation(message.conversation_id, {
                "type": MessageTypeWS.TYPING.value,
                "content": {
                    "user_id": connection.user_id,
                    "is_typing": message.content.get("is_typing", False)
                }
            }, exclude_connections={connection.connection_id})
    
    async def _heartbeat_loop(self):
        """心跳检测循环"""
        
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(minutes=5)  # 5分钟超时
                
                timeout_connections = []
                for connection_id, connection in self.active_connections.items():
                    if connection.last_activity < timeout_threshold:
                        timeout_connections.append(connection_id)
                
                # 断开超时连接
                for connection_id in timeout_connections:
                    await self.disconnect(connection_id, "timeout")
                
                if timeout_connections:
                    logger.info(f"清理超时连接: {len(timeout_connections)}")
                    
            except Exception as e:
                logger.error(f"心跳检测错误: {e}")
    
    async def _cleanup_loop(self):
        """清理循环"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                
                # 清理Redis中的过期连接记录
                if self.redis_client:
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    # TODO: 添加Redis清理逻辑
                    _ = cutoff_time  # 占位符，避免未使用变量警告
                    
            except Exception as e:
                logger.error(f"清理任务错误: {e}")
    
    async def _process_message_queue(self):
        """处理消息队列"""
        
        self.is_processing = True
        try:
            while True:
                # 这里可以实现消息队列处理逻辑
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"消息队列处理错误: {e}")
        finally:
            self.is_processing = False
    
    async def _save_connection_to_redis(self, connection: WebSocketConnection):
        """保存连接信息到Redis"""

        if not self.redis_client:
            return

        try:
            key = f"websocket_connection:{connection.connection_id}"
            data = json.dumps(connection.to_dict())
            # 使用同步方法，因为 self.redis_client 是同步客户端
            self.redis_client.setex(key, 86400, data)  # 24小时过期
        except Exception as e:
            logger.error(f"保存连接信息到Redis失败: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        
        return {
            "total_connections": len(self.active_connections),
            "users_online": len(self.user_connections),
            "active_conversations": len(self.conversation_connections),
            "connections_by_status": {
                status.value: sum(1 for conn in self.active_connections.values() 
                                if conn.status == status)
                for status in ConnectionStatus
            }
        }
    
    async def shutdown(self):
        """关闭WebSocket管理器"""
        
        # 取消后台任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # 断开所有连接
        for connection_id in list(self.active_connections.keys()):
            await self.disconnect(connection_id, "server_shutdown")
        
        # 关闭Redis连接
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("WebSocket管理器已关闭")


# 全局WebSocket管理器实例
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager(conversation_manager: ConversationManager = None) -> WebSocketManager:
    """获取WebSocket管理器实例"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager(conversation_manager)
    return _websocket_manager 