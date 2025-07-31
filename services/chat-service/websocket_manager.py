"""
WebSocket管理器
实现WebSocket连接管理、消息路由、实时流式响应处理、连接状态监控和消息队列处理
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import structlog

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .conversation_manager import get_conversation_manager, MessageType

logger = get_logger(__name__)
settings = get_settings()


class ConnectionStatus(Enum):
    """连接状态枚举"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class MessageCategory(Enum):
    """消息类别枚举"""
    CHAT = "chat"
    SYSTEM = "system"
    HEARTBEAT = "heartbeat"
    STREAM = "stream"
    ERROR = "error"
    NOTIFICATION = "notification"


@dataclass
class WebSocketConnection:
    """WebSocket连接信息"""
    connection_id: str
    user_id: str
    conversation_id: Optional[str]
    websocket: WebSocket
    status: ConnectionStatus
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    message_id: str
    connection_id: str
    category: MessageCategory
    message_type: str
    data: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "connection_id": self.connection_id,
            "category": self.category.value,
            "type": self.message_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class ConnectionPool:
    """连接池管理器"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.conversation_connections: Dict[str, Set[str]] = {}  # conversation_id -> connection_ids
        
    def add_connection(self, connection: WebSocketConnection) -> None:
        """添加连接"""
        self.connections[connection.connection_id] = connection
        
        # 添加到用户连接映射
        if connection.user_id not in self.user_connections:
            self.user_connections[connection.user_id] = set()
        self.user_connections[connection.user_id].add(connection.connection_id)
        
        # 添加到对话连接映射
        if connection.conversation_id:
            if connection.conversation_id not in self.conversation_connections:
                self.conversation_connections[connection.conversation_id] = set()
            self.conversation_connections[connection.conversation_id].add(connection.connection_id)
    
    def remove_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """移除连接"""
        connection = self.connections.pop(connection_id, None)
        if not connection:
            return None
        
        # 从用户连接映射中移除
        if connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # 从对话连接映射中移除
        if connection.conversation_id and connection.conversation_id in self.conversation_connections:
            self.conversation_connections[connection.conversation_id].discard(connection_id)
            if not self.conversation_connections[connection.conversation_id]:
                del self.conversation_connections[connection.conversation_id]
        
        return connection
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """获取连接"""
        return self.connections.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """获取用户的所有连接"""
        connection_ids = self.user_connections.get(user_id, set())
        return [self.connections[conn_id] for conn_id in connection_ids if conn_id in self.connections]
    
    def get_conversation_connections(self, conversation_id: str) -> List[WebSocketConnection]:
        """获取对话的所有连接"""
        connection_ids = self.conversation_connections.get(conversation_id, set())
        return [self.connections[conn_id] for conn_id in connection_ids if conn_id in self.connections]
    
    def get_all_connections(self) -> List[WebSocketConnection]:
        """获取所有连接"""
        return list(self.connections.values())
    
    def update_connection_activity(self, connection_id: str) -> None:
        """更新连接活动时间"""
        if connection_id in self.connections:
            self.connections[connection_id].last_activity = datetime.now()


class MessageRouter:
    """消息路由器"""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """注册消息处理器"""
        self.handlers[message_type] = handler
        logger.info(f"注册消息处理器: {message_type}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self.middleware.append(middleware)
    
    async def route_message(self, connection: WebSocketConnection, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """路由消息"""
        message_type = message.get("type")
        if not message_type:
            return {"error": "缺少消息类型"}
        
        # 执行中间件
        for middleware in self.middleware:
            try:
                message = await middleware(connection, message)
                if message is None:
                    return None  # 中间件拦截了消息
            except Exception as e:
                logger.error(f"中间件执行失败: {e}")
                return {"error": "消息处理失败"}
        
        # 查找处理器
        handler = self.handlers.get(message_type)
        if not handler:
            return {"error": f"未知消息类型: {message_type}"}
        
        try:
            # 执行处理器
            result = await handler(connection, message)
            return result
        except Exception as e:
            logger.error(f"消息处理器执行失败: {e}")
            return {"error": "消息处理失败"}


class StreamingManager:
    """流式响应管理器"""
    
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    def start_stream(self, stream_id: str, connection_id: str, metadata: Dict[str, Any] = None) -> None:
        """开始流式响应"""
        self.active_streams[stream_id] = {
            "connection_id": connection_id,
            "started_at": datetime.now(),
            "metadata": metadata or {},
            "chunk_count": 0
        }
    
    def end_stream(self, stream_id: str) -> None:
        """结束流式响应"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
    
    def is_stream_active(self, stream_id: str) -> bool:
        """检查流是否活跃"""
        return stream_id in self.active_streams
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流信息"""
        return self.active_streams.get(stream_id)
    
    def update_stream_progress(self, stream_id: str, chunk_data: Any) -> None:
        """更新流进度"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["chunk_count"] += 1
            self.active_streams[stream_id]["last_chunk_at"] = datetime.now()


class HeartbeatManager:
    """心跳管理器"""
    
    def __init__(self, interval: int = 30, timeout: int = 60):
        self.interval = interval  # 心跳间隔（秒）
        self.timeout = timeout    # 超时时间（秒）
        self.connection_heartbeats: Dict[str, datetime] = {}
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    def start(self) -> None:
        """启动心跳检查"""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._heartbeat_loop())
            logger.info("心跳管理器已启动")
    
    def stop(self) -> None:
        """停止心跳检查"""
        self.running = False
        if self.task:
            self.task.cancel()
            logger.info("心跳管理器已停止")
    
    def update_heartbeat(self, connection_id: str) -> None:
        """更新连接心跳"""
        self.connection_heartbeats[connection_id] = datetime.now()
    
    def remove_connection(self, connection_id: str) -> None:
        """移除连接心跳"""
        self.connection_heartbeats.pop(connection_id, None)
    
    def is_connection_alive(self, connection_id: str) -> bool:
        """检查连接是否存活"""
        last_heartbeat = self.connection_heartbeats.get(connection_id)
        if not last_heartbeat:
            return False
        
        return (datetime.now() - last_heartbeat).total_seconds() < self.timeout
    
    async def _heartbeat_loop(self) -> None:
        """心跳检查循环"""
        while self.running:
            try:
                await asyncio.sleep(self.interval)
                
                current_time = datetime.now()
                dead_connections = []
                
                for connection_id, last_heartbeat in self.connection_heartbeats.items():
                    if (current_time - last_heartbeat).total_seconds() > self.timeout:
                        dead_connections.append(connection_id)
                
                # 通知死连接
                for connection_id in dead_connections:
                    logger.warning(f"检测到死连接: {connection_id}")
                    # 这里可以添加死连接处理逻辑
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳检查异常: {e}")


class WebSocketManager:
    """WebSocket管理器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or self._create_redis_client()
        self.conversation_manager = get_conversation_manager(self.redis_client)
        
        # 组件初始化
        self.connection_pool = ConnectionPool()
        self.message_router = MessageRouter()
        self.streaming_manager = StreamingManager()
        self.heartbeat_manager = HeartbeatManager()
        
        # 消息队列
        self.message_queue = asyncio.Queue()
        self.broadcast_queue = asyncio.Queue()
        
        # 统计信息
        self.stats = {
            "total_connections": 0,
            "total_messages": 0,
            "total_errors": 0,
            "start_time": datetime.now()
        }
        
        # 初始化消息处理器
        self._register_default_handlers()
        
        # 启动后台任务
        self.running = False
        self.tasks: List[asyncio.Task] = []
    
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
    
    def start(self) -> None:
        """启动WebSocket管理器"""
        if not self.running:
            self.running = True
            
            # 启动心跳管理器
            self.heartbeat_manager.start()
            
            # 启动后台任务
            self.tasks = [
                asyncio.create_task(self._message_queue_worker()),
                asyncio.create_task(self._broadcast_queue_worker()),
                asyncio.create_task(self._connection_cleanup_worker()),
                asyncio.create_task(self._redis_subscriber())
            ]
            
            logger.info("WebSocket管理器已启动")
    
    def stop(self) -> None:
        """停止WebSocket管理器"""
        self.running = False
        
        # 停止心跳管理器
        self.heartbeat_manager.stop()
        
        # 取消后台任务
        for task in self.tasks:
            task.cancel()
        
        logger.info("WebSocket管理器已停止")
    
    async def connect(self, websocket: WebSocket, user_id: str, conversation_id: Optional[str] = None) -> str:
        """建立WebSocket连接"""
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                connection_id=connection_id,
                user_id=user_id,
                conversation_id=conversation_id,
                websocket=websocket,
                status=ConnectionStatus.CONNECTED,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                metadata={}
            )
            
            # 添加到连接池
            self.connection_pool.add_connection(connection)
            
            # 更新心跳
            self.heartbeat_manager.update_heartbeat(connection_id)
            
            # 更新统计
            self.stats["total_connections"] += 1
            
            # 发送连接成功消息
            await self.send_message(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"WebSocket连接建立: {connection_id}, 用户: {user_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"WebSocket连接失败: {e}")
            raise
    
    async def disconnect(self, connection_id: str) -> None:
        """断开WebSocket连接"""
        connection = self.connection_pool.remove_connection(connection_id)
        if connection:
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.error(f"关闭WebSocket连接失败: {e}")
            
            # 移除心跳
            self.heartbeat_manager.remove_connection(connection_id)
            
            # 结束活跃的流
            for stream_id in list(self.streaming_manager.active_streams.keys()):
                stream_info = self.streaming_manager.get_stream_info(stream_id)
                if stream_info and stream_info["connection_id"] == connection_id:
                    self.streaming_manager.end_stream(stream_id)
            
            logger.info(f"WebSocket连接断开: {connection_id}")
    
    async def handle_message(self, connection_id: str, message_data: str) -> None:
        """处理接收到的消息"""
        connection = self.connection_pool.get_connection(connection_id)
        if not connection:
            logger.warning(f"未找到连接: {connection_id}")
            return
        
        try:
            # 解析消息
            message = json.loads(message_data)
            
            # 更新连接活动时间
            self.connection_pool.update_connection_activity(connection_id)
            self.heartbeat_manager.update_heartbeat(connection_id)
            
            # 更新统计
            self.stats["total_messages"] += 1
            
            # 路由消息
            response = await self.message_router.route_message(connection, message)
            
            # 发送响应
            if response:
                await self.send_message(connection_id, response)
                
        except json.JSONDecodeError:
            await self.send_error(connection_id, "无效的JSON格式")
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self.send_error(connection_id, "消息处理失败")
            self.stats["total_errors"] += 1
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """发送消息"""
        connection = self.connection_pool.get_connection(connection_id)
        if not connection:
            return False
        
        try:
            message_with_meta = {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                **message
            }
            
            await connection.websocket.send_text(json.dumps(message_with_meta))
            return True
            
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            await self.disconnect(connection_id)
            return False
    
    async def send_error(self, connection_id: str, error_message: str) -> bool:
        """发送错误消息"""
        return await self.send_message(connection_id, {
            "type": "error",
            "error": error_message,
            "category": MessageCategory.ERROR.value
        })
    
    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """向用户的所有连接广播消息"""
        connections = self.connection_pool.get_user_connections(user_id)
        sent_count = 0
        
        for connection in connections:
            if await self.send_message(connection.connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Dict[str, Any]) -> int:
        """向对话的所有连接广播消息"""
        connections = self.connection_pool.get_conversation_connections(conversation_id)
        sent_count = 0
        
        for connection in connections:
            if await self.send_message(connection.connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def start_streaming_response(self, connection_id: str, stream_id: str) -> bool:
        """开始流式响应"""
        connection = self.connection_pool.get_connection(connection_id)
        if not connection:
            return False
        
        # 启动流
        self.streaming_manager.start_stream(stream_id, connection_id)
        
        # 发送流开始消息
        await self.send_message(connection_id, {
            "type": "stream_start",
            "stream_id": stream_id,
            "category": MessageCategory.STREAM.value
        })
        
        return True
    
    async def send_streaming_chunk(self, stream_id: str, chunk_data: Any) -> bool:
        """发送流式数据块"""
        stream_info = self.streaming_manager.get_stream_info(stream_id)
        if not stream_info:
            return False
        
        connection_id = stream_info["connection_id"]
        
        # 更新流进度
        self.streaming_manager.update_stream_progress(stream_id, chunk_data)
        
        # 发送数据块
        return await self.send_message(connection_id, {
            "type": "stream_chunk",
            "stream_id": stream_id,
            "data": chunk_data,
            "category": MessageCategory.STREAM.value
        })
    
    async def end_streaming_response(self, stream_id: str) -> bool:
        """结束流式响应"""
        stream_info = self.streaming_manager.get_stream_info(stream_id)
        if not stream_info:
            return False
        
        connection_id = stream_info["connection_id"]
        
        # 发送流结束消息
        await self.send_message(connection_id, {
            "type": "stream_end",
            "stream_id": stream_id,
            "category": MessageCategory.STREAM.value
        })
        
        # 结束流
        self.streaming_manager.end_stream(stream_id)
        
        return True
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        active_connections = len(self.connection_pool.get_all_connections())
        active_streams = len(self.streaming_manager.active_streams)
        
        return {
            "active_connections": active_connections,
            "active_streams": active_streams,
            "total_connections": self.stats["total_connections"],
            "total_messages": self.stats["total_messages"],
            "total_errors": self.stats["total_errors"],
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "connections_by_user": len(self.connection_pool.user_connections),
            "connections_by_conversation": len(self.connection_pool.conversation_connections)
        }
    
    # 默认消息处理器
    def _register_default_handlers(self) -> None:
        """注册默认消息处理器"""
        
        @self.message_router.register_handler("ping")
        async def handle_ping(connection: WebSocketConnection, message: Dict[str, Any]) -> Dict[str, Any]:
            """处理ping消息"""
            return {
                "type": "pong",
                "category": MessageCategory.HEARTBEAT.value,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.message_router.register_handler("chat_message")
        async def handle_chat_message(connection: WebSocketConnection, message: Dict[str, Any]) -> Dict[str, Any]:
            """处理聊天消息"""
            content = message.get("content", "")
            if not content:
                return {"error": "消息内容不能为空"}
            
            # 添加消息到对话管理器
            if connection.conversation_id:
                await self.conversation_manager.add_message(
                    conversation_id=connection.conversation_id,
                    content=content,
                    message_type=MessageType.USER_TEXT,
                    user_id=connection.user_id,
                    metadata=message.get("metadata", {})
                )
            
            return {
                "type": "message_received",
                "category": MessageCategory.CHAT.value,
                "message_id": message.get("message_id")
            }
        
        @self.message_router.register_handler("join_conversation")
        async def handle_join_conversation(connection: WebSocketConnection, message: Dict[str, Any]) -> Dict[str, Any]:
            """处理加入对话"""
            conversation_id = message.get("conversation_id")
            if not conversation_id:
                return {"error": "缺少对话ID"}
            
            # 更新连接的对话ID
            connection.conversation_id = conversation_id
            
            # 重新添加到连接池（更新映射）
            self.connection_pool.remove_connection(connection.connection_id)
            self.connection_pool.add_connection(connection)
            
            return {
                "type": "conversation_joined",
                "category": MessageCategory.SYSTEM.value,
                "conversation_id": conversation_id
            }
    
    # 后台任务
    async def _message_queue_worker(self) -> None:
        """消息队列工作器"""
        while self.running:
            try:
                # 这里可以实现消息队列处理逻辑
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"消息队列工作器异常: {e}")
    
    async def _broadcast_queue_worker(self) -> None:
        """广播队列工作器"""
        while self.running:
            try:
                # 这里可以实现广播队列处理逻辑
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"广播队列工作器异常: {e}")
    
    async def _connection_cleanup_worker(self) -> None:
        """连接清理工作器"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次
                
                # 清理死连接
                all_connections = self.connection_pool.get_all_connections()
                for connection in all_connections:
                    if not self.heartbeat_manager.is_connection_alive(connection.connection_id):
                        await self.disconnect(connection.connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"连接清理工作器异常: {e}")
    
    async def _redis_subscriber(self) -> None:
        """Redis订阅器"""
        if not self.redis_client:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("websocket_broadcast")
            
            while self.running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "message":
                        # 处理Redis广播消息
                        data = json.loads(message["data"])
                        await self._handle_redis_broadcast(data)
                        
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                    
        except Exception as e:
            logger.error(f"Redis订阅器异常: {e}")
    
    async def _handle_redis_broadcast(self, data: Dict[str, Any]) -> None:
        """处理Redis广播消息"""
        broadcast_type = data.get("type")
        target = data.get("target")
        message = data.get("message", {})
        
        if broadcast_type == "user":
            await self.broadcast_to_user(target, message)
        elif broadcast_type == "conversation":
            await self.broadcast_to_conversation(target, message)


# 全局实例
websocket_manager = None

def get_websocket_manager(redis_client=None) -> WebSocketManager:
    """获取WebSocket管理器实例"""
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = WebSocketManager(redis_client=redis_client)
    return websocket_manager 