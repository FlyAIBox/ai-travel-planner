"""
Chat服务主入口
整合上下文工程、对话管理、WebSocket通信、MCP服务器等组件
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import redis.asyncio as redis
from pydantic import BaseModel, Field

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from .context_engine import get_context_engine
from .websocket_manager import get_websocket_manager, WebSocketManager
from .conversation_manager import get_conversation_manager, MessageType, ConversationStatus
from .mcp_server import get_mcp_server
from .mcp_tools import create_mcp_tools

logger = get_logger(__name__)
settings = get_settings()


# Pydantic模型
class ChatMessage(BaseModel):
    """聊天消息模型"""
    content: str = Field(..., description="消息内容")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    user_id: str = Field(..., description="用户ID")
    message_type: str = Field(default="text", description="消息类型")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    message_id: str
    conversation_id: str
    content: str
    response_type: str = "text"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str


class ConversationInfo(BaseModel):
    """对话信息模型"""
    conversation_id: str
    user_id: str
    status: str
    message_count: int
    created_at: str
    updated_at: str
    current_intent: Optional[str] = None


# 依赖注入
async def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=True
    )


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动Chat服务...")
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 初始化组件
    context_engine = get_context_engine(redis_client)
    conversation_manager = get_conversation_manager(redis_client)
    websocket_manager = get_websocket_manager(redis_client)
    mcp_server = get_mcp_server()
    
    # 启动WebSocket管理器
    websocket_manager.start()
    
    # 注册MCP工具
    tools, tool_cache, tool_monitor = create_mcp_tools(redis_client)
    for tool in tools:
        mcp_server.tool_registry.register_tool(tool)
    
    # 存储到应用状态
    app.state.redis_client = redis_client
    app.state.context_engine = context_engine
    app.state.conversation_manager = conversation_manager
    app.state.websocket_manager = websocket_manager
    app.state.mcp_server = mcp_server
    app.state.tool_cache = tool_cache
    app.state.tool_monitor = tool_monitor
    
    logger.info("Chat服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭Chat服务...")
    websocket_manager.stop()
    await redis_client.close()
    logger.info("Chat服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner Chat Service",
    description="智能旅行规划聊天服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket端点
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, conversation_id: Optional[str] = None):
    """WebSocket连接端点"""
    connection_id = None
    
    try:
        # 建立连接
        connection_id = await websocket_manager.connect(
            websocket=websocket,
            user_id=user_id,
            conversation_id=conversation_id,
            metadata={"user_agent": websocket.headers.get("user-agent", "")}
        )
        
        logger.info(f"WebSocket连接建立: {connection_id}")
        
        # 监听消息
        while True:
            try:
                # 接收消息
                raw_message = await websocket.receive_text()
                
                # 处理消息
                await websocket_manager.handle_message(connection_id, raw_message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket客户端断开连接: {connection_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {e}")
                await websocket_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "content": {"error": f"消息处理错误: {str(e)}"}
                })
    
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
        if connection_id:
            await websocket_manager.disconnect(connection_id, f"connection_error: {str(e)}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id, "connection_closed")


# REST API端点
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """发送聊天消息"""
    try:
        conversation_manager = app.state.conversation_manager
        
        # 创建或获取对话
        if message.conversation_id:
            conversation = await conversation_manager.get_conversation(message.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="对话不存在")
        else:
            message.conversation_id = await conversation_manager.create_conversation(
                user_id=message.user_id,
                metadata=message.metadata or {}
            )
        
        # 添加用户消息
        user_message = await conversation_manager.add_message(
            conversation_id=message.conversation_id,
            content=message.content,
            message_type=MessageType.USER_TEXT,
            user_id=message.user_id,
            metadata=message.metadata or {}
        )
        
        # 获取AI响应上下文
        context = await conversation_manager.get_context_for_ai(
            conversation_id=message.conversation_id,
            query=message.content,
            max_tokens=4096
        )
        
        # 模拟AI响应（实际应该调用AI模型）
        ai_response_content = await generate_ai_response(message.content, context)
        
        # 添加AI响应消息
        ai_message = await conversation_manager.add_message(
            conversation_id=message.conversation_id,
            content=ai_response_content,
            message_type=MessageType.AI_RESPONSE,
            user_id=message.user_id
        )
        
        # 广播到WebSocket连接
        background_tasks.add_task(
            broadcast_message_to_websockets,
            message.user_id,
            message.conversation_id,
            ai_response_content
        )
        
        return ChatResponse(
            message_id=ai_message.message_id,
            conversation_id=message.conversation_id,
            content=ai_response_content,
            response_type="text",
            timestamp=ai_message.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"聊天处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream(message: ChatMessage):
    """流式聊天响应"""
    try:
        conversation_manager = app.state.conversation_manager
        
        # 创建或获取对话
        if message.conversation_id:
            conversation = await conversation_manager.get_conversation(message.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="对话不存在")
        else:
            message.conversation_id = await conversation_manager.create_conversation(
                user_id=message.user_id,
                metadata=message.metadata or {}
            )
        
        # 添加用户消息
        await conversation_manager.add_message(
            conversation_id=message.conversation_id,
            content=message.content,
            message_type=MessageType.USER_TEXT,
            user_id=message.user_id,
            metadata=message.metadata or {}
        )
        
        # 获取上下文
        context = await conversation_manager.get_context_for_ai(
            conversation_id=message.conversation_id,
            query=message.content,
            max_tokens=4096
        )
        
        # 流式响应
        return StreamingResponse(
            generate_streaming_response(message.content, context, message.conversation_id, message.user_id),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"流式聊天处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationInfo)
async def get_conversation(conversation_id: str):
    """获取对话信息"""
    try:
        conversation_manager = app.state.conversation_manager
        conversation = await conversation_manager.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        return ConversationInfo(
            conversation_id=conversation.conversation_id,
            user_id=conversation.user_id,
            status=conversation.status.value,
            message_count=conversation.message_count,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            current_intent=conversation.current_intent.value if conversation.current_intent else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversations/{conversation_id}/messages")
async def get_conversation_history(conversation_id: str, limit: int = 50, offset: int = 0):
    """获取对话历史"""
    try:
        conversation_manager = app.state.conversation_manager
        messages = await conversation_manager.get_conversation_history(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "message_id": msg.message_id,
                    "content": msg.content,
                    "message_type": msg.message_type.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ],
            "total": len(messages)
        }
        
    except Exception as e:
        logger.error(f"获取对话历史错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/users/{user_id}/conversations")
async def get_user_conversations(user_id: str, status: Optional[str] = None, limit: int = 20):
    """获取用户对话列表"""
    try:
        conversation_manager = app.state.conversation_manager
        
        # 转换状态
        status_filter = None
        if status:
            try:
                status_filter = ConversationStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail="无效的状态值")
        
        conversations = await conversation_manager.get_user_conversations(
            user_id=user_id,
            status=status_filter,
            limit=limit
        )
        
        return {
            "user_id": user_id,
            "conversations": [
                ConversationInfo(
                    conversation_id=conv.conversation_id,
                    user_id=conv.user_id,
                    status=conv.status.value,
                    message_count=conv.message_count,
                    created_at=conv.created_at.isoformat(),
                    updated_at=conv.updated_at.isoformat(),
                    current_intent=conv.current_intent.value if conv.current_intent else None
                )
                for conv in conversations
            ],
            "total": len(conversations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户对话列表错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除对话"""
    try:
        conversation_manager = app.state.conversation_manager
        success = await conversation_manager.end_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        return {"message": "对话已删除", "conversation_id": conversation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/mcp/tools")
async def list_mcp_tools():
    """列出MCP工具"""
    try:
        mcp_server = app.state.mcp_server
        tools = mcp_server.tool_registry.list_tools()
        
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in tools
            ],
            "total": len(tools)
        }
        
    except Exception as e:
        logger.error(f"获取MCP工具列表错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/mcp/tools/{tool_name}/call")
async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]):
    """调用MCP工具"""
    try:
        mcp_server = app.state.mcp_server
        
        # 模拟客户端ID
        client_id = "api_client"
        
        # 构造MCP消息
        from .mcp_server import MCPMessage
        message = MCPMessage(
            id="api_call",
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        # 调用工具
        response_data = await mcp_server.handle_message(client_id, json.dumps(message.model_dump()))
        
        if response_data:
            response = json.loads(response_data)
            return response
        else:
            raise HTTPException(status_code=500, detail="工具调用失败")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"调用MCP工具错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_service_stats():
    """获取服务统计信息"""
    try:
        websocket_manager = app.state.websocket_manager
        mcp_server = app.state.mcp_server
        tool_cache = app.state.tool_cache
        tool_monitor = app.state.tool_monitor
        
        return {
            "websocket": websocket_manager.get_connection_stats(),
            "mcp_server": mcp_server.get_server_stats(),
            "tool_cache": tool_cache.get_stats(),
            "tool_monitor": tool_monitor.get_all_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取服务统计错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    try:
        redis_client = app.state.redis_client
        
        # 检查Redis连接
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "chat-service",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# 辅助函数
async def generate_ai_response(user_input: str, context: str) -> str:
    """生成AI响应（模拟实现）"""
    # 这里应该调用真实的AI模型
    # 为了演示，返回简单的响应
    
    await asyncio.sleep(0.5)  # 模拟处理时间
    
    if "机票" in user_input or "航班" in user_input:
        return "我可以帮您搜索航班信息。请告诉我您的出发地、目的地和出发时间，我会为您查找最优惠的机票。"
    elif "酒店" in user_input or "住宿" in user_input:
        return "我来帮您查找合适的住宿。请提供您的目的地、入住日期、退房日期和客人数量，我会推荐性价比高的酒店。"
    elif "天气" in user_input:
        return "我可以为您查询天气预报。请告诉我您想了解哪个城市的天气情况。"
    elif "旅行计划" in user_input or "行程" in user_input:
        return "我很乐意帮您制定旅行计划！请告诉我您的目的地、旅行时间、预算和兴趣爱好，我会为您规划详细的行程。"
    else:
        return f"我理解您的问题：{user_input}。作为您的旅行助手，我可以帮您查询航班、酒店、天气信息，制定旅行计划等。请告诉我您需要什么帮助？"


async def generate_streaming_response(user_input: str, context: str, conversation_id: str, user_id: str):
    """生成流式响应"""
    response = await generate_ai_response(user_input, context)
    
    # 分块发送
    words = response.split()
    chunk_size = 3
    
    full_response = ""
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        full_response += chunk + " "
        
        # 发送数据块
        yield f"data: {json.dumps({'chunk': chunk, 'type': 'text'})}\n\n"
        
        # 模拟延迟
        await asyncio.sleep(0.1)
    
    # 发送结束标记
    yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    # 保存完整响应到对话历史
    conversation_manager = app.state.conversation_manager
    await conversation_manager.add_message(
        conversation_id=conversation_id,
        content=full_response.strip(),
        message_type=MessageType.AI_RESPONSE,
        user_id=user_id
    )


async def broadcast_message_to_websockets(user_id: str, conversation_id: str, content: str):
    """广播消息到WebSocket连接"""
    try:
        websocket_manager = app.state.websocket_manager
        
        message = {
            "type": "ai_response",
            "conversation_id": conversation_id,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_to_user(user_id, message)
        
    except Exception as e:
        logger.error(f"WebSocket广播失败: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    ) 