"""
MCP服务器架构
实现Model Context Protocol服务器，支持工具注册、资源管理、协议处理和安全验证
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import inspect

from pydantic import BaseModel, Field, ValidationError
import structlog

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MCPMessageType(Enum):
    """MCP消息类型"""
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    PING = "ping"
    PONG = "pong"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    NOTIFICATIONS = "notifications"
    ERROR = "error"


class MCPCapability(Enum):
    """MCP能力"""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    LOGGING = "logging"


@dataclass
class MCPServerInfo:
    """MCP服务器信息"""
    name: str
    version: str
    capabilities: List[str]
    instructions: str
    metadata: Dict[str, Any]


class MCPMessage(BaseModel):
    """MCP消息模型"""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC版本")
    id: Optional[Union[str, int]] = Field(None, description="消息ID")
    method: Optional[str] = Field(None, description="方法名")
    params: Optional[Dict[str, Any]] = Field(None, description="参数")
    result: Optional[Any] = Field(None, description="结果")
    error: Optional[Dict[str, Any]] = Field(None, description="错误信息")


class MCPTool(BaseModel):
    """MCP工具定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    inputSchema: Dict[str, Any] = Field(..., description="输入JSON Schema")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="工具元数据")


class MCPResource(BaseModel):
    """MCP资源定义"""
    uri: str = Field(..., description="资源URI")
    name: str = Field(..., description="资源名称") 
    description: str = Field(..., description="资源描述")
    mimeType: Optional[str] = Field(None, description="MIME类型")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="资源元数据")


class MCPPrompt(BaseModel):
    """MCP提示词定义"""
    name: str = Field(..., description="提示词名称")
    description: str = Field(..., description="提示词描述")
    arguments: Optional[List[Dict[str, Any]]] = Field(default=None, description="参数定义")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="提示词元数据")


class MCPError(Exception):
    """MCP错误"""
    def __init__(self, code: int, message: str, data: Dict[str, Any] = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(f"MCP Error {code}: {message}")


class MCPToolRegistry:
    """MCP工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        
    def register_tool(self, tool: MCPTool, handler: Callable) -> None:
        """注册工具"""
        self.tools[tool.name] = tool
        self.tool_handlers[tool.name] = handler
        logger.info(f"注册MCP工具: {tool.name}")
    
    def unregister_tool(self, name: str) -> None:
        """注销工具"""
        self.tools.pop(name, None)
        self.tool_handlers.pop(name, None)
        logger.info(f"注销MCP工具: {name}")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """获取工具定义"""
        return self.tools.get(name)
    
    def get_tool_handler(self, name: str) -> Optional[Callable]:
        """获取工具处理器"""
        return self.tool_handlers.get(name)
    
    def list_tools(self) -> List[MCPTool]:
        """列出所有工具"""
        return list(self.tools.values())
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        handler = self.tool_handlers.get(name)
        if not handler:
            raise MCPError(-32601, f"工具不存在: {name}")
        
        try:
            # 检查处理器是否为异步函数
            if inspect.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ],
                "isError": False
            }
            
        except Exception as e:
            logger.error(f"工具调用失败 {name}: {e}")
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": f"工具调用失败: {str(e)}"
                    }
                ],
                "isError": True
            }


class MCPResourceManager:
    """MCP资源管理器"""
    
    def __init__(self):
        self.resources: Dict[str, MCPResource] = {}
        self.resource_handlers: Dict[str, Callable] = {}
    
    def register_resource(self, resource: MCPResource, handler: Callable) -> None:
        """注册资源"""
        self.resources[resource.uri] = resource
        self.resource_handlers[resource.uri] = handler
        logger.info(f"注册MCP资源: {resource.uri}")
    
    def unregister_resource(self, uri: str) -> None:
        """注销资源"""
        self.resources.pop(uri, None)
        self.resource_handlers.pop(uri, None)
        logger.info(f"注销MCP资源: {uri}")
    
    def get_resource(self, uri: str) -> Optional[MCPResource]:
        """获取资源定义"""
        return self.resources.get(uri)
    
    def list_resources(self) -> List[MCPResource]:
        """列出所有资源"""
        return list(self.resources.values())
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """读取资源内容"""
        handler = self.resource_handlers.get(uri)
        if not handler:
            raise MCPError(-32601, f"资源不存在: {uri}")
        
        try:
            if inspect.iscoroutinefunction(handler):
                content = await handler()
            else:
                content = handler()
            
            resource = self.resources[uri]
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource.mimeType or "text/plain",
                        "text": str(content)
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"资源读取失败 {uri}: {e}")
            raise MCPError(-32603, f"资源读取失败: {str(e)}")


class MCPPromptManager:
    """MCP提示词管理器"""
    
    def __init__(self):
        self.prompts: Dict[str, MCPPrompt] = {}
        self.prompt_handlers: Dict[str, Callable] = {}
    
    def register_prompt(self, prompt: MCPPrompt, handler: Callable) -> None:
        """注册提示词"""
        self.prompts[prompt.name] = prompt
        self.prompt_handlers[prompt.name] = handler
        logger.info(f"注册MCP提示词: {prompt.name}")
    
    def unregister_prompt(self, name: str) -> None:
        """注销提示词"""
        self.prompts.pop(name, None)
        self.prompt_handlers.pop(name, None)
        logger.info(f"注销MCP提示词: {name}")
    
    def get_prompt(self, name: str) -> Optional[MCPPrompt]:
        """获取提示词定义"""
        return self.prompts.get(name)
    
    def list_prompts(self) -> List[MCPPrompt]:
        """列出所有提示词"""
        return list(self.prompts.values())

    async def get_prompt_content(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取提示词内容"""
        handler = self.prompt_handlers.get(name)
        if not handler:
            raise MCPError(-32601, f"提示词不存在: {name}")
        
        try:
            arguments = arguments or {}
            if inspect.iscoroutinefunction(handler):
                messages = await handler(**arguments)
            else:
                messages = handler(**arguments)
            
            return {
                "description": self.prompts[name].description,
                "messages": messages if isinstance(messages, list) else [{"role": "user", "content": {"type": "text", "text": str(messages)}}]
            }
            
        except Exception as e:
            logger.error(f"提示词生成失败 {name}: {e}")
            raise MCPError(-32603, f"提示词生成失败: {str(e)}")


class MCPSecurityManager:
    """MCP安全管理器"""
    
    def __init__(self):
        self.allowed_tools: Set[str] = set()
        self.allowed_resources: Set[str] = set()
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
    
    def set_tool_permissions(self, tools: List[str]) -> None:
        """设置工具权限"""
        self.allowed_tools = set(tools)
    
    def set_resource_permissions(self, resources: List[str]) -> None:
        """设置资源权限"""
        self.allowed_resources = set(resources)
    
    def check_tool_permission(self, tool_name: str, client_id: str = None) -> bool:
        """检查工具权限"""
        if not self.allowed_tools:  # 如果没有设置限制，则允许所有
            return True
        return tool_name in self.allowed_tools
    
    def check_resource_permission(self, resource_uri: str, client_id: str = None) -> bool:
        """检查资源权限"""
        if not self.allowed_resources:  # 如果没有设置限制，则允许所有
            return True
        return resource_uri in self.allowed_resources
    
    def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """检查访问频率限制"""
        # 简化的频率限制实现
        now = datetime.now()
        key = f"{client_id}:{endpoint}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {"count": 1, "reset_time": now + timedelta(minutes=1)}
            return True
        
        limit_info = self.rate_limits[key]
        if now > limit_info["reset_time"]:
            # 重置计数
            limit_info["count"] = 1
            limit_info["reset_time"] = now + timedelta(minutes=1)
            return True
        
        if limit_info["count"] >= 60:  # 每分钟最多60次请求
            return False
        
        limit_info["count"] += 1
        return True


class MCPServer:
    """MCP服务器"""
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.tool_registry = MCPToolRegistry()
        self.resource_manager = MCPResourceManager()
        self.prompt_manager = MCPPromptManager()
        self.security_manager = MCPSecurityManager()
        
        # 客户端连接管理
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        
        # 注册默认处理器
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """注册默认处理器"""
        # 这里可以注册一些默认的处理器
        pass
    
    async def handle_message(self, client_id: str, raw_message: str) -> Optional[str]:
        """处理MCP消息"""
        try:
            message_data = json.loads(raw_message)
            message = MCPMessage(**message_data)
            
            # 更新客户端活动时间
            if client_id in self.clients:
                self.clients[client_id]["last_activity"] = datetime.now()
            
            # 路由消息
            response = await self._route_message(client_id, message)

            if response:
                return json.dumps(response.model_dump(exclude_none=True), ensure_ascii=False)

            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"MCP消息JSON解析错误: {e}")
            return self._create_error_response(None, -32700, "Parse error")
        except ValidationError as e:
            logger.error(f"MCP消息验证错误: {e}")
            return self._create_error_response(None, -32600, "Invalid Request")
        except Exception as e:
            logger.error(f"MCP消息处理错误: {e}")
            return self._create_error_response(None, -32603, "Internal error")
    
    async def _route_message(self, client_id: str, message: MCPMessage) -> Optional[MCPMessage]:
        """路由MCP消息"""
        
        # 检查频率限制
        if not self.security_manager.check_rate_limit(client_id, message.method):
            return self._create_error_response(message.id, -32429, "Too Many Requests")
        
        try:
            if message.method == MCPMessageType.INITIALIZE.value:
                return await self._handle_initialize(client_id, message)
            elif message.method == MCPMessageType.PING.value:
                return await self._handle_ping(client_id, message)
            elif message.method == MCPMessageType.TOOLS_LIST.value:
                return await self._handle_tools_list(client_id, message)
            elif message.method == MCPMessageType.TOOLS_CALL.value:
                return await self._handle_tools_call(client_id, message)
            elif message.method == MCPMessageType.RESOURCES_LIST.value:
                return await self._handle_resources_list(client_id, message)
            elif message.method == MCPMessageType.RESOURCES_READ.value:
                return await self._handle_resources_read(client_id, message)
            elif message.method == MCPMessageType.PROMPTS_LIST.value:
                return await self._handle_prompts_list(client_id, message)
            elif message.method == MCPMessageType.PROMPTS_GET.value:
                return await self._handle_prompts_get(client_id, message)
            else:
                return self._create_error_response(message.id, -32601, f"Method not found: {message.method}")
                
        except MCPError as e:
            return self._create_error_response(message.id, e.code, e.message, e.data)
        except Exception as e:
            logger.error(f"MCP方法处理错误 {message.method}: {e}")
            return self._create_error_response(message.id, -32603, "Internal error")
    
    async def _handle_initialize(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理初始化请求"""
        params = message.params or {}
        client_info = params.get("clientInfo", {})

        # 注册客户端
        self.clients[client_id] = {
            "client_info": client_info,
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "capabilities": params.get("capabilities", {})
        }

        self.is_initialized = True

        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    capability.value: {} for capability in MCPCapability
                },
                "serverInfo": {
                    "name": self.server_info.name,
                    "version": self.server_info.version
                },
                "instructions": self.server_info.instructions
            }
        )
        
    async def _handle_ping(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理ping请求"""
        return MCPMessage(
            id=message.id,
            result={}
        )
        
    async def _handle_tools_list(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理工具列表请求"""
        tools = []
        for tool in self.tool_registry.list_tools():
            if self.security_manager.check_tool_permission(tool.name, client_id):
                tools.append(tool.model_dump())

        return MCPMessage(
            id=message.id,
            result={"tools": tools}
        )
        
    async def _handle_tools_call(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理工具调用请求"""
        params = message.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            raise MCPError(-32602, "Missing tool name")

        if not self.security_manager.check_tool_permission(tool_name, client_id):
            raise MCPError(-32603, f"Access denied for tool: {tool_name}")

        result = await self.tool_registry.call_tool(tool_name, arguments)

        return MCPMessage(
            id=message.id,
            result=result
        )
    
    async def _handle_resources_list(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理资源列表请求"""
        resources = []
        for resource in self.resource_manager.list_resources():
            if self.security_manager.check_resource_permission(resource.uri, client_id):
                resources.append(resource.model_dump())

        return MCPMessage(
            id=message.id,
            result={"resources": resources}
        )
        
    async def _handle_resources_read(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理资源读取请求"""
        params = message.params or {}
        uri = params.get("uri")

        if not uri:
            raise MCPError(-32602, "Missing resource URI")

        if not self.security_manager.check_resource_permission(uri, client_id):
            raise MCPError(-32603, f"Access denied for resource: {uri}")

        result = await self.resource_manager.read_resource(uri)

        return MCPMessage(
            id=message.id,
            result=result
        )
    
    async def _handle_prompts_list(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理提示词列表请求"""
        prompts = [prompt.model_dump() for prompt in self.prompt_manager.list_prompts()]

        return MCPMessage(
            id=message.id,
            result={"prompts": prompts}
        )
        
    async def _handle_prompts_get(self, client_id: str, message: MCPMessage) -> MCPMessage:
        """处理提示词获取请求"""
        params = message.params or {}
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not name:
            raise MCPError(-32602, "Missing prompt name")
        
        result = await self.prompt_manager.get_prompt_content(name, arguments)

        return MCPMessage(
            id=message.id,
            result=result
        )
    
    def _create_error_response(self, message_id: Optional[Union[str, int]], code: int, message: str, data: Dict[str, Any] = None) -> str:
        """创建错误响应"""
        error_response = {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        
        if data:
            error_response["error"]["data"] = data
        
        return json.dumps(error_response, ensure_ascii=False)
    
    def register_tool(self, tool: MCPTool, handler: Callable) -> None:
        """注册工具"""
        self.tool_registry.register_tool(tool, handler)
    
    def register_resource(self, resource: MCPResource, handler: Callable) -> None:
        """注册资源"""
        self.resource_manager.register_resource(resource, handler)
    
    def register_prompt(self, prompt: MCPPrompt, handler: Callable) -> None:
        """注册提示词"""
        self.prompt_manager.register_prompt(prompt, handler)
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return {
            "server_info": self.server_info.__dict__,
            "is_initialized": self.is_initialized,
            "connected_clients": len(self.clients),
            "registered_tools": len(self.tool_registry.tools),
            "registered_resources": len(self.resource_manager.resources),
            "registered_prompts": len(self.prompt_manager.prompts),
            "clients": {
                client_id: {
                    "client_info": client_data["client_info"],
                    "connected_at": client_data["connected_at"].isoformat(),
                    "last_activity": client_data["last_activity"].isoformat()
                }
                for client_id, client_data in self.clients.items()
            }
        }
    
    async def disconnect_client(self, client_id: str) -> None:
        """断开客户端连接"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"MCP客户端断开连接: {client_id}")


# 全局MCP服务器实例
_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """获取MCP服务器实例"""
    global _mcp_server
    if _mcp_server is None:
        server_info = MCPServerInfo(
            name="AI Travel Planner MCP Server",
            version="1.0.0",
            capabilities=[capability.value for capability in MCPCapability],
            instructions="AI智能旅行规划助手MCP服务器，提供旅行相关的工具、资源和提示词。",
            metadata={
                "description": "提供航班搜索、酒店查询、天气信息、路线规划等旅行相关功能",
                "author": "AI Travel Planner Team",
                "created_at": datetime.now().isoformat()
            }
        )
        _mcp_server = MCPServer(server_info)
    return _mcp_server 