"""
MCP (Model Context Protocol) 服务器实现
实现MCP协议处理、工具注册表、资源管理器、客户端连接和状态管理、安全验证
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import inspect
import traceback

from pydantic import BaseModel, Field, ValidationError
import structlog

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# MCP协议数据模型
class MCPMessageType(Enum):
    """MCP消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(Enum):
    """MCP方法"""
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    PING = "ping"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    COMPLETE = "completion/complete"
    SUBSCRIBE = "notifications/subscribe"
    UNSUBSCRIBE = "notifications/unsubscribe"


class MCPCapability(Enum):
    """MCP能力"""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    COMPLETION = "completion"
    NOTIFICATIONS = "notifications"


@dataclass
class MCPClientInfo:
    """MCP客户端信息"""
    name: str
    version: str
    capabilities: List[MCPCapability]
    metadata: Dict[str, Any] = None


@dataclass
class MCPServerInfo:
    """MCP服务器信息"""
    name: str = "AI Travel Planner MCP Server"
    version: str = "1.0.0"
    capabilities: List[MCPCapability] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [
                MCPCapability.TOOLS,
                MCPCapability.RESOURCES,
                MCPCapability.PROMPTS,
                MCPCapability.COMPLETION
            ]


class MCPMessage(BaseModel):
    """MCP消息基类"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPError(BaseModel):
    """MCP错误"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPTool(BaseModel):
    """MCP工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class MCPResource(BaseModel):
    """MCP资源定义"""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPPrompt(BaseModel):
    """MCP提示词定义"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


# 工具基类
class MCPToolBase(ABC):
    """MCP工具基类"""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.metadata = {}
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """执行工具"""
        pass
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> bool:
        """验证参数"""
        # 这里可以添加更复杂的参数验证逻辑
        return True
    
    def get_definition(self) -> MCPTool:
        """获取工具定义"""
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
            metadata=self.metadata
        )


# 资源基类
class MCPResourceBase(ABC):
    """MCP资源基类"""
    
    def __init__(self, uri: str, name: str, description: Optional[str] = None, mime_type: Optional[str] = None):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self.metadata = {}
    
    @abstractmethod
    async def read(self) -> Any:
        """读取资源"""
        pass
    
    def get_definition(self) -> MCPResource:
        """获取资源定义"""
        return MCPResource(
            uri=self.uri,
            name=self.name,
            description=self.description,
            mimeType=self.mime_type,
            metadata=self.metadata
        )


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, MCPToolBase] = {}
        self.tool_permissions: Dict[str, List[str]] = {}  # tool_name -> allowed_clients
        
    def register_tool(self, tool: MCPToolBase, allowed_clients: List[str] = None) -> None:
        """注册工具"""
        self.tools[tool.name] = tool
        if allowed_clients:
            self.tool_permissions[tool.name] = allowed_clients
        logger.info(f"注册MCP工具: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """注销工具"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            if tool_name in self.tool_permissions:
                del self.tool_permissions[tool_name]
            logger.info(f"注销MCP工具: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[MCPToolBase]:
        """获取工具"""
        return self.tools.get(tool_name)
    
    def list_tools(self, client_id: Optional[str] = None) -> List[MCPTool]:
        """列出工具"""
        tools = []
        for tool_name, tool in self.tools.items():
            # 检查权限
            if client_id and tool_name in self.tool_permissions:
                if client_id not in self.tool_permissions[tool_name]:
                    continue
            tools.append(tool.get_definition())
        return tools
    
    def has_permission(self, tool_name: str, client_id: str) -> bool:
        """检查权限"""
        if tool_name not in self.tool_permissions:
            return True  # 没有权限限制
        return client_id in self.tool_permissions[tool_name]


class ResourceRegistry:
    """资源注册表"""
    
    def __init__(self):
        self.resources: Dict[str, MCPResourceBase] = {}
        self.resource_permissions: Dict[str, List[str]] = {}  # resource_uri -> allowed_clients
    
    def register_resource(self, resource: MCPResourceBase, allowed_clients: List[str] = None) -> None:
        """注册资源"""
        self.resources[resource.uri] = resource
        if allowed_clients:
            self.resource_permissions[resource.uri] = allowed_clients
        logger.info(f"注册MCP资源: {resource.uri}")
    
    def unregister_resource(self, resource_uri: str) -> bool:
        """注销资源"""
        if resource_uri in self.resources:
            del self.resources[resource_uri]
            if resource_uri in self.resource_permissions:
                del self.resource_permissions[resource_uri]
            logger.info(f"注销MCP资源: {resource_uri}")
            return True
        return False
    
    def get_resource(self, resource_uri: str) -> Optional[MCPResourceBase]:
        """获取资源"""
        return self.resources.get(resource_uri)
    
    def list_resources(self, client_id: Optional[str] = None) -> List[MCPResource]:
        """列出资源"""
        resources = []
        for resource_uri, resource in self.resources.items():
            # 检查权限
            if client_id and resource_uri in self.resource_permissions:
                if client_id not in self.resource_permissions[resource_uri]:
                    continue
            resources.append(resource.get_definition())
        return resources
    
    def has_permission(self, resource_uri: str, client_id: str) -> bool:
        """检查权限"""
        if resource_uri not in self.resource_permissions:
            return True  # 没有权限限制
        return client_id in self.resource_permissions[resource_uri]


class PromptRegistry:
    """提示词注册表"""
    
    def __init__(self):
        self.prompts: Dict[str, MCPPrompt] = {}
    
    def register_prompt(self, prompt: MCPPrompt) -> None:
        """注册提示词"""
        self.prompts[prompt.name] = prompt
        logger.info(f"注册MCP提示词: {prompt.name}")
    
    def unregister_prompt(self, prompt_name: str) -> bool:
        """注销提示词"""
        if prompt_name in self.prompts:
            del self.prompts[prompt_name]
            logger.info(f"注销MCP提示词: {prompt_name}")
            return True
        return False
    
    def get_prompt(self, prompt_name: str) -> Optional[MCPPrompt]:
        """获取提示词"""
        return self.prompts.get(prompt_name)
    
    def list_prompts(self) -> List[MCPPrompt]:
        """列出提示词"""
        return list(self.prompts.values())


@dataclass
class MCPClient:
    """MCP客户端"""
    client_id: str
    client_info: MCPClientInfo
    connected_at: datetime
    last_activity: datetime
    status: str = "connected"
    session_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.session_data is None:
            self.session_data = {}


class MCPServer:
    """MCP服务器"""
    
    def __init__(self, server_info: MCPServerInfo = None):
        self.server_info = server_info or MCPServerInfo()
        
        # 注册表
        self.tool_registry = ToolRegistry()
        self.resource_registry = ResourceRegistry()
        self.prompt_registry = PromptRegistry()
        
        # 客户端管理
        self.clients: Dict[str, MCPClient] = {}
        self.client_timeout = timedelta(hours=1)
        
        # 消息处理器
        self.message_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # 中间件
        self.middleware: List[Callable] = []
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "start_time": datetime.now()
        }
        
        # 运行状态
        self.running = False
    
    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self.middleware.append(middleware)
    
    def register_message_handler(self, method: str, handler: Callable) -> None:
        """注册消息处理器"""
        self.message_handlers[method] = handler
        logger.info(f"注册MCP消息处理器: {method}")
    
    async def handle_message(self, client_id: str, message_data: str) -> Optional[str]:
        """处理客户端消息"""
        try:
            # 解析消息
            message_dict = json.loads(message_data)
            message = MCPMessage(**message_dict)
            
            # 更新统计
            self.stats["total_requests"] += 1
            
            # 更新客户端活动时间
            if client_id in self.clients:
                self.clients[client_id].last_activity = datetime.now()
            
            # 执行中间件
            for middleware in self.middleware:
                try:
                    message_dict = await middleware(client_id, message_dict)
                    if message_dict is None:
                        return None  # 中间件拦截了消息
                except Exception as e:
                    logger.error(f"中间件执行失败: {e}")
                    return self._create_error_response(message.id, -32603, "中间件处理失败")
            
            # 路由消息
            if message.method:
                handler = self.message_handlers.get(message.method)
                if handler:
                    try:
                        response = await handler(client_id, message)
                        if response:
                            return json.dumps(response.model_dump(exclude_none=True))
                    except Exception as e:
                        logger.error(f"处理消息失败: {e}")
                        self.stats["total_errors"] += 1
                        return self._create_error_response(message.id, -32603, f"处理失败: {str(e)}")
                else:
                    return self._create_error_response(message.id, -32601, f"未知方法: {message.method}")
            
            return None
            
        except json.JSONDecodeError:
            return self._create_error_response(None, -32700, "JSON解析错误")
        except ValidationError as e:
            return self._create_error_response(None, -32600, f"消息格式错误: {e}")
        except Exception as e:
            logger.error(f"处理消息异常: {e}")
            self.stats["total_errors"] += 1
            return self._create_error_response(None, -32603, "服务器内部错误")
    
    def _create_error_response(self, request_id: Optional[Union[str, int]], code: int, message: str) -> str:
        """创建错误响应"""
        error_response = MCPMessage(
            id=request_id,
            error=MCPError(code=code, message=message).model_dump()
        )
        return json.dumps(error_response.model_dump(exclude_none=True))
    
    def _register_default_handlers(self) -> None:
        """注册默认消息处理器"""
        
        async def handle_initialize(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理初始化请求"""
            params = message.params or {}
            client_info_data = params.get("clientInfo", {})
            
            client_info = MCPClientInfo(
                name=client_info_data.get("name", "Unknown"),
                version=client_info_data.get("version", "1.0.0"),
                capabilities=[MCPCapability(cap) for cap in client_info_data.get("capabilities", [])],
                metadata=client_info_data.get("metadata", {})
            )
            
            # 注册客户端
            client = MCPClient(
                client_id=client_id,
                client_info=client_info,
                connected_at=datetime.now(),
                last_activity=datetime.now()
            )
            self.clients[client_id] = client
            
            logger.info(f"MCP客户端已连接: {client_id}, 名称: {client_info.name}")
            
            return MCPMessage(
                id=message.id,
                result={
                    "serverInfo": {
                        "name": self.server_info.name,
                        "version": self.server_info.version,
                        "capabilities": [cap.value for cap in self.server_info.capabilities]
                    },
                    "protocolVersion": "2024-11-05"
                }
            )
        
        async def handle_ping(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理ping请求"""
            return MCPMessage(
                id=message.id,
                result={"status": "ok", "timestamp": datetime.now().isoformat()}
            )
        
        async def handle_list_tools(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理工具列表请求"""
            tools = self.tool_registry.list_tools(client_id)
            return MCPMessage(
                id=message.id,
                result={
                    "tools": [tool.model_dump() for tool in tools]
                }
            )
        
        async def handle_call_tool(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理工具调用请求"""
            params = message.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32602, message="缺少工具名称").model_dump()
                )
            
            # 检查权限
            if not self.tool_registry.has_permission(tool_name, client_id):
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32601, message="没有权限调用此工具").model_dump()
                )
            
            # 获取工具
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32601, message=f"工具不存在: {tool_name}").model_dump()
                )
            
            try:
                # 验证参数
                if not tool.validate_arguments(arguments):
                    return MCPMessage(
                        id=message.id,
                        error=MCPError(code=-32602, message="参数验证失败").model_dump()
                    )
                
                # 执行工具
                result = await tool.execute(arguments)
                
                return MCPMessage(
                    id=message.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                )
                
            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32603, message=f"工具执行失败: {str(e)}").model_dump()
                )
        
        async def handle_list_resources(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理资源列表请求"""
            resources = self.resource_registry.list_resources(client_id)
            return MCPMessage(
                id=message.id,
                result={
                    "resources": [resource.model_dump() for resource in resources]
                }
            )
        
        async def handle_read_resource(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理资源读取请求"""
            params = message.params or {}
            resource_uri = params.get("uri")
            
            if not resource_uri:
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32602, message="缺少资源URI").model_dump()
                )
            
            # 检查权限
            if not self.resource_registry.has_permission(resource_uri, client_id):
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32601, message="没有权限访问此资源").model_dump()
                )
            
            # 获取资源
            resource = self.resource_registry.get_resource(resource_uri)
            if not resource:
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32601, message=f"资源不存在: {resource_uri}").model_dump()
                )
            
            try:
                # 读取资源
                content = await resource.read()
                
                return MCPMessage(
                    id=message.id,
                    result={
                        "contents": [
                            {
                                "uri": resource_uri,
                                "mimeType": resource.mime_type or "text/plain",
                                "text": str(content)
                            }
                        ]
                    }
                )
                
            except Exception as e:
                logger.error(f"资源读取失败: {e}")
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32603, message=f"资源读取失败: {str(e)}").model_dump()
                )
        
        async def handle_list_prompts(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理提示词列表请求"""
            prompts = self.prompt_registry.list_prompts()
            return MCPMessage(
                id=message.id,
                result={
                    "prompts": [prompt.model_dump() for prompt in prompts]
                }
            )
        
        async def handle_get_prompt(client_id: str, message: MCPMessage) -> MCPMessage:
            """处理获取提示词请求"""
            params = message.params or {}
            prompt_name = params.get("name")
            
            if not prompt_name:
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32602, message="缺少提示词名称").model_dump()
                )
            
            prompt = self.prompt_registry.get_prompt(prompt_name)
            if not prompt:
                return MCPMessage(
                    id=message.id,
                    error=MCPError(code=-32601, message=f"提示词不存在: {prompt_name}").model_dump()
                )
            
            return MCPMessage(
                id=message.id,
                result={
                    "description": prompt.description,
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": f"提示词: {prompt.name}\n描述: {prompt.description}"
                            }
                        }
                    ]
                }
            )
        
        # 注册处理器
        self.register_message_handler("initialize", handle_initialize)
        self.register_message_handler("ping", handle_ping)
        self.register_message_handler("tools/list", handle_list_tools)
        self.register_message_handler("tools/call", handle_call_tool)
        self.register_message_handler("resources/list", handle_list_resources)
        self.register_message_handler("resources/read", handle_read_resource)
        self.register_message_handler("prompts/list", handle_list_prompts)
        self.register_message_handler("prompts/get", handle_get_prompt)
    
    def disconnect_client(self, client_id: str) -> bool:
        """断开客户端连接"""
        if client_id in self.clients:
            client = self.clients.pop(client_id)
            logger.info(f"MCP客户端已断开: {client_id}")
            return True
        return False
    
    def get_client_info(self, client_id: str) -> Optional[MCPClient]:
        """获取客户端信息"""
        return self.clients.get(client_id)
    
    def list_clients(self) -> List[MCPClient]:
        """列出所有客户端"""
        return list(self.clients.values())
    
    async def cleanup_expired_clients(self) -> None:
        """清理过期客户端"""
        cutoff_time = datetime.now() - self.client_timeout
        expired_clients = []
        
        for client_id, client in self.clients.items():
            if client.last_activity < cutoff_time:
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            self.disconnect_client(client_id)
        
        if expired_clients:
            logger.info(f"清理了 {len(expired_clients)} 个过期MCP客户端")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        active_clients = len(self.clients)
        total_tools = len(self.tool_registry.tools)
        total_resources = len(self.resource_registry.resources)
        total_prompts = len(self.prompt_registry.prompts)
        
        return {
            "server_info": {
                "name": self.server_info.name,
                "version": self.server_info.version,
                "capabilities": [cap.value for cap in self.server_info.capabilities]
            },
            "active_clients": active_clients,
            "total_tools": total_tools,
            "total_resources": total_resources,
            "total_prompts": total_prompts,
            "total_requests": self.stats["total_requests"],
            "total_errors": self.stats["total_errors"],
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "error_rate": self.stats["total_errors"] / max(self.stats["total_requests"], 1)
        }


# 全局实例
mcp_server = None

def get_mcp_server() -> MCPServer:
    """获取MCP服务器实例"""
    global mcp_server
    if mcp_server is None:
        mcp_server = MCPServer()
    return mcp_server 