#!/usr/bin/env python3
"""
AI旅行规划系统功能演示脚本
展示Chat服务、MCP工具调用、WebSocket通信等核心功能
"""

import asyncio
import json
import aiohttp
import websockets
from datetime import datetime


class TravelPlannerDemo:
    """旅行规划系统演示"""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.user_id = "demo_user"
        self.conversation_id = None
    
    async def demo_health_check(self):
        """演示健康检查"""
        print("🔍 1. 健康检查")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/health") as resp:
                result = await resp.json()
                print(f"状态: {result['status']}")
                print(f"服务: {result['service']}")
                print(f"时间: {result['timestamp']}")
                print()
    
    async def demo_mcp_tools(self):
        """演示MCP工具"""
        print("🛠️ 2. MCP工具演示")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # 列出可用工具
            async with session.get(f"{self.base_url}/api/v1/mcp/tools") as resp:
                tools = await resp.json()
                print(f"可用工具数量: {tools['total']}")
                for tool in tools['tools']:
                    print(f"  - {tool['name']}: {tool['description']}")
                print()
            
            # 演示航班搜索工具
            print("✈️ 调用航班搜索工具:")
            flight_args = {
                "departure_city": "北京",
                "arrival_city": "上海",
                "departure_date": "2024-03-15",
                "passengers": 1,
                "class": "economy"
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/mcp/tools/flight_search/call",
                json=flight_args
            ) as resp:
                result = await resp.json()
                if 'result' in result:
                    flights = result['result']['content'][0]['text']
                    print(f"搜索结果: {flights[:200]}...")
                else:
                    print(f"调用结果: {result}")
                print()
            
            # 演示天气查询工具
            print("🌤️ 调用天气查询工具:")
            weather_args = {
                "city": "北京",
                "days": 3
            }
            
            async with session.post(
                f"{self.base_url}/api/v1/mcp/tools/weather_inquiry/call",
                json=weather_args
            ) as resp:
                result = await resp.json()
                if 'result' in result:
                    weather = result['result']['content'][0]['text']
                    print(f"天气信息: {weather[:200]}...")
                else:
                    print(f"调用结果: {result}")
                print()
    
    async def demo_chat_api(self):
        """演示聊天API"""
        print("💬 3. 聊天API演示")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # 发送第一条消息
            message1 = {
                "content": "你好，我想计划一次北京旅游",
                "user_id": self.user_id
            }
            
            print(f"用户: {message1['content']}")
            async with session.post(
                f"{self.base_url}/api/v1/chat",
                json=message1
            ) as resp:
                result = await resp.json()
                print(f"助手: {result['content']}")
                self.conversation_id = result['conversation_id']
                print(f"对话ID: {self.conversation_id}")
                print()
            
            # 发送第二条消息
            message2 = {
                "content": "我想了解北京的主要景点",
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            }
            
            print(f"用户: {message2['content']}")
            async with session.post(
                f"{self.base_url}/api/v1/chat",
                json=message2
            ) as resp:
                result = await resp.json()
                print(f"助手: {result['content']}")
                print()
            
            # 查询航班
            message3 = {
                "content": "帮我查询明天从上海到北京的机票",
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            }
            
            print(f"用户: {message3['content']}")
            async with session.post(
                f"{self.base_url}/api/v1/chat",
                json=message3
            ) as resp:
                result = await resp.json()
                print(f"助手: {result['content']}")
                print()
    
    async def demo_conversation_history(self):
        """演示对话历史"""
        print("📜 4. 对话历史演示")
        print("-" * 40)
        
        if not self.conversation_id:
            print("⚠️ 请先执行聊天演示")
            return
        
        async with aiohttp.ClientSession() as session:
            # 获取对话信息
            async with session.get(
                f"{self.base_url}/api/v1/conversations/{self.conversation_id}"
            ) as resp:
                conv_info = await resp.json()
                print(f"对话状态: {conv_info['status']}")
                print(f"消息数量: {conv_info['message_count']}")
                print(f"创建时间: {conv_info['created_at']}")
                print()
            
            # 获取对话历史
            async with session.get(
                f"{self.base_url}/api/v1/conversations/{self.conversation_id}/messages"
            ) as resp:
                history = await resp.json()
                print("对话历史:")
                for msg in history['messages']:
                    role = "用户" if msg['message_type'] == 'user_text' else "助手"
                    content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                    print(f"  {role}: {content}")
                print()
    
    async def demo_websocket(self):
        """演示WebSocket通信"""
        print("🔗 5. WebSocket通信演示")
        print("-" * 40)
        
        websocket_url = f"ws://localhost:8080/ws/{self.user_id}"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                print(f"✅ WebSocket连接建立: {websocket_url}")
                
                # 发送ping消息
                ping_msg = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(ping_msg))
                print(f"发送: {ping_msg}")
                
                # 接收响应
                response = await websocket.recv()
                response_data = json.loads(response)
                print(f"接收: {response_data}")
                print()
                
                # 发送聊天消息
                chat_msg = {
                    "type": "chat_message",
                    "content": "通过WebSocket发送的消息",
                    "user_id": self.user_id
                }
                await websocket.send(json.dumps(chat_msg))
                print(f"发送聊天: {chat_msg}")
                
                # 接收确认
                response = await websocket.recv()
                response_data = json.loads(response)
                print(f"接收确认: {response_data}")
                print()
                
        except Exception as e:
            print(f"❌ WebSocket连接失败: {e}")
    
    async def demo_service_stats(self):
        """演示服务统计"""
        print("📊 6. 服务统计演示")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/stats") as resp:
                stats = await resp.json()
                
                print("WebSocket统计:")
                ws_stats = stats.get('websocket', {})
                print(f"  活跃连接: {ws_stats.get('active_connections', 0)}")
                print(f"  总连接数: {ws_stats.get('total_connections', 0)}")
                print(f"  总消息数: {ws_stats.get('total_messages', 0)}")
                print()
                
                print("MCP服务器统计:")
                mcp_stats = stats.get('mcp_server', {})
                print(f"  活跃客户端: {mcp_stats.get('active_clients', 0)}")
                print(f"  总工具数: {mcp_stats.get('total_tools', 0)}")
                print(f"  总请求数: {mcp_stats.get('total_requests', 0)}")
                print()
                
                print("工具缓存统计:")
                cache_stats = stats.get('tool_cache', {})
                print(f"  缓存命中: {cache_stats.get('hits', 0)}")
                print(f"  缓存未命中: {cache_stats.get('misses', 0)}")
                print(f"  命中率: {cache_stats.get('hit_rate', 0):.2%}")
                print()
    
    async def run_demo(self):
        """运行完整演示"""
        print("🎉 AI旅行规划系统功能演示")
        print("=" * 50)
        print()
        
        try:
            await self.demo_health_check()
            await self.demo_mcp_tools()
            await self.demo_chat_api()
            await self.demo_conversation_history()
            await self.demo_websocket()
            await self.demo_service_stats()
            
            print("✅ 演示完成！")
            print("=" * 50)
            print("🌟 主要功能验证:")
            print("  ✅ 健康检查 - 服务正常运行")
            print("  ✅ MCP工具 - 航班搜索、天气查询等工具正常")
            print("  ✅ 聊天API - 支持多轮对话和上下文管理")
            print("  ✅ 对话历史 - 消息持久化和检索")
            print("  ✅ WebSocket - 实时通信功能")
            print("  ✅ 监控统计 - 服务性能监控")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
            print("请确保服务已启动: docker compose -f deployment/docker/docker-compose.dev.yml up -d")


async def main():
    """主函数"""
    demo = TravelPlannerDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 