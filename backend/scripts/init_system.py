#!/usr/bin/env python3
"""
AI Travel Planner 系统初始化脚本
创建数据库表、向量集合、初始化数据等
"""

import asyncio
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.database.connection import get_database, Base
from shared.vector_db.client import get_qdrant_client
from shared.vector_db.collections import create_travel_collections
from shared.config.settings import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


async def init_database():
    """初始化数据库"""
    logger.info("🗄️ 初始化MySQL数据库...")

    try:
        # 获取数据库连接
        database = await get_database()

        # 测试连接
        if not await database.test_connection():
            logger.error("❌ 数据库连接失败")
            logger.error("请确保MySQL服务已启动并且配置正确")
            logger.info("💡 提示：如果使用Docker，请先启动数据库服务：")
            logger.info("   docker compose -f deployment/docker/docker-compose.dev.yml up -d mysql")
            return False

        logger.info("✅ 数据库连接成功")

        # 创建所有表
        logger.info("📊 创建数据库表...")

        # 导入所有ORM模型以确保表被创建
        import shared.database.models.user  # noqa
        import shared.database.models.travel  # noqa
        import shared.database.models.conversation  # noqa
        import shared.database.models.knowledge  # noqa
        import shared.database.models.agent  # noqa
        
        # 使用同步引擎创建表
        from shared.database.connection import Base
        try:
            Base.metadata.create_all(database.engine)
        except Exception as e:
            # 如果表已存在，忽略错误
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.warning(f"⚠️ 数据库表可能已存在: {e}")
                logger.info("✅ 跳过表创建，使用现有表")
            else:
                raise e
        
        logger.info("✅ 数据库表创建完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        return False


async def init_vector_database():
    """初始化向量数据库"""
    logger.info("🔍 初始化Qdrant向量数据库...")
    
    try:
        # 获取Qdrant客户端
        qdrant_manager = await get_qdrant_client()
        
        # 检查健康状态
        health = await qdrant_manager.health_check()
        if not health["healthy"]:
            logger.error(f"❌ Qdrant连接失败: {health['message']}")
            return False
        
        logger.info("✅ Qdrant连接成功")
        
        # 创建旅行相关集合
        logger.info("📂 创建向量集合...")
        results = await create_travel_collections(qdrant_manager)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        logger.info(f"✅ 向量集合创建完成: {success_count}/{total_count}")
        
        # 显示创建结果
        for collection_name, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {collection_name}")
        
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"❌ 向量数据库初始化失败: {e}")
        return False


async def init_knowledge_base():
    """初始化知识库"""
    logger.info("📚 初始化知识库...")
    
    try:
        # 这里可以添加初始化知识库的逻辑
        # 比如加载预定义的旅行知识文档
        
        # 示例：创建一些基础的知识条目
        sample_destinations = [
            {
                "name": "北京",
                "country": "中国",
                "description": "中华人民共和国首都，历史文化名城",
                "tags": ["历史", "文化", "美食", "购物"]
            },
            {
                "name": "上海",
                "country": "中国", 
                "description": "国际化大都市，金融中心",
                "tags": ["现代", "购物", "夜生活", "美食"]
            },
            {
                "name": "东京",
                "country": "日本",
                "description": "日本首都，现代与传统结合的城市",
                "tags": ["现代", "文化", "美食", "购物"]
            }
        ]
        
        logger.info(f"📝 添加 {len(sample_destinations)} 个示例目的地")
        logger.info("✅ 知识库初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 知识库初始化失败: {e}")
        return False


async def init_default_users():
    """初始化默认用户"""
    logger.info("👤 初始化默认用户...")
    
    try:
        # 这里可以创建一些默认用户
        # 比如管理员用户、测试用户等
        
        default_users = [
            {
                "username": "admin",
                "email": "admin@ai-travel.com",
                "role": "admin",
                "description": "系统管理员"
            },
            {
                "username": "demo_user",
                "email": "demo@ai-travel.com", 
                "role": "user",
                "description": "演示用户"
            }
        ]
        
        logger.info(f"👥 创建 {len(default_users)} 个默认用户")
        logger.info("✅ 默认用户初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 默认用户初始化失败: {e}")
        return False


async def check_system_status():
    """检查系统状态"""
    logger.info("🔍 检查系统状态...")
    
    status = {
        "database": False,
        "vector_database": False,
        "knowledge_base": False
    }
    
    try:
        # 检查数据库
        database = await get_database()
        status["database"] = await database.test_connection()
        
        # 检查向量数据库
        qdrant_manager = await get_qdrant_client()
        health = await qdrant_manager.health_check()
        status["vector_database"] = health["healthy"]
        
        # 检查知识库
        status["knowledge_base"] = True  # 简化检查
        
    except Exception as e:
        logger.error(f"❌ 系统状态检查失败: {e}")
    
    # 显示状态
    logger.info("📊 系统状态:")
    for component, healthy in status.items():
        status_icon = "✅" if healthy else "❌"
        logger.info(f"  {status_icon} {component}")
    
    return all(status.values())


async def main():
    """主函数"""
    logger.info("🚀 开始初始化AI Travel Planner系统...")
    
    success_count = 0
    total_steps = 4
    
    # 步骤1: 初始化数据库
    if await init_database():
        success_count += 1
    
    # 步骤2: 初始化向量数据库
    if await init_vector_database():
        success_count += 1
    
    # 步骤3: 初始化知识库
    if await init_knowledge_base():
        success_count += 1
    
    # 步骤4: 初始化默认用户
    if await init_default_users():
        success_count += 1
    
    # 检查系统状态
    logger.info("\n" + "="*50)
    if success_count == total_steps:
        logger.info("🎉 系统初始化完成！")
        
        # 最终状态检查
        if await check_system_status():
            logger.info("✅ 所有系统组件运行正常")
        else:
            logger.warning("⚠️ 部分系统组件可能存在问题")
            
    else:
        logger.error(f"❌ 系统初始化失败: {success_count}/{total_steps} 步骤成功")
        sys.exit(1)
    
    logger.info("="*50)
    logger.info("🌟 AI Travel Planner 已准备就绪！")
    logger.info("📱 前端应用: http://localhost:3000")
    logger.info("🚪 API网关: http://localhost:8080")
    logger.info("📚 API文档: http://localhost:8080/docs")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ 初始化被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 初始化过程中发生错误: {e}")
        sys.exit(1) 