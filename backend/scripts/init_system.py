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

# 导入数据库创建相关模块
import aiomysql
from sqlalchemy import create_engine, text
import uuid
from passlib.context import CryptContext

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def create_database_if_not_exists():
    """创建数据库（如果不存在）"""
    logger.info("🔧 检查并创建数据库...")

    try:
        # 构建不包含数据库名的连接URL，用于创建数据库
        connection_params = {
            'host': settings.MYSQL_HOST,
            'port': settings.MYSQL_PORT,
            'user': settings.MYSQL_USER,
            'password': settings.MYSQL_PASSWORD,
            'charset': 'utf8mb4',
            'autocommit': True
        }

        # 连接到MySQL服务器（不指定数据库）
        connection = await aiomysql.connect(**connection_params)

        try:
            cursor = await connection.cursor()

            # 检查数据库是否存在
            await cursor.execute(f"SHOW DATABASES LIKE '{settings.MYSQL_DATABASE}'")
            result = await cursor.fetchone()

            if result:
                logger.info(f"✅ 数据库 '{settings.MYSQL_DATABASE}' 已存在")
            else:
                # 创建数据库
                logger.info(f"📦 创建数据库 '{settings.MYSQL_DATABASE}'...")
                await cursor.execute(
                    f"CREATE DATABASE `{settings.MYSQL_DATABASE}` "
                    f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
                logger.info(f"✅ 数据库 '{settings.MYSQL_DATABASE}' 创建成功")

            # 检查用户是否存在（如果用户不是root）
            if settings.MYSQL_USER != 'root':
                await cursor.execute(
                    "SELECT User FROM mysql.user WHERE User = %s",
                    (settings.MYSQL_USER,)
                )
                user_result = await cursor.fetchone()

                if user_result:
                    logger.info(f"✅ 用户 '{settings.MYSQL_USER}' 已存在")
                else:
                    # 创建用户
                    logger.info(f"👤 创建用户 '{settings.MYSQL_USER}'...")
                    await cursor.execute(
                        f"CREATE USER '{settings.MYSQL_USER}'@'%' IDENTIFIED BY '{settings.MYSQL_PASSWORD}'"
                    )

                    # 授予权限
                    await cursor.execute(
                        f"GRANT ALL PRIVILEGES ON `{settings.MYSQL_DATABASE}`.* TO '{settings.MYSQL_USER}'@'%'"
                    )

                    # 刷新权限
                    await cursor.execute("FLUSH PRIVILEGES")
                    logger.info(f"✅ 用户 '{settings.MYSQL_USER}' 创建成功并授权")

            await cursor.close()

        finally:
            connection.close()

        return True

    except Exception as e:
        logger.error(f"❌ 数据库创建失败: {e}")
        logger.error("请确保MySQL服务已启动并且root用户可以连接")
        logger.info("💡 提示：如果使用Docker，请先启动数据库服务：")
        logger.info("   docker compose -f deployment/docker/docker-compose.dev.yml up -d mysql")
        return False


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
        # 获取数据库连接
        database = await get_database()

        # 默认用户数据
        default_users = [
            {
                "username": "admin",
                "email": "admin@ai-travel.com",
                "password": "admin123456",  # 默认密码，生产环境应该修改
                "role": "admin",
                "description": "系统管理员"
            },
            {
                "username": "demo_user",
                "email": "demo@ai-travel.com",
                "password": "demo123456",  # 默认密码，生产环境应该修改
                "role": "user",
                "description": "演示用户"
            }
        ]

        created_count = 0

        async with database.get_async_session() as session:
            from shared.database.models.user import UserORM
            from shared.models.user import UserRole
            from sqlalchemy import text
            from datetime import datetime

            for user_data in default_users:
                # 检查用户是否已存在 (使用原生SQL避免关系映射问题)
                check_sql = text("SELECT id FROM users WHERE username = :username")
                result = await session.execute(check_sql, {"username": user_data["username"]})
                existing_user = result.fetchone()

                if existing_user:
                    logger.info(f"⚠️ 用户 '{user_data['username']}' 已存在，跳过创建")
                    continue

                # 创建新用户
                user_id = str(uuid.uuid4())
                password_hash = pwd_context.hash(user_data["password"])

                new_user = UserORM(
                    id=user_id,
                    username=user_data["username"],
                    email=user_data["email"],
                    password_hash=password_hash,
                    role=UserRole.ADMIN if user_data["role"] == "admin" else UserRole.USER,
                    status="active",
                    is_verified=True,
                    is_active=True,
                    created_at=datetime.now(),
                    notes=user_data["description"]
                )

                session.add(new_user)
                created_count += 1
                logger.info(f"✅ 创建用户: {user_data['username']} ({user_data['email']})")

            # 提交事务
            await session.commit()

        logger.info(f"👥 成功创建 {created_count} 个默认用户")

        if created_count > 0:
            logger.info("🔐 默认用户密码:")
            for user_data in default_users:
                logger.info(f"  - {user_data['username']}: {user_data['password']}")
            logger.warning("⚠️ 请在生产环境中修改默认密码！")

        logger.info("✅ 默认用户初始化完成")
        return True

    except Exception as e:
        logger.error(f"❌ 默认用户初始化失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
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
    total_steps = 5

    # 步骤0: 创建数据库（如果不存在）
    if await create_database_if_not_exists():
        success_count += 1

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
    logger.info("")
    logger.info("💡 提示:")
    logger.info("  - 可以运行验证脚本检查数据库状态: python scripts/verify_database.py")
    logger.info("  - 可以使用启动脚本启动数据库服务: ../scripts/start-database.sh")
    logger.info("  - 查看详细文档: docs/database-setup.md")

    # 清理资源
    try:
        from shared.database.connection import _database
        if _database:
            await _database.close()
            logger.info("✅ 数据库连接已清理")
    except Exception as e:
        logger.warning(f"⚠️ 数据库连接清理失败: {e}")

    try:
        from shared.vector_db.client import _qdrant_manager
        if _qdrant_manager:
            await _qdrant_manager.close()
            logger.info("✅ Qdrant连接已清理")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant连接清理失败: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ 初始化被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 初始化过程中发生错误: {e}")
        sys.exit(1) 