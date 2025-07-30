#!/usr/bin/env python3
"""
数据库初始化脚本
用于创建数据库、运行迁移和初始化数据
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config.settings import get_settings
from shared.database.connection import (
    get_async_engine, 
    check_database_connection,
    get_database_info,
    create_database_tables,
    drop_database_tables
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def init_database():
    """初始化数据库"""
    try:
        logger.info("🚀 开始数据库初始化...")
        
        # 检查数据库连接
        logger.info("📡 检查数据库连接...")
        if not await check_database_connection():
            logger.error("❌ 数据库连接失败，请检查配置")
            return False
        
        # 获取数据库信息
        db_info = await get_database_info()
        logger.info(f"✅ 数据库连接成功")
        logger.info(f"📊 数据库信息: {db_info}")
        
        # 创建数据库表
        logger.info("🏗️ 创建数据库表...")
        await create_database_tables()
        logger.info("✅ 数据库表创建完成")
        
        # 运行Alembic迁移（可选）
        logger.info("🔄 数据库迁移完成（使用 'alembic upgrade head' 来运行迁移）")
        
        logger.info("🎉 数据库初始化完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def reset_database():
    """重置数据库（危险操作）"""
    if not settings.DEBUG:
        logger.error("❌ 只能在调试模式下重置数据库")
        return False
    
    try:
        logger.warning("⚠️ 开始重置数据库（这将删除所有数据）...")
        
        # 删除所有表
        logger.info("🗑️ 删除现有数据库表...")
        await drop_database_tables()
        
        # 重新创建表
        logger.info("🏗️ 重新创建数据库表...")
        await create_database_tables()
        
        logger.info("✅ 数据库重置完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据库重置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_database_status():
    """检查数据库状态"""
    try:
        logger.info("🔍 检查数据库状态...")
        
        # 检查连接
        is_connected = await check_database_connection()
        logger.info(f"📡 数据库连接: {'✅ 正常' if is_connected else '❌ 失败'}")
        
        if is_connected:
            # 获取数据库信息
            db_info = await get_database_info()
            logger.info(f"📊 数据库版本: {db_info.get('version', '未知')}")
            logger.info(f"📂 数据库名称: {db_info.get('database_name', '未知')}")
            logger.info(f"🔤 字符集: {db_info.get('charset', '未知')}")
            logger.info(f"🔗 连接URL: {db_info.get('url', '未知')}")
        
        return is_connected
        
    except Exception as e:
        logger.error(f"❌ 检查数据库状态失败: {e}")
        return False


async def create_sample_data():
    """创建示例数据"""
    try:
        from shared.database.connection import get_async_session
        from shared.database.models import UserORM
        from shared.models.user import UserStatus
        from uuid import uuid4
        
        logger.info("📝 创建示例数据...")
        
        async with get_async_session() as session:
            # 创建示例用户
            sample_user = UserORM(
                id=str(uuid4()),
                username="demo_user",
                email="demo@example.com",
                first_name="演示",
                last_name="用户",
                status=UserStatus.ACTIVE,
                is_verified=True
            )
            
            session.add(sample_user)
            await session.commit()
            
            logger.info(f"✅ 创建示例用户: {sample_user.username}")
        
        logger.info("🎉 示例数据创建完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建示例数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库管理工具")
    parser.add_argument(
        "action", 
        choices=["init", "reset", "check", "sample"],
        help="要执行的操作: init=初始化, reset=重置, check=检查状态, sample=创建示例数据"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="强制执行操作（用于重置数据库）"
    )
    
    args = parser.parse_args()
    
    if args.action == "reset" and not args.force:
        print("⚠️ 重置数据库是危险操作，需要使用 --force 参数确认")
        return
    
    if args.action == "init":
        success = asyncio.run(init_database())
    elif args.action == "reset":
        success = asyncio.run(reset_database())
    elif args.action == "check":
        success = asyncio.run(check_database_status())
    elif args.action == "sample":
        success = asyncio.run(create_sample_data())
    else:
        print(f"❌ 未知操作: {args.action}")
        return
    
    if success:
        print("✅ 操作完成")
        sys.exit(0)
    else:
        print("❌ 操作失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 