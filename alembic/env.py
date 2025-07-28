"""
Alembic环境配置文件
配置数据库连接和模型元数据
"""

from logging.config import fileConfig
import asyncio
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 导入我们的模型基类和所有ORM模型
from shared.database.connection import Base
from shared.database import models  # 这会导入所有模型
from shared.config.settings import get_settings

# 这是 Alembic Config 对象，它提供对 .ini 文件中值的访问
config = context.config

# 获取应用配置
settings = get_settings()

# 从配置中读取数据库URL，如果没有则使用默认配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 设置数据库URL
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# 添加模型的MetaData对象以支持自动生成迁移
target_metadata = Base.metadata

# 其他值从配置文件中获取，在这里可以设置默认值
def render_item(type_, obj, autogen_context):
    """自定义渲染函数，用于改进自动生成的迁移"""
    # 可以在这里添加自定义的渲染逻辑
    return False


def include_name(name, type_, parent_names):
    """决定是否包含某个名称在迁移中"""
    # 跳过某些临时表或系统表
    if type_ == "table":
        return not name.startswith("temp_") and not name.startswith("sys_")
    return True


def include_object(object, name, type_, reflected, compare_to):
    """决定是否包含某个对象在迁移中"""
    # 可以在这里添加过滤逻辑
    return True


def run_migrations_offline() -> None:
    """在'离线'模式下运行迁移。
    
    这会配置上下文只使用URL，而不是Engine，
    尽管这里也可以接受Engine。
    通过跳过Engine的创建，我们甚至不需要DBAPI可用。
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
        include_schemas=True,
        include_name=include_name,
        include_object=include_object,
        # 添加MySQL特定的渲染选项
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """实际运行迁移的函数"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=render_item,
        include_schemas=True,
        include_name=include_name,
        include_object=include_object,
        # 添加MySQL特定的渲染选项
        compare_type=True,
        compare_server_default=True,
        # MySQL特定配置
        render_as_batch=True,  # 支持批量操作
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在'在线'模式下运行异步迁移。
    
    在这种情况下，我们需要创建一个Engine
    并将连接与上下文关联。
    """
    # 创建异步引擎
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在'在线'模式下运行迁移"""
    asyncio.run(run_async_migrations())


# 判断是在线模式还是离线模式
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
