#!/usr/bin/env python3
"""
AI旅行规划系统初始化脚本
设置数据库、创建向量集合、构建知识库等
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.http import models

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from services.rag_service.vector_database import get_vector_database, VectorIndexConfig
from services.rag_service.knowledge_builder import get_knowledge_builder

logger = get_logger(__name__)
settings = get_settings()


class SystemInitializer:
    """系统初始化器"""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.vector_db = None
        self.knowledge_builder = None
        
    async def initialize(self):
        """初始化系统"""
        logger.info("🚀 开始初始化AI旅行规划系统...")
        
        try:
            # 1. 检查并初始化Redis
            await self._init_redis()
            
            # 2. 检查并初始化向量数据库
            await self._init_vector_database()
            
            # 3. 初始化知识库构建器
            await self._init_knowledge_builder()
            
            # 4. 构建旅行知识库
            await self._build_travel_knowledge_base()
            
            # 5. 验证系统状态
            await self._verify_system()
            
            logger.info("✅ 系统初始化完成！")
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            raise
        finally:
            # 清理资源
            if self.redis_client:
                await self.redis_client.close()
    
    async def _init_redis(self):
        """初始化Redis连接"""
        logger.info("📡 连接Redis...")
        
        try:
            self.redis_client = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                db=self.settings.REDIS_DB,
                decode_responses=True
            )
            
            # 测试连接
            await self.redis_client.ping()
            logger.info("✅ Redis连接成功")
            
            # 设置一些初始配置
            await self.redis_client.set("system:initialized", "true", ex=3600)
            
        except Exception as e:
            logger.error(f"❌ Redis连接失败: {e}")
            raise
    
    async def _init_vector_database(self):
        """初始化向量数据库"""
        logger.info("🗄️ 初始化向量数据库...")
        
        try:
            # 获取向量数据库实例
            self.vector_db = get_vector_database()
            
            # 检查健康状态
            health_info = await self.vector_db.health_check()
            if health_info["status"] != "healthy":
                raise Exception(f"向量数据库健康检查失败: {health_info}")
            
            logger.info("✅ 向量数据库连接成功")
            
            # 检查集合是否存在
            collection_name = self.settings.QDRANT_COLLECTION_NAME
            collection_info = await self.vector_db.get_collection_info(collection_name)
            
            if collection_info is None:
                logger.info(f"📝 创建向量集合: {collection_name}")
                
                # 创建集合配置
                vector_config = VectorIndexConfig(
                    vector_size=384,  # sentence-transformers模型维度
                    distance=models.Distance.COSINE,
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 200,
                        "full_scan_threshold": 10000,
                        "max_indexing_threads": 4
                    },
                    quantization_config={
                        "scalar": {
                            "type": "int8",
                            "quantile": 0.99,
                            "always_ram": True
                        }
                    }
                )
                
                # 创建集合
                success = await self.vector_db.create_collection(
                    collection_name=collection_name,
                    vector_config=vector_config,
                    shard_number=1,
                    replication_factor=1
                )
                
                if success:
                    logger.info(f"✅ 向量集合 {collection_name} 创建成功")
                else:
                    raise Exception("向量集合创建失败")
            else:
                logger.info(f"✅ 向量集合 {collection_name} 已存在，包含 {collection_info.points_count} 个向量")
                
        except Exception as e:
            logger.error(f"❌ 向量数据库初始化失败: {e}")
            raise
    
    async def _init_knowledge_builder(self):
        """初始化知识库构建器"""
        logger.info("🧠 初始化知识库构建器...")
        
        try:
            self.knowledge_builder = get_knowledge_builder()
            
            # 检查处理统计
            stats = self.knowledge_builder.get_processing_stats()
            logger.info(f"📊 知识库统计: {stats}")
            
            logger.info("✅ 知识库构建器初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 知识库构建器初始化失败: {e}")
            raise
    
    async def _build_travel_knowledge_base(self):
        """构建旅行知识库"""
        logger.info("📖 构建旅行知识库...")
        
        try:
            # 检查是否已有知识库数据
            collection_info = await self.vector_db.get_collection_info(
                self.settings.QDRANT_COLLECTION_NAME
            )
            
            if collection_info and collection_info.points_count > 0:
                logger.info(f"✅ 知识库已存在，包含 {collection_info.points_count} 个文档块")
                return
            
            # 构建知识库
            logger.info("🔨 开始构建旅行知识库...")
            version = await self.knowledge_builder.build_travel_knowledge_base()
            
            if version:
                logger.info(f"✅ 旅行知识库构建完成，版本: {version}")
                
                # 验证构建结果
                collection_info = await self.vector_db.get_collection_info(
                    self.settings.QDRANT_COLLECTION_NAME,
                    use_cache=False
                )
                
                if collection_info:
                    logger.info(f"📊 知识库包含 {collection_info.points_count} 个文档块")
                else:
                    logger.warning("⚠️ 知识库构建完成但无法获取统计信息")
            else:
                logger.warning("⚠️ 知识库构建未返回版本信息")
                
        except Exception as e:
            logger.error(f"❌ 知识库构建失败: {e}")
            raise
    
    async def _verify_system(self):
        """验证系统状态"""
        logger.info("🔍 验证系统状态...")
        
        try:
            # 验证Redis
            redis_ok = await self.redis_client.ping()
            logger.info(f"✅ Redis状态: {'正常' if redis_ok else '异常'}")
            
            # 验证向量数据库
            health_info = await self.vector_db.health_check()
            logger.info(f"✅ 向量数据库状态: {health_info['status']}")
            
            # 验证知识库
            collection_info = await self.vector_db.get_collection_info(
                self.settings.QDRANT_COLLECTION_NAME
            )
            if collection_info:
                logger.info(f"✅ 知识库状态: {collection_info.points_count} 个文档块")
            
            # 验证性能统计
            performance_stats = self.vector_db.get_performance_stats()
            logger.info(f"📊 向量数据库性能: {performance_stats}")
            
            knowledge_stats = self.knowledge_builder.get_processing_stats()
            logger.info(f"📊 知识库处理统计: {knowledge_stats}")
            
            logger.info("✅ 系统验证完成")
            
        except Exception as e:
            logger.error(f"❌ 系统验证失败: {e}")
            raise


async def main():
    """主函数"""
    try:
        initializer = SystemInitializer()
        await initializer.initialize()
        
        print("\n" + "="*60)
        print("🎉 AI旅行规划系统初始化完成!")
        print("="*60)
        print("🌐 服务端点:")
        print(f"  - Chat服务: http://localhost:8080")
        print(f"  - API文档: http://localhost:8080/docs")
        print(f"  - WebSocket: ws://localhost:8080/ws/{{user_id}}")
        print(f"  - Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        print(f"  - Qdrant: {settings.QDRANT_URL}")
        print("="*60)
        print("📝 使用说明:")
        print("  1. 启动服务: docker compose -f deployment/docker/docker-compose.dev.yml up -d")
        print("  2. 测试API: curl http://localhost:8080/api/v1/health")
        print("  3. 查看日志: docker compose logs -f chat-service")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 