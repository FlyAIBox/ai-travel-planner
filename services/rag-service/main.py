"""
RAG服务主入口
提供向量检索、知识库管理、文档处理等API端点
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from pydantic import BaseModel, Field

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

# 导入当前服务的模块
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vector_database import get_vector_database, VectorIndexConfig, VectorSearchResult
from knowledge_builder import get_knowledge_builder, DocumentMetadata, ProcessingResult

logger = get_logger(__name__)
settings = get_settings()


# Pydantic模型
class SearchQuery(BaseModel):
    """搜索查询模型"""
    query: str = Field(..., description="搜索查询")
    collection_name: Optional[str] = Field(None, description="集合名称")
    limit: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度阈值")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    with_vectors: bool = Field(default=False, description="是否返回向量")


class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str
    results: List[Dict[str, Any]]
    total: int
    processing_time: float
    timestamp: str


class DocumentUpload(BaseModel):
    """文档上传模型"""
    title: Optional[str] = Field(None, description="文档标题")
    content: str = Field(..., description="文档内容")
    source_type: str = Field(default="manual", description="来源类型")
    category: str = Field(default="general", description="文档分类")
    tags: List[str] = Field(default=[], description="标签")
    priority: int = Field(default=1, ge=1, le=5, description="优先级")
    language: str = Field(default="zh", description="语言")


class CollectionCreate(BaseModel):
    """创建集合模型"""
    name: str = Field(..., description="集合名称")
    vector_size: int = Field(default=384, ge=1, le=2048, description="向量维度")
    description: Optional[str] = Field(None, description="集合描述")


class KnowledgeStats(BaseModel):
    """知识库统计模型"""
    total_documents: int
    total_chunks: int
    collections: Dict[str, Any]
    processing_stats: Dict[str, Any]
    last_updated: str


# 依赖注入
async def get_redis_client():
    """获取Redis客户端"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB_CACHE,  # 使用缓存数据库
        decode_responses=True
    )


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动RAG服务...")
    
    # 获取Redis客户端
    redis_client = await get_redis_client()
    
    # 初始化组件
    vector_db = get_vector_database()
    knowledge_builder = get_knowledge_builder()
    
    # 存储到应用状态
    app.state.redis_client = redis_client
    app.state.vector_db = vector_db
    app.state.knowledge_builder = knowledge_builder
    
    # 检查向量数据库健康状态
    health_info = await vector_db.health_check()
    if health_info["status"] == "healthy":
        logger.info("向量数据库连接正常")
    else:
        logger.warning(f"向量数据库状态异常: {health_info}")
    
    logger.info("RAG服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("关闭RAG服务...")
    await redis_client.close()
    logger.info("RAG服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI Travel Planner RAG Service",
    description="检索增强生成服务，提供向量检索和知识库管理",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 向量搜索端点
@app.post("/api/v1/search", response_model=SearchResponse)
async def search_vectors(query: SearchQuery):
    """向量搜索"""
    start_time = datetime.now()
    
    try:
        vector_db = app.state.vector_db
        knowledge_builder = app.state.knowledge_builder
        
        # 生成查询向量
        query_vectors = await knowledge_builder.embedding_generator.generate_embeddings(
            [query.query]
        )
        query_vector = query_vectors[0]
        
        # 执行搜索
        collection_name = query.collection_name or settings.QDRANT_COLLECTION_NAME
        results = await vector_db.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=query.limit,
            score_threshold=query.score_threshold,
            filter_conditions=query.filters,
            with_vectors=query.with_vectors
        )
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "title": result.payload.get("title"),
                "source": result.payload.get("source_path"),
                "category": result.payload.get("category"),
                "tags": result.payload.get("tags", []),
                "metadata": result.payload
            }
            
            if query.with_vectors and result.vector:
                formatted_result["vector"] = result.vector
            
            formatted_results.append(formatted_result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SearchResponse(
            query=query.query,
            results=formatted_results,
            total=len(formatted_results),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search/similar")
async def search_similar_documents(query: SearchQuery):
    """搜索相似文档"""
    try:
        vector_db = app.state.vector_db
        knowledge_builder = app.state.knowledge_builder
        
        # 生成查询向量
        query_vectors = await knowledge_builder.embedding_generator.generate_embeddings(
            [query.query]
        )
        query_vector = query_vectors[0]
        
        # 搜索相似文档
        collection_name = query.collection_name or settings.QDRANT_COLLECTION_NAME
        results = await vector_db.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=query.limit,
            score_threshold=query.score_threshold or 0.7,
            filter_conditions=query.filters
        )
        
        # 按文档ID聚合结果
        document_groups = {}
        for result in results:
            doc_id = result.payload.get("document_id")
            if doc_id not in document_groups:
                document_groups[doc_id] = {
                    "document_id": doc_id,
                    "title": result.payload.get("title"),
                    "source": result.payload.get("source_path"),
                    "category": result.payload.get("category"),
                    "max_score": result.score,
                    "chunks": []
                }
            
            document_groups[doc_id]["chunks"].append({
                "chunk_id": result.id,
                "content": result.payload.get("content", ""),
                "score": result.score,
                "chunk_index": result.payload.get("chunk_index")
            })
            
            # 更新最大分数
            if result.score > document_groups[doc_id]["max_score"]:
                document_groups[doc_id]["max_score"] = result.score
        
        # 按最大分数排序
        sorted_documents = sorted(
            document_groups.values(),
            key=lambda x: x["max_score"],
            reverse=True
        )
        
        return {
            "query": query.query,
            "documents": sorted_documents,
            "total": len(sorted_documents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"相似文档搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 文档管理端点
@app.post("/api/v1/documents/upload")
async def upload_document(doc: DocumentUpload, background_tasks: BackgroundTasks):
    """上传文档"""
    try:
        knowledge_builder = app.state.knowledge_builder
        
        # 创建文档元数据
        metadata = DocumentMetadata(
            source_id=f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source_type=doc.source_type,
            source_path=f"manual_upload/{datetime.now().strftime('%Y/%m/%d')}",
            title=doc.title,
            category=doc.category,
            tags=doc.tags,
            priority=doc.priority,
            language=doc.language
        )
        
        # 后台处理文档
        background_tasks.add_task(
            process_document_background,
            doc.content,
            metadata
        )
        
        return {
            "message": "文档上传成功，正在后台处理",
            "document_id": metadata.source_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/batch-upload")
async def batch_upload_documents(files: List[UploadFile] = File(...)):
    """批量上传文档"""
    try:
        knowledge_builder = app.state.knowledge_builder
        upload_tasks = []
        
        for file in files:
            # 读取文件内容
            content = await file.read()
            text_content = content.decode('utf-8')
            
            # 创建元数据
            metadata = DocumentMetadata(
                source_id=f"batch_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_type="file",
                source_path=f"batch_upload/{file.filename}",
                title=file.filename,
                category="uploaded",
                tags=["batch_upload"],
                priority=3
            )
            
            upload_tasks.append((text_content, metadata))
        
        # 批量处理
        results = await knowledge_builder.process_batch(upload_tasks, max_concurrent=3)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return {
            "message": f"批量上传完成",
            "total": len(files),
            "successful": len(successful),
            "failed": len(failed),
            "results": [
                {
                    "document_id": r.document_id,
                    "success": r.success,
                    "chunks_count": r.chunks_count,
                    "error": r.error
                }
                for r in results
            ]
        }
        
    except Exception as e:
        logger.error(f"批量上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    try:
        knowledge_builder = app.state.knowledge_builder
        success = await knowledge_builder.delete_document(document_id)
        
        if success:
            return {"message": "文档删除成功", "document_id": document_id}
        else:
            raise HTTPException(status_code=404, detail="文档不存在")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 集合管理端点
@app.post("/api/v1/collections")
async def create_collection(collection: CollectionCreate):
    """创建向量集合"""
    try:
        vector_db = app.state.vector_db
        
        # 创建向量配置
        vector_config = VectorIndexConfig(
            vector_size=collection.vector_size,
            distance="cosine"
        )
        
        success = await vector_db.create_collection(
            collection_name=collection.name,
            vector_config=vector_config
        )
        
        if success:
            return {
                "message": "集合创建成功",
                "collection_name": collection.name,
                "vector_size": collection.vector_size
            }
        else:
            raise HTTPException(status_code=500, detail="集合创建失败")
        
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collections")
async def list_collections():
    """列出所有集合"""
    try:
        vector_db = app.state.vector_db
        
        # 这里需要实现获取所有集合的方法
        # 目前返回默认集合信息
        collection_info = await vector_db.get_collection_info(settings.QDRANT_COLLECTION_NAME)
        
        collections = []
        if collection_info:
            collections.append({
                "name": collection_info.name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            })
        
        return {
            "collections": collections,
            "total": len(collections)
        }
        
    except Exception as e:
        logger.error(f"获取集合列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """获取集合信息"""
    try:
        vector_db = app.state.vector_db
        collection_info = await vector_db.get_collection_info(collection_name)
        
        if not collection_info:
            raise HTTPException(status_code=404, detail="集合不存在")
        
        return {
            "name": collection_info.name,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "points_count": collection_info.points_count,
            "segments_count": collection_info.segments_count,
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "disk_usage": collection_info.disk_usage
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取集合信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 知识库管理端点
@app.post("/api/v1/knowledge-base/build")
async def build_knowledge_base(background_tasks: BackgroundTasks):
    """构建知识库"""
    try:
        knowledge_builder = app.state.knowledge_builder
        
        # 后台构建知识库
        background_tasks.add_task(build_knowledge_base_background)
        
        return {
            "message": "知识库构建已启动",
            "status": "building",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"启动知识库构建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/knowledge-base/stats", response_model=KnowledgeStats)
async def get_knowledge_stats():
    """获取知识库统计"""
    try:
        vector_db = app.state.vector_db
        knowledge_builder = app.state.knowledge_builder
        
        # 获取集合信息
        collection_info = await vector_db.get_collection_info(settings.QDRANT_COLLECTION_NAME)
        
        collections = {}
        if collection_info:
            collections[collection_info.name] = {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        
        # 获取处理统计
        processing_stats = knowledge_builder.get_processing_stats()
        
        return KnowledgeStats(
            total_documents=processing_stats.get("total_documents", 0),
            total_chunks=processing_stats.get("total_chunks", 0),
            collections=collections,
            processing_stats=processing_stats,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/knowledge-base/versions")
async def list_knowledge_versions():
    """列出知识库版本"""
    try:
        knowledge_builder = app.state.knowledge_builder
        versions = knowledge_builder.version_manager.list_versions()
        
        return {
            "versions": versions,
            "current_version": knowledge_builder.version_manager.current_version,
            "total": len(versions)
        }
        
    except Exception as e:
        logger.error(f"获取知识库版本失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查和统计
@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    try:
        vector_db = app.state.vector_db
        redis_client = app.state.redis_client
        
        # 检查Redis连接
        await redis_client.ping()
        
        # 检查向量数据库
        health_info = await vector_db.health_check()
        
        return {
            "status": "healthy" if health_info["status"] == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "service": "rag-service",
            "version": "1.0.0",
            "vector_db_status": health_info["status"],
            "redis_status": "connected"
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/v1/stats")
async def get_service_stats():
    """获取服务统计"""
    try:
        vector_db = app.state.vector_db
        knowledge_builder = app.state.knowledge_builder
        
        # 获取性能统计
        performance_stats = vector_db.get_performance_stats()
        processing_stats = knowledge_builder.get_processing_stats()
        
        return {
            "vector_database": performance_stats,
            "knowledge_processing": processing_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取服务统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 后台任务
async def process_document_background(content: str, metadata: DocumentMetadata):
    """后台处理文档"""
    try:
        knowledge_builder = app.state.knowledge_builder
        result = await knowledge_builder.process_document(content, metadata)
        
        if result.success:
            logger.info(f"文档处理成功: {metadata.source_id}, 生成 {result.chunks_count} 个块")
        else:
            logger.error(f"文档处理失败: {metadata.source_id}, 错误: {result.error}")
            
    except Exception as e:
        logger.error(f"后台文档处理异常: {e}")


async def build_knowledge_base_background():
    """后台构建知识库"""
    try:
        knowledge_builder = app.state.knowledge_builder
        version = await knowledge_builder.build_travel_knowledge_base()
        
        if version:
            logger.info(f"知识库构建完成，版本: {version}")
        else:
            logger.error("知识库构建失败")
            
    except Exception as e:
        logger.error(f"后台知识库构建异常: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 