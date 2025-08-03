# 数据库设置指南

本文档介绍如何设置和初始化AI Travel Planner项目的数据库。

## 概述

AI Travel Planner使用以下数据库：
- **MySQL 8.0**: 主数据库，存储用户、旅行计划、对话等结构化数据
- **Redis 7**: 缓存和会话存储
- **Qdrant**: 向量数据库，用于存储和检索嵌入向量

## 快速开始

### 1. 环境准备

确保已安装以下软件：
- Docker Desktop (推荐) 或 Docker + Docker Compose
- Python 3.9+

### 2. 配置环境变量

复制环境变量模板：
```bash
cp .env.example .env
```

编辑 `.env` 文件，修改数据库相关配置（如果需要）：
```env
# MySQL配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=ai_travel_db
MYSQL_USER=ai_travel_user
MYSQL_PASSWORD=ai_travel_pass
MYSQL_ROOT_PASSWORD=ai_travel_root

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=ai_travel_redis

# Qdrant配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. 启动数据库服务

#### 方法一：使用启动脚本（推荐）

**Linux/macOS:**
```bash
chmod +x scripts/start-database.sh
./scripts/start-database.sh
```

**Windows:**
```cmd
scripts\start-database.bat
```

#### 方法二：手动启动

```bash
# 启动数据库服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d mysql redis qdrant

# 查看服务状态
docker compose -f deployment/docker/docker-compose.dev.yml ps
```

### 4. 初始化数据库

等待数据库服务启动完成后，运行初始化脚本：

```bash
cd backend
python scripts/init_system.py
```

## 详细说明

### 数据库初始化过程

初始化脚本 `backend/scripts/init_system.py` 会执行以下步骤：

1. **创建数据库**: 如果数据库不存在，会自动创建
2. **创建用户**: 创建应用专用的数据库用户
3. **创建表结构**: 根据ORM模型创建所有必要的表
4. **初始化向量数据库**: 创建Qdrant集合
5. **初始化知识库**: 添加示例数据
6. **创建默认用户**: 创建管理员和演示用户

### 数据库结构

#### MySQL表结构

主要表包括：
- `users`: 用户信息
- `travel_plans`: 旅行计划
- `conversations`: 对话记录
- `knowledge_base`: 知识库
- `agents`: 智能体配置

#### Qdrant集合

- `travel_knowledge`: 旅行知识向量
- `user_preferences`: 用户偏好向量
- `destination_embeddings`: 目的地嵌入

### 故障排除

#### 常见问题

1. **Docker服务未启动**
   ```
   错误: Cannot connect to the Docker daemon
   解决: 启动Docker Desktop或Docker服务
   ```

2. **端口冲突**
   ```
   错误: Port 3306 is already in use
   解决: 停止占用端口的服务或修改配置文件中的端口
   ```

3. **权限问题**
   ```
   错误: Access denied for user
   解决: 检查数据库用户名和密码配置
   ```

4. **数据库连接失败**
   ```
   错误: Can't connect to MySQL server
   解决: 确保MySQL服务已启动并等待几秒钟
   ```

#### 重置数据库

如果需要重置数据库：

```bash
# 停止并删除容器
docker compose -f deployment/docker/docker-compose.dev.yml down -v

# 删除数据卷
docker volume rm docker_mysql_data docker_redis_data docker_qdrant_data

# 重新启动
./scripts/start-database.sh
cd backend && python scripts/init_system.py
```

### 生产环境部署

生产环境建议：

1. **使用外部数据库服务**: 如云数据库服务
2. **配置备份策略**: 定期备份数据
3. **启用SSL连接**: 加密数据传输
4. **设置监控**: 监控数据库性能和健康状态
5. **优化配置**: 根据负载调整连接池和缓存设置

### 监控和维护

#### 查看服务状态

```bash
# 查看容器状态
docker ps

# 查看服务日志
docker logs ai-travel-mysql-dev
docker logs ai-travel-redis-dev
docker logs ai-travel-qdrant-dev
```

#### 数据库连接测试

```bash
# 测试MySQL连接
docker exec -it ai-travel-mysql-dev mysql -u ai_travel_user -p ai_travel_db

# 测试Redis连接
docker exec -it ai-travel-redis-dev redis-cli

# 测试Qdrant连接
curl http://localhost:6333/health
```

## 相关文档

- [项目架构文档](./architecture.md)
- [API文档](./api.md)
- [部署指南](./deployment.md)
