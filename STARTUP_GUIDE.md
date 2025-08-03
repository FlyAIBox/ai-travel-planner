# AI Travel Planner 系统启动指南

## 📋 目录
- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [快速启动](#快速启动)
- [详细启动流程](#详细启动流程)
- [服务验证](#服务验证)
- [常见问题](#常见问题)

## 🔧 系统要求

### 硬件要求
- **CPU**: 4核心以上
- **内存**: 8GB以上（推荐16GB）
- **存储**: 20GB可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: Linux/macOS/Windows
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.10+ (可选，用于本地开发)

## 🚀 环境准备

### 1. 安装Docker和Docker Compose
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose-plugin

# CentOS/RHEL
sudo yum install docker docker-compose-plugin

# macOS (使用Homebrew)
brew install docker docker-compose

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. 验证安装
```bash
docker --version
docker compose version
```

### 3. 配置环境变量(后台服务)
```bash
# 复制环境变量模板
cd backend
cp .env.example .env

# 编辑环境变量
vim .env
```

**重要环境变量配置：**
```env
# 数据库配置
MYSQL_ROOT_PASSWORD=your_secure_root_password
MYSQL_DATABASE=ai_travel_db
MYSQL_USER=ai_travel_user
MYSQL_PASSWORD=your_secure_password

# Redis配置
REDIS_PASSWORD=your_redis_password

# JWT配置
JWT_SECRET_KEY=your_super_secret_jwt_key_change_in_production

# API密钥
OPENAI_API_KEY=your_openai_api_key
ZHIPU_API_KEY=your_zhipu_api_key

# 第三方服务
HEWEATHER_API_KEY=your_weather_api_key
```

## ⚡ 快速启动

### 开发环境一键启动
```bash
# 进入项目根目录
cd /path/to/ai-travel-planner

# 使用启动脚本
chmod +x backend/scripts/start_system.sh
./backend/scripts/start_system.sh

# 或者直接使用Docker Compose
cd deployment/docker
docker compose -f docker-compose.dev.yml up -d
```

### 生产环境启动
```bash
# 使用部署脚本
chmod +x backend/scripts/deploy.sh
./backend/scripts/deploy.sh --env production
```

## 📝 详细启动流程

### 第一步：准备数据目录
```bash
# 创建必要的数据目录
mkdir -p data/{mysql,redis,qdrant,logs,uploads,backups}
mkdir -p logs/{api,chat,agent,rag,user,nginx}
```

### 第二步：启动基础服务
```bash
cd deployment/docker

# 启动数据库和缓存服务
docker compose -f docker-compose.dev.yml up -d redis mysql qdrant

# 等待服务就绪（约30-60秒）
docker compose -f docker-compose.dev.yml logs -f mysql
```

### 第三步：启动应用服务
```bash
# 启动所有后端服务
docker compose -f docker-compose.dev.yml up -d \
  chat-service \
  rag-service \
  agent-service \
  user-service \
  planning-service \
  integration-service \
  api-gateway

# 启动前端服务
docker compose -f docker-compose.dev.yml up -d frontend
```

### 第四步：启动监控服务（可选）
```bash
# 启动监控和可视化服务
docker compose -f docker-compose.dev.yml up -d \
  prometheus \
  grafana \
  n8n
```

## 🔍 服务验证

### 1. 检查容器状态
```bash
# 查看所有容器状态
docker compose -f docker-compose.dev.yml ps

# 查看服务日志
docker compose -f docker-compose.dev.yml logs -f [service-name]
```

### 2. 健康检查端点
```bash
# API网关健康检查
curl http://localhost:8080/gateway/health

# 各个服务健康检查
curl http://localhost:8080/api/v1/health  # Chat服务
curl http://localhost:8001/api/v1/health  # RAG服务
curl http://localhost:8002/api/v1/health  # Agent服务
curl http://localhost:8003/api/v1/health  # User服务
curl http://localhost:8004/api/v1/health  # Planning服务
curl http://localhost:8005/api/v1/health  # Integration服务
```

### 3. 前端访问
```bash
# 前端应用
http://localhost:3000

# API文档
http://localhost:8080/docs

# 监控面板
http://localhost:3000  # Grafana (admin/ai_travel_grafana)
http://localhost:5678  # n8n (admin/ai_travel_n8n)
```

## 🔧 服务端口映射

| 服务 | 端口 | 描述 |
|------|------|------|
| 前端应用 | 3000 | React前端界面 |
| API网关 | 8080 | 统一API入口 |
| Chat服务 | 8080 | 对话服务 |
| RAG服务 | 8001 | 检索增强生成 |
| Agent服务 | 8002 | 智能体服务 |
| User服务 | 8003 | 用户管理 |
| Planning服务 | 8004 | 行程规划 |
| Integration服务 | 8005 | 外部集成 |
| MySQL | 3306 | 主数据库 |
| Redis | 6379 | 缓存数据库 |
| Qdrant | 6333 | 向量数据库 |
| Prometheus | 9090 | 监控数据收集 |
| Grafana | 3000 | 监控可视化 |
| n8n | 5678 | 工作流引擎 |

## 🛠️ 管理命令

### 启动/停止服务
```bash
# 启动所有服务
docker compose -f docker-compose.dev.yml up -d

# 停止所有服务
docker compose -f docker-compose.dev.yml down

# 重启特定服务
docker compose -f docker-compose.dev.yml restart chat-service

# 查看服务状态
docker compose -f docker-compose.dev.yml ps
```

### 日志管理
```bash
# 查看所有服务日志
docker compose -f docker-compose.dev.yml logs -f

# 查看特定服务日志
docker compose -f docker-compose.dev.yml logs -f chat-service

# 查看最近100行日志
docker compose -f docker-compose.dev.yml logs --tail=100 chat-service
```

### 数据管理
```bash
# 备份数据库
docker compose -f docker-compose.dev.yml exec mysql mysqldump -u root -p ai_travel_db > backup.sql

# 清理未使用的Docker资源
docker system prune -f

# 重建服务（清除缓存）
docker compose -f docker-compose.dev.yml build --no-cache
docker compose -f docker-compose.dev.yml up -d
```

## ❗ 常见问题

### 1. 端口冲突
**问题**: 端口已被占用
**解决**: 
```bash
# 查看端口占用
netstat -tulpn | grep :8080
# 或者修改docker-compose.dev.yml中的端口映射
```

### 2. 内存不足
**问题**: 容器启动失败，内存不足
**解决**: 
- 增加系统内存
- 减少并发启动的服务数量
- 调整Docker内存限制

### 3. 数据库连接失败
**问题**: 应用无法连接数据库
**解决**: 
```bash
# 检查数据库容器状态
docker compose -f docker-compose.dev.yml logs mysql

# 重启数据库服务
docker compose -f docker-compose.dev.yml restart mysql

# 检查环境变量配置
```

### 4. 服务启动顺序问题
**问题**: 服务依赖导致启动失败
**解决**: 
```bash
# 按顺序启动服务
docker compose -f docker-compose.dev.yml up -d redis mysql qdrant
sleep 30
docker compose -f docker-compose.dev.yml up -d chat-service rag-service
sleep 15
docker compose -f docker-compose.dev.yml up -d api-gateway frontend
```

## 📞 技术支持

如果遇到问题，请：
1. 检查日志文件：`docker compose logs [service-name]`
2. 验证环境变量配置
3. 确认系统资源充足
4. 查看GitHub Issues或提交新问题

---

**注意**: 首次启动可能需要较长时间来下载Docker镜像和初始化数据库。请耐心等待。
