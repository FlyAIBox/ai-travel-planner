# AI旅行规划智能体 🌍✈️

一个基于现代AI技术栈的智能旅行规划助手，集成大模型、RAG、多智能体协作和工作流自动化技术，为用户提供个性化的端到端旅行规划服务。


### 📊 技术架构

```
┌─────────────────────────────────────────┐
│ 🗄️ 数据层                               │
│ ├── MySQL (用户数据、计划数据)           │
│ ├── Redis (缓存、会话管理)              │
│ └── Qdrant (向量数据库、知识库)         │
├─────────────────────────────────────────┤
│ 🔧 服务层                               │
│ ├── Chat Service (对话管理、WebSocket) │
│ ├── RAG Service (检索增强生成)          │
│ ├── Agent Service (多智能体协作)        │
│ ├── Planning Service (旅行规划引擎)     │
│ ├── Integration Service (外部API集成)   │
│ └── API Gateway (统一网关、认证)        │
├─────────────────────────────────────────┤
│ 🎨 前端层                               │
│ ├── React 18 + TypeScript              │
│ ├── Redux Toolkit (状态管理)           │
│ ├── Ant Design (UI组件库)              │
│ └── React Router (路由管理)             │
└─────────────────────────────────────────┘
```

### 🌟 系统特性亮点

- **🤖 智能对话**: 基于大模型的自然语言交互
- **📚 知识检索**: RAG技术支持的智能问答
- **🎯 个性化推荐**: 基于用户偏好的旅行建议
- **📱 实时交互**: WebSocket支持的流式响应
- **🔄 多智能体协作**: 专业智能体团队协同工作
- **🎨 现代化UI**: 响应式设计、暗黑模式支持
- **🔒 安全认证**: JWT认证、受保护路由
- **⚡ 高性能**: 向量检索、缓存优化、负载均衡

## 🚀 项目特性

- **🤖 多角色智能体协作**: 基于LangChain的专业智能体团队（航班、酒店、行程、预算、当地向导）
- **🧠 先进AI推理**: 集成vLLM高性能推理引擎，支持Qwen2.5-7B-Instruct等大模型
- **📚 RAG知识增强**: 基于向量数据库的旅行知识检索系统
- **🔄 工作流自动化**: n8n驱动的智能工作流，自动化复杂旅行规划流程
- **💬 自然语言交互**: 支持中英文的多轮对话，智能意图识别
- **🎯 个性化推荐**: 基于用户偏好和历史数据的智能推荐算法
- **🔌 MCP协议集成**: 现代化的模型上下文协议支持
- **🐳 容器化部署**: 完整的Docker微服务架构

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    用户界面      │    │   API网关       │    │   认证服务      │
│  (React/Vue)    │───▶│  (FastAPI)     │───▶│   (JWT)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      对话管理器 (WebSocket)                      │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   智能体协调器 (LangChain)                        │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ 航班智能体   │ 酒店智能体   │ 行程智能体   │ 推荐智能体   │ 预算分析师│
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI推理层 (vLLM + RAG)                       │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│ 提示词引擎   │ 上下文管理   │ 向量检索     │    工作流引擎 (n8n)      │
└─────────────┴─────────────┴─────────────┴─────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         数据层                                  │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│ MySQL      │ Redis缓存   │ Qdrant向量库│    外部API服务           │
│ (主数据库)  │ (会话状态)   │ (知识库)    │ (航班/酒店/天气/地图)     │
└─────────────┴─────────────┴─────────────┴─────────────────────────┘
```

## 🛠️ 技术栈

### 核心框架
- **Python 3.10**: 主要开发语言
- **FastAPI**: 高性能Web框架和API网关
- **LangChain 0.3.12**: 多智能体框架和工具链
- **Pydantic**: 数据验证和设置管理

### AI和ML
- **vLLM**: 高性能大模型推理引擎
- **Qwen2.5-7B-Instruct**: 主要对话模型
- **sentence-transformers**: 文本向量化

### 数据存储
- **MySQL**: 主数据库（用户数据、旅行计划）
- **Redis**: 缓存和会话管理
- **Qdrant**: 向量数据库（知识库）

### 工作流和集成
- **n8n**: 工作流自动化引擎
- **MCP**: 模型上下文协议
- **WebSocket**: 实时通信

### 部署和运维
- **Docker & Docker Compose**: 容器化部署
- **Ubuntu 22.04.4**: 目标部署环境
- **Prometheus + Grafana**: 监控系统

## 📋 项目结构

```
ai-travel-planner/
├── services/                  # 微服务目录
│   ├── api-gateway/          # API网关服务
│   ├── chat-service/         # 对话管理服务
│   ├── agent-service/        # 智能体服务
│   ├── vllm-service/         # vLLM推理服务
│   ├── rag-service/          # RAG检索服务
│   └── user-service/         # 用户管理服务
├── shared/                   # 共享组件
│   ├── models/              # 数据模型
│   ├── utils/               # 工具函数
│   └── config/              # 配置管理
├── frontend/                # 前端应用
├── docs/                    # 项目文档
├── tests/                   # 测试代码
├── deployment/              # 部署配置
│   ├── docker/              # Docker配置
│   ├── k8s/                 # Kubernetes配置
│   └── monitoring/          # 监控配置
├── data/                    # 数据文件
│   ├── knowledge-base/      # 知识库数据
│   └── models/              # AI模型文件
└── scripts/                 # 部署和维护脚本
```

## 🚀 快速开始

### 系统要求

#### 硬件要求
- **CPU**: 4核心以上
- **内存**: 8GB以上（推荐16GB）
- **存储**: 20GB可用空间
- **网络**: 稳定的互联网连接
- **GPU**: NVIDIA GPU (推荐，用于AI推理)

#### 软件要求
- **操作系统**: Linux/macOS/Windows
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.10+ (本地开发)
- **Node.js**: 16+ (前端开发)

### 开发环境启动流程

#### 🎯 推荐方法：一键启动脚本

```bash
# 克隆项目
git clone https://github.com/FlyAIBox/ai-travel-planner.git
cd ai-travel-planner

# 一键启动整个系统
chmod +x backend/scripts/start_system.sh
./backend/scripts/start_system.sh

# 查看系统状态
./backend/scripts/start_system.sh status

# 查看实时日志
./backend/scripts/start_system.sh logs

# 停止系统
./backend/scripts/start_system.sh stop
```

#### 🔧 手动启动方法（开发环境）

按照以下8个步骤手动启动开发环境：

##### 步骤1：启动基础服务
```bash
# 启动数据库和缓存服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d redis qdrant mysql

# 等待服务就绪（约30-60秒）
docker compose -f deployment/docker/docker-compose.dev.yml logs -f mysql
```

##### 步骤2：初始化数据库数据
```bash
# 等待基础服务完全启动
sleep 30

# 运行系统初始化脚本
cd backend
python scripts/init_system.py
```

##### 步骤3：安装Python依赖
```bash
# 在backend目录下安装Python依赖
cd backend
pip install -r requirements.txt

# 或使用虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

##### 步骤4：修改配置文件
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量配置
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

##### 步骤5：启动后端服务
```bash
cd backend

# 方法1：使用统一启动脚本（推荐）
chmod +x scripts/start_backend_services.sh
./scripts/start_backend_services.sh

./backend/scripts/start_backend_services.sh
# 方法2：手动启动各个服务（分别在不同终端）
cd services/chat-service && python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
cd services/rag-service && python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
cd services/agent-service && python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
cd services/user-service && python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
cd services/planning-service && python -m uvicorn main:app --host 0.0.0.0 --port 8004 --reload
cd services/integration-service && python -m uvicorn main:app --host 0.0.0.0 --port 8005 --reload
cd services/api-gateway && python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

##### 步骤6：验证后端服务
```bash
# 检查各服务健康状态
curl http://localhost:8080/api/v1/health  # Chat服务
curl http://localhost:8001/api/v1/health  # RAG服务
curl http://localhost:8002/api/v1/health  # Agent服务
curl http://localhost:8003/api/v1/health  # User服务
curl http://localhost:8004/api/v1/health  # Planning服务
curl http://localhost:8005/api/v1/health  # Integration服务
curl http://localhost:8080/gateway/health # API网关

# 测试聊天API
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "我想去北京旅游", "user_id": "test_user"}'
```

##### 步骤7：启动前端服务
```bash
# 安装前端依赖并启动
cd frontend
npm install
npm run dev

# 或使用yarn
yarn install
yarn dev
```

##### 步骤8：验证前端服务
```bash
# 访问前端应用
http://localhost:3000

# 检查前端健康状态
curl http://localhost:3000/health
```

### 服务端口映射

| 服务 | 端口 | 访问地址 | 描述 |
|------|------|----------|------|
| 前端应用 | 3000 | http://localhost:3000 | React前端界面 |
| API网关 | 8080 | http://localhost:8080 | 统一API入口 |
| Chat服务 | 8080 | http://localhost:8080/docs | 对话服务API文档 |
| RAG服务 | 8001 | http://localhost:8001/docs | 检索增强生成 |
| Agent服务 | 8002 | http://localhost:8002/docs | 智能体服务 |
| User服务 | 8003 | http://localhost:8003/docs | 用户管理 |
| Planning服务 | 8004 | http://localhost:8004/docs | 行程规划 |
| Integration服务 | 8005 | http://localhost:8005/docs | 外部集成 |
| MySQL | 3306 | localhost:3306 | 主数据库 |
| Redis | 6379 | localhost:6379 | 缓存数据库 |
| Qdrant | 6333 | http://localhost:6333 | 向量数据库 |
| n8n工作流 | 5678 | http://localhost:5678 | 工作流引擎 (admin/ai_travel_n8n) |
| Prometheus | 9090 | http://localhost:9090 | 监控数据收集 |
| Grafana | 3000 | http://localhost:3000 | 监控可视化 (admin/ai_travel_grafana) |

### 管理命令

#### 启动/停止服务
```bash
# 启动所有服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d

# 停止所有服务
docker compose -f deployment/docker/docker-compose.dev.yml down

# 重启特定服务
docker compose -f deployment/docker/docker-compose.dev.yml restart chat-service

# 查看服务状态
docker compose -f deployment/docker/docker-compose.dev.yml ps
```

#### 日志管理
```bash
# 查看所有服务日志
docker compose -f deployment/docker/docker-compose.dev.yml logs -f

# 查看特定服务日志
docker compose -f deployment/docker/docker-compose.dev.yml logs -f chat-service

# 查看最近100行日志
docker compose -f deployment/docker/docker-compose.dev.yml logs --tail=100 chat-service
```

### 常见问题解决

#### 1. 端口冲突
**问题**: 端口已被占用
**解决**:
```bash
# 查看端口占用
netstat -tulpn | grep :8080
# 或者修改docker-compose.dev.yml中的端口映射
```

#### 2. 内存不足
**问题**: 容器启动失败，内存不足
**解决**:
- 增加系统内存
- 减少并发启动的服务数量
- 调整Docker内存限制

#### 3. 数据库连接失败
**问题**: 应用无法连接数据库
**解决**:
```bash
# 检查数据库容器状态
docker compose -f deployment/docker/docker-compose.dev.yml logs mysql

# 重启数据库服务
docker compose -f deployment/docker/docker-compose.dev.yml restart mysql

# 检查环境变量配置
```

#### 4. 服务启动顺序问题
**问题**: 服务依赖导致启动失败
**解决**:
```bash
# 按顺序启动服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d redis mysql qdrant
sleep 30
docker compose -f deployment/docker/docker-compose.dev.yml up -d chat-service rag-service
sleep 15
docker compose -f deployment/docker/docker-compose.dev.yml up -d api-gateway frontend
```

## 📚 文档

- [需求文档](.kiro/specs/ai-travel-planner/requirements.md)
- [设计文档](.kiro/specs/ai-travel-planner/design.md)
- [任务列表](.kiro/specs/ai-travel-planner/tasks.md)
- [API文档](docs/api.md)
- [部署指南](docs/deployment.md)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持！⭐**

[![Star History Chart](https://api.star-history.com/svg?repos=FlyAIBox/ai-travel-planner&type=Date)](https://www.star-history.com/#FlyAIBox/ai-travel-planner&Date)

**🔗 更多访问：[大模型实战101](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkzODUxMTY1Mg==&action=getalbum&album_id=3945699220593803270#wechat_redirect)**

</div>

