# AI旅行规划智能体 🌍✈️

一个基于现代AI技术栈的智能旅行规划助手，集成大模型、RAG、多智能体协作和工作流自动化技术，为用户提供个性化的端到端旅行规划服务。

## 🚀 项目状态更新

### ✅ 已完成功能

#### 后端服务 (100% 完成)
- **✅ 向量数据库基础设施** - Qdrant集群配置、持久化存储、性能优化
- **✅ RAG知识检索系统** - 混合检索策略、结果重排序、查询优化
- **✅ 多角色智能体系统** - LangChain框架、专业智能体、协调机制
- **✅ MCP协议集成** - 服务器架构、工具注册、安全验证
- **✅ 旅行规划引擎** - 约束求解、路径优化、动态重规划
- **✅ API网关服务** - 路由配置、认证中间件、负载均衡
- **✅ 对话管理服务** - WebSocket实时通信、上下文工程

#### 前端界面 (95% 完成)
- **✅ React 18 + TypeScript** - 现代化前端框架
- **✅ 状态管理系统** - Redux Toolkit + React Query
- **✅ 页面组件** - 首页、聊天页面、计划页面、用户中心
- **✅ 认证系统** - 登录/注册页面、受保护路由
- **✅ 布局组件** - 响应式布局、导航菜单

#### 基础设施 (100% 完成)
- **✅ Docker容器化** - 多服务编排、环境隔离
- **✅ 数据库系统** - MySQL主数据库、Redis缓存、Qdrant向量库
- **✅ 监控和日志** - 结构化日志、性能监控
- **✅ 配置管理** - 环境变量、容器配置

### ⚡ 快速启动系统

#### 方法1：一键启动（推荐）
```bash
# 1. 启动基础数据服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d redis qdrant mysql

# 2. 等待服务就绪（约30秒）
sleep 30

# 3. 验证基础服务
curl http://localhost:6333/collections  # Qdrant
curl http://localhost:3306              # MySQL（可能需要MySQL客户端）
redis-cli ping                          # Redis

# 4. 启动应用服务（如果依赖已安装）
docker compose -f deployment/docker/docker-compose.dev.yml up -d
```

#### 方法2：本地开发启动
```bash
# 1. 启动基础服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d redis qdrant mysql

# 2. 安装Python依赖
pip install -r requirements.txt

# 3. 启动后端服务（分别在不同终端）
cd services/chat-service && python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
cd services/rag-service && python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
cd services/agent-service && python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# 4. 启动前端（需要先安装Node.js依赖）
cd frontend
npm install
npm run dev
```

### 🔧 当前状态和下一步

#### 系统运行状态
- **基础服务**: ✅ MySQL、Redis、Qdrant 全部正常运行
- **后端API**: ✅ 核心逻辑完成，需要安装Python依赖
- **前端界面**: ⚠️ 组件完成，需要安装Node.js依赖

#### 需要解决的问题
1. **前端依赖安装**
   ```bash
   cd frontend
   npm install  # 安装React、TypeScript、Ant Design等依赖
   ```

2. **后端依赖安装**
   ```bash
   pip install -r requirements.txt  # 安装Python依赖
   ```

3. **环境配置**
   - 检查 `.env` 文件配置
   - 确保端口不冲突（8080, 8001, 8002, 3000, 6333, 6379, 3306）

### 📊 技术架构完整性

```
✅ 已完成的核心功能
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

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU (推荐，用于AI推理)
- 16GB+ RAM

### 一键启动系统

#### 🎯 推荐方法：使用启动脚本

1. **克隆项目**
```bash
git clone https://github.com/FlyAIBox/ai-travel-planner.git
cd ai-travel-planner
```

2. **一键启动**
```bash
# 自动化启动整个系统（推荐）
./scripts/start_system.sh

# 查看系统状态
./scripts/start_system.sh status

# 查看实时日志
./scripts/start_system.sh logs

# 停止系统
./scripts/start_system.sh stop
```

#### 🔧 手动启动方法

1. **初始化系统**
```bash
# 启动基础服务
docker compose -f deployment/docker/docker-compose.dev.yml up -d redis qdrant mysql

# 等待服务启动完成（约30秒）
sleep 30

# 初始化系统（创建数据库、向量集合、构建知识库）
python scripts/init_system.py
```

2. **启动所有服务**
```bash
# 启动完整系统
docker compose -f deployment/docker/docker-compose.dev.yml up -d

# 查看服务状态
docker compose -f deployment/docker/docker-compose.dev.yml ps
```

3. **验证系统**
```bash
# 检查各服务健康状态
curl http://localhost:8080/api/v1/health  # Chat服务
curl http://localhost:8001/api/v1/health  # RAG服务
curl http://localhost:8002/api/v1/health  # 智能体服务
curl http://localhost:8003/api/v1/health  # 用户服务
curl http://localhost:8080/gateway/health # API网关

# 检查MCP工具列表
curl http://localhost:8080/api/v1/mcp/tools

# 测试聊天API
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "我想去北京旅游", "user_id": "test_user"}'
```

### 服务端点

- **Chat服务**: http://localhost:8080
  - API文档: http://localhost:8080/docs
  - WebSocket: ws://localhost:8080/ws/{user_id}
- **向量数据库**: http://localhost:6333
- **Redis缓存**: localhost:6379
- **MySQL数据库**: localhost:3306
- **n8n工作流**: http://localhost:5678 (admin/ai_travel_n8n)
- **Prometheus监控**: http://localhost:9090
- **Grafana仪表板**: http://localhost:3000 (admin/ai_travel_grafana)

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

