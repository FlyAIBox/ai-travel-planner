# AI旅行规划智能体 🌍✈️

一个基于现代AI技术栈的智能旅行规划助手，集成大模型、RAG、多智能体协作和工作流自动化技术，为用户提供个性化的端到端旅行规划服务。

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

