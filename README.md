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

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/FlyAIBox/ai-travel-planner.git
cd ai-travel-planner
```

2. **设置Python环境**
```bash
# 使用Conda管理环境
conda create -n ai-travel-planner python=3.10 -y
conda activate ai-travel-planner
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，配置数据库、API密钥等
```

4. **启动服务**
```bash
# 开发环境
docker-compose -f docker-compose.dev.yml up -d

# 生产环境
docker-compose -f docker-compose.prod.yml up -d
```

5. **访问应用**
- Web界面: http://localhost:3000
- API文档: http://localhost:8000/docs
- n8n工作流: http://localhost:5678

## 📚 文档

- [需求文档](.kiro/specs/ai-travel-planner/requirements.md)
- [设计文档](.kiro/specs/ai-travel-planner/design.md)
- [任务列表](.kiro/specs/ai-travel-planner/tasks.md)
- [API文档](docs/api.md)
- [部署指南](docs/deployment.md)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持！⭐**

<a href="https://star-history.com/#FlyAIBox/ai-travel-planner&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FlyAIBox/ai-travel-planner&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FlyAIBox/ai-travel-planner&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FlyAIBox/ai-travel-planner&type=Date" />
  </picture>
</a>

**🔗 更多访问：[大模型实战101](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkzODUxMTY1Mg==&action=getalbum&album_id=3945699220593803270#wechat_redirect)**

</div>

