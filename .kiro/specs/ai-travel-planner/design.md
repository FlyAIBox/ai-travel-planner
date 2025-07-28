# AI旅行规划智能体 - 设计文档

## 概述

本设计文档描述了基于Ubuntu22.04.4和Python3.10的AI旅行规划智能体系统架构。系统集成了现代AI技术栈，包括大模型提示词工程、上下文工程、MCP（Model Context Protocol）、RAG（检索增强生成）、vLLM推理、n8n工作流自动化和LangChain多角色智能体技术，通过Docker容器化部署。

## 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph "用户层"
        UI[Web界面]
        API[REST API]
    end
    
    subgraph "应用层"
        Gateway[API网关]
        Auth[认证服务]
        Chat[对话管理器]
    end
    
    subgraph "智能体层"
        Coordinator[协调智能体]
        FlightAgent[航班智能体]
        HotelAgent[酒店智能体]
        ItineraryAgent[行程智能体]
        RecommendAgent[推荐智能体]
    end
    
    subgraph "AI推理层"
        vLLM[vLLM推理引擎]
        RAG[RAG系统]
        PromptEngine[提示词引擎]
        ContextManager[上下文管理器]
    end
    
    subgraph "工作流层"
        N8N[n8n工作流引擎]
        TaskQueue[任务队列]
    end
    
    subgraph "数据层"
        VectorDB[向量数据库]
        RelationalDB[关系数据库]
        Cache[缓存层]
        KnowledgeBase[知识库]
    end
    
    subgraph "外部服务"
        FlightAPI[航班API]
        HotelAPI[酒店API]
        WeatherAPI[天气API]
        MapsAPI[地图API]
    end
    
    UI --> Gateway
    API --> Gateway
    Gateway --> Auth
    Gateway --> Chat
    Chat --> Coordinator
    Coordinator --> FlightAgent
    Coordinator --> HotelAgent
    Coordinator --> ItineraryAgent
    Coordinator --> RecommendAgent
    
    FlightAgent --> vLLM
    HotelAgent --> vLLM
    ItineraryAgent --> vLLM
    RecommendAgent --> vLLM
    
    vLLM --> RAG
    vLLM --> PromptEngine
    vLLM --> ContextManager
    
    RAG --> VectorDB
    RAG --> KnowledgeBase
    
    Coordinator --> N8N
    N8N --> TaskQueue
    
    Chat --> RelationalDB
    Chat --> Cache
    
    FlightAgent --> FlightAPI
    HotelAgent --> HotelAPI
    ItineraryAgent --> WeatherAPI
    ItineraryAgent --> MapsAPI
```

## 核心组件设计

### 1. 对话管理器 (Chat Manager)

**职责：** 处理用户对话，维护会话状态，协调智能体交互

**技术实现：**
- 使用LangChain的ConversationChain管理对话历史
- 实现基于Redis的会话状态存储
- 集成MCP协议进行模型上下文管理

**关键特性：**
- 多轮对话上下文保持
- 意图识别和实体提取
- 多语言支持（中文/英文）
- 对话状态持久化

### 2. 多角色智能体系统

**架构模式：** 采用协调者模式，由协调智能体统一调度专业智能体

#### 2.1 协调智能体 (Coordinator Agent)
- **职责：** 任务分解、智能体调度、结果整合
- **技术：** LangChain的MultiAgentExecutor
- **决策逻辑：** 基于用户需求类型路由到相应专业智能体

#### 2.2 专业智能体

**航班智能体 (Flight Agent):**
- 专门处理航班搜索、比价、预订建议
- 集成多个航班API（如Amadeus、Skyscanner）
- 实现价格趋势分析和最佳购买时机推荐

**酒店智能体 (Hotel Agent):**
- 处理酒店搜索、筛选、推荐
- 基于用户偏好和预算进行个性化推荐
- 集成Booking.com、Expedia等API

**行程智能体 (Itinerary Agent):**
- 生成详细的日程安排
- 考虑交通时间、景点开放时间等约束
- 优化路线规划

**推荐智能体 (Recommendation Agent):**
- 基于用户画像和历史数据进行个性化推荐
- 实现协同过滤和内容推荐算法
- 集成RAG系统提供基于知识库的推荐

### 3. vLLM推理引擎

**部署架构：**
- 使用vLLM作为高性能LLM推理后端
- 支持多模型并行推理（如Qwen、ChatGLM等）
- 实现动态批处理和KV缓存优化

**模型配置：**
```python
# vLLM配置示例
vllm_config = {
    "model": "Qwen/Qwen2-7B-Instruct",
    "tensor_parallel_size": 2,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.8,
    "enable_prefix_caching": True
}
```

### 4. RAG系统设计

**向量数据库：** 使用qdrant或Qdrant存储旅行知识向量

**知识库构建：**
- 旅行目的地信息
- 酒店和景点评价数据
- 旅行攻略和经验分享
- 实时政策和签证信息

**检索策略：**
- 混合检索（向量检索 + 关键词检索）
- 重排序机制提升检索精度
- 上下文窗口管理

### 5. 提示词工程系统

**提示词模板管理：**
```python
# 提示词模板示例
FLIGHT_SEARCH_PROMPT = """
你是一个专业的航班搜索专家。根据用户需求搜索最适合的航班选项。

用户需求：
- 出发地：{departure}
- 目的地：{destination}
- 出发日期：{departure_date}
- 返程日期：{return_date}
- 预算范围：{budget}
- 偏好：{preferences}

请提供3-5个最佳航班选项，包括：
1. 航班号和航空公司
2. 起飞和到达时间
3. 价格和舱位等级
4. 推荐理由

输出格式：JSON
"""
```

**动态提示词生成：**
- 基于用户画像调整提示词风格
- 根据任务类型选择合适的提示词模板
- 实现提示词版本管理和A/B测试

### 6. n8n工作流自动化

**工作流设计：**

**旅行规划工作流：**
1. 用户需求解析
2. 并行调用多个智能体
3. 结果汇总和冲突解决
4. 生成最终旅行计划
5. 用户确认和修改

**预订工作流：**
1. 用户确认预订意向
2. 检查可用性和价格
3. 生成预订链接或引导
4. 发送确认邮件
5. 添加日历提醒

**监控和提醒工作流：**
1. 监控价格变化
2. 天气预警通知
3. 行程提醒
4. 紧急情况处理

## 数据模型设计

### 用户模型
```python
class User:
    id: str
    username: str
    email: str
    preferences: UserPreferences
    travel_history: List[TravelRecord]
    created_at: datetime
    updated_at: datetime

class UserPreferences:
    budget_range: Tuple[int, int]
    preferred_airlines: List[str]
    hotel_star_rating: int
    travel_style: str  # 家庭、情侣、商务等
    dietary_restrictions: List[str]
    language: str
```

### 旅行计划模型
```python
class TravelPlan:
    id: str
    user_id: str
    destination: str
    start_date: date
    end_date: date
    budget: int
    status: str  # 规划中、已确认、已完成
    flights: List[FlightOption]
    hotels: List[HotelOption]
    itinerary: List[ItineraryItem]
    created_at: datetime
    updated_at: datetime

class FlightOption:
    flight_number: str
    airline: str
    departure_time: datetime
    arrival_time: datetime
    price: float
    booking_url: str

class HotelOption:
    name: str
    address: str
    star_rating: int
    price_per_night: float
    amenities: List[str]
    booking_url: str

class ItineraryItem:
    day: int
    time: str
    activity: str
    location: str
    duration: int
    cost: float
    notes: str
```

### 对话模型
```python
class Conversation:
    id: str
    user_id: str
    messages: List[Message]
    context: Dict
    status: str
    created_at: datetime
    updated_at: datetime

class Message:
    id: str
    role: str  # user, assistant, system
    content: str
    metadata: Dict
    timestamp: datetime
```

## 接口设计

### REST API接口

**用户管理：**
- `POST /api/users/register` - 用户注册
- `POST /api/users/login` - 用户登录
- `GET /api/users/profile` - 获取用户信息
- `PUT /api/users/preferences` - 更新用户偏好

**对话接口：**
- `POST /api/chat/message` - 发送消息
- `GET /api/chat/history/{conversation_id}` - 获取对话历史
- `DELETE /api/chat/{conversation_id}` - 删除对话

**旅行规划：**
- `POST /api/travel/plan` - 创建旅行计划
- `GET /api/travel/plans` - 获取用户的旅行计划列表
- `PUT /api/travel/plan/{plan_id}` - 更新旅行计划
- `DELETE /api/travel/plan/{plan_id}` - 删除旅行计划

**搜索接口：**
- `POST /api/search/flights` - 搜索航班
- `POST /api/search/hotels` - 搜索酒店
- `POST /api/search/attractions` - 搜索景点

### WebSocket接口

**实时对话：**
- `ws://api/chat/ws/{user_id}` - 实时对话连接

**状态更新：**
- `ws://api/status/ws/{user_id}` - 任务状态更新推送

## 错误处理策略

### 分层错误处理

**API层错误：**
- 参数验证错误
- 认证授权错误
- 限流错误

**业务逻辑层错误：**
- 数据不存在错误
- 业务规则违反错误
- 外部服务调用失败

**基础设施层错误：**
- 数据库连接错误
- 缓存服务错误
- 消息队列错误

### 错误恢复机制

**重试策略：**
- 指数退避重试
- 断路器模式
- 降级服务

**数据一致性：**
- 分布式事务管理
- 最终一致性保证
- 补偿机制

## 测试策略

### 单元测试
- 智能体逻辑测试
- 提示词效果测试
- 数据模型验证测试

### 集成测试
- API接口测试
- 数据库集成测试
- 外部服务集成测试

### 端到端测试
- 完整旅行规划流程测试
- 多智能体协作测试
- 用户体验测试

### 性能测试
- 并发用户测试
- 响应时间测试
- 资源使用率测试

## 部署架构

### Docker容器化

**服务容器：**
- `ai-travel-api`: 主API服务
- `ai-travel-agents`: 智能体服务
- `vllm-inference`: vLLM推理服务
- `n8n-workflow`: 工作流引擎
- `rag-service`: RAG检索服务

**数据容器：**
- `mysql`: 关系数据库
- `redis`: 缓存和会话存储
- `qdrant`: 向量数据库
- `elasticsearch`: 搜索引擎

### 容器编排

**docker-compose.yml结构：**
```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - mysql
      - redis
      - vllm
    environment:
      - DATABASE_URL=MySQL://user:pass@mysql:5432/travel_db
      - REDIS_URL=redis://redis:6379
      - VLLM_URL=http://vllm:8001
  
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8001:8000"
    volumes:
      - ./models:/models
    command: --model /models/Qwen2-7B-Instruct
    
  mysql:
    image: mysql:15
    environment:
      - mysql_DB=travel_db
      - mysql_USER=user
      - mysql_PASSWORD=pass
    volumes:
      - mysql_data:/var/lib/MySQL/data
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
      
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=admin
    volumes:
      - n8n_data:/home/node/.n8n
```

### 监控和日志

**监控指标：**
- 系统资源使用率
- API响应时间
- 错误率统计
- 用户活跃度

**日志管理：**
- 结构化日志输出
- 集中式日志收集
- 日志分析和告警

## 安全设计

### 认证和授权
- JWT令牌认证
- OAuth2.0集成
- 角色基础访问控制

### 数据安全
- 敏感数据加密存储
- 传输层安全（TLS）
- 数据脱敏处理

### API安全
- 请求限流
- 输入验证和过滤
- SQL注入防护

### 隐私保护
- 数据最小化原则
- 用户数据删除权
- 隐私政策透明化