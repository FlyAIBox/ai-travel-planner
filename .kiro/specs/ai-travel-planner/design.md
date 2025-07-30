# AI旅行规划智能体 - 设计文档

## 概述

本设计文档描述了基于Ubuntu22.04.4和Python3.10的AI旅行规划智能体系统架构。系统集成了现代AI技术栈，包括大模型提示词工程、上下文工程、MCP（Model Context Protocol）、RAG（检索增强生成）、vLLM推理、n8n工作流自动化和LangChain多角色智能体技术，通过Docker容器化部署。

## 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph "用户层"
        UI[Web界面<br/>React/Vue.js]
        MobileApp[移动端<br/>Progressive Web App]
        API_Client[API客户端<br/>OpenAPI/Swagger]
    end
    
    subgraph "网关层"
        Gateway[API网关<br/>FastAPI + Nginx]
        LoadBalancer[负载均衡器<br/>Nginx/HAProxy]
        RateLimit[限流器<br/>Redis + Lua]
    end
    
    subgraph "应用服务层"
        Auth[认证服务<br/>JWT + OAuth2]
        ChatService[对话服务<br/>WebSocket + REST]
        UserService[用户服务<br/>用户管理 + 偏好]
        PlanService[规划服务<br/>旅行计划管理]
    end
    
    subgraph "智能体协作层"
        MCPServer[MCP服务器<br/>工具注册 + 调用]
        Coordinator[协调智能体<br/>任务分发 + 结果聚合]
        
        subgraph "专业智能体集群"
            FlightAgent[航班智能体<br/>搜索 + 比价]
            HotelAgent[酒店智能体<br/>推荐 + 筛选]
            ItineraryAgent[行程智能体<br/>路线 + 时间优化]
            BudgetAgent[预算智能体<br/>成本控制 + 优化]
            LocalGuideAgent[当地向导<br/>本地化推荐]
            RecommendAgent[推荐智能体<br/>个性化算法]
        end
    end
    
    subgraph "AI推理层"
        vLLMCluster[vLLM推理集群<br/>Qwen2.5-7B + 负载均衡]
        PromptEngine[提示词引擎<br/>模板管理 + 动态生成]
        ContextManager[上下文管理器<br/>MCP协议 + 会话状态]
        
        subgraph "RAG系统"
            Retriever[检索器<br/>混合检索 + 重排序]
            Embedder[向量化器<br/>sentence-transformers]
            Reranker[重排序器<br/>Cross-encoder]
        end
    end
    
    subgraph "工作流自动化层"
        N8NCluster[n8n工作流集群<br/>高可用部署]
        TaskQueue[任务队列<br/>Redis + Celery]
        ScheduleEngine[调度引擎<br/>Crontab + APScheduler]
        NotificationService[通知服务<br/>邮件 + 短信 + 推送]
    end
    
    subgraph "数据存储层"
        subgraph "主数据库"
            MySQL_Master[MySQL主库<br/>用户数据 + 旅行计划]
            MySQL_Slave[MySQL从库<br/>读取副本]
        end
        
        subgraph "缓存系统"
            Redis_Session[Redis会话<br/>用户会话 + 临时数据]
            Redis_Cache[Redis缓存<br/>API响应 + 计算结果]
        end
        
        subgraph "向量存储"
            Qdrant_Cluster[Qdrant集群<br/>向量数据库]
            KnowledgeBase[知识库<br/>旅行攻略 + 实时信息]
        end
        
        subgraph "搜索引擎"
            Elasticsearch[Elasticsearch<br/>全文搜索 + 日志分析]
        end
    end
    
    subgraph "外部服务集成"
        subgraph "旅行API"
            FlightAPI[航班API<br/>Amadeus + Skyscanner]
            HotelAPI[酒店API<br/>Booking + Expedia]
            WeatherAPI[天气API<br/>OpenWeather + 和风]
            MapsAPI[地图API<br/>Google Maps + 高德]
        end
        
        subgraph "支付服务"
            PaymentGateway[支付网关<br/>Stripe + 支付宝]
        end
        
        subgraph "通知服务"
            EmailService[邮件服务<br/>SendGrid + 腾讯云]
            SMSService[短信服务<br/>Twilio + 阿里云]
        end
    end
    
    subgraph "监控运维层"
        Prometheus[Prometheus<br/>指标采集]
        Grafana[Grafana<br/>监控面板]
        ELKStack[ELK Stack<br/>日志分析]
        AlertManager[告警管理<br/>多渠道通知]
    end

    %% 连接关系
    UI --> LoadBalancer
    MobileApp --> LoadBalancer
    API_Client --> LoadBalancer
    
    LoadBalancer --> Gateway
    Gateway --> RateLimit
    RateLimit --> Auth
    
    Gateway --> ChatService
    Gateway --> UserService
    Gateway --> PlanService
    
    ChatService --> MCPServer
    MCPServer --> Coordinator
    
    Coordinator --> FlightAgent
    Coordinator --> HotelAgent
    Coordinator --> ItineraryAgent
    Coordinator --> BudgetAgent
    Coordinator --> LocalGuideAgent
    Coordinator --> RecommendAgent
    
    FlightAgent --> vLLMCluster
    HotelAgent --> vLLMCluster
    ItineraryAgent --> vLLMCluster
    BudgetAgent --> vLLMCluster
    LocalGuideAgent --> vLLMCluster
    RecommendAgent --> vLLMCluster
    
    vLLMCluster --> PromptEngine
    vLLMCluster --> ContextManager
    vLLMCluster --> Retriever
    
    Retriever --> Embedder
    Retriever --> Reranker
    Retriever --> Qdrant_Cluster
    
    Coordinator --> N8NCluster
    N8NCluster --> TaskQueue
    N8NCluster --> ScheduleEngine
    N8NCluster --> NotificationService
    
    UserService --> MySQL_Master
    PlanService --> MySQL_Master
    ChatService --> Redis_Session
    
    MySQL_Master --> MySQL_Slave
    
    FlightAgent --> FlightAPI
    HotelAgent --> HotelAPI
    ItineraryAgent --> WeatherAPI
    ItineraryAgent --> MapsAPI
    
    NotificationService --> EmailService
    NotificationService --> SMSService
    
    %% 监控连接
    Gateway -.-> Prometheus
    ChatService -.-> Prometheus
    vLLMCluster -.-> Prometheus
    N8NCluster -.-> Prometheus
    MySQL_Master -.-> Prometheus
    Redis_Session -.-> Prometheus
    Qdrant_Cluster -.-> Prometheus
```

### 微服务架构详解

#### 1. 服务拆分策略

**按业务领域拆分：**
- **用户域**：用户管理、认证授权、偏好设置
- **对话域**：会话管理、消息路由、上下文维护
- **智能体域**：多智能体协作、任务调度、结果聚合
- **规划域**：旅行计划生成、优化、版本管理
- **推荐域**：个性化推荐、协同过滤、内容推荐
- **集成域**：外部API集成、数据同步、缓存管理

#### 2. 服务间通信

**同步通信：**
- HTTP/REST：用户请求、API调用
- gRPC：高性能内部服务通信
- WebSocket：实时双向通信

**异步通信：**
- 消息队列（Redis Streams）：事件驱动
- 事件总线：松耦合服务协作
- 工作流引擎：复杂业务流程

#### 3. 数据一致性策略

**最终一致性：**
- Saga模式：分布式事务管理
- 事件溯源：状态变更追踪
- CQRS：读写分离优化

**强一致性：**
- 分布式锁：关键资源保护
- 两阶段提交：关键业务事务

## 核心组件设计

### 1. MCP (Model Context Protocol) 集成架构

**MCP服务器设计：**
```python
# MCP服务器架构
class MCPServer:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.resource_manager = ResourceManager()
        self.prompt_registry = PromptRegistry()
        
    async def handle_tools_list(self) -> ListToolsResult:
        """返回可用工具列表"""
        return ListToolsResult(tools=[
            Tool(
                name="search_flights",
                description="搜索航班信息",
                inputSchema=FlightSearchSchema
            ),
            Tool(
                name="search_hotels", 
                description="搜索酒店信息",
                inputSchema=HotelSearchSchema
            ),
            Tool(
                name="get_weather",
                description="获取天气信息",
                inputSchema=WeatherQuerySchema
            ),
            Tool(
                name="plan_itinerary",
                description="规划行程路线",
                inputSchema=ItineraryPlanSchema
            )
        ])
    
    async def handle_tool_call(self, request: CallToolRequest) -> CallToolResult:
        """处理工具调用"""
        tool = self.tool_registry.get_tool(request.params.name)
        result = await tool.execute(request.params.arguments)
        return CallToolResult(content=[TextContent(type="text", text=result)])

# 工具注册表
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self._register_travel_tools()
    
    def _register_travel_tools(self):
        self.register_tool("search_flights", FlightSearchTool())
        self.register_tool("search_hotels", HotelSearchTool())
        self.register_tool("get_weather", WeatherTool())
        self.register_tool("plan_itinerary", ItineraryPlanTool())
        self.register_tool("calculate_budget", BudgetCalculatorTool())
        self.register_tool("get_local_attractions", LocalAttractionsTool())
```

**MCP工具实现：**
```python
class FlightSearchTool(BaseTool):
    async def execute(self, arguments: Dict[str, Any]) -> str:
        search_params = FlightSearchParams(**arguments)
        
        # 调用航班智能体
        flight_agent = self.get_agent("flight_agent")
        results = await flight_agent.search_flights(search_params)
        
        # 格式化结果
        return self._format_flight_results(results)

class HotelSearchTool(BaseTool):
    async def execute(self, arguments: Dict[str, Any]) -> str:
        search_params = HotelSearchParams(**arguments)
        
        # 调用酒店智能体
        hotel_agent = self.get_agent("hotel_agent")
        results = await hotel_agent.search_hotels(search_params)
        
        return self._format_hotel_results(results)
```

**MCP资源管理：**
```python
class ResourceManager:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.user_preferences = UserPreferenceStore()
        
    async def handle_resources_list(self) -> ListResourcesResult:
        """返回可用资源列表"""
        return ListResourcesResult(resources=[
            Resource(
                uri="travel://destinations/popular",
                name="热门目的地信息",
                description="包含热门旅行目的地的详细信息",
                mimeType="application/json"
            ),
            Resource(
                uri="travel://user/preferences", 
                name="用户偏好设置",
                description="当前用户的旅行偏好和历史记录",
                mimeType="application/json"
            )
        ])
    
    async def handle_resource_read(self, uri: str) -> ReadResourceResult:
        """读取资源内容"""
        if uri.startswith("travel://destinations/"):
            content = await self.knowledge_base.get_destinations()
        elif uri.startswith("travel://user/"):
            content = await self.user_preferences.get_preferences()
        
        return ReadResourceResult(
            contents=[TextContent(type="text", text=content)]
        )
```

### 2. 对话管理器 (Chat Manager)

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

### 1. 用户域模型

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime, date
from enum import Enum

class TravelStyle(str, Enum):
    FAMILY = "family"
    COUPLE = "couple"
    BUSINESS = "business"
    SOLO = "solo"
    GROUP = "group"
    ADVENTURE = "adventure"
    LUXURY = "luxury"
    BUDGET = "budget"

class Language(str, Enum):
    ZH_CN = "zh-cn"
    EN_US = "en-us"
    JA_JP = "ja-jp"
    KO_KR = "ko-kr"

class User(BaseModel):
    id: str = Field(..., description="用户唯一标识")
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: "UserPreferences"
    travel_history: List["TravelRecord"] = []
    loyalty_programs: List["LoyaltyProgram"] = []
    payment_methods: List["PaymentMethod"] = []
    is_verified: bool = False
    is_premium: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

class UserPreferences(BaseModel):
    # 基础偏好
    language: Language = Language.ZH_CN
    currency: str = Field(default="CNY", description="偏好货币")
    timezone: str = Field(default="Asia/Shanghai")
    
    # 预算偏好
    budget_range: tuple[int, int] = Field(default=(1000, 10000), description="预算范围(元)")
    budget_flexibility: float = Field(default=0.2, ge=0, le=1, description="预算灵活度")
    
    # 旅行偏好
    travel_style: TravelStyle = TravelStyle.FAMILY
    preferred_season: List[str] = Field(default=["spring", "autumn"])
    trip_duration_preference: tuple[int, int] = Field(default=(3, 7), description="行程天数偏好")
    
    # 交通偏好
    preferred_airlines: List[str] = []
    seat_class_preference: str = Field(default="economy", description="舱位偏好")
    max_layovers: int = Field(default=1, description="最大转机次数")
    departure_time_preference: List[str] = Field(default=["morning"], description="起飞时间偏好")
    
    # 住宿偏好
    hotel_star_rating: tuple[int, int] = Field(default=(3, 5), description="酒店星级范围")
    room_type_preference: str = Field(default="standard", description="房型偏好")
    amenities_importance: Dict[str, int] = Field(default={
        "wifi": 5, "breakfast": 4, "gym": 2, "pool": 3, "spa": 2
    })
    
    # 饮食偏好
    dietary_restrictions: List[str] = []
    cuisine_preferences: List[str] = []
    food_adventurousness: int = Field(default=3, ge=1, le=5, description="美食冒险精神")
    
    # 活动偏好
    activity_preferences: List[str] = []
    interest_categories: List[str] = []
    physical_activity_level: int = Field(default=3, ge=1, le=5, description="体力活动水平")
    
    # 通知偏好
    notification_preferences: Dict[str, bool] = Field(default={
        "email": True, "sms": False, "push": True, "price_alerts": True
    })

class LoyaltyProgram(BaseModel):
    program_name: str
    membership_number: str
    tier_status: str
    points_balance: Optional[int] = None

class PaymentMethod(BaseModel):
    id: str
    type: str  # card, alipay, wechat_pay, paypal
    masked_info: str  # 脱敏后的信息
    is_default: bool = False
    is_verified: bool = False
```

### 2. 旅行计划域模型

```python
class PlanStatus(str, Enum):
    DRAFTING = "drafting"         # 草拟中
    PLANNING = "planning"         # 规划中
    OPTIMIZING = "optimizing"     # 优化中
    CONFIRMED = "confirmed"       # 已确认
    BOOKED = "booked"            # 已预订
    IN_PROGRESS = "in_progress"   # 进行中
    COMPLETED = "completed"       # 已完成
    CANCELLED = "cancelled"       # 已取消

class TravelPlan(BaseModel):
    id: str
    user_id: str
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    
    # 基本信息
    destinations: List["Destination"]
    start_date: date
    end_date: date
    total_budget: float
    traveler_count: int = Field(default=1, ge=1)
    traveler_details: List["TravelerInfo"] = []
    
    # 计划状态
    status: PlanStatus = PlanStatus.DRAFTING
    completion_percentage: float = Field(default=0.0, ge=0, le=100)
    
    # 组件
    flights: List["FlightBooking"] = []
    accommodations: List["AccommodationBooking"] = []
    itinerary: List["ItineraryDay"] = []
    activities: List["ActivityBooking"] = []
    
    # 预算分解
    budget_breakdown: "BudgetBreakdown"
    
    # 元数据
    tags: List[str] = []
    is_public: bool = False
    share_token: Optional[str] = None
    version: int = Field(default=1)
    created_at: datetime
    updated_at: datetime
    confirmed_at: Optional[datetime] = None

class Destination(BaseModel):
    id: str
    name: str
    country: str
    city: str
    coordinates: tuple[float, float]  # (latitude, longitude)
    timezone: str
    visit_order: int
    arrival_date: date
    departure_date: date
    is_primary: bool = False

class TravelerInfo(BaseModel):
    name: str
    age: int
    gender: Optional[str] = None
    passport_country: str
    special_needs: List[str] = []
    is_primary: bool = False

class BudgetBreakdown(BaseModel):
    total_budget: float
    flights: float = 0.0
    accommodation: float = 0.0
    activities: float = 0.0
    food: float = 0.0
    transportation: float = 0.0
    shopping: float = 0.0
    insurance: float = 0.0
    emergency: float = 0.0
    
class FlightBooking(BaseModel):
    id: str
    flight_number: str
    airline: str
    aircraft_type: Optional[str] = None
    
    # 航程信息
    departure_airport: str
    arrival_airport: str
    departure_time: datetime
    arrival_time: datetime
    flight_duration: int  # minutes
    
    # 价格和座位
    price: float
    seat_class: str
    seat_number: Optional[str] = None
    baggage_allowance: str
    
    # 预订状态
    booking_reference: Optional[str] = None
    booking_status: str = "selected"  # selected, booked, cancelled
    booking_url: Optional[str] = None
    
    # 附加信息
    layovers: List["Layover"] = []
    amenities: List[str] = []
    cancellation_policy: Optional[str] = None

class Layover(BaseModel):
    airport: str
    duration: int  # minutes
    is_overnight: bool = False

class AccommodationBooking(BaseModel):
    id: str
    name: str
    type: str  # hotel, apartment, hostel, resort
    brand: Optional[str] = None
    
    # 位置信息
    address: str
    coordinates: tuple[float, float]
    district: Optional[str] = None
    
    # 房间信息
    room_type: str
    bed_configuration: str
    max_occupancy: int
    room_size: Optional[int] = None  # square meters
    
    # 入住信息
    check_in_date: date
    check_out_date: date
    nights: int
    price_per_night: float
    total_price: float
    
    # 设施服务
    star_rating: Optional[int] = None
    amenities: List[str] = []
    policies: Dict[str, str] = {}
    
    # 预订状态
    booking_reference: Optional[str] = None
    booking_status: str = "selected"
    booking_url: Optional[str] = None
    
    # 评价信息
    rating: Optional[float] = None
    reviews_count: Optional[int] = None

class ItineraryDay(BaseModel):
    day_number: int
    date: date
    destination: str
    theme: Optional[str] = None  # 当日主题
    
    # 时间段活动
    activities: List["ItineraryActivity"] = []
    
    # 当日统计
    total_cost: float = 0.0
    total_walking_distance: float = 0.0  # kilometers
    estimated_duration: int = 0  # minutes
    
    # 实用信息
    weather_forecast: Optional["WeatherInfo"] = None
    transportation_notes: Optional[str] = None
    tips: List[str] = []

class ItineraryActivity(BaseModel):
    id: str
    name: str
    type: str  # sightseeing, dining, shopping, transportation, rest
    
    # 时间信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: int  # minutes
    
    # 位置信息
    location: str
    address: Optional[str] = None
    coordinates: Optional[tuple[float, float]] = None
    
    # 成本信息
    estimated_cost: float = 0.0
    booking_required: bool = False
    booking_url: Optional[str] = None
    
    # 描述信息
    description: Optional[str] = None
    tips: List[str] = []
    photos: List[str] = []
    
    # 评级信息
    popularity_score: Optional[float] = None
    rating: Optional[float] = None
    difficulty_level: Optional[int] = None  # 1-5

class ActivityBooking(BaseModel):
    id: str
    activity_id: str
    name: str
    category: str  # tour, experience, ticket, transport
    
    # 时间信息
    date: date
    start_time: Optional[datetime] = None
    duration: int  # minutes
    
    # 价格信息
    price_per_person: float
    total_price: float
    participants: int
    
    # 预订信息
    booking_reference: Optional[str] = None
    booking_status: str = "selected"
    confirmation_required: bool = False
    cancellation_deadline: Optional[datetime] = None
    
    # 详细信息
    provider: str
    meeting_point: Optional[str] = None
    what_to_bring: List[str] = []
    included: List[str] = []
    excluded: List[str] = []

class WeatherInfo(BaseModel):
    date: date
    temperature_high: int
    temperature_low: int
    condition: str  # sunny, cloudy, rainy, snowy
    humidity: int
    wind_speed: int
    precipitation_chance: int
    uv_index: int
```

### 3. 对话域模型

```python
class ConversationStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class Conversation(BaseModel):
    id: str
    user_id: str
    title: str = Field(default="新对话")
    
    # 会话状态
    status: ConversationStatus = ConversationStatus.ACTIVE
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    # 关联信息
    travel_plan_id: Optional[str] = None
    agent_session_id: Optional[str] = None
    
    # 时间信息
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
    
    # 统计信息
    message_count: int = 0
    total_tokens: int = 0

class Message(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    
    # 消息元数据
    metadata: Dict[str, Any] = {}
    attachments: List["MessageAttachment"] = []
    
    # AI相关
    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    
    # 工具调用
    tool_calls: List["ToolCall"] = []
    tool_call_id: Optional[str] = None
    
    # 时间信息
    timestamp: datetime
    edited_at: Optional[datetime] = None
    
    # 状态信息
    is_deleted: bool = False
    is_edited: bool = False

class MessageAttachment(BaseModel):
    id: str
    type: str  # image, file, location, contact
    name: str
    url: str
    size: Optional[int] = None
    mime_type: Optional[str] = None

class ToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    execution_time: Optional[float] = None
    status: str = "pending"  # pending, completed, failed
```

### 4. 知识库和向量模型

```python
class KnowledgeDocument(BaseModel):
    id: str
    title: str
    content: str
    category: str  # destination, activity, hotel, restaurant, guide
    subcategory: Optional[str] = None
    
    # 向量化信息
    embedding_vector: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # 地理信息
    locations: List[str] = []
    coordinates: Optional[tuple[float, float]] = None
    
    # 元数据
    tags: List[str] = []
    language: str = "zh-cn"
    source: str  # official, user_generated, scraped, manual
    source_url: Optional[str] = None
    author: Optional[str] = None
    
    # 质量指标
    quality_score: float = 0.0
    popularity_score: float = 0.0
    relevance_score: float = 0.0
    
    # 版本控制
    version: int = 1
    created_at: datetime
    updated_at: datetime
    last_verified: Optional[datetime] = None
    
    # 使用统计
    view_count: int = 0
    reference_count: int = 0

class VectorSearchQuery(BaseModel):
    query_text: str
    query_vector: Optional[List[float]] = None
    filters: Dict[str, Any] = {}
    limit: int = Field(default=10, le=100)
    threshold: float = Field(default=0.7, ge=0, le=1)
    
class VectorSearchResult(BaseModel):
    document_id: str
    document: KnowledgeDocument
    similarity_score: float
    rank: int

class UserVector(BaseModel):
    user_id: str
    preference_vector: List[float]
    travel_history_vector: List[float]
    interaction_vector: List[float]
    combined_vector: List[float]
    last_updated: datetime
```

### 5. 智能体域模型

```python
class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    FLIGHT = "flight"
    HOTEL = "hotel"
    ITINERARY = "itinerary"
    BUDGET = "budget"
    LOCAL_GUIDE = "local_guide"
    RECOMMENDATION = "recommendation"

class AgentSession(BaseModel):
    id: str
    conversation_id: str
    user_id: str
    active_agents: List[AgentType] = []
    
    # 会话状态
    current_task: Optional[str] = None
    task_progress: Dict[str, float] = {}
    shared_context: Dict[str, Any] = {}
    
    # 协作信息
    agent_interactions: List["AgentInteraction"] = []
    
    created_at: datetime
    updated_at: datetime

class AgentInteraction(BaseModel):
    id: str
    session_id: str
    from_agent: AgentType
    to_agent: AgentType
    interaction_type: str  # request, response, notification, delegation
    content: Dict[str, Any]
    timestamp: datetime
    
class TaskResult(BaseModel):
    task_id: str
    agent_type: AgentType
    result_type: str  # success, partial, failed
    data: Dict[str, Any]
    confidence_score: float
    execution_time: float
    created_at: datetime
```

## 接口设计

### REST API 接口规范

#### 1. 基础架构

**API版本控制：**
- Base URL: `https://api.ai-travel-planner.com/v1`
- 版本号在URL路径中：`/v1/`, `/v2/`
- 向后兼容策略：保留旧版本至少6个月

**通用响应格式：**
```json
{
  "success": true,
  "data": {},
  "message": "操作成功",
  "code": "SUCCESS",
  "timestamp": "2025-01-27T10:30:00Z",
  "request_id": "req_1234567890abcdef"
}
```

**错误响应格式：**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": [
      {
        "field": "email",
        "message": "邮箱格式不正确"
      }
    ]
  },
  "timestamp": "2025-01-27T10:30:00Z",
  "request_id": "req_1234567890abcdef"
}
```

#### 2. 认证和授权

**JWT Token认证：**
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**OAuth2.0 社交登录：**
```http
POST /v1/auth/oauth/{provider}/callback
Content-Type: application/json

{
  "code": "auth_code_from_provider",
  "state": "random_state_string"
}
```

#### 3. 用户管理接口

```http
POST /v1/auth/register
Content-Type: application/json

{
  "username": "travel_lover",
  "email": "user@example.com",
  "password": "SecurePass123!",
  "phone": "+86-13800138000",
  "verification_code": "123456"
}
```

```http
POST /v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "remember_me": true
}
```

```http
GET /v1/users/profile
Authorization: Bearer {token}

Response:
{
  "success": true,
  "data": {
    "id": "user_123",
    "username": "travel_lover",
    "email": "user@example.com",
    "avatar_url": "https://cdn.example.com/avatars/123.jpg",
    "is_verified": true,
    "is_premium": false,
    "preferences": {
      "language": "zh-cn",
      "currency": "CNY",
      "travel_style": "family",
      "budget_range": [5000, 15000]
    },
    "travel_stats": {
      "trips_planned": 12,
      "countries_visited": 8,
      "total_savings": 25000
    }
  }
}
```

```http
PUT /v1/users/preferences
Authorization: Bearer {token}
Content-Type: application/json

{
  "language": "zh-cn",
  "currency": "CNY",
  "travel_style": "family",
  "budget_range": [8000, 20000],
  "preferred_airlines": ["CA", "MU"],
  "hotel_star_rating": [4, 5],
  "dietary_restrictions": ["vegetarian"],
  "activity_preferences": ["culture", "nature", "food"]
}
```

#### 4. 对话接口

```http
POST /v1/conversations
Authorization: Bearer {token}
Content-Type: application/json

{
  "title": "巴黎7日游规划",
  "initial_message": "我想计划一次去巴黎的7日游，预算大概1.5万元，有什么推荐吗？"
}
```

```http
POST /v1/conversations/{conversation_id}/messages
Authorization: Bearer {token}
Content-Type: application/json

{
  "content": "我希望住在市中心附近，方便游览景点",
  "attachments": [
    {
      "type": "image",
      "url": "https://cdn.example.com/uploads/preference.jpg"
    }
  ]
}
```

```http
GET /v1/conversations/{conversation_id}/messages
Authorization: Bearer {token}
Query Parameters:
- page: 1
- limit: 20
- before_message_id: msg_123

Response:
{
  "success": true,
  "data": {
    "messages": [
      {
        "id": "msg_123",
        "role": "user",
        "content": "我想计划一次去巴黎的7日游",
        "timestamp": "2025-01-27T10:30:00Z",
        "attachments": []
      },
      {
        "id": "msg_124",
        "role": "assistant",
        "content": "很高兴为您规划巴黎之旅！...",
        "timestamp": "2025-01-27T10:30:15Z",
        "tool_calls": [
          {
            "tool_name": "search_flights",
            "arguments": {"from": "BJS", "to": "CDG"},
            "result": "找到3个航班选项..."
          }
        ]
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 45,
      "has_next": true
    }
  }
}
```

#### 5. 旅行规划接口

```http
POST /v1/travel-plans
Authorization: Bearer {token}
Content-Type: application/json

{
  "title": "巴黎浪漫7日游",
  "destinations": [
    {
      "name": "巴黎",
      "country": "法国",
      "city": "巴黎",
      "coordinates": [48.8566, 2.3522],
      "arrival_date": "2025-06-01",
      "departure_date": "2025-06-07",
      "is_primary": true
    }
  ],
  "start_date": "2025-06-01",
  "end_date": "2025-06-07",
  "total_budget": 15000,
  "traveler_count": 2,
  "traveler_details": [
    {
      "name": "张三",
      "age": 28,
      "passport_country": "CN",
      "is_primary": true
    },
    {
      "name": "李四",
      "age": 26,
      "passport_country": "CN"
    }
  ]
}
```

```http
GET /v1/travel-plans
Authorization: Bearer {token}
Query Parameters:
- status: confirmed
- page: 1
- limit: 10
- sort: created_at_desc

Response:
{
  "success": true,
  "data": {
    "plans": [
      {
        "id": "plan_123",
        "title": "巴黎浪漫7日游",
        "status": "confirmed",
        "destinations": ["巴黎"],
        "start_date": "2025-06-01",
        "end_date": "2025-06-07",
        "total_budget": 15000,
        "completion_percentage": 85,
        "created_at": "2025-01-27T10:30:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 5,
      "has_next": false
    }
  }
}
```

```http
PUT /v1/travel-plans/{plan_id}/optimize
Authorization: Bearer {token}
Content-Type: application/json

{
  "optimization_type": "budget",
  "constraints": {
    "max_budget": 12000,
    "must_include": ["埃菲尔铁塔", "卢浮宫"],
    "preferences": {
      "accommodation_type": "hotel",
      "transport_preference": "public"
    }
  }
}
```

#### 6. 搜索和推荐接口

```http
POST /v1/search/flights
Authorization: Bearer {token}
Content-Type: application/json

{
  "departure": {
    "city": "北京",
    "airport": "PEK"
  },
  "arrival": {
    "city": "巴黎",
    "airport": "CDG"
  },
  "departure_date": "2025-06-01",
  "return_date": "2025-06-07",
  "passengers": 2,
  "cabin_class": "economy",
  "filters": {
    "max_stops": 1,
    "airlines": ["CA", "AF"],
    "departure_time": ["06:00", "12:00"]
  }
}
```

```http
POST /v1/search/hotels
Authorization: Bearer {token}
Content-Type: application/json

{
  "destination": "巴黎",
  "check_in": "2025-06-01",
  "check_out": "2025-06-07",
  "guests": 2,
  "rooms": 1,
  "filters": {
    "star_rating": [4, 5],
    "price_range": [500, 1500],
    "amenities": ["wifi", "breakfast"],
    "location": "市中心"
  }
}
```

```http
GET /v1/recommendations/destinations
Authorization: Bearer {token}
Query Parameters:
- season: summer
- budget: 15000
- style: romantic
- duration: 7
- limit: 10

Response:
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "destination": "巴黎",
        "country": "法国",
        "confidence_score": 0.95,
        "match_reasons": ["浪漫氛围", "文化艺术", "美食体验"],
        "estimated_budget": 14500,
        "best_season": "春秋",
        "highlights": ["埃菲尔铁塔", "卢浮宫", "塞纳河游船"],
        "photos": ["url1", "url2"]
      }
    ]
  }
}
```

#### 7. 智能体和工作流接口

```http
POST /v1/agents/sessions
Authorization: Bearer {token}
Content-Type: application/json

{
  "conversation_id": "conv_123",
  "task_type": "comprehensive_planning",
  "requirements": {
    "destination": "巴黎",
    "budget": 15000,
    "duration": 7,
    "travelers": 2
  }
}
```

```http
GET /v1/agents/sessions/{session_id}/status
Authorization: Bearer {token}

Response:
{
  "success": true,
  "data": {
    "session_id": "session_123",
    "status": "processing",
    "progress": {
      "overall": 65,
      "flight_search": 100,
      "hotel_search": 80,
      "itinerary_planning": 30,
      "budget_optimization": 0
    },
    "active_agents": ["flight", "hotel", "itinerary"],
    "estimated_completion": "2025-01-27T10:35:00Z"
  }
}
```

### WebSocket 实时接口

#### 1. 连接管理

**建立连接：**
```javascript
const ws = new WebSocket('wss://api.ai-travel-planner.com/v1/ws/chat/{user_id}?token={jwt_token}');
```

**连接认证：**
```json
{
  "type": "auth",
  "data": {
    "token": "jwt_token_here"
  }
}
```

#### 2. 消息类型

**用户消息：**
```json
{
  "type": "user_message",
  "data": {
    "conversation_id": "conv_123",
    "content": "帮我找找巴黎的酒店",
    "message_id": "msg_456"
  }
}
```

**流式响应：**
```json
{
  "type": "stream_chunk",
  "data": {
    "conversation_id": "conv_123",
    "message_id": "msg_457",
    "chunk": "我为您找到了几家",
    "chunk_index": 5,
    "is_final": false
  }
}
```

**工具调用状态：**
```json
{
  "type": "tool_call_status",
  "data": {
    "conversation_id": "conv_123",
    "tool_call_id": "call_123",
    "tool_name": "search_hotels",
    "status": "executing",
    "progress": 45
  }
}
```

**智能体状态更新：**
```json
{
  "type": "agent_status",
  "data": {
    "session_id": "session_123",
    "agent_type": "flight",
    "status": "completed",
    "result": {
      "flights_found": 5,
      "best_price": 4580
    }
  }
}
```

#### 3. 错误处理

**连接错误：**
```json
{
  "type": "error",
  "data": {
    "code": "AUTH_FAILED",
    "message": "Token已过期，请重新登录"
  }
}
```

**消息错误：**
```json
{
  "type": "message_error",
  "data": {
    "message_id": "msg_456",
    "error_code": "PROCESSING_FAILED",
    "error_message": "消息处理失败，请重试"
  }
}
```

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

**生产环境 docker-compose.prod.yml：**
```yaml
version: '3.8'

networks:
  ai-travel-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  mysql_data:
    driver: local
  redis_data:
    driver: local
  qdrant_data:
    driver: local
  n8n_data:
    driver: local
  elasticsearch_data:
    driver: local
  models_data:
    driver: local

services:
  # 负载均衡器
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./deployment/nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api-gateway
    networks:
      - ai-travel-network
    restart: unless-stopped

  # API网关
  api-gateway:
    build: 
      context: ./services/api-gateway
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - DATABASE_URL=mysql+aiomysql://travel_user:${MYSQL_PASSWORD}@mysql:3306/travel_db
      - REDIS_URL=redis://redis:6379/0
      - VLLM_URL=http://vllm:8001
      - QDRANT_URL=http://qdrant:6333
      - JWT_SECRET=${JWT_SECRET}
      - RATE_LIMIT_PER_MINUTE=60
    depends_on:
      - mysql
      - redis
      - vllm
      - qdrant
    networks:
      - ai-travel-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 对话服务
  chat-service:
    build: 
      context: ./services/chat-service
      dockerfile: Dockerfile.prod
    environment:
      - ENV=production
      - REDIS_URL=redis://redis:6379/1
      - MCP_SERVER_URL=http://mcp-server:8002
      - VLLM_URL=http://vllm:8001
    depends_on:
      - redis
      - mcp-server
      - vllm
    networks:
      - ai-travel-network
    restart: unless-stopped
    deploy:
      replicas: 3

  # 智能体服务
  agent-service:
    build: 
      context: ./services/agent-service
      dockerfile: Dockerfile.prod
    environment:
      - ENV=production
      - DATABASE_URL=mysql+aiomysql://travel_user:${MYSQL_PASSWORD}@mysql:3306/travel_db
      - REDIS_URL=redis://redis:6379/2
      - RAG_SERVICE_URL=http://rag-service:8003
    depends_on:
      - mysql
      - redis
      - rag-service
    networks:
      - ai-travel-network
    restart: unless-stopped
    deploy:
      replicas: 2

  # MCP服务器
  mcp-server:
    build: 
      context: ./services/mcp-server
      dockerfile: Dockerfile.prod
    ports:
      - "8002:8002"
    environment:
      - ENV=production
      - EXTERNAL_API_TIMEOUT=30
      - FLIGHT_API_KEY=${FLIGHT_API_KEY}
      - HOTEL_API_KEY=${HOTEL_API_KEY}
      - WEATHER_API_KEY=${WEATHER_API_KEY}
    networks:
      - ai-travel-network
    restart: unless-stopped

  # RAG服务
  rag-service:
    build: 
      context: ./services/rag-service
      dockerfile: Dockerfile.prod
    ports:
      - "8003:8003"
    environment:
      - ENV=production
      - QDRANT_URL=http://qdrant:6333
      - EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    depends_on:
      - qdrant
    volumes:
      - models_data:/app/models
    networks:
      - ai-travel-network
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # vLLM推理服务
  vllm:
    image: vllm/vllm-openai:v0.6.4
    ports:
      - "8001:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    volumes:
      - models_data:/models
      - /dev/shm:/dev/shm
    command: >
      --model /models/Qwen/Qwen3-32B
      --tensor-parallel-size 2
      --max-model-len 4096
      --gpu-memory-utilization 0.8
      --served-model-name qwen2.5-7b
      --enable-prefix-caching
      --disable-log-requests
    networks:
      - ai-travel-network
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]

  # 用户服务
  user-service:
    build: 
      context: ./services/user-service
      dockerfile: Dockerfile.prod
    environment:
      - ENV=production
      - DATABASE_URL=mysql+aiomysql://travel_user:${MYSQL_PASSWORD}@mysql:3306/travel_db
      - REDIS_URL=redis://redis:6379/3
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - mysql
      - redis
    networks:
      - ai-travel-network
    restart: unless-stopped

  # MySQL数据库
  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
      - MYSQL_DATABASE=travel_db
      - MYSQL_USER=travel_user
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./deployment/mysql/init.sql:/docker-entrypoint-initdb.d/init.sql
      - ./deployment/mysql/my.cnf:/etc/mysql/conf.d/my.cnf
    ports:
      - "3306:3306"
    networks:
      - ai-travel-network
    restart: unless-stopped
    command: --default-authentication-plugin=mysql_native_password

  # Redis缓存
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./deployment/redis/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    networks:
      - ai-travel-network
    restart: unless-stopped

  # Qdrant向量数据库
  qdrant:
    image: qdrant/qdrant:v1.11.3
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
      - ./deployment/qdrant/config.yaml:/qdrant/config/production.yaml
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - ai-travel-network
    restart: unless-stopped

  # Elasticsearch
  elasticsearch:
    image: elasticsearch:8.15.0
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms2g -Xmx2g
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - ai-travel-network
    restart: unless-stopped

  # n8n工作流引擎
  n8n:
    image: n8nio/n8n:1.65.1
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=${N8N_HOST}
      - WEBHOOK_URL=${N8N_WEBHOOK_URL}
      - GENERIC_TIMEZONE=Asia/Shanghai
    volumes:
      - n8n_data:/home/node/.n8n
      - ./deployment/n8n/workflows:/home/node/.n8n/workflows
    networks:
      - ai-travel-network
    restart: unless-stopped

  # 监控服务
  prometheus:
    image: prom/prometheus:v2.54.0
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./deployment/monitoring/rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - ai-travel-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - ./deployment/monitoring/grafana:/etc/grafana/provisioning
      - ./deployment/monitoring/dashboards:/var/lib/grafana/dashboards
    networks:
      - ai-travel-network
    restart: unless-stopped

  # 日志收集
  filebeat:
    image: elastic/filebeat:8.15.0
    user: root
    volumes:
      - ./deployment/logging/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: filebeat -e -strict.perms=false
    networks:
      - ai-travel-network
    restart: unless-stopped
```

**开发环境 docker-compose.dev.yml：**
```yaml
version: '3.8'

services:
  # 简化的开发环境配置
  api-gateway:
    build: 
      context: ./services/api-gateway
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./services/api-gateway:/app
      - /app/node_modules
    environment:
      - ENV=development
      - DEBUG=true
      - HOT_RELOAD=true
    depends_on:
      - mysql-dev
      - redis-dev

  mysql-dev:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      - MYSQL_ROOT_PASSWORD=devpass
      - MYSQL_DATABASE=travel_db_dev
      - MYSQL_USER=dev_user
      - MYSQL_PASSWORD=devpass
    volumes:
      - ./data/mysql-dev:/var/lib/mysql

  redis-dev:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - ./data/redis-dev:/data
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