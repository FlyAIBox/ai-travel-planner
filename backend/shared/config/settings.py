"""
共享配置设置模块
管理所有微服务的配置参数
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置设置"""
    
    # model_config 是 Pydantic v2 配置模型的专用属性，用于定义 Settings 类的全局配置行为。
    # 下面详细解释每个配置项的作用：
    model_config = SettingsConfigDict(
        env_file=".env",                   # 指定环境变量文件路径，优先从 .env 文件加载环境变量
        env_file_encoding="utf-8",         # 指定 .env 文件的编码格式为 UTF-8，确保中文等字符正常读取
        case_sensitive=False,              # 环境变量名不区分大小写，方便在不同操作系统下使用
        extra="ignore",                    # 忽略未在模型中声明的额外字段，防止因多余配置报错
        env_parse_none_str="None",         # 当环境变量值为字符串 "None" 时，自动解析为 Python 的 None
        env_parse_enums=True               # 支持将环境变量自动解析为枚举类型，提升类型安全
    )
    # ==================== 应用基础配置 ====================
    APP_NAME: str = Field(default="AI Travel Planner", description="应用名称")
    APP_VERSION: str = Field(default="1.0.0", description="应用版本")
    ENVIRONMENT: str = Field(default="development", description="运行环境")
    DEBUG: bool = Field(default=True, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    HOST: str = Field(default="0.0.0.0", description="服务主机")
    PORT: int = Field(default=8080, description="服务端口")
    
    # ==================== 安全配置 ====================
    JWT_SECRET: str = Field(default="your_super_secret_jwt_key_here", description="JWT密钥")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT算法")
    JWT_EXPIRE_HOURS: int = Field(default=24, description="JWT过期时间(小时)")
    JWT_REFRESH_EXPIRE_DAYS: int = Field(default=30, description="刷新令牌过期时间(天)")
    
    ENCRYPTION_KEY: str = Field(default="your_encryption_key_here_exactly_32", description="加密密钥")
    PASSWORD_SALT_ROUNDS: int = Field(default=12, description="密码盐轮数")
    
    # CORS和主机配置
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="允许的主机")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="CORS源")

    @field_validator('ALLOWED_HOSTS', 'CORS_ORIGINS', mode='before')
    @classmethod
    def parse_comma_separated_list(cls, v):
        """解析逗号分隔的字符串为列表"""
        if isinstance(v, str):
            # 如果是逗号分隔的字符串，分割为列表
            if ',' in v:
                return [item.strip() for item in v.split(',') if item.strip()]
            # 如果是单个值，返回包含该值的列表
            return [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            return v
        return v
    
    # ==================== 数据库配置 ====================
    DATABASE_URL: str = Field(
        default="mysql+aiomysql://travel_user:travel_pass@localhost:3306/travel_db",
        description="数据库连接URL"
    )
    DATABASE_HOST: str = Field(default="localhost", description="数据库主机")
    DATABASE_PORT: int = Field(default=3306, description="数据库端口")
    DATABASE_NAME: str = Field(default="travel_db", description="数据库名称")
    DATABASE_USER: str = Field(default="travel_user", description="数据库用户")
    DATABASE_PASSWORD: str = Field(default="travel_pass", description="数据库密码")
    DATABASE_POOL_SIZE: int = Field(default=20, description="连接池大小")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, description="连接池最大溢出")
    
    # ==================== 缓存配置 ====================
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis连接URL")
    REDIS_HOST: str = Field(default="localhost", description="Redis主机")
    REDIS_PORT: int = Field(default=6379, description="Redis端口")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis密码")
    REDIS_DB_SESSION: int = Field(default=0, description="会话Redis数据库")
    REDIS_DB_CACHE: int = Field(default=1, description="缓存Redis数据库")
    REDIS_DB_QUEUE: int = Field(default=2, description="队列Redis数据库")
    REDIS_DB_AGENT: int = Field(default=3, description="智能体Redis数据库")
    
    # ==================== AI服务配置 ====================
    # OpenAI配置
    OPENAI_API_KEY: str = Field(default="your_openai_api_key_here", description="OpenAI API密钥")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI API基础URL")
    OPENAI_MODEL: str = Field(default="gpt-4-0125-preview", description="OpenAI模型")
    OPENAI_MAX_TOKENS: int = Field(default=4096, description="OpenAI最大令牌数")
    OPENAI_TEMPERATURE: float = Field(default=0.7, description="OpenAI温度")
    
    # vLLM配置
    VLLM_URL: str = Field(default="http://localhost:8001", description="vLLM服务URL")
    VLLM_MODEL: str = Field(default="Qwen/Qwen3-32B", description="vLLM模型")
    VLLM_MAX_TOKENS: int = Field(default=4096, description="vLLM最大令牌数")
    VLLM_TEMPERATURE: float = Field(default=0.7, description="vLLM温度")
    VLLM_TOP_P: float = Field(default=0.95, description="vLLM Top-P")
    
    # ==================== 向量数据库配置 ====================
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API密钥")
    QDRANT_COLLECTION_NAME: str = Field(default="travel_knowledge", description="Qdrant集合名称")
    QDRANT_VECTOR_SIZE: int = Field(default=384, description="向量维度")
    
    # Embedding模型配置
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="嵌入模型"
    )
    EMBEDDING_BATCH_SIZE: int = Field(default=32, description="嵌入批次大小")
    
    # ==================== 搜索引擎配置 ====================
    ELASTICSEARCH_URL: str = Field(default="http://localhost:9200", description="Elasticsearch URL")
    ELASTICSEARCH_USERNAME: Optional[str] = Field(default=None, description="Elasticsearch用户名")
    ELASTICSEARCH_PASSWORD: Optional[str] = Field(default=None, description="Elasticsearch密码")
    ELASTICSEARCH_INDEX_PREFIX: str = Field(default="travel", description="索引前缀")
    
    # ==================== 外部API配置 - 中国大陆服务 ====================
    # 在线旅游平台API
    CTRIP_API_KEY: str = Field(default="", description="携程API密钥")
    CTRIP_API_SECRET: str = Field(default="", description="携程API密钥")
    CTRIP_API_URL: str = Field(default="https://openapi.ctrip.com", description="携程API地址")
    
    QUNAR_API_KEY: str = Field(default="", description="去哪儿API密钥")
    QUNAR_API_SECRET: str = Field(default="", description="去哪儿API密钥")
    QUNAR_API_URL: str = Field(default="https://open.qunar.com", description="去哪儿API地址")
    
    FLIGGY_API_KEY: str = Field(default="", description="飞猪API密钥")
    FLIGGY_APP_SECRET: str = Field(default="", description="飞猪APP密钥")
    FLIGGY_API_URL: str = Field(default="https://eco.taobao.com/router/rest", description="飞猪API地址")
    
    MEITUAN_API_KEY: str = Field(default="", description="美团API密钥")
    MEITUAN_API_SECRET: str = Field(default="", description="美团API密钥")
    MEITUAN_API_URL: str = Field(default="https://api.meituan.com", description="美团API地址")
    
    # 地图服务API (中国大陆)
    BAIDU_MAP_API_KEY: str = Field(default="", description="百度地图API密钥")
    BAIDU_MAP_API_URL: str = Field(default="https://api.map.baidu.com", description="百度地图API地址")
    
    AMAP_API_KEY: str = Field(default="", description="高德地图API密钥")
    AMAP_API_URL: str = Field(default="https://restapi.amap.com", description="高德地图API地址")
    
    TENCENT_MAP_API_KEY: str = Field(default="", description="腾讯地图API密钥")
    TENCENT_MAP_API_URL: str = Field(default="https://apis.map.qq.com", description="腾讯地图API地址")
    
    # 天气服务API (中国大陆)
    CAIYUN_WEATHER_API_KEY: str = Field(default="", description="彩云天气API密钥")
    CAIYUN_WEATHER_API_URL: str = Field(default="https://api.caiyunapp.com", description="彩云天气API地址")
    
    HEWEATHER_API_KEY: str = Field(default="", description="和风天气API密钥")
    HEWEATHER_API_URL: str = Field(default="https://api.qweather.com", description="和风天气API地址")
    
    XINZHI_WEATHER_API_KEY: str = Field(default="", description="心知天气API密钥")
    XINZHI_WEATHER_API_URL: str = Field(default="https://api.seniverse.com", description="心知天气API地址")
    
    # ==================== MCP服务配置 ====================
    MCP_SERVER_URL: str = Field(default="http://localhost:8002", description="MCP服务器URL")
    MCP_SERVER_HOST: str = Field(default="0.0.0.0", description="MCP服务器主机")
    MCP_SERVER_PORT: int = Field(default=8002, description="MCP服务器端口")
    MCP_TOOL_TIMEOUT: int = Field(default=30, description="工具调用超时时间")
    MCP_MAX_CONCURRENT_TOOLS: int = Field(default=10, description="最大并发工具数")
    
    # ==================== 工作流配置 ====================
    # n8n配置
    N8N_HOST: str = Field(default="localhost", description="n8n主机")
    N8N_PORT: int = Field(default=5678, description="n8n端口")
    N8N_USER: str = Field(default="admin", description="n8n用户")
    N8N_PASSWORD: str = Field(default="your_n8n_password", description="n8n密码")
    N8N_WEBHOOK_URL: str = Field(default="http://localhost:5678/webhook", description="n8n Webhook URL")
    
    # Celery配置
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/2", description="Celery代理URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", description="Celery结果后端")
    CELERY_MAX_RETRIES: int = Field(default=3, description="最大重试次数")
    CELERY_RETRY_DELAY: int = Field(default=60, description="重试延迟(秒)")
    
    # ==================== 监控配置 ====================
    PROMETHEUS_URL: str = Field(default="http://localhost:9090", description="Prometheus URL")
    METRICS_PORT: int = Field(default=8080, description="指标端口")
    
    GRAFANA_URL: str = Field(default="http://localhost:3000", description="Grafana URL")
    GRAFANA_USER: str = Field(default="admin", description="Grafana用户")
    GRAFANA_PASSWORD: str = Field(default="your_grafana_password", description="Grafana密码")
    
    # ==================== 限流配置 ====================
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="每分钟请求限制")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, description="每小时请求限制")
    RATE_LIMIT_PER_DAY: int = Field(default=10000, description="每天请求限制")
    
    # ==================== 开发工具配置 ====================
    TEST_DATABASE_URL: str = Field(
        default="mysql+aiomysql://test_user:test_pass@localhost:3306/travel_test_db",
        description="测试数据库URL"
    )
    ENABLE_PROFILER: bool = Field(default=False, description="启用性能分析器")
    ENABLE_QUERY_LOG: bool = Field(default=False, description="启用查询日志")
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN")
    
    @property
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """判断是否为测试环境"""
        return self.ENVIRONMENT.lower() == "testing"
    
    # ==================== 中国大陆服务配置扩展 ====================
    # 国产大模型配置
    BAIDU_QIANFAN_API_KEY: str = Field(default="", description="百度千帆API密钥")
    BAIDU_QIANFAN_SECRET_KEY: str = Field(default="", description="百度千帆密钥")
    ALIBABA_DASHSCOPE_API_KEY: str = Field(default="", description="阿里云通义千问API密钥")
    TENCENT_HUNYUAN_SECRET_ID: str = Field(default="", description="腾讯混元SecretId")
    TENCENT_HUNYUAN_SECRET_KEY: str = Field(default="", description="腾讯混元SecretKey")
    
    # 社交登录配置
    WECHAT_APP_ID: str = Field(default="", description="微信应用ID")
    WECHAT_APP_SECRET: str = Field(default="", description="微信应用密钥")
    QQ_APP_ID: str = Field(default="", description="QQ应用ID")
    QQ_APP_KEY: str = Field(default="", description="QQ应用密钥")
    WEIBO_APP_KEY: str = Field(default="", description="微博应用密钥")
    WEIBO_APP_SECRET: str = Field(default="", description="微博应用密钥")
    
    # 支付服务配置
    ALIPAY_APP_ID: str = Field(default="", description="支付宝应用ID")
    ALIPAY_PRIVATE_KEY: str = Field(default="", description="支付宝私钥")
    ALIPAY_PUBLIC_KEY: str = Field(default="", description="支付宝公钥")
    WECHAT_PAY_APP_ID: str = Field(default="", description="微信支付应用ID")
    WECHAT_PAY_MCH_ID: str = Field(default="", description="微信支付商户号")
    WECHAT_PAY_API_KEY: str = Field(default="", description="微信支付API密钥")
    
    # 云存储配置
    ALIYUN_OSS_ACCESS_KEY_ID: str = Field(default="", description="阿里云OSS AccessKey")
    ALIYUN_OSS_ACCESS_KEY_SECRET: str = Field(default="", description="阿里云OSS Secret")
    TENCENT_COS_SECRET_ID: str = Field(default="", description="腾讯云COS SecretId")
    TENCENT_COS_SECRET_KEY: str = Field(default="", description="腾讯云COS SecretKey")
    QINIU_ACCESS_KEY: str = Field(default="", description="七牛云AccessKey")
    QINIU_SECRET_KEY: str = Field(default="", description="七牛云SecretKey")
    
    # 地区和语言设置
    DEFAULT_COUNTRY: str = Field(default="CN", description="默认国家")
    DEFAULT_LANGUAGE: str = Field(default="zh-CN", description="默认语言")
    DEFAULT_CURRENCY: str = Field(default="CNY", description="默认货币")
    TIMEZONE: str = Field(default="Asia/Shanghai", description="时区")
    
    # 合规配置
    ICP_LICENSE: str = Field(default="", description="ICP备案号")
    CONTENT_MODERATION_ENABLED: bool = Field(default=True, description="启用内容审核")
    BAIDU_TEXT_CENSOR_API_KEY: str = Field(default="", description="百度内容审核API密钥")


@lru_cache()
def get_settings() -> Settings:
    """获取配置设置（带缓存）"""
    return Settings()


# 获取当前配置
settings = get_settings()

# 导出常用配置
__all__ = ["Settings", "get_settings", "settings"] 