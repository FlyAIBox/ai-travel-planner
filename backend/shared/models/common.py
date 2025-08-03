"""
通用基础数据模型
定义系统中通用的基础模型和数据结构
"""

from datetime import datetime
from typing import Optional, Any, Dict, List, Union
from enum import Enum
import uuid

from pydantic import BaseModel as PydanticBaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings


class BaseModel(PydanticBaseModel):
    """基础模型类"""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class TimestampMixin(BaseModel):
    """时间戳混入"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class IDMixin(BaseModel):
    """ID混入"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Location(BaseModel):
    """位置信息"""
    latitude: float = Field(..., ge=-90, le=90, description="纬度")
    longitude: float = Field(..., ge=-180, le=180, description="经度")
    address: Optional[str] = Field(None, description="地址")
    city: Optional[str] = Field(None, description="城市")
    country: Optional[str] = Field(None, description="国家")
    postal_code: Optional[str] = Field(None, description="邮政编码")


class DateRange(BaseModel):
    """日期范围"""
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")
    
    @validator('end_date')
    def end_date_must_be_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('结束日期必须晚于开始日期')
        return v


class PaginationModel(BaseModel):
    """分页模型"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    total: Optional[int] = Field(None, description="总数量")
    total_pages: Optional[int] = Field(None, description="总页数")


class FilterModel(BaseModel):
    """过滤模型"""
    keyword: Optional[str] = Field(None, description="关键词")
    category: Optional[str] = Field(None, description="类别")
    tags: Optional[List[str]] = Field(None, description="标签")
    date_range: Optional[DateRange] = Field(None, description="日期范围")


class ResponseStatus(str, Enum):
    """响应状态"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class BaseResponse(BaseModel):
    """基础响应模型"""
    status: ResponseStatus = Field(..., description="响应状态")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")


class SuccessResponse(BaseResponse):
    """成功响应"""
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Optional[Any] = Field(None, description="响应数据")


class ErrorResponse(BaseResponse):
    """错误响应"""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: Optional[str] = Field(None, description="错误代码")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


class PaginatedResponse(SuccessResponse):
    """分页响应"""
    pagination: PaginationModel = Field(..., description="分页信息")


class Currency(str, Enum):
    """货币类型"""
    CNY = "CNY"  # 人民币
    USD = "USD"  # 美元
    EUR = "EUR"  # 欧元
    JPY = "JPY"  # 日元
    GBP = "GBP"  # 英镑
    AUD = "AUD"  # 澳元
    CAD = "CAD"  # 加元


class Money(BaseModel):
    """金钱模型"""
    amount: float = Field(..., ge=0, description="金额")
    currency: Currency = Field(Currency.CNY, description="货币类型")
    
    def __str__(self):
        return f"{self.amount} {self.currency.value}"


class Priority(str, Enum):
    """优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Status(str, Enum):
    """通用状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class FileInfo(BaseModel):
    """文件信息"""
    filename: str = Field(..., description="文件名")
    file_size: int = Field(..., ge=0, description="文件大小（字节）")
    file_type: str = Field(..., description="文件类型")
    mime_type: str = Field(..., description="MIME类型")
    file_url: Optional[str] = Field(None, description="文件URL")
    file_hash: Optional[str] = Field(None, description="文件哈希")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")


class Contact(BaseModel):
    """联系方式"""
    email: Optional[str] = Field(None, description="邮箱")
    phone: Optional[str] = Field(None, description="电话")
    wechat: Optional[str] = Field(None, description="微信")
    qq: Optional[str] = Field(None, description="QQ")
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('邮箱格式不正确')
        return v


class Rating(BaseModel):
    """评分模型"""
    score: float = Field(..., ge=0, le=5, description="评分(0-5)")
    count: int = Field(0, ge=0, description="评分数量")
    
    @property
    def average(self) -> float:
        return round(self.score, 1)


class Tag(BaseModel):
    """标签模型"""
    name: str = Field(..., min_length=1, max_length=50, description="标签名称")
    color: Optional[str] = Field(None, description="标签颜色")
    description: Optional[str] = Field(None, description="标签描述")


class Metrics(BaseModel):
    """指标模型"""
    views: int = Field(0, ge=0, description="浏览次数")
    likes: int = Field(0, ge=0, description="点赞次数")
    shares: int = Field(0, ge=0, description="分享次数")
    comments: int = Field(0, ge=0, description="评论次数")
    bookmarks: int = Field(0, ge=0, description="收藏次数")


class Language(str, Enum):
    """语言类型"""
    ZH_CN = "zh-CN"  # 简体中文
    ZH_TW = "zh-TW"  # 繁体中文
    EN_US = "en-US"  # 英语（美国）
    EN_GB = "en-GB"  # 英语（英国）
    JA_JP = "ja-JP"  # 日语
    KO_KR = "ko-KR"  # 韩语
    FR_FR = "fr-FR"  # 法语
    DE_DE = "de-DE"  # 德语
    ES_ES = "es-ES"  # 西班牙语
    IT_IT = "it-IT"  # 意大利语


class TimeZone(str, Enum):
    """时区"""
    UTC = "UTC"
    ASIA_SHANGHAI = "Asia/Shanghai"
    ASIA_TOKYO = "Asia/Tokyo"
    EUROPE_LONDON = "Europe/London"
    EUROPE_PARIS = "Europe/Paris"
    AMERICA_NEW_YORK = "America/New_York"
    AMERICA_LOS_ANGELES = "America/Los_Angeles"


class WeatherCondition(str, Enum):
    """天气状况"""
    SUNNY = "sunny"          # 晴天
    CLOUDY = "cloudy"        # 多云
    OVERCAST = "overcast"    # 阴天
    RAINY = "rainy"          # 雨天
    SNOWY = "snowy"          # 雪天
    FOGGY = "foggy"          # 雾天
    WINDY = "windy"          # 风天
    STORMY = "stormy"        # 暴风雨


class Weather(BaseModel):
    """天气信息"""
    condition: WeatherCondition = Field(..., description="天气状况")
    temperature: float = Field(..., description="温度（摄氏度）")
    humidity: float = Field(..., ge=0, le=100, description="湿度（%）")
    wind_speed: float = Field(..., ge=0, description="风速（km/h）")
    description: Optional[str] = Field(None, description="天气描述")
    icon: Optional[str] = Field(None, description="天气图标")


class ValidationError(BaseModel):
    """验证错误"""
    field: str = Field(..., description="字段名")
    message: str = Field(..., description="错误消息")
    code: str = Field(..., description="错误代码")


class BulkOperationResult(BaseModel):
    """批量操作结果"""
    total: int = Field(..., ge=0, description="总数")
    success: int = Field(..., ge=0, description="成功数")
    failed: int = Field(..., ge=0, description="失败数")
    errors: List[ValidationError] = Field(default_factory=list, description="错误列表")
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.success / self.total * 100, 2) 