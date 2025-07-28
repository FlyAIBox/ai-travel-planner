"""
用户域数据模型
包含用户、用户偏好、忠诚度计划、支付方式等模型
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, EmailStr, validator, root_validator


# ==================== 枚举类型 ====================
class TravelStyle(str, Enum):
    """旅行风格"""
    LUXURY = "luxury"              # 奢华
    BUDGET = "budget"              # 经济
    ADVENTURE = "adventure"        # 冒险
    CULTURAL = "cultural"          # 文化
    FAMILY = "family"              # 家庭
    BUSINESS = "business"          # 商务
    ROMANTIC = "romantic"          # 浪漫
    SOLO = "solo"                  # 独自
    ECO_FRIENDLY = "eco_friendly"  # 环保


class Language(str, Enum):
    """支持的语言"""
    ZH_CN = "zh-cn"    # 简体中文
    ZH_TW = "zh-tw"    # 繁体中文
    EN_US = "en-us"    # 英语
    JA_JP = "ja-jp"    # 日语
    KO_KR = "ko-kr"    # 韩语
    FR_FR = "fr-fr"    # 法语
    DE_DE = "de-de"    # 德语
    ES_ES = "es-es"    # 西班牙语


class Currency(str, Enum):
    """货币类型"""
    CNY = "CNY"    # 人民币
    USD = "USD"    # 美元
    EUR = "EUR"    # 欧元
    JPY = "JPY"    # 日元
    KRW = "KRW"    # 韩元
    GBP = "GBP"    # 英镑


class UserStatus(str, Enum):
    """用户状态"""
    ACTIVE = "active"        # 活跃
    INACTIVE = "inactive"    # 非活跃
    SUSPENDED = "suspended"  # 暂停
    BANNED = "banned"        # 禁用


class PaymentMethodType(str, Enum):
    """支付方式类型"""
    CREDIT_CARD = "credit_card"    # 信用卡
    DEBIT_CARD = "debit_card"      # 借记卡
    ALIPAY = "alipay"              # 支付宝
    WECHAT_PAY = "wechat_pay"      # 微信支付
    PAYPAL = "paypal"              # PayPal
    BANK_TRANSFER = "bank_transfer" # 银行转账


class LoyaltyTier(str, Enum):
    """忠诚度等级"""
    BRONZE = "bronze"      # 铜牌
    SILVER = "silver"      # 银牌
    GOLD = "gold"          # 金牌
    PLATINUM = "platinum"  # 白金
    DIAMOND = "diamond"    # 钻石


# ==================== 基础模型 ====================
class BaseUser(BaseModel):
    """用户基础模型"""
    
    class Config:
        # 支持任意类型
        arbitrary_types_allowed = True
        # 使用枚举值而非名称
        use_enum_values = True
        # 验证赋值
        validate_assignment = True
        # JSON编码器
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            UUID: lambda v: str(v),
        }


# ==================== 用户模型 ====================
class User(BaseUser):
    """用户模型"""
    
    # 基本信息
    id: UUID = Field(default_factory=uuid4, description="用户ID")
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")
    phone: Optional[str] = Field(None, description="手机号码")
    
    # 个人信息
    first_name: str = Field(..., min_length=1, max_length=50, description="名字")
    last_name: str = Field(..., min_length=1, max_length=50, description="姓氏")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    birth_date: Optional[date] = Field(None, description="出生日期")
    gender: Optional[str] = Field(None, description="性别")
    
    # 地址信息
    country: Optional[str] = Field(None, description="国家")
    city: Optional[str] = Field(None, description="城市")
    address: Optional[str] = Field(None, description="详细地址")
    postal_code: Optional[str] = Field(None, description="邮政编码")
    
    # 系统信息
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="用户状态")
    is_verified: bool = Field(default=False, description="是否验证")
    is_premium: bool = Field(default=False, description="是否高级用户")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    last_login_at: Optional[datetime] = Field(None, description="最后登录时间")
    
    @validator('phone')
    def validate_phone(cls, v):
        """验证手机号码"""
        if v and not v.replace('+', '').replace('-', '').replace(' ', '').isdigit():
            raise ValueError('手机号码格式无效')
        return v
    
    @validator('birth_date')
    def validate_birth_date(cls, v):
        """验证出生日期"""
        if v and v > date.today():
            raise ValueError('出生日期不能是未来日期')
        return v
    
    @property
    def full_name(self) -> str:
        """完整姓名"""
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def age(self) -> Optional[int]:
        """年龄"""
        if not self.birth_date:
            return None
        today = date.today()
        return today.year - self.birth_date.year - (
            (today.month, today.day) < (self.birth_date.month, self.birth_date.day)
        )


class UserPreferences(BaseUser):
    """用户偏好设置"""
    
    id: UUID = Field(default_factory=uuid4, description="偏好ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 旅行偏好
    travel_styles: List[TravelStyle] = Field(default=[], description="旅行风格")
    preferred_budget_min: Optional[Decimal] = Field(None, ge=0, description="最小预算")
    preferred_budget_max: Optional[Decimal] = Field(None, ge=0, description="最大预算")
    preferred_currency: Currency = Field(default=Currency.CNY, description="偏好货币")
    
    # 住宿偏好
    preferred_hotel_star: Optional[int] = Field(None, ge=1, le=5, description="偏好酒店星级")
    preferred_room_type: Optional[str] = Field(None, description="偏好房型")
    smoking_preference: Optional[bool] = Field(None, description="吸烟偏好")
    
    # 交通偏好
    preferred_flight_class: Optional[str] = Field(None, description="偏好航班舱位")
    preferred_airlines: List[str] = Field(default=[], description="偏好航空公司")
    seat_preference: Optional[str] = Field(None, description="座位偏好")
    
    # 餐饮偏好
    dietary_restrictions: List[str] = Field(default=[], description="饮食限制")
    food_preferences: List[str] = Field(default=[], description="美食偏好")
    
    # 活动偏好
    activity_interests: List[str] = Field(default=[], description="活动兴趣")
    physical_activity_level: Optional[str] = Field(None, description="体力活动水平")
    
    # 系统偏好
    language: Language = Field(default=Language.ZH_CN, description="界面语言")
    timezone: str = Field(default="Asia/Shanghai", description="时区")
    notification_preferences: Dict[str, bool] = Field(default={}, description="通知偏好")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    
    @validator('preferred_budget_max')
    def validate_budget_range(cls, v, values):
        """验证预算范围"""
        if v and 'preferred_budget_min' in values and values['preferred_budget_min']:
            if v < values['preferred_budget_min']:
                raise ValueError('最大预算必须大于等于最小预算')
        return v


class LoyaltyProgram(BaseUser):
    """忠诚度计划"""
    
    id: UUID = Field(default_factory=uuid4, description="忠诚度计划ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 积分信息
    points_balance: int = Field(default=0, ge=0, description="积分余额")
    lifetime_points: int = Field(default=0, ge=0, description="累计积分")
    points_expiring_soon: int = Field(default=0, ge=0, description="即将过期积分")
    
    # 等级信息
    current_tier: LoyaltyTier = Field(default=LoyaltyTier.BRONZE, description="当前等级")
    tier_progress: int = Field(default=0, ge=0, le=100, description="等级进度百分比")
    next_tier_points_needed: int = Field(default=0, ge=0, description="下一等级所需积分")
    
    # 统计信息
    total_bookings: int = Field(default=0, ge=0, description="总预订数")
    total_spending: Decimal = Field(default=Decimal('0'), ge=0, description="总消费金额")
    
    # 特权
    benefits: List[str] = Field(default=[], description="享有特权")
    discount_rate: Decimal = Field(default=Decimal('0'), ge=0, le=1, description="折扣率")
    
    # 时间戳
    joined_at: datetime = Field(default_factory=datetime.utcnow, description="加入时间")
    tier_achieved_at: Optional[datetime] = Field(None, description="达到当前等级时间")
    last_activity_at: Optional[datetime] = Field(None, description="最后活动时间")


class PaymentMethod(BaseUser):
    """支付方式"""
    
    id: UUID = Field(default_factory=uuid4, description="支付方式ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 基本信息
    type: PaymentMethodType = Field(..., description="支付方式类型")
    name: str = Field(..., min_length=1, max_length=100, description="支付方式名称")
    is_default: bool = Field(default=False, description="是否默认支付方式")
    is_active: bool = Field(default=True, description="是否启用")
    
    # 卡片信息（加密存储，这里仅做字段定义）
    card_last_four: Optional[str] = Field(None, description="卡号后四位")
    card_brand: Optional[str] = Field(None, description="卡片品牌")
    card_type: Optional[str] = Field(None, description="卡片类型")
    expires_at: Optional[date] = Field(None, description="到期日期")
    
    # 第三方支付信息
    external_id: Optional[str] = Field(None, description="外部支付ID")
    provider_data: Dict[str, Union[str, int, bool]] = Field(default={}, description="第三方数据")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    verified_at: Optional[datetime] = Field(None, description="验证时间")
    
    @validator('expires_at')
    def validate_expiry(cls, v):
        """验证到期日期"""
        if v and v < date.today():
            raise ValueError('支付方式已过期')
        return v


# ==================== 请求/响应模型 ====================
class UserCreate(BaseUser):
    """创建用户请求"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    phone: Optional[str] = None
    birth_date: Optional[date] = None


class UserUpdate(BaseUser):
    """更新用户请求"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    phone: Optional[str] = None
    birth_date: Optional[date] = None
    country: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None


class UserResponse(BaseUser):
    """用户响应"""
    id: UUID
    username: str
    email: EmailStr
    first_name: str
    last_name: str
    avatar_url: Optional[str]
    status: UserStatus
    is_verified: bool
    is_premium: bool
    created_at: datetime
    last_login_at: Optional[datetime]
    
    # 不包含敏感信息
    class Config(BaseUser.Config):
        # 允许从ORM实例创建
        orm_mode = True


class UserPreferencesUpdate(BaseUser):
    """用户偏好更新请求"""
    travel_styles: Optional[List[TravelStyle]] = None
    preferred_budget_min: Optional[Decimal] = Field(None, ge=0)
    preferred_budget_max: Optional[Decimal] = Field(None, ge=0)
    preferred_currency: Optional[Currency] = None
    language: Optional[Language] = None
    timezone: Optional[str] = None


class PaymentMethodCreate(BaseUser):
    """创建支付方式请求"""
    type: PaymentMethodType
    name: str = Field(..., min_length=1, max_length=100)
    is_default: bool = Field(default=False)
    # 实际实现中，敏感信息应通过安全渠道传输
    card_token: Optional[str] = Field(None, description="卡片令牌")
    external_id: Optional[str] = None


# ==================== 批量操作模型 ====================
class UserListResponse(BaseUser):
    """用户列表响应"""
    users: List[UserResponse]
    total: int = Field(..., ge=0, description="总数量")
    page: int = Field(..., ge=1, description="当前页码")
    size: int = Field(..., ge=1, le=100, description="每页大小")
    
    @property
    def total_pages(self) -> int:
        """总页数"""
        return (self.total + self.size - 1) // self.size


# ==================== 统计模型 ====================
class UserStats(BaseUser):
    """用户统计"""
    total_users: int = Field(..., ge=0, description="总用户数")
    active_users: int = Field(..., ge=0, description="活跃用户数")
    verified_users: int = Field(..., ge=0, description="已验证用户数")
    premium_users: int = Field(..., ge=0, description="高级用户数")
    new_users_today: int = Field(..., ge=0, description="今日新用户")
    new_users_this_month: int = Field(..., ge=0, description="本月新用户") 