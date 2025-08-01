"""
用户领域数据模型
定义用户、用户偏好、忠诚度计划等模型
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import Field, validator, EmailStr
from .common import BaseModel, IDMixin, TimestampMixin, Contact, Language, Currency, Status


class UserRole(str, Enum):
    """用户角色"""
    GUEST = "guest"          # 游客
    USER = "user"            # 普通用户
    VIP = "vip"              # VIP用户
    ADMIN = "admin"          # 管理员
    SUPER_ADMIN = "super_admin"  # 超级管理员


class Gender(str, Enum):
    """性别"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_SAY = "prefer_not_say"


class TravelStyle(str, Enum):
    """旅行风格"""
    BUDGET = "budget"            # 经济型
    COMFORT = "comfort"          # 舒适型
    LUXURY = "luxury"            # 豪华型
    ADVENTURE = "adventure"      # 冒险型
    CULTURAL = "cultural"        # 文化型
    RELAXATION = "relaxation"    # 休闲型
    BUSINESS = "business"        # 商务型
    FAMILY = "family"            # 家庭型


class TravelFrequency(str, Enum):
    """旅行频率"""
    RARELY = "rarely"        # 很少（1年少于1次）
    OCCASIONALLY = "occasionally"  # 偶尔（1年1-2次）
    REGULARLY = "regularly"  # 经常（1年3-5次）
    FREQUENTLY = "frequently"  # 频繁（1年6次以上）


class AccommodationType(str, Enum):
    """住宿类型偏好"""
    HOTEL = "hotel"              # 酒店
    RESORT = "resort"            # 度假村
    HOSTEL = "hostel"            # 青年旅社
    APARTMENT = "apartment"      # 公寓
    VILLA = "villa"              # 别墅
    BNB = "bnb"                  # 民宿
    CAMPING = "camping"          # 露营


class TransportationType(str, Enum):
    """交通方式偏好"""
    FLIGHT = "flight"            # 飞机
    TRAIN = "train"              # 火车
    BUS = "bus"                  # 巴士
    CAR = "car"                  # 汽车
    MOTORCYCLE = "motorcycle"   # 摩托车
    BICYCLE = "bicycle"          # 自行车
    WALKING = "walking"          # 步行


class ActivityType(str, Enum):
    """活动类型偏好"""
    SIGHTSEEING = "sightseeing"      # 观光
    MUSEUM = "museum"                # 博物馆
    SHOPPING = "shopping"            # 购物
    FOOD = "food"                    # 美食
    NIGHTLIFE = "nightlife"          # 夜生活
    NATURE = "nature"                # 自然风光
    SPORTS = "sports"                # 运动
    PHOTOGRAPHY = "photography"      # 摄影
    HISTORY = "history"              # 历史文化
    ART = "art"                      # 艺术
    MUSIC = "music"                  # 音乐
    FESTIVAL = "festival"            # 节庆活动


class UserProfile(BaseModel):
    """用户资料"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50, description="名")
    last_name: Optional[str] = Field(None, min_length=1, max_length=50, description="姓")
    display_name: Optional[str] = Field(None, min_length=1, max_length=100, description="显示名称")
    gender: Optional[Gender] = Field(None, description="性别")
    birth_date: Optional[date] = Field(None, description="出生日期")
    nationality: Optional[str] = Field(None, description="国籍")
    passport_number: Optional[str] = Field(None, description="护照号码")
    id_number: Optional[str] = Field(None, description="身份证号码")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    
    @property
    def full_name(self) -> Optional[str]:
        if self.first_name and self.last_name:
            return f"{self.last_name} {self.first_name}"
        return self.display_name
    
    @property
    def age(self) -> Optional[int]:
        if self.birth_date:
            today = date.today()
            return today.year - self.birth_date.year - ((today.month, today.day) < (self.birth_date.month, self.birth_date.day))
        return None


class UserPreferences(BaseModel):
    """用户偏好设置"""
    
    # 基础偏好
    language: Language = Field(Language.ZH_CN, description="语言偏好")
    currency: Currency = Field(Currency.CNY, description="货币偏好")
    
    # 旅行偏好
    travel_style: Optional[TravelStyle] = Field(None, description="旅行风格")
    travel_frequency: Optional[TravelFrequency] = Field(None, description="旅行频率")
    budget_range_min: Optional[float] = Field(None, ge=0, description="预算下限")
    budget_range_max: Optional[float] = Field(None, ge=0, description="预算上限")
    
    # 住宿偏好
    accommodation_types: List[AccommodationType] = Field(default_factory=list, description="住宿类型偏好")
    accommodation_rating_min: float = Field(3.0, ge=0, le=5, description="住宿评分最低要求")
    room_type_preference: Optional[str] = Field(None, description="房间类型偏好")
    
    # 交通偏好
    transportation_types: List[TransportationType] = Field(default_factory=list, description="交通方式偏好")
    flight_class_preference: Optional[str] = Field(None, description="航班舱位偏好")
    
    # 活动偏好
    activity_types: List[ActivityType] = Field(default_factory=list, description="活动类型偏好")
    activity_intensity: Optional[str] = Field(None, description="活动强度偏好")
    
    # 饮食偏好
    dietary_restrictions: List[str] = Field(default_factory=list, description="饮食限制")
    cuisine_preferences: List[str] = Field(default_factory=list, description="菜系偏好")
    
    # 目的地偏好
    preferred_destinations: List[str] = Field(default_factory=list, description="偏好目的地")
    avoided_destinations: List[str] = Field(default_factory=list, description="不喜欢的目的地")
    preferred_climate: Optional[str] = Field(None, description="偏好气候")
    
    # 时间偏好
    preferred_trip_duration_min: Optional[int] = Field(None, ge=1, description="偏好行程最短天数")
    preferred_trip_duration_max: Optional[int] = Field(None, ge=1, description="偏好行程最长天数")
    preferred_travel_months: List[int] = Field(default_factory=list, description="偏好旅行月份")
    
    # 同行偏好
    typical_group_size: Optional[int] = Field(None, ge=1, description="典型团队大小")
    travel_with_children: bool = Field(False, description="是否带儿童旅行")
    travel_with_pets: bool = Field(False, description="是否带宠物旅行")
    
    # 通知偏好
    email_notifications: bool = Field(True, description="邮件通知")
    sms_notifications: bool = Field(False, description="短信通知")
    push_notifications: bool = Field(True, description="推送通知")
    price_alert_enabled: bool = Field(True, description="价格提醒")
    
    @validator('budget_range_max')
    def budget_max_must_be_greater_than_min(cls, v, values):
        if v is not None and 'budget_range_min' in values and values['budget_range_min'] is not None:
            if v <= values['budget_range_min']:
                raise ValueError('预算上限必须大于下限')
        return v


class LoyaltyTier(str, Enum):
    """忠诚度等级"""
    BRONZE = "bronze"        # 青铜
    SILVER = "silver"        # 银牌
    GOLD = "gold"            # 金牌
    PLATINUM = "platinum"    # 白金
    DIAMOND = "diamond"      # 钻石


class LoyaltyProgram(BaseModel):
    """忠诚度计划"""
    points: int = Field(0, ge=0, description="积分")
    tier: LoyaltyTier = Field(LoyaltyTier.BRONZE, description="等级")
    tier_progress: float = Field(0, ge=0, le=100, description="等级进度（%）")
    lifetime_points: int = Field(0, ge=0, description="终身积分")
    points_expiry_date: Optional[datetime] = Field(None, description="积分过期时间")
    
    # 统计信息
    total_trips: int = Field(0, ge=0, description="总旅行次数")
    total_spent: float = Field(0, ge=0, description="总消费金额")
    countries_visited: int = Field(0, ge=0, description="访问国家数")
    cities_visited: int = Field(0, ge=0, description="访问城市数")
    
    # 权益
    free_upgrades: int = Field(0, ge=0, description="免费升级次数")
    lounge_access: bool = Field(False, description="贵宾厅权限")
    priority_booking: bool = Field(False, description="优先预订")
    dedicated_support: bool = Field(False, description="专属客服")


class UserSettings(BaseModel):
    """用户设置"""
    
    # 隐私设置
    profile_public: bool = Field(False, description="公开个人资料")
    show_travel_history: bool = Field(False, description="显示旅行历史")
    allow_friend_requests: bool = Field(True, description="允许好友请求")
    
    # 通知设置
    marketing_emails: bool = Field(False, description="营销邮件")
    newsletter: bool = Field(True, description="新闻通讯")
    travel_tips: bool = Field(True, description="旅行贴士")
    
    # 功能设置
    auto_save_searches: bool = Field(True, description="自动保存搜索")
    smart_recommendations: bool = Field(True, description="智能推荐")
    location_tracking: bool = Field(False, description="位置追踪")
    
    # 安全设置
    two_factor_auth: bool = Field(False, description="双因素认证")
    login_notifications: bool = Field(True, description="登录通知")
    password_change_date: Optional[datetime] = Field(None, description="密码修改时间")


class User(IDMixin, TimestampMixin):
    """用户模型"""
    
    # 基础信息
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    phone: Optional[str] = Field(None, description="手机号")
    password_hash: str = Field(..., description="密码哈希")
    
    # 状态信息
    role: UserRole = Field(UserRole.USER, description="用户角色")
    status: Status = Field(Status.ACTIVE, description="账户状态")
    is_verified: bool = Field(False, description="是否已验证")
    is_active: bool = Field(True, description="是否激活")
    
    # 登录信息
    last_login: Optional[datetime] = Field(None, description="最后登录时间")
    login_count: int = Field(0, ge=0, description="登录次数")
    failed_login_attempts: int = Field(0, ge=0, description="失败登录次数")
    locked_until: Optional[datetime] = Field(None, description="锁定到期时间")
    
    # 关联信息
    profile: Optional[UserProfile] = Field(None, description="用户资料")
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="用户偏好")
    loyalty: LoyaltyProgram = Field(default_factory=LoyaltyProgram, description="忠诚度计划")
    settings: UserSettings = Field(default_factory=UserSettings, description="用户设置")
    contact: Optional[Contact] = Field(None, description="联系方式")
    
    # 其他信息
    referral_code: Optional[str] = Field(None, description="推荐码")
    referred_by: Optional[str] = Field(None, description="推荐人ID")
    tags: List[str] = Field(default_factory=list, description="用户标签")
    notes: Optional[str] = Field(None, description="备注")
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.replace('_', '').replace('-', '').isalnum(), '用户名只能包含字母、数字、下划线和连字符'
        return v.lower()
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and not v.replace('+', '').replace('-', '').replace(' ', '').isdigit():
            raise ValueError('手机号格式不正确')
        return v
    
    @property
    def is_locked(self) -> bool:
        """检查账户是否被锁定"""
        if self.locked_until is None:
            return False
        return datetime.now() < self.locked_until
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        if self.profile and self.profile.display_name:
            return self.profile.display_name
        if self.profile and self.profile.full_name:
            return self.profile.full_name
        return self.username


class UserSession(IDMixin, TimestampMixin):
    """用户会话"""
    user_id: str = Field(..., description="用户ID")
    session_token: str = Field(..., description="会话令牌")
    refresh_token: Optional[str] = Field(None, description="刷新令牌")
    expires_at: datetime = Field(..., description="过期时间")
    last_activity: datetime = Field(default_factory=datetime.now, description="最后活动时间")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")
    device_info: Optional[Dict[str, Any]] = Field(None, description="设备信息")
    is_active: bool = Field(True, description="是否激活")
    
    @property
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.now() > self.expires_at


class UserActivity(IDMixin, TimestampMixin):
    """用户活动记录"""
    user_id: str = Field(..., description="用户ID")
    activity_type: str = Field(..., description="活动类型")
    description: str = Field(..., description="活动描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")


class UserFeedback(IDMixin, TimestampMixin):
    """用户反馈"""
    user_id: str = Field(..., description="用户ID")
    type: str = Field(..., description="反馈类型")
    subject: str = Field(..., min_length=1, max_length=200, description="主题")
    content: str = Field(..., min_length=1, max_length=2000, description="内容")
    rating: Optional[int] = Field(None, ge=1, le=5, description="评分")
    status: Status = Field(Status.PENDING, description="处理状态")
    admin_response: Optional[str] = Field(None, description="管理员回复")
    responded_at: Optional[datetime] = Field(None, description="回复时间")
    responded_by: Optional[str] = Field(None, description="回复人ID") 