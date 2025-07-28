"""
旅行计划域数据模型
包含旅行计划、目的地、预订、行程等模型
"""

from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator

from .user import BaseUser, Currency


# ==================== 枚举类型 ====================
class PlanStatus(str, Enum):
    """旅行计划状态"""
    DRAFT = "draft"              # 草稿
    PLANNING = "planning"        # 规划中
    CONFIRMED = "confirmed"      # 已确认
    BOOKED = "booked"           # 已预订
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"      # 已完成
    CANCELLED = "cancelled"      # 已取消


class BookingStatus(str, Enum):
    """预订状态"""
    PENDING = "pending"          # 待确认
    CONFIRMED = "confirmed"      # 已确认
    CANCELLED = "cancelled"      # 已取消
    REFUNDED = "refunded"        # 已退款
    NO_SHOW = "no_show"         # 未出现


class FlightClass(str, Enum):
    """航班舱位"""
    ECONOMY = "economy"          # 经济舱
    PREMIUM_ECONOMY = "premium_economy"  # 高端经济舱
    BUSINESS = "business"        # 商务舱
    FIRST = "first"             # 头等舱


class AccommodationType(str, Enum):
    """住宿类型"""
    HOTEL = "hotel"             # 酒店
    RESORT = "resort"           # 度假村
    APARTMENT = "apartment"     # 公寓
    HOSTEL = "hostel"          # 青年旅社
    VILLA = "villa"            # 别墅
    B_AND_B = "bnb"            # 民宿
    CAMPING = "camping"        # 露营
    OTHER = "other"            # 其他


class ActivityType(str, Enum):
    """活动类型"""
    SIGHTSEEING = "sightseeing"    # 观光
    ADVENTURE = "adventure"        # 冒险
    CULTURAL = "cultural"          # 文化
    FOOD = "food"                 # 美食
    SHOPPING = "shopping"         # 购物
    ENTERTAINMENT = "entertainment" # 娱乐
    RELAXATION = "relaxation"     # 休闲
    SPORT = "sport"               # 运动
    BUSINESS = "business"         # 商务
    OTHER = "other"               # 其他


class WeatherCondition(str, Enum):
    """天气状况"""
    SUNNY = "sunny"           # 晴朗
    CLOUDY = "cloudy"         # 多云
    RAINY = "rainy"           # 雨天
    STORMY = "stormy"         # 暴风雨
    SNOWY = "snowy"           # 下雪
    FOGGY = "foggy"           # 雾天
    WINDY = "windy"           # 大风


# ==================== 旅行计划模型 ====================
class TravelPlan(BaseUser):
    """旅行计划"""
    
    id: UUID = Field(default_factory=uuid4, description="计划ID")
    user_id: UUID = Field(..., description="用户ID")
    
    # 基本信息
    title: str = Field(..., min_length=1, max_length=200, description="计划标题")
    description: Optional[str] = Field(None, max_length=2000, description="计划描述")
    status: PlanStatus = Field(default=PlanStatus.DRAFT, description="计划状态")
    
    # 时间信息
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    duration_days: Optional[int] = Field(None, ge=1, description="持续天数")
    
    # 预算信息
    total_budget: Optional[Decimal] = Field(None, ge=0, description="总预算")
    currency: Currency = Field(default=Currency.CNY, description="货币")
    estimated_cost: Optional[Decimal] = Field(None, ge=0, description="预估费用")
    actual_cost: Optional[Decimal] = Field(None, ge=0, description="实际费用")
    
    # 旅行者信息
    traveler_count: int = Field(default=1, ge=1, le=20, description="旅行者数量")
    adult_count: int = Field(default=1, ge=1, description="成人数量")
    child_count: int = Field(default=0, ge=0, description="儿童数量")
    infant_count: int = Field(default=0, ge=0, description="婴儿数量")
    
    # 偏好设置
    preferences: Dict[str, Any] = Field(default={}, description="偏好设置")
    tags: List[str] = Field(default=[], description="标签")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """验证日期范围"""
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('结束日期必须大于等于开始日期')
        return v
    
    @validator('duration_days', always=True)
    def calculate_duration(cls, v, values):
        """计算持续天数"""
        if 'start_date' in values and 'end_date' in values:
            delta = values['end_date'] - values['start_date']
            return delta.days + 1
        return v
    
    @validator('adult_count')
    def validate_traveler_count(cls, v, values):
        """验证旅行者数量"""
        if 'traveler_count' in values and v > values['traveler_count']:
            raise ValueError('成人数量不能超过总旅行者数量')
        return v


class Destination(BaseUser):
    """目的地"""
    
    id: UUID = Field(default_factory=uuid4, description="目的地ID")
    plan_id: UUID = Field(..., description="计划ID")
    
    # 位置信息
    name: str = Field(..., min_length=1, max_length=200, description="目的地名称")
    country: str = Field(..., description="国家")
    city: str = Field(..., description="城市")
    region: Optional[str] = Field(None, description="地区")
    
    # 地理坐标
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="纬度")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="经度")
    timezone: Optional[str] = Field(None, description="时区")
    
    # 访问信息
    arrival_date: date = Field(..., description="到达日期")
    departure_date: date = Field(..., description="离开日期")
    stay_duration: Optional[int] = Field(None, ge=1, description="停留天数")
    
    # 描述信息
    description: Optional[str] = Field(None, max_length=1000, description="目的地描述")
    highlights: List[str] = Field(default=[], description="亮点")
    notes: Optional[str] = Field(None, max_length=500, description="备注")
    
    # 排序
    order: int = Field(default=0, ge=0, description="访问顺序")
    
    @validator('departure_date')
    def validate_stay_dates(cls, v, values):
        """验证停留日期"""
        if 'arrival_date' in values and v < values['arrival_date']:
            raise ValueError('离开日期必须大于等于到达日期')
        return v


class TravelerInfo(BaseUser):
    """旅行者信息"""
    
    id: UUID = Field(default_factory=uuid4, description="旅行者ID")
    plan_id: UUID = Field(..., description="计划ID")
    
    # 基本信息
    first_name: str = Field(..., min_length=1, max_length=50, description="名字")
    last_name: str = Field(..., min_length=1, max_length=50, description="姓氏")
    birth_date: date = Field(..., description="出生日期")
    gender: Optional[str] = Field(None, description="性别")
    
    # 身份信息
    passport_number: Optional[str] = Field(None, description="护照号码")
    passport_expiry: Optional[date] = Field(None, description="护照到期日")
    nationality: Optional[str] = Field(None, description="国籍")
    
    # 联系信息
    email: Optional[str] = Field(None, description="邮箱")
    phone: Optional[str] = Field(None, description="电话")
    
    # 特殊需求
    dietary_restrictions: List[str] = Field(default=[], description="饮食限制")
    medical_conditions: List[str] = Field(default=[], description="医疗状况")
    special_requests: List[str] = Field(default=[], description="特殊要求")
    
    # 关系
    relationship_to_primary: Optional[str] = Field(None, description="与主要旅行者关系")
    is_primary: bool = Field(default=False, description="是否主要旅行者")
    
    @property
    def full_name(self) -> str:
        """完整姓名"""
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def age(self) -> int:
        """年龄"""
        today = date.today()
        return today.year - self.birth_date.year - (
            (today.month, today.day) < (self.birth_date.month, self.birth_date.day)
        )


# ==================== 预算模型 ====================
class BudgetBreakdown(BaseUser):
    """预算明细"""
    
    id: UUID = Field(default_factory=uuid4, description="预算明细ID")
    plan_id: UUID = Field(..., description="计划ID")
    
    # 预算分类
    accommodation_budget: Decimal = Field(default=Decimal('0'), ge=0, description="住宿预算")
    transportation_budget: Decimal = Field(default=Decimal('0'), ge=0, description="交通预算")
    food_budget: Decimal = Field(default=Decimal('0'), ge=0, description="餐饮预算")
    activity_budget: Decimal = Field(default=Decimal('0'), ge=0, description="活动预算")
    shopping_budget: Decimal = Field(default=Decimal('0'), ge=0, description="购物预算")
    emergency_budget: Decimal = Field(default=Decimal('0'), ge=0, description="应急预算")
    other_budget: Decimal = Field(default=Decimal('0'), ge=0, description="其他预算")
    
    # 实际花费
    accommodation_spent: Decimal = Field(default=Decimal('0'), ge=0, description="住宿花费")
    transportation_spent: Decimal = Field(default=Decimal('0'), ge=0, description="交通花费")
    food_spent: Decimal = Field(default=Decimal('0'), ge=0, description="餐饮花费")
    activity_spent: Decimal = Field(default=Decimal('0'), ge=0, description="活动花费")
    shopping_spent: Decimal = Field(default=Decimal('0'), ge=0, description="购物花费")
    emergency_spent: Decimal = Field(default=Decimal('0'), ge=0, description="应急花费")
    other_spent: Decimal = Field(default=Decimal('0'), ge=0, description="其他花费")
    
    currency: Currency = Field(default=Currency.CNY, description="货币")
    
    @property
    def total_budget(self) -> Decimal:
        """总预算"""
        return (self.accommodation_budget + self.transportation_budget + 
                self.food_budget + self.activity_budget + self.shopping_budget +
                self.emergency_budget + self.other_budget)
    
    @property
    def total_spent(self) -> Decimal:
        """总花费"""
        return (self.accommodation_spent + self.transportation_spent +
                self.food_spent + self.activity_spent + self.shopping_spent +
                self.emergency_spent + self.other_spent)
    
    @property
    def remaining_budget(self) -> Decimal:
        """剩余预算"""
        return self.total_budget - self.total_spent


# ==================== 航班预订模型 ====================
class FlightBooking(BaseUser):
    """航班预订"""
    
    id: UUID = Field(default_factory=uuid4, description="航班预订ID")
    plan_id: UUID = Field(..., description="计划ID")
    
    # 预订信息
    booking_reference: str = Field(..., description="预订参考号")
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="预订状态")
    
    # 航班信息
    airline: str = Field(..., description="航空公司")
    flight_number: str = Field(..., description="航班号")
    aircraft_type: Optional[str] = Field(None, description="机型")
    
    # 出发信息
    departure_airport: str = Field(..., description="出发机场")
    departure_city: str = Field(..., description="出发城市")
    departure_datetime: datetime = Field(..., description="出发时间")
    departure_terminal: Optional[str] = Field(None, description="出发航站楼")
    departure_gate: Optional[str] = Field(None, description="出发登机口")
    
    # 到达信息
    arrival_airport: str = Field(..., description="到达机场")
    arrival_city: str = Field(..., description="到达城市")
    arrival_datetime: datetime = Field(..., description="到达时间")
    arrival_terminal: Optional[str] = Field(None, description="到达航站楼")
    arrival_gate: Optional[str] = Field(None, description="到达登机口")
    
    # 舱位和座位
    flight_class: FlightClass = Field(..., description="舱位等级")
    seat_numbers: List[str] = Field(default=[], description="座位号")
    
    # 价格信息
    base_price: Decimal = Field(..., ge=0, description="基础价格")
    taxes_and_fees: Decimal = Field(default=Decimal('0'), ge=0, description="税费")
    total_price: Decimal = Field(..., ge=0, description="总价格")
    currency: Currency = Field(default=Currency.CNY, description="货币")
    
    # 行李信息
    checked_baggage_allowance: Optional[str] = Field(None, description="托运行李额度")
    carry_on_allowance: Optional[str] = Field(None, description="随身行李额度")
    
    # 时间戳
    booked_at: datetime = Field(default_factory=datetime.utcnow, description="预订时间")
    check_in_opens_at: Optional[datetime] = Field(None, description="值机开放时间")
    
    @property
    def flight_duration(self) -> int:
        """航班时长（分钟）"""
        delta = self.arrival_datetime - self.departure_datetime
        return int(delta.total_seconds() / 60)


class Layover(BaseUser):
    """中转"""
    
    id: UUID = Field(default_factory=uuid4, description="中转ID")
    flight_booking_id: UUID = Field(..., description="航班预订ID")
    
    # 中转机场信息
    airport: str = Field(..., description="中转机场")
    city: str = Field(..., description="中转城市")
    country: str = Field(..., description="中转国家")
    
    # 时间信息
    arrival_datetime: datetime = Field(..., description="到达时间")
    departure_datetime: datetime = Field(..., description="离开时间")
    duration_minutes: int = Field(..., ge=0, description="中转时长（分钟）")
    
    # 是否需要签证
    visa_required: bool = Field(default=False, description="是否需要签证")
    can_exit_airport: bool = Field(default=True, description="是否可以出机场")


# ==================== 住宿预订模型 ====================
class AccommodationBooking(BaseUser):
    """住宿预订"""
    
    id: UUID = Field(default_factory=uuid4, description="住宿预订ID")
    plan_id: UUID = Field(..., description="计划ID")
    destination_id: Optional[UUID] = Field(None, description="目的地ID")
    
    # 预订信息
    booking_reference: str = Field(..., description="预订参考号")
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="预订状态")
    
    # 住宿信息
    accommodation_type: AccommodationType = Field(..., description="住宿类型")
    name: str = Field(..., description="住宿名称")
    brand: Optional[str] = Field(None, description="品牌")
    star_rating: Optional[int] = Field(None, ge=1, le=5, description="星级")
    
    # 地址信息
    address: str = Field(..., description="地址")
    city: str = Field(..., description="城市")
    country: str = Field(..., description="国家")
    postal_code: Optional[str] = Field(None, description="邮政编码")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="纬度")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="经度")
    
    # 入住信息
    check_in_date: date = Field(..., description="入住日期")
    check_out_date: date = Field(..., description="退房日期")
    nights: int = Field(..., ge=1, description="住宿夜数")
    
    # 房间信息
    room_type: str = Field(..., description="房型")
    room_count: int = Field(default=1, ge=1, description="房间数量")
    guest_count: int = Field(..., ge=1, description="客人数量")
    
    # 价格信息
    nightly_rate: Decimal = Field(..., ge=0, description="每晚价格")
    total_price: Decimal = Field(..., ge=0, description="总价格")
    taxes_and_fees: Decimal = Field(default=Decimal('0'), ge=0, description="税费")
    currency: Currency = Field(default=Currency.CNY, description="货币")
    
    # 设施和服务
    amenities: List[str] = Field(default=[], description="设施")
    included_services: List[str] = Field(default=[], description="包含服务")
    
    # 联系信息
    phone: Optional[str] = Field(None, description="电话")
    email: Optional[str] = Field(None, description="邮箱")
    website: Optional[str] = Field(None, description="网站")
    
    # 特殊要求
    special_requests: List[str] = Field(default=[], description="特殊要求")
    
    # 时间戳
    booked_at: datetime = Field(default_factory=datetime.utcnow, description="预订时间")
    
    @validator('check_out_date')
    def validate_stay_dates(cls, v, values):
        """验证住宿日期"""
        if 'check_in_date' in values and v <= values['check_in_date']:
            raise ValueError('退房日期必须大于入住日期')
        return v
    
    @validator('nights', always=True)
    def calculate_nights(cls, v, values):
        """计算住宿夜数"""
        if 'check_in_date' in values and 'check_out_date' in values:
            delta = values['check_out_date'] - values['check_in_date']
            return delta.days
        return v


# ==================== 行程安排模型 ====================
class ItineraryDay(BaseUser):
    """每日行程"""
    
    id: UUID = Field(default_factory=uuid4, description="每日行程ID")
    plan_id: UUID = Field(..., description="计划ID")
    destination_id: Optional[UUID] = Field(None, description="目的地ID")
    
    # 日期信息
    date: date = Field(..., description="日期")
    day_number: int = Field(..., ge=1, description="第几天")
    
    # 基本信息
    title: Optional[str] = Field(None, max_length=200, description="标题")
    description: Optional[str] = Field(None, max_length=1000, description="描述")
    notes: Optional[str] = Field(None, max_length=500, description="备注")
    
    # 预算
    planned_budget: Decimal = Field(default=Decimal('0'), ge=0, description="计划预算")
    actual_spending: Decimal = Field(default=Decimal('0'), ge=0, description="实际花费")
    
    # 天气信息
    weather_condition: Optional[WeatherCondition] = Field(None, description="天气状况")
    temperature_high: Optional[int] = Field(None, description="最高温度")
    temperature_low: Optional[int] = Field(None, description="最低温度")
    
    # 完成状态
    is_completed: bool = Field(default=False, description="是否完成")
    completion_notes: Optional[str] = Field(None, max_length=500, description="完成备注")


class ItineraryActivity(BaseUser):
    """行程活动"""
    
    id: UUID = Field(default_factory=uuid4, description="活动ID")
    itinerary_day_id: UUID = Field(..., description="每日行程ID")
    
    # 基本信息
    title: str = Field(..., min_length=1, max_length=200, description="活动标题")
    description: Optional[str] = Field(None, max_length=1000, description="活动描述")
    activity_type: ActivityType = Field(..., description="活动类型")
    
    # 时间信息
    start_time: Optional[time] = Field(None, description="开始时间")
    end_time: Optional[time] = Field(None, description="结束时间")
    duration_minutes: Optional[int] = Field(None, ge=0, description="时长（分钟）")
    
    # 位置信息
    location_name: Optional[str] = Field(None, description="位置名称")
    address: Optional[str] = Field(None, description="地址")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="纬度")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="经度")
    
    # 费用信息
    estimated_cost: Decimal = Field(default=Decimal('0'), ge=0, description="预估费用")
    actual_cost: Decimal = Field(default=Decimal('0'), ge=0, description="实际费用")
    currency: Currency = Field(default=Currency.CNY, description="货币")
    
    # 优先级和状态
    priority: int = Field(default=0, ge=0, le=10, description="优先级")
    order: int = Field(default=0, ge=0, description="顺序")
    is_confirmed: bool = Field(default=False, description="是否确认")
    is_completed: bool = Field(default=False, description="是否完成")
    
    # 预订信息
    booking_required: bool = Field(default=False, description="是否需要预订")
    booking_url: Optional[str] = Field(None, description="预订链接")
    contact_info: Optional[str] = Field(None, description="联系信息")
    
    # 标签和备注
    tags: List[str] = Field(default=[], description="标签")
    notes: Optional[str] = Field(None, max_length=500, description="备注")


class ActivityBooking(BaseUser):
    """活动预订"""
    
    id: UUID = Field(default_factory=uuid4, description="活动预订ID")
    activity_id: UUID = Field(..., description="活动ID")
    
    # 预订信息
    booking_reference: str = Field(..., description="预订参考号")
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="预订状态")
    
    # 服务提供商
    provider_name: str = Field(..., description="服务提供商")
    provider_contact: Optional[str] = Field(None, description="联系方式")
    
    # 参与者信息
    participant_count: int = Field(..., ge=1, description="参与者数量")
    participant_names: List[str] = Field(default=[], description="参与者姓名")
    
    # 价格信息
    unit_price: Decimal = Field(..., ge=0, description="单价")
    total_price: Decimal = Field(..., ge=0, description="总价")
    currency: Currency = Field(default=Currency.CNY, description="货币")
    
    # 时间信息
    booking_datetime: datetime = Field(..., description="预订时间")
    activity_datetime: datetime = Field(..., description="活动时间")
    
    # 取消政策
    cancellation_policy: Optional[str] = Field(None, description="取消政策")
    cancellable_until: Optional[datetime] = Field(None, description="可取消截止时间")


# ==================== 天气信息模型 ====================
class WeatherInfo(BaseUser):
    """天气信息"""
    
    id: UUID = Field(default_factory=uuid4, description="天气信息ID")
    destination_id: UUID = Field(..., description="目的地ID")
    
    # 日期和时间
    date: date = Field(..., description="日期")
    forecast_datetime: datetime = Field(..., description="预报时间")
    
    # 温度信息
    temperature_high: int = Field(..., description="最高温度")
    temperature_low: int = Field(..., description="最低温度")
    feels_like: Optional[int] = Field(None, description="体感温度")
    
    # 天气状况
    condition: WeatherCondition = Field(..., description="天气状况")
    description: str = Field(..., description="天气描述")
    
    # 其他天气数据
    humidity: Optional[int] = Field(None, ge=0, le=100, description="湿度百分比")
    wind_speed: Optional[float] = Field(None, ge=0, description="风速")
    wind_direction: Optional[str] = Field(None, description="风向")
    precipitation_chance: Optional[int] = Field(None, ge=0, le=100, description="降水概率")
    uv_index: Optional[int] = Field(None, ge=0, le=11, description="紫外线指数")
    
    # 数据来源
    source: str = Field(..., description="数据来源")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新时间")


# ==================== 请求/响应模型 ====================
class TravelPlanCreate(BaseUser):
    """创建旅行计划请求"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    start_date: date
    end_date: date
    total_budget: Optional[Decimal] = Field(None, ge=0)
    currency: Currency = Field(default=Currency.CNY)
    traveler_count: int = Field(default=1, ge=1, le=20)
    adult_count: int = Field(default=1, ge=1)
    child_count: int = Field(default=0, ge=0)
    infant_count: int = Field(default=0, ge=0)
    preferences: Dict[str, Any] = Field(default={})
    tags: List[str] = Field(default=[])


class TravelPlanUpdate(BaseUser):
    """更新旅行计划请求"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_budget: Optional[Decimal] = Field(None, ge=0)
    status: Optional[PlanStatus] = None
    preferences: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class TravelPlanResponse(BaseUser):
    """旅行计划响应"""
    id: UUID
    title: str
    description: Optional[str]
    status: PlanStatus
    start_date: date
    end_date: date
    duration_days: int
    total_budget: Optional[Decimal]
    currency: Currency
    estimated_cost: Optional[Decimal]
    actual_cost: Optional[Decimal]
    traveler_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config(BaseUser.Config):
        orm_mode = True 