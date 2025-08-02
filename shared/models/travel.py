"""
旅行领域数据模型
定义旅行计划、目的地、航班预订、住宿预订、活动等模型
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from decimal import Decimal

from pydantic import Field, validator
from .common import (
    BaseModel, IDMixin, TimestampMixin, Location, Money, Currency, 
    Status, Priority, Rating, Weather, Tag
)


class TravelPlanStatus(str, Enum):
    """旅行计划状态"""
    DRAFT = "draft"              # 草稿
    PLANNING = "planning"        # 规划中
    CONFIRMED = "confirmed"      # 已确认
    BOOKED = "booked"           # 已预订
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"      # 已完成
    CANCELLED = "cancelled"      # 已取消


class TripType(str, Enum):
    """行程类型"""
    ROUND_TRIP = "round_trip"    # 往返
    ONE_WAY = "one_way"          # 单程
    MULTI_CITY = "multi_city"    # 多城市
    OPEN_JAW = "open_jaw"        # 缺口行程


class FlightClass(str, Enum):
    """航班舱位"""
    ECONOMY = "economy"          # 经济舱
    PREMIUM_ECONOMY = "premium_economy"  # 超经舱
    BUSINESS = "business"        # 商务舱
    FIRST = "first"              # 头等舱


class AccommodationClass(str, Enum):
    """住宿等级"""
    BUDGET = "budget"            # 经济型
    STANDARD = "standard"        # 标准型
    SUPERIOR = "superior"        # 高级型
    DELUXE = "deluxe"           # 豪华型
    LUXURY = "luxury"           # 奢华型


class TransportationType(str, Enum):
    """交通方式"""
    FLIGHT = "flight"            # 飞机
    TRAIN = "train"              # 火车
    BUS = "bus"                  # 巴士
    CAR_RENTAL = "car_rental"    # 租车
    TAXI = "taxi"                # 出租车
    PRIVATE_CAR = "private_car"  # 私家车
    BOAT = "boat"                # 船只
    WALKING = "walking"          # 步行
    BICYCLE = "bicycle"          # 自行车


class ActivityCategory(str, Enum):
    """活动分类"""
    SIGHTSEEING = "sightseeing"      # 观光
    CULTURAL = "cultural"            # 文化
    ADVENTURE = "adventure"          # 探险
    RELAXATION = "relaxation"        # 休闲
    DINING = "dining"                # 用餐
    SHOPPING = "shopping"            # 购物
    ENTERTAINMENT = "entertainment"   # 娱乐
    SPORTS = "sports"                # 运动
    NATURE = "nature"                # 自然
    BUSINESS = "business"            # 商务


class BookingStatus(str, Enum):
    """预订状态"""
    PENDING = "pending"          # 待确认
    CONFIRMED = "confirmed"      # 已确认
    PAID = "paid"               # 已付款
    CANCELLED = "cancelled"      # 已取消
    REFUNDED = "refunded"       # 已退款
    COMPLETED = "completed"      # 已完成


class Destination(IDMixin, TimestampMixin):
    """目的地模型"""
    
    # 基本信息
    name: str = Field(..., min_length=1, max_length=200, description="目的地名称")
    name_en: Optional[str] = Field(None, description="英文名称")
    description: Optional[str] = Field(None, max_length=2000, description="描述")
    
    # 地理位置
    location: Location = Field(..., description="地理位置")
    timezone: str = Field(..., description="时区")
    country_code: str = Field(..., min_length=2, max_length=3, description="国家代码")
    region: Optional[str] = Field(None, description="地区")
    
    # 分类标签
    tags: List[Tag] = Field(default_factory=list, description="标签")
    categories: List[str] = Field(default_factory=list, description="分类")
    
    # 评价信息
    rating: Optional[Rating] = Field(None, description="评分")
    popularity_score: float = Field(0, ge=0, le=100, description="热门度评分")
    
    # 最佳旅行时间
    best_months: List[int] = Field(default_factory=list, description="最佳旅行月份")
    peak_season: Optional[str] = Field(None, description="旺季")
    
    # 成本信息
    cost_level: int = Field(3, ge=1, le=5, description="消费水平(1-5)")
    avg_daily_cost: Optional[Money] = Field(None, description="平均日消费")
    
    # 媒体资源
    images: List[str] = Field(default_factory=list, description="图片URL列表")
    videos: List[str] = Field(default_factory=list, description="视频URL列表")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    is_active: bool = Field(True, description="是否激活")


class Activity(IDMixin, TimestampMixin):
    """活动模型"""
    
    # 基本信息
    name: str = Field(..., min_length=1, max_length=200, description="活动名称")
    description: Optional[str] = Field(None, max_length=2000, description="活动描述")
    category: ActivityCategory = Field(..., description="活动分类")
    
    # 位置信息
    destination_id: str = Field(..., description="目的地ID")
    location: Optional[Location] = Field(None, description="具体位置")
    address: Optional[str] = Field(None, description="地址")
    
    # 时间信息
    duration_hours: float = Field(..., ge=0, description="持续时间（小时）")
    opening_hours: Optional[str] = Field(None, description="开放时间")
    available_dates: Optional[List[date]] = Field(None, description="可预订日期")
    
    # 价格信息
    price: Optional[Money] = Field(None, description="价格")
    price_per_person: Optional[Money] = Field(None, description="每人价格")
    is_free: bool = Field(False, description="是否免费")
    
    # 要求和限制
    min_age: Optional[int] = Field(None, ge=0, description="最小年龄")
    max_participants: Optional[int] = Field(None, ge=1, description="最大参与人数")
    difficulty_level: int = Field(1, ge=1, le=5, description="难度等级")
    physical_requirement: Optional[str] = Field(None, description="体力要求")
    
    # 评价信息
    rating: Optional[Rating] = Field(None, description="评分")
    
    # 标签和分类
    tags: List[str] = Field(default_factory=list, description="标签")
    
    # 媒体资源
    images: List[str] = Field(default_factory=list, description="图片URL列表")
    
    # 预订信息
    booking_required: bool = Field(False, description="是否需要预订")
    booking_url: Optional[str] = Field(None, description="预订链接")
    
    # 元数据
    provider: Optional[str] = Field(None, description="服务提供商")
    external_id: Optional[str] = Field(None, description="外部ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    is_active: bool = Field(True, description="是否激活")


class Transportation(IDMixin, TimestampMixin):
    """交通模型"""
    
    # 基本信息
    type: TransportationType = Field(..., description="交通方式")
    name: Optional[str] = Field(None, description="交通工具名称")
    
    # 路线信息
    from_location: Location = Field(..., description="出发地")
    to_location: Location = Field(..., description="目的地")
    from_name: str = Field(..., description="出发地名称")
    to_name: str = Field(..., description="目的地名称")
    
    # 时间信息
    departure_time: datetime = Field(..., description="出发时间")
    arrival_time: datetime = Field(..., description="到达时间")
    duration_minutes: int = Field(..., ge=0, description="持续时间（分钟）")
    
    # 价格信息
    price: Money = Field(..., description="价格")
    currency: Currency = Field(..., description="货币")
    
    # 运营商信息
    operator: Optional[str] = Field(None, description="运营商")
    vehicle_number: Optional[str] = Field(None, description="车次/航班号")
    
    # 座位/舱位信息
    class_type: Optional[str] = Field(None, description="舱位/座位类型")
    seat_number: Optional[str] = Field(None, description="座位号")
    
    # 预订信息
    booking_reference: Optional[str] = Field(None, description="预订参考号")
    booking_status: BookingStatus = Field(BookingStatus.PENDING, description="预订状态")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @property
    def duration_hours(self) -> float:
        """持续时间（小时）"""
        return self.duration_minutes / 60.0


class FlightBooking(IDMixin, TimestampMixin):
    """航班预订模型"""
    
    # 基本信息
    airline: str = Field(..., description="航空公司")
    flight_number: str = Field(..., description="航班号")
    aircraft_type: Optional[str] = Field(None, description="机型")
    
    # 路线信息
    departure_airport: str = Field(..., description="出发机场代码")
    arrival_airport: str = Field(..., description="到达机场代码")
    departure_city: str = Field(..., description="出发城市")
    arrival_city: str = Field(..., description="到达城市")
    
    # 时间信息
    departure_time: datetime = Field(..., description="出发时间")
    arrival_time: datetime = Field(..., description="到达时间")
    flight_duration: int = Field(..., ge=0, description="飞行时间（分钟）")
    
    # 舱位信息
    cabin_class: FlightClass = Field(..., description="舱位等级")
    seat_number: Optional[str] = Field(None, description="座位号")
    baggage_allowance: Optional[str] = Field(None, description="行李额度")
    
    # 价格信息
    base_price: Money = Field(..., description="基础价格")
    taxes_and_fees: Money = Field(..., description="税费")
    total_price: Money = Field(..., description="总价格")
    
    # 乘客信息
    passengers: List[Dict[str, Any]] = Field(default_factory=list, description="乘客信息")
    
    # 预订信息
    booking_reference: str = Field(..., description="预订参考号")
    ticket_number: Optional[str] = Field(None, description="票号")
    status: BookingStatus = Field(BookingStatus.PENDING, description="预订状态")
    
    # 服务信息
    meal_preference: Optional[str] = Field(None, description="餐食偏好")
    special_requests: List[str] = Field(default_factory=list, description="特殊要求")
    
    # 元数据
    booking_source: Optional[str] = Field(None, description="预订来源")
    external_booking_id: Optional[str] = Field(None, description="外部预订ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @property
    def flight_duration_hours(self) -> float:
        """飞行时间（小时）"""
        return self.flight_duration / 60.0


class AccommodationBooking(IDMixin, TimestampMixin):
    """住宿预订模型"""
    
    # 基本信息
    hotel_name: str = Field(..., description="酒店名称")
    hotel_chain: Optional[str] = Field(None, description="酒店集团")
    hotel_category: AccommodationClass = Field(..., description="酒店等级")
    
    # 位置信息
    location: Location = Field(..., description="酒店位置")
    address: str = Field(..., description="酒店地址")
    city: str = Field(..., description="城市")
    country: str = Field(..., description="国家")
    
    # 时间信息
    check_in_date: date = Field(..., description="入住日期")
    check_out_date: date = Field(..., description="退房日期")
    nights: int = Field(..., ge=1, description="住宿夜数")
    
    # 房间信息
    room_type: str = Field(..., description="房间类型")
    room_count: int = Field(1, ge=1, description="房间数量")
    guests: int = Field(1, ge=1, description="客人数量")
    
    # 价格信息
    rate_per_night: Money = Field(..., description="每晚价格")
    total_rate: Money = Field(..., description="总房费")
    taxes_and_fees: Money = Field(..., description="税费")
    total_price: Money = Field(..., description="总价格")
    
    # 预订信息
    booking_reference: str = Field(..., description="预订参考号")
    confirmation_number: Optional[str] = Field(None, description="确认号")
    status: BookingStatus = Field(BookingStatus.PENDING, description="预订状态")
    
    # 客人信息
    guest_details: List[Dict[str, Any]] = Field(default_factory=list, description="客人详情")
    
    # 服务和设施
    included_services: List[str] = Field(default_factory=list, description="包含服务")
    amenities: List[str] = Field(default_factory=list, description="设施")
    breakfast_included: bool = Field(False, description="是否含早餐")
    wifi_included: bool = Field(True, description="是否含WiFi")
    parking_included: bool = Field(False, description="是否含停车")
    
    # 政策信息
    cancellation_policy: Optional[str] = Field(None, description="取消政策")
    check_in_time: Optional[str] = Field(None, description="入住时间")
    check_out_time: Optional[str] = Field(None, description="退房时间")
    
    # 特殊要求
    special_requests: List[str] = Field(default_factory=list, description="特殊要求")
    
    # 评价信息
    hotel_rating: Optional[Rating] = Field(None, description="酒店评分")
    
    # 元数据
    booking_source: Optional[str] = Field(None, description="预订来源")
    external_booking_id: Optional[str] = Field(None, description="外部预订ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @validator('check_out_date')
    def check_out_after_check_in(cls, v, values):
        if 'check_in_date' in values and v <= values['check_in_date']:
            raise ValueError('退房日期必须晚于入住日期')
        return v


class Budget(BaseModel):
    """预算模型"""
    
    # 总预算
    total_budget: Money = Field(..., description="总预算")
    currency: Currency = Field(..., description="货币")
    
    # 分类预算
    accommodation_budget: Optional[Money] = Field(None, description="住宿预算")
    transportation_budget: Optional[Money] = Field(None, description="交通预算")
    food_budget: Optional[Money] = Field(None, description="餐饮预算")
    activity_budget: Optional[Money] = Field(None, description="活动预算")
    shopping_budget: Optional[Money] = Field(None, description="购物预算")
    emergency_budget: Optional[Money] = Field(None, description="应急预算")
    
    # 实际支出
    actual_spent: Money = Field(Money(amount=0, currency=Currency.CNY), description="实际支出")
    
    # 统计信息
    remaining_budget: Optional[Money] = Field(None, description="剩余预算")
    budget_utilization: float = Field(0, ge=0, le=100, description="预算使用率（%）")
    
    @property
    def is_over_budget(self) -> bool:
        """是否超预算"""
        return self.actual_spent.amount > self.total_budget.amount


class Itinerary(IDMixin, TimestampMixin):
    """行程安排模型"""
    
    # 基本信息
    date: "date" = Field(..., description="日期")
    day_number: int = Field(..., ge=1, description="第几天")
    title: Optional[str] = Field(None, description="行程标题")
    description: Optional[str] = Field(None, description="行程描述")
    
    # 位置信息
    destination_id: str = Field(..., description="目的地ID")
    city: str = Field(..., description="城市")
    
    # 活动列表
    activities: List[str] = Field(default_factory=list, description="活动ID列表")
    
    # 交通安排
    transportations: List[str] = Field(default_factory=list, description="交通ID列表")
    
    # 住宿信息
    accommodation_id: Optional[str] = Field(None, description="住宿ID")
    
    # 时间安排
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    
    # 预算信息
    estimated_cost: Optional[Money] = Field(None, description="预估费用")
    actual_cost: Optional[Money] = Field(None, description="实际费用")
    
    # 天气信息
    weather: Optional[Weather] = Field(None, description="天气信息")
    
    # 备注和提醒
    notes: Optional[str] = Field(None, description="备注")
    reminders: List[str] = Field(default_factory=list, description="提醒事项")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TravelPlan(IDMixin, TimestampMixin):
    """旅行计划模型"""
    
    # 基本信息
    title: str = Field(..., min_length=1, max_length=200, description="计划标题")
    description: Optional[str] = Field(None, max_length=2000, description="计划描述")
    user_id: str = Field(..., description="用户ID")
    
    # 行程信息
    trip_type: TripType = Field(..., description="行程类型")
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    duration_days: int = Field(..., ge=1, description="行程天数")
    
    # 参与者信息
    travelers_count: int = Field(1, ge=1, description="旅行者数量")
    adults: int = Field(1, ge=1, description="成人数量")
    children: int = Field(0, ge=0, description="儿童数量")
    infants: int = Field(0, ge=0, description="婴儿数量")
    
    # 目的地信息
    destinations: List[str] = Field(..., min_items=1, description="目的地ID列表")
    primary_destination: str = Field(..., description="主要目的地ID")
    
    # 预算信息
    budget: Budget = Field(..., description="预算信息")
    
    # 预订信息
    flight_bookings: List[str] = Field(default_factory=list, description="航班预订ID列表")
    accommodation_bookings: List[str] = Field(default_factory=list, description="住宿预订ID列表")
    activity_bookings: List[str] = Field(default_factory=list, description="活动预订ID列表")
    
    # 行程安排
    itineraries: List[str] = Field(default_factory=list, description="行程安排ID列表")
    
    # 状态信息
    status: TravelPlanStatus = Field(TravelPlanStatus.DRAFT, description="计划状态")
    priority: Priority = Field(Priority.MEDIUM, description="优先级")
    
    # 分享和权限
    is_public: bool = Field(False, description="是否公开")
    shared_with: List[str] = Field(default_factory=list, description="分享用户ID列表")
    
    # 标签和分类
    tags: List[str] = Field(default_factory=list, description="标签")
    categories: List[str] = Field(default_factory=list, description="分类")
    
    # 偏好设置
    travel_style: Optional[str] = Field(None, description="旅行风格")
    accommodation_preference: Optional[str] = Field(None, description="住宿偏好")
    transportation_preference: Optional[str] = Field(None, description="交通偏好")
    
    # 特殊要求
    special_requirements: List[str] = Field(default_factory=list, description="特殊要求")
    dietary_requirements: List[str] = Field(default_factory=list, description="饮食要求")
    accessibility_needs: List[str] = Field(default_factory=list, description="无障碍需求")
    
    # 联系信息
    emergency_contact: Optional[Dict[str, str]] = Field(None, description="紧急联系人")
    
    # 文档和资料
    documents: List[str] = Field(default_factory=list, description="相关文档")
    
    # 评价和反馈
    rating: Optional[Rating] = Field(None, description="计划评分")
    reviews: List[str] = Field(default_factory=list, description="评价ID列表")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('结束日期必须晚于开始日期')
        return v
    
    @validator('duration_days', pre=True, always=True)
    def calculate_duration(cls, v, values):
        if 'start_date' in values and 'end_date' in values:
            return (values['end_date'] - values['start_date']).days + 1
        return v
    
    @property
    def is_past(self) -> bool:
        """是否已过期"""
        return self.end_date < date.today()
    
    @property
    def is_current(self) -> bool:
        """是否正在进行"""
        today = date.today()
        return self.start_date <= today <= self.end_date
    
    @property
    def days_until_trip(self) -> int:
        """距离出行天数"""
        if self.is_past:
            return -1
        return (self.start_date - date.today()).days


class TravelDocument(IDMixin, TimestampMixin):
    """旅行文档模型"""
    
    # 基本信息
    travel_plan_id: str = Field(..., description="旅行计划ID")
    document_type: str = Field(..., description="文档类型")
    title: str = Field(..., description="文档标题")
    
    # 文件信息
    file_url: str = Field(..., description="文件URL")
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., ge=0, description="文件大小")
    file_type: str = Field(..., description="文件类型")
    
    # 文档内容
    content: Optional[str] = Field(None, description="文档内容")
    summary: Optional[str] = Field(None, description="摘要")
    
    # 分类和标签
    category: str = Field(..., description="分类")
    tags: List[str] = Field(default_factory=list, description="标签")
    
    # 权限设置
    is_private: bool = Field(True, description="是否私密")
    access_users: List[str] = Field(default_factory=list, description="访问用户列表")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据") 