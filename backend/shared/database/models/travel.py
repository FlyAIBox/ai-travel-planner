"""
旅行计划域 SQLAlchemy ORM 模型
"""

from datetime import datetime, date, time
from decimal import Decimal
from typing import Optional, List

from sqlalchemy import (
    Boolean, DateTime, String, Text, Integer, Numeric, Date, Time, JSON, 
    Enum as SQLEnum, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from shared.database.connection import Base
from shared.models.travel import (
    TravelPlanStatus, BookingStatus, FlightClass, AccommodationClass, 
    ActivityCategory, TransportationType
)
from shared.models.common import Currency, WeatherCondition

# ==================== 旅行计划模型 ====================
class TravelPlanORM(Base):
    """旅行计划表"""
    __tablename__ = "travel_plans"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="计划ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="用户ID"
    )
    
    # 基本信息
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment="计划标题")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="计划描述")
    status: Mapped[TravelPlanStatus] = mapped_column(
        SQLEnum(TravelPlanStatus), 
        nullable=False, 
        default=TravelPlanStatus.DRAFT,
        comment="计划状态"
    )
    
    # 时间信息
    start_date: Mapped[date] = mapped_column(Date, nullable=False, comment="开始日期")
    end_date: Mapped[date] = mapped_column(Date, nullable=False, comment="结束日期")
    duration_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="持续天数")
    
    # 预算信息
    total_budget: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 2), 
        nullable=True,
        comment="总预算"
    )
    currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="货币"
    )
    estimated_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 2), 
        nullable=True,
        comment="预估费用"
    )
    actual_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 2), 
        nullable=True,
        comment="实际费用"
    )
    
    # 旅行者信息
    traveler_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="旅行者数量")
    adult_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="成人数量")
    child_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="儿童数量")
    infant_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="婴儿数量")
    
    # 偏好设置
    preferences: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="偏好设置JSON")
    tags: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="标签JSON")
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    
    # 关联关系
    destinations: Mapped[List["DestinationORM"]] = relationship(
        "DestinationORM", 
        back_populates="travel_plan",
        cascade="all, delete-orphan",
        order_by="DestinationORM.order"
    )
    travelers: Mapped[List["TravelerInfoORM"]] = relationship(
        "TravelerInfoORM", 
        back_populates="travel_plan",
        cascade="all, delete-orphan"
    )
    budget_breakdown: Mapped["BudgetBreakdownORM"] = relationship(
        "BudgetBreakdownORM", 
        back_populates="travel_plan",
        uselist=False,
        cascade="all, delete-orphan"
    )
    flight_bookings: Mapped[List["FlightBookingORM"]] = relationship(
        "FlightBookingORM", 
        back_populates="travel_plan",
        cascade="all, delete-orphan"
    )
    accommodation_bookings: Mapped[List["AccommodationBookingORM"]] = relationship(
        "AccommodationBookingORM", 
        back_populates="travel_plan",
        cascade="all, delete-orphan"
    )
    itinerary_days: Mapped[List["ItineraryDayORM"]] = relationship(
        "ItineraryDayORM", 
        back_populates="travel_plan",
        cascade="all, delete-orphan",
        order_by="ItineraryDayORM.day_number"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_travel_plans_user_id", "user_id"),
        Index("idx_travel_plans_status", "status"),
        Index("idx_travel_plans_dates", "start_date", "end_date"),
        Index("idx_travel_plans_created_at", "created_at"),
        CheckConstraint("end_date >= start_date", name="check_date_range"),
        CheckConstraint("traveler_count >= 1", name="check_traveler_count_positive"),
        CheckConstraint("adult_count >= 1", name="check_adult_count_positive"),
        CheckConstraint("total_budget IS NULL OR total_budget >= 0", name="check_budget_positive"),
    )

class DestinationORM(Base):
    """目的地表"""
    __tablename__ = "destinations"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="目的地ID")
    plan_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="计划ID"
    )
    
    # 位置信息
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="目的地名称")
    country: Mapped[str] = mapped_column(String(100), nullable=False, comment="国家")
    city: Mapped[str] = mapped_column(String(100), nullable=False, comment="城市")
    region: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="地区")
    
    # 地理坐标
    latitude: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 7), 
        nullable=True,
        comment="纬度"
    )
    longitude: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 7), 
        nullable=True,
        comment="经度"
    )
    timezone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="时区")
    
    # 访问信息
    arrival_date: Mapped[date] = mapped_column(Date, nullable=False, comment="到达日期")
    departure_date: Mapped[date] = mapped_column(Date, nullable=False, comment="离开日期")
    stay_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="停留天数")
    
    # 描述信息
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="目的地描述")
    highlights: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="亮点JSON")
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="备注")
    
    # 排序
    order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="访问顺序")
    
    # 关联关系
    travel_plan: Mapped["TravelPlanORM"] = relationship("TravelPlanORM", back_populates="destinations")
    accommodations: Mapped[List["AccommodationBookingORM"]] = relationship(
        "AccommodationBookingORM", 
        back_populates="destination"
    )
    itinerary_days: Mapped[List["ItineraryDayORM"]] = relationship(
        "ItineraryDayORM", 
        back_populates="destination"
    )
    weather_info: Mapped[List["WeatherInfoORM"]] = relationship(
        "WeatherInfoORM", 
        back_populates="destination",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_destinations_plan_id", "plan_id"),
        Index("idx_destinations_country_city", "country", "city"),
        Index("idx_destinations_dates", "arrival_date", "departure_date"),
        Index("idx_destinations_order", "plan_id", "order"),
        CheckConstraint("departure_date >= arrival_date", name="check_stay_date_range"),
        CheckConstraint("latitude IS NULL OR (latitude >= -90 AND latitude <= 90)", name="check_latitude_range"),
        CheckConstraint("longitude IS NULL OR (longitude >= -180 AND longitude <= 180)", name="check_longitude_range"),
    )

class TravelerInfoORM(Base):
    """旅行者信息表"""
    __tablename__ = "traveler_info"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="旅行者ID")
    plan_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="计划ID"
    )
    
    # 基本信息
    first_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="名字")
    last_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="姓氏")
    birth_date: Mapped[date] = mapped_column(Date, nullable=False, comment="出生日期")
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="性别")
    
    # 身份信息
    passport_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="护照号码")
    passport_expiry: Mapped[Optional[date]] = mapped_column(Date, nullable=True, comment="护照到期日")
    nationality: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="国籍")
    
    # 联系信息
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="邮箱")
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, comment="电话")
    
    # 特殊需求
    dietary_restrictions: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="饮食限制JSON")
    medical_conditions: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="医疗状况JSON")
    special_requests: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="特殊要求JSON")
    
    # 关系
    relationship_to_primary: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="与主要旅行者关系")
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否主要旅行者")
    
    # 关联关系
    travel_plan: Mapped["TravelPlanORM"] = relationship("TravelPlanORM", back_populates="travelers")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_traveler_info_plan_id", "plan_id"),
        Index("idx_traveler_info_primary", "plan_id", "is_primary"),
        Index("idx_traveler_info_passport", "passport_number"),
    )

# ==================== 预算模型 ====================
class BudgetBreakdownORM(Base):
    """预算明细表"""
    __tablename__ = "budget_breakdowns"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="预算明细ID")
    plan_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        unique=True,
        comment="计划ID"
    )
    
    # 预算分类
    accommodation_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="住宿预算"
    )
    transportation_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="交通预算"
    )
    food_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="餐饮预算"
    )
    activity_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="活动预算"
    )
    shopping_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="购物预算"
    )
    emergency_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="应急预算"
    )
    other_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="其他预算"
    )
    
    # 实际花费
    accommodation_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="住宿花费"
    )
    transportation_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="交通花费"
    )
    food_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="餐饮花费"
    )
    activity_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="活动花费"
    )
    shopping_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="购物花费"
    )
    emergency_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="应急花费"
    )
    other_spent: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="其他花费"
    )
    
    currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="货币"
    )
    
    # 关联关系
    travel_plan: Mapped["TravelPlanORM"] = relationship("TravelPlanORM", back_populates="budget_breakdown")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_budget_breakdowns_plan_id", "plan_id"),
        # 所有预算和花费必须非负
        CheckConstraint("accommodation_budget >= 0", name="check_accommodation_budget_positive"),
        CheckConstraint("transportation_budget >= 0", name="check_transportation_budget_positive"),
        CheckConstraint("food_budget >= 0", name="check_food_budget_positive"),
        CheckConstraint("activity_budget >= 0", name="check_activity_budget_positive"),
        CheckConstraint("shopping_budget >= 0", name="check_shopping_budget_positive"),
        CheckConstraint("emergency_budget >= 0", name="check_emergency_budget_positive"),
        CheckConstraint("other_budget >= 0", name="check_other_budget_positive"),
        CheckConstraint("accommodation_spent >= 0", name="check_accommodation_spent_positive"),
        CheckConstraint("transportation_spent >= 0", name="check_transportation_spent_positive"),
        CheckConstraint("food_spent >= 0", name="check_food_spent_positive"),
        CheckConstraint("activity_spent >= 0", name="check_activity_spent_positive"),
        CheckConstraint("shopping_spent >= 0", name="check_shopping_spent_positive"),
        CheckConstraint("emergency_spent >= 0", name="check_emergency_spent_positive"),
        CheckConstraint("other_spent >= 0", name="check_other_spent_positive"),
    )

# ==================== 预订模型 ====================
class FlightBookingORM(Base):
    """航班预订表"""
    __tablename__ = "flight_bookings"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="航班预订ID")
    plan_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="计划ID"
    )
    
    # 预订信息
    booking_reference: Mapped[str] = mapped_column(String(50), nullable=False, comment="预订参考号")
    status: Mapped[BookingStatus] = mapped_column(
        SQLEnum(BookingStatus), 
        nullable=False, 
        default=BookingStatus.PENDING,
        comment="预订状态"
    )
    
    # 航班信息
    airline: Mapped[str] = mapped_column(String(100), nullable=False, comment="航空公司")
    flight_number: Mapped[str] = mapped_column(String(20), nullable=False, comment="航班号")
    aircraft_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="机型")
    
    # 出发信息
    departure_airport: Mapped[str] = mapped_column(String(10), nullable=False, comment="出发机场")
    departure_city: Mapped[str] = mapped_column(String(100), nullable=False, comment="出发城市")
    departure_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="出发时间")
    departure_terminal: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="出发航站楼")
    departure_gate: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="出发登机口")
    
    # 到达信息
    arrival_airport: Mapped[str] = mapped_column(String(10), nullable=False, comment="到达机场")
    arrival_city: Mapped[str] = mapped_column(String(100), nullable=False, comment="到达城市")
    arrival_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="到达时间")
    arrival_terminal: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="到达航站楼")
    arrival_gate: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="到达登机口")
    
    # 舱位和座位
    flight_class: Mapped[FlightClass] = mapped_column(
        SQLEnum(FlightClass), 
        nullable=False,
        comment="舱位等级"
    )
    seat_numbers: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="座位号JSON")
    
    # 价格信息
    base_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, comment="基础价格")
    taxes_and_fees: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="税费"
    )
    total_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, comment="总价格")
    currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="货币"
    )
    
    # 行李信息
    checked_baggage_allowance: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="托运行李额度")
    carry_on_allowance: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="随身行李额度")
    
    # 时间戳
    booked_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="预订时间"
    )
    check_in_opens_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="值机开放时间")
    
    # 关联关系
    travel_plan: Mapped["TravelPlanORM"] = relationship("TravelPlanORM", back_populates="flight_bookings")
    layovers: Mapped[List["LayoverORM"]] = relationship(
        "LayoverORM", 
        back_populates="flight_booking",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_flight_bookings_plan_id", "plan_id"),
        Index("idx_flight_bookings_reference", "booking_reference"),
        Index("idx_flight_bookings_flight", "airline", "flight_number"),
        Index("idx_flight_bookings_departure", "departure_datetime"),
        Index("idx_flight_bookings_status", "status"),
        CheckConstraint("arrival_datetime > departure_datetime", name="check_flight_duration_positive"),
        CheckConstraint("base_price >= 0", name="check_base_price_positive"),
        CheckConstraint("taxes_and_fees >= 0", name="check_taxes_fees_positive"),
        CheckConstraint("total_price >= 0", name="check_total_price_positive"),
    )

class LayoverORM(Base):
    """中转表"""
    __tablename__ = "layovers"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="中转ID")
    flight_booking_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="航班预订ID"
    )
    
    # 中转机场信息
    airport: Mapped[str] = mapped_column(String(10), nullable=False, comment="中转机场")
    city: Mapped[str] = mapped_column(String(100), nullable=False, comment="中转城市")
    country: Mapped[str] = mapped_column(String(100), nullable=False, comment="中转国家")
    
    # 时间信息
    arrival_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="到达时间")
    departure_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="离开时间")
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, comment="中转时长（分钟）")
    
    # 是否需要签证
    visa_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否需要签证")
    can_exit_airport: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否可以出机场")
    
    # 关联关系
    flight_booking: Mapped["FlightBookingORM"] = relationship("FlightBookingORM", back_populates="layovers")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_layovers_flight_booking_id", "flight_booking_id"),
        Index("idx_layovers_airport", "airport"),
        CheckConstraint("departure_datetime > arrival_datetime", name="check_layover_duration_positive"),
        CheckConstraint("duration_minutes >= 0", name="check_duration_minutes_positive"),
    )

class AccommodationBookingORM(Base):
    """住宿预订表"""
    __tablename__ = "accommodation_bookings"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="住宿预订ID")
    plan_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="计划ID"
    )
    destination_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),        nullable=True,
        comment="目的地ID"
    )
    
    # 预订信息
    booking_reference: Mapped[str] = mapped_column(String(50), nullable=False, comment="预订参考号")
    status: Mapped[BookingStatus] = mapped_column(
        SQLEnum(BookingStatus), 
        nullable=False, 
        default=BookingStatus.PENDING,
        comment="预订状态"
    )
    
    # 住宿信息
    accommodation_type: Mapped[AccommodationClass] = mapped_column(
        SQLEnum(AccommodationClass), 
        nullable=False,
        comment="住宿类型"
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="住宿名称")
    brand: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="品牌")
    star_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="星级")
    
    # 地址信息
    address: Mapped[str] = mapped_column(Text, nullable=False, comment="地址")
    city: Mapped[str] = mapped_column(String(100), nullable=False, comment="城市")
    country: Mapped[str] = mapped_column(String(100), nullable=False, comment="国家")
    postal_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, comment="邮政编码")
    latitude: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 7), 
        nullable=True,
        comment="纬度"
    )
    longitude: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 7), 
        nullable=True,
        comment="经度"
    )
    
    # 入住信息
    check_in_date: Mapped[date] = mapped_column(Date, nullable=False, comment="入住日期")
    check_out_date: Mapped[date] = mapped_column(Date, nullable=False, comment="退房日期")
    nights: Mapped[int] = mapped_column(Integer, nullable=False, comment="住宿夜数")
    
    # 房间信息
    room_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="房型")
    room_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="房间数量")
    guest_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="客人数量")
    
    # 价格信息
    nightly_rate: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, comment="每晚价格")
    total_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, comment="总价格")
    taxes_and_fees: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="税费"
    )
    currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="货币"
    )
    
    # 设施和服务
    amenities: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="设施JSON")
    included_services: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="包含服务JSON")
    
    # 联系信息
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, comment="电话")
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="邮箱")
    website: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="网站")
    
    # 特殊要求
    special_requests: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="特殊要求JSON")
    
    # 时间戳
    booked_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="预订时间"
    )
    
    # 关联关系
    travel_plan: Mapped["TravelPlanORM"] = relationship("TravelPlanORM", back_populates="accommodation_bookings")
    destination: Mapped[Optional["DestinationORM"]] = relationship("DestinationORM", back_populates="accommodations")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_accommodation_bookings_plan_id", "plan_id"),
        Index("idx_accommodation_bookings_destination_id", "destination_id"),
        Index("idx_accommodation_bookings_reference", "booking_reference"),
        Index("idx_accommodation_bookings_dates", "check_in_date", "check_out_date"),
        Index("idx_accommodation_bookings_status", "status"),
        Index("idx_accommodation_bookings_location", "city", "country"),
        CheckConstraint("check_out_date > check_in_date", name="check_stay_duration_positive"),
        CheckConstraint("nights >= 1", name="check_nights_positive"),
        CheckConstraint("room_count >= 1", name="check_room_count_positive"),
        CheckConstraint("guest_count >= 1", name="check_guest_count_positive"),
        CheckConstraint("star_rating IS NULL OR (star_rating >= 1 AND star_rating <= 5)", name="check_star_rating_range"),
        CheckConstraint("nightly_rate >= 0", name="check_nightly_rate_positive"),
        CheckConstraint("total_price >= 0", name="check_total_price_positive"),
        CheckConstraint("taxes_and_fees >= 0", name="check_taxes_fees_positive"),
    )

# ==================== 行程安排模型 ====================
class ItineraryDayORM(Base):
    """每日行程表"""
    __tablename__ = "itinerary_days"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="每日行程ID")
    plan_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="计划ID"
    )
    destination_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),        nullable=True,
        comment="目的地ID"
    )
    
    # 日期信息
    date: Mapped[date] = mapped_column(Date, nullable=False, comment="日期")
    day_number: Mapped[int] = mapped_column(Integer, nullable=False, comment="第几天")
    
    # 基本信息
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="标题")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="描述")
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="备注")
    
    # 预算
    planned_budget: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="计划预算"
    )
    actual_spending: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="实际花费"
    )
    
    # 天气信息
    weather_condition: Mapped[Optional[WeatherCondition]] = mapped_column(
        SQLEnum(WeatherCondition), 
        nullable=True,
        comment="天气状况"
    )
    temperature_high: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="最高温度")
    temperature_low: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="最低温度")
    
    # 完成状态
    is_completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否完成")
    completion_notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="完成备注")
    
    # 关联关系
    travel_plan: Mapped["TravelPlanORM"] = relationship("TravelPlanORM", back_populates="itinerary_days")
    destination: Mapped[Optional["DestinationORM"]] = relationship("DestinationORM", back_populates="itinerary_days")
    activities: Mapped[List["ItineraryActivityORM"]] = relationship(
        "ItineraryActivityORM", 
        back_populates="itinerary_day",
        cascade="all, delete-orphan",
        order_by="ItineraryActivityORM.order"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_itinerary_days_plan_id", "plan_id"),
        Index("idx_itinerary_days_destination_id", "destination_id"),
        Index("idx_itinerary_days_date", "date"),
        Index("idx_itinerary_days_day_number", "plan_id", "day_number"),
        CheckConstraint("day_number >= 1", name="check_day_number_positive"),
        CheckConstraint("planned_budget >= 0", name="check_planned_budget_positive"),
        CheckConstraint("actual_spending >= 0", name="check_actual_spending_positive"),
    )

class ItineraryActivityORM(Base):
    """行程活动表"""
    __tablename__ = "itinerary_activities"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="活动ID")
    itinerary_day_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="每日行程ID"
    )
    
    # 基本信息
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment="活动标题")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="活动描述")
    activity_type: Mapped[ActivityCategory] = mapped_column(
        SQLEnum(ActivityCategory), 
        nullable=False,
        comment="活动类型"
    )
    
    # 时间信息
    start_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True, comment="开始时间")
    end_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True, comment="结束时间")
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="时长（分钟）")
    
    # 位置信息
    location_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="位置名称")
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="地址")
    latitude: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 7), 
        nullable=True,
        comment="纬度"
    )
    longitude: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 7), 
        nullable=True,
        comment="经度"
    )
    
    # 费用信息
    estimated_cost: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="预估费用"
    )
    actual_cost: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="实际费用"
    )
    currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="货币"
    )
    
    # 优先级和状态
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="优先级")
    order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="顺序")
    is_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否确认")
    is_completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否完成")
    
    # 预订信息
    booking_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否需要预订")
    booking_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="预订链接")
    contact_info: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="联系信息")
    
    # 标签和备注
    tags: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="标签JSON")
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="备注")
    
    # 关联关系
    itinerary_day: Mapped["ItineraryDayORM"] = relationship("ItineraryDayORM", back_populates="activities")
    activity_bookings: Mapped[List["ActivityBookingORM"]] = relationship(
        "ActivityBookingORM", 
        back_populates="activity",
        cascade="all, delete-orphan"
    )
    
    # 约束和索引
    __table_args__ = (
        Index("idx_itinerary_activities_day_id", "itinerary_day_id"),
        Index("idx_itinerary_activities_type", "activity_type"),
        Index("idx_itinerary_activities_time", "start_time", "end_time"),
        Index("idx_itinerary_activities_order", "itinerary_day_id", "order"),
        Index("idx_itinerary_activities_priority", "priority"),
        CheckConstraint("end_time IS NULL OR start_time IS NULL OR end_time >= start_time", name="check_activity_time_range"),
        CheckConstraint("duration_minutes IS NULL OR duration_minutes >= 0", name="check_duration_positive"),
        CheckConstraint("priority >= 0 AND priority <= 10", name="check_priority_range"),
        CheckConstraint("estimated_cost >= 0", name="check_estimated_cost_positive"),
        CheckConstraint("actual_cost >= 0", name="check_actual_cost_positive"),
    )

class ActivityBookingORM(Base):
    """活动预订表"""
    __tablename__ = "activity_bookings"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="活动预订ID")
    activity_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="活动ID"
    )
    
    # 预订信息
    booking_reference: Mapped[str] = mapped_column(String(50), nullable=False, comment="预订参考号")
    status: Mapped[BookingStatus] = mapped_column(
        SQLEnum(BookingStatus), 
        nullable=False, 
        default=BookingStatus.PENDING,
        comment="预订状态"
    )
    
    # 服务提供商
    provider_name: Mapped[str] = mapped_column(String(200), nullable=False, comment="服务提供商")
    provider_contact: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="联系方式")
    
    # 参与者信息
    participant_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="参与者数量")
    participant_names: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="参与者姓名JSON")
    
    # 价格信息
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, comment="单价")
    total_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, comment="总价")
    currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="货币"
    )
    
    # 时间信息
    booking_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="预订时间")
    activity_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="活动时间")
    
    # 取消政策
    cancellation_policy: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="取消政策")
    cancellable_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="可取消截止时间")
    
    # 关联关系
    activity: Mapped["ItineraryActivityORM"] = relationship("ItineraryActivityORM", back_populates="activity_bookings")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_activity_bookings_activity_id", "activity_id"),
        Index("idx_activity_bookings_reference", "booking_reference"),
        Index("idx_activity_bookings_status", "status"),
        Index("idx_activity_bookings_datetime", "activity_datetime"),
        CheckConstraint("participant_count >= 1", name="check_participant_count_positive"),
        CheckConstraint("unit_price >= 0", name="check_unit_price_positive"),
        CheckConstraint("total_price >= 0", name="check_total_price_positive"),
    )

# ==================== 天气信息模型 ====================
class WeatherInfoORM(Base):
    """天气信息表"""
    __tablename__ = "weather_info"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="天气信息ID")
    destination_id: Mapped[str] = mapped_column(
        CHAR(36),        nullable=False,
        comment="目的地ID"
    )
    
    # 日期和时间
    date: Mapped[date] = mapped_column(Date, nullable=False, comment="日期")
    forecast_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="预报时间")
    
    # 温度信息
    temperature_high: Mapped[int] = mapped_column(Integer, nullable=False, comment="最高温度")
    temperature_low: Mapped[int] = mapped_column(Integer, nullable=False, comment="最低温度")
    feels_like: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="体感温度")
    
    # 天气状况
    condition: Mapped[WeatherCondition] = mapped_column(
        SQLEnum(WeatherCondition), 
        nullable=False,
        comment="天气状况"
    )
    description: Mapped[str] = mapped_column(String(200), nullable=False, comment="天气描述")
    
    # 其他天气数据
    humidity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="湿度百分比")
    wind_speed: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 1), nullable=True, comment="风速")
    wind_direction: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="风向")
    precipitation_chance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="降水概率")
    uv_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="紫外线指数")
    
    # 数据来源
    source: Mapped[str] = mapped_column(String(100), nullable=False, comment="数据来源")
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="最后更新时间"
    )
    
    # 关联关系
    destination: Mapped["DestinationORM"] = relationship("DestinationORM", back_populates="weather_info")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_weather_info_destination_id", "destination_id"),
        Index("idx_weather_info_date", "date"),
        Index("idx_weather_info_forecast_datetime", "forecast_datetime"),
        CheckConstraint("humidity IS NULL OR (humidity >= 0 AND humidity <= 100)", name="check_humidity_range"),
        CheckConstraint("wind_speed IS NULL OR wind_speed >= 0", name="check_wind_speed_positive"),
        CheckConstraint("precipitation_chance IS NULL OR (precipitation_chance >= 0 AND precipitation_chance <= 100)", name="check_precipitation_range"),
        CheckConstraint("uv_index IS NULL OR (uv_index >= 0 AND uv_index <= 11)", name="check_uv_index_range"),
    ) 