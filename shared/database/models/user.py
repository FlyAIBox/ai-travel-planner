"""
用户域 SQLAlchemy ORM 模型
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List

from sqlalchemy import (
    Boolean, DateTime, String, Text, Integer, Numeric, Date, JSON, Enum as SQLEnum,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from shared.database.connection import Base
from shared.models.user import (
    TravelStyle, Language, Currency, UserStatus, PaymentMethodType, LoyaltyTier
)


# ==================== 用户模型 ====================
class UserORM(Base):
    """用户表"""
    __tablename__ = "users"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="用户ID")
    
    # 基本信息
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="用户名")
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, comment="邮箱地址")
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, comment="手机号码")
    
    # 个人信息
    first_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="名字")
    last_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="姓氏")
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="头像URL")
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, comment="出生日期")
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="性别")
    
    # 地址信息
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="国家")
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="城市")
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="详细地址")
    postal_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, comment="邮政编码")
    
    # 系统信息
    status: Mapped[UserStatus] = mapped_column(
        SQLEnum(UserStatus), 
        nullable=False, 
        default=UserStatus.ACTIVE,
        comment="用户状态"
    )
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否验证")
    is_premium: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否高级用户")
    
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
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="最后登录时间")
    
    # 关联关系
    preferences: Mapped["UserPreferencesORM"] = relationship(
        "UserPreferencesORM", 
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    loyalty_program: Mapped["LoyaltyProgramORM"] = relationship(
        "LoyaltyProgramORM", 
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    payment_methods: Mapped[List["PaymentMethodORM"]] = relationship(
        "PaymentMethodORM", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # 索引
    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_username", "username"),
        Index("idx_users_status", "status"),
        Index("idx_users_created_at", "created_at"),
        Index("idx_users_phone", "phone"),
        CheckConstraint("email LIKE '%@%'", name="check_email_format"),
    )


class UserPreferencesORM(Base):
    """用户偏好设置表"""
    __tablename__ = "user_preferences"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="偏好ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True,
        comment="用户ID"
    )
    
    # 旅行偏好
    travel_styles: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="旅行风格JSON")
    preferred_budget_min: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2), 
        nullable=True,
        comment="最小预算"
    )
    preferred_budget_max: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2), 
        nullable=True,
        comment="最大预算"
    )
    preferred_currency: Mapped[Currency] = mapped_column(
        SQLEnum(Currency), 
        nullable=False, 
        default=Currency.CNY,
        comment="偏好货币"
    )
    
    # 住宿偏好
    preferred_hotel_star: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="偏好酒店星级")
    preferred_room_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="偏好房型")
    smoking_preference: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, comment="吸烟偏好")
    
    # 交通偏好
    preferred_flight_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="偏好航班舱位")
    preferred_airlines: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="偏好航空公司JSON")
    seat_preference: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="座位偏好")
    
    # 餐饮偏好
    dietary_restrictions: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="饮食限制JSON")
    food_preferences: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="美食偏好JSON")
    
    # 活动偏好
    activity_interests: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="活动兴趣JSON")
    physical_activity_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="体力活动水平")
    
    # 系统偏好
    language: Mapped[Language] = mapped_column(
        SQLEnum(Language), 
        nullable=False, 
        default=Language.ZH_CN,
        comment="界面语言"
    )
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="Asia/Shanghai", comment="时区")
    notification_preferences: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="通知偏好JSON")
    
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
    user: Mapped["UserORM"] = relationship("UserORM", back_populates="preferences")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_user_preferences_user_id", "user_id"),
        CheckConstraint(
            "preferred_budget_min IS NULL OR preferred_budget_min >= 0", 
            name="check_budget_min_positive"
        ),
        CheckConstraint(
            "preferred_budget_max IS NULL OR preferred_budget_max >= 0", 
            name="check_budget_max_positive"
        ),
        CheckConstraint(
            "preferred_hotel_star IS NULL OR (preferred_hotel_star >= 1 AND preferred_hotel_star <= 5)", 
            name="check_hotel_star_range"
        ),
    )


class LoyaltyProgramORM(Base):
    """忠诚度计划表"""
    __tablename__ = "loyalty_programs"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="忠诚度计划ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True,
        comment="用户ID"
    )
    
    # 积分信息
    points_balance: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="积分余额")
    lifetime_points: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="累计积分")
    points_expiring_soon: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="即将过期积分")
    
    # 等级信息
    current_tier: Mapped[LoyaltyTier] = mapped_column(
        SQLEnum(LoyaltyTier), 
        nullable=False, 
        default=LoyaltyTier.BRONZE,
        comment="当前等级"
    )
    tier_progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="等级进度百分比")
    next_tier_points_needed: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="下一等级所需积分")
    
    # 统计信息
    total_bookings: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="总预订数")
    total_spending: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), 
        nullable=False, 
        default=Decimal('0'),
        comment="总消费金额"
    )
    
    # 特权
    benefits: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="享有特权JSON")
    discount_rate: Mapped[Decimal] = mapped_column(
        Numeric(5, 4), 
        nullable=False, 
        default=Decimal('0'),
        comment="折扣率"
    )
    
    # 时间戳
    joined_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        server_default=func.now(),
        comment="加入时间"
    )
    tier_achieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="达到当前等级时间")
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="最后活动时间")
    
    # 关联关系
    user: Mapped["UserORM"] = relationship("UserORM", back_populates="loyalty_program")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_loyalty_programs_user_id", "user_id"),
        Index("idx_loyalty_programs_tier", "current_tier"),
        Index("idx_loyalty_programs_points", "points_balance"),
        CheckConstraint("points_balance >= 0", name="check_points_balance_positive"),
        CheckConstraint("lifetime_points >= 0", name="check_lifetime_points_positive"),
        CheckConstraint("tier_progress >= 0 AND tier_progress <= 100", name="check_tier_progress_range"),
        CheckConstraint("discount_rate >= 0 AND discount_rate <= 1", name="check_discount_rate_range"),
    )


class PaymentMethodORM(Base):
    """支付方式表"""
    __tablename__ = "payment_methods"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="支付方式ID")
    user_id: Mapped[str] = mapped_column(
        CHAR(36), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        comment="用户ID"
    )
    
    # 基本信息
    type: Mapped[PaymentMethodType] = mapped_column(
        SQLEnum(PaymentMethodType), 
        nullable=False,
        comment="支付方式类型"
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="支付方式名称")
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否默认支付方式")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否启用")
    
    # 卡片信息（敏感信息需要加密）
    card_last_four: Mapped[Optional[str]] = mapped_column(String(4), nullable=True, comment="卡号后四位")
    card_brand: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="卡片品牌")
    card_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="卡片类型")
    expires_at: Mapped[Optional[date]] = mapped_column(Date, nullable=True, comment="到期日期")
    
    # 第三方支付信息
    external_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="外部支付ID")
    provider_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True, comment="第三方数据JSON")
    
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
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, comment="验证时间")
    
    # 关联关系
    user: Mapped["UserORM"] = relationship("UserORM", back_populates="payment_methods")
    
    # 约束和索引
    __table_args__ = (
        Index("idx_payment_methods_user_id", "user_id"),
        Index("idx_payment_methods_type", "type"),
        Index("idx_payment_methods_active", "is_active"),
        Index("idx_payment_methods_default", "is_default"),
        UniqueConstraint("user_id", "external_id", name="uq_user_external_payment"),
    ) 