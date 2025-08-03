"""
用户相关ORM模型
包含用户、用户偏好、会话等数据库映射
"""

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Integer, DateTime, Date, Boolean, Text,
    JSON, Enum as SQLEnum, Float, Index
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.mysql import CHAR, VARCHAR

from shared.database.connection import Base
from shared.models.user import UserRole, Gender, TravelStyle, LoyaltyTier

class UserORM(Base):
    """用户ORM模型"""
    __tablename__ = "users"
    
    # 主键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    
    # 基础信息
    username: Mapped[str] = mapped_column(VARCHAR(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(VARCHAR(255), unique=True, nullable=False, index=True)
    phone: Mapped[Optional[str]] = mapped_column(VARCHAR(20), nullable=True)
    password_hash: Mapped[str] = mapped_column(VARCHAR(255), nullable=False)
    
    # 状态信息
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    status: Mapped[str] = mapped_column(VARCHAR(20), default="active", nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # 登录信息
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 其他信息
    referral_code: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True, unique=True)
    referred_by: Mapped[Optional[str]] = mapped_column(CHAR(36), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON字符串
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除所有关系映射
    # 如果需要关联数据，请使用手动查询方式
    
    # 索引
    __table_args__ = (
        Index('idx_user_email_status', 'email', 'status'),
        Index('idx_user_username_active', 'username', 'is_active'),
        Index('idx_user_role_status', 'role', 'status'),
        Index('idx_user_created_at', 'created_at'),
    )

class UserProfileORM(Base):
    """用户资料ORM模型"""
    __tablename__ = "user_profiles"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False, unique=True)
    
    # 个人信息
    first_name: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(VARCHAR(100), nullable=True)
    gender: Mapped[Optional[Gender]] = mapped_column(SQLEnum(Gender), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    nationality: Mapped[Optional[str]] = mapped_column(VARCHAR(100), nullable=True)
    passport_number: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True)
    id_number: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(VARCHAR(500), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(VARCHAR(500), nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射

class UserPreferencesORM(Base):
    """用户偏好ORM模型"""
    __tablename__ = "user_preferences"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False, unique=True)
    
    # 基础偏好
    language: Mapped[str] = mapped_column(VARCHAR(10), default="zh-CN", nullable=False)
    currency: Mapped[str] = mapped_column(VARCHAR(3), default="CNY", nullable=False)
    
    # 旅行偏好
    travel_style: Mapped[Optional[TravelStyle]] = mapped_column(SQLEnum(TravelStyle), nullable=True)
    travel_frequency: Mapped[Optional[str]] = mapped_column(VARCHAR(20), nullable=True)
    budget_range_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    budget_range_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # 住宿偏好
    accommodation_types: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    accommodation_rating_min: Mapped[float] = mapped_column(Float, default=3.0, nullable=False)
    room_type_preference: Mapped[Optional[str]] = mapped_column(VARCHAR(100), nullable=True)
    
    # 交通偏好
    transportation_types: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    flight_class_preference: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True)
    
    # 活动偏好
    activity_types: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    activity_intensity: Mapped[Optional[str]] = mapped_column(VARCHAR(50), nullable=True)
    
    # 饮食偏好
    dietary_restrictions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    cuisine_preferences: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    
    # 目的地偏好
    preferred_destinations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    avoided_destinations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    preferred_climate: Mapped[Optional[str]] = mapped_column(VARCHAR(100), nullable=True)
    
    # 时间偏好
    preferred_trip_duration_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    preferred_trip_duration_max: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    preferred_travel_months: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    
    # 同行偏好
    typical_group_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    travel_with_children: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    travel_with_pets: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # 通知偏好
    email_notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sms_notifications: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    push_notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    price_alert_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射

class LoyaltyProgramORM(Base):
    """忠诚度计划ORM模型"""
    __tablename__ = "loyalty_programs"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False, unique=True)
    
    # 积分信息
    points: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tier: Mapped[LoyaltyTier] = mapped_column(SQLEnum(LoyaltyTier), default=LoyaltyTier.BRONZE, nullable=False)
    tier_progress: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    lifetime_points: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    points_expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 统计信息
    total_trips: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_spent: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    countries_visited: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cities_visited: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # 权益
    free_upgrades: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lounge_access: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    priority_booking: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    dedicated_support: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射

class UserSettingsORM(Base):
    """用户设置ORM模型"""
    __tablename__ = "user_settings"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False, unique=True)
    
    # 隐私设置
    profile_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    show_travel_history: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    allow_friend_requests: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # 通知设置
    marketing_emails: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    newsletter: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    travel_tips: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # 功能设置
    auto_save_searches: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    smart_recommendations: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    location_tracking: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # 安全设置
    two_factor_auth: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    login_notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    password_change_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射

class UserSessionORM(Base):
    """用户会话ORM模型"""
    __tablename__ = "user_sessions"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False)
    
    # 会话信息
    session_token: Mapped[str] = mapped_column(VARCHAR(255), nullable=False, unique=True, index=True)
    refresh_token: Mapped[Optional[str]] = mapped_column(VARCHAR(255), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    
    # 设备信息
    ip_address: Mapped[Optional[str]] = mapped_column(VARCHAR(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device_info: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射
    
    # 索引
    __table_args__ = (
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_expires_at', 'expires_at'),
        Index('idx_session_last_activity', 'last_activity'),
    )

class UserActivityORM(Base):
    """用户活动记录ORM模型"""
    __tablename__ = "user_activities"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False)
    
    # 活动信息
    activity_type: Mapped[str] = mapped_column(VARCHAR(100), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    activity_metadata: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # 请求信息
    ip_address: Mapped[Optional[str]] = mapped_column(VARCHAR(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False, index=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射
    
    # 索引
    __table_args__ = (
        Index('idx_activity_user_type', 'user_id', 'activity_type'),
        Index('idx_activity_type_created', 'activity_type', 'created_at'),
    )

class UserFeedbackORM(Base):
    """用户反馈ORM模型"""
    __tablename__ = "user_feedback"
    
    # 主键和外键
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(CHAR(36), nullable=False)
    
    # 反馈信息
    type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(VARCHAR(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(VARCHAR(20), default="pending", nullable=False, index=True)
    
    # 回复信息
    admin_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    responded_by: Mapped[Optional[str]] = mapped_column(CHAR(36), nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False, index=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # 注意：为了避免外键依赖，暂时移除关系映射
    
    # 索引
    __table_args__ = (
        Index('idx_feedback_user_status', 'user_id', 'status'),
        Index('idx_feedback_type_created', 'type', 'created_at'),
    ) 