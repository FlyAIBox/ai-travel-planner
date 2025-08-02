"""
日志工具模块
提供统一的日志配置和管理功能
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class LoggerManager:
    """日志管理器"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def configure(cls, 
                  log_level: str = "INFO",
                  log_dir: str = "logs",
                  max_file_size: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5,
                  enable_console: bool = True,
                  enable_file: bool = True,
                  enable_json: bool = False,
                  enable_colors: bool = True):
        """配置日志系统"""
        
        if cls._configured:
            return
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 设置根日志级别
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            
            if enable_colors and sys.stdout.isatty():
                console_format = ColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                console_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            console_handler.setFormatter(console_format)
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if enable_file:
            # 应用日志文件
            app_log_file = log_path / "app.log"
            file_handler = logging.handlers.RotatingFileHandler(
                app_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, log_level.upper()))
            
            if enable_json:
                file_format = JSONFormatter()
            else:
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)
            
            # 错误日志文件
            error_log_file = log_path / "error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_format)
            root_logger.addHandler(error_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取日志器"""
        if name not in cls._loggers:
            # 确保日志系统已配置
            if not cls._configured:
                cls.configure()
            
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


def get_logger(name: str = None, **kwargs) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称，通常使用 __name__
        **kwargs: 日志配置参数
    
    Returns:
        logging.Logger: 配置好的日志器实例
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条信息日志")
        >>> logger.error("这是一条错误日志")
    """
    if name is None:
        name = __name__
    
    # 如果传入了配置参数，重新配置日志系统
    if kwargs:
        LoggerManager.configure(**kwargs)
    
    return LoggerManager.get_logger(name)


def configure_logging(config: Dict[str, Any] = None):
    """
    配置日志系统
    
    Args:
        config: 日志配置字典
    """
    if config is None:
        config = {}
    
    # 默认配置
    default_config = {
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_dir': os.getenv('LOG_DIR', 'logs'),
        'max_file_size': int(os.getenv('LOG_MAX_FILE_SIZE', 10 * 1024 * 1024)),
        'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
        'enable_console': os.getenv('LOG_ENABLE_CONSOLE', 'true').lower() == 'true',
        'enable_file': os.getenv('LOG_ENABLE_FILE', 'true').lower() == 'true',
        'enable_json': os.getenv('LOG_ENABLE_JSON', 'false').lower() == 'true',
        'enable_colors': os.getenv('LOG_ENABLE_COLORS', 'true').lower() == 'true'
    }
    
    # 合并配置
    final_config = {**default_config, **config}
    
    LoggerManager.configure(**final_config)


# 创建一些常用的日志器
def get_service_logger(service_name: str) -> logging.Logger:
    """获取服务专用日志器"""
    return get_logger(f"service.{service_name}")


def get_api_logger() -> logging.Logger:
    """获取API日志器"""
    return get_logger("api")


def get_db_logger() -> logging.Logger:
    """获取数据库日志器"""
    return get_logger("database")


def get_cache_logger() -> logging.Logger:
    """获取缓存日志器"""
    return get_logger("cache")


# 日志装饰器
def log_function_call(logger: logging.Logger = None):
    """
    函数调用日志装饰器
    
    Args:
        logger: 日志器实例，如果为None则使用默认日志器
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            func_logger.debug(f"调用函数: {func.__name__}, args: {args}, kwargs: {kwargs}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"函数 {func.__name__} 执行成功")
                return result
            except Exception as e:
                func_logger.error(f"函数 {func.__name__} 执行失败: {e}")
                raise
        
        return wrapper
    return decorator


# 异步日志装饰器
def log_async_function_call(logger: logging.Logger = None):
    """
    异步函数调用日志装饰器
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            func_logger.debug(f"调用异步函数: {func.__name__}, args: {args}, kwargs: {kwargs}")
            
            try:
                result = await func(*args, **kwargs)
                func_logger.debug(f"异步函数 {func.__name__} 执行成功")
                return result
            except Exception as e:
                func_logger.error(f"异步函数 {func.__name__} 执行失败: {e}")
                raise
        
        return wrapper
    return decorator


# 初始化日志系统（在模块导入时自动配置）
configure_logging()
