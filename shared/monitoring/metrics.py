"""
Prometheus监控指标模块
"""

import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info
from fastapi import FastAPI, Request, Response


# 定义指标
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    ['service']
)

SERVICE_INFO = Info(
    'service_info',
    'Service information'
)


def setup_metrics(app: FastAPI):
    """设置监控指标"""
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """监控中间件"""
        start_time = time.time()
        
        response = await call_next(request)
        
        # 记录请求指标
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    
    # 设置服务信息
    SERVICE_INFO.info({
        'version': '1.0.0',
        'service': 'ai-travel-planner'
    })


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.custom_metrics: Dict[str, Any] = {}
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """增加计数器"""
        if labels:
            REQUEST_COUNT.labels(**labels).inc()
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图"""
        if labels:
            REQUEST_DURATION.labels(**labels).observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表"""
        if labels:
            ACTIVE_CONNECTIONS.labels(**labels).set(value)


# 全局指标收集器
metrics_collector = MetricsCollector() 