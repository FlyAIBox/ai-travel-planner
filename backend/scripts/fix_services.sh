#!/bin/bash

# 修复并启动后端服务的脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 项目根目录
PROJECT_ROOT="/root/AI-BOX/code/fly/ai-travel-planner"
BACKEND_ROOT="$PROJECT_ROOT/backend"

# 设置Python路径
export PYTHONPATH="$BACKEND_ROOT:$PYTHONPATH"

# 加载本地环境变量
if [ -f "$BACKEND_ROOT/.env.local" ]; then
    export $(grep -v '^#' "$BACKEND_ROOT/.env.local" | xargs)
    log_info "已加载本地环境配置"
fi

# 停止所有服务
log_info "停止所有后端服务..."
pkill -f "uvicorn main:app" || true
sleep 2

# 启动单个服务的函数
start_service() {
    local service_name=$1
    local service_port=$2
    local service_dir="$BACKEND_ROOT/services/$service_name"
    
    log_info "启动 $service_name 服务 (端口: $service_port)..."
    
    if [ ! -d "$service_dir" ]; then
        log_error "服务目录不存在: $service_dir"
        return 1
    fi
    
    cd "$service_dir"
    mkdir -p logs
    
    # 检查端口是否被占用
    if netstat -tuln 2>/dev/null | grep -q ":$service_port "; then
        log_error "端口 $service_port 已被占用"
        return 1
    fi
    
    # 启动服务
    nohup python -m uvicorn main:app --host 0.0.0.0 --port $service_port --reload > "logs/$service_name.log" 2>&1 &
    local pid=$!
    
    # 等待服务启动
    sleep 3
    
    # 检查服务是否启动成功
    if kill -0 $pid 2>/dev/null; then
        log_success "$service_name 服务启动成功 (PID: $pid)"
        echo $pid > "logs/$service_name.pid"
        return 0
    else
        log_error "$service_name 服务启动失败"
        return 1
    fi
}

# 启动所有服务
log_info "🚀 启动所有后端服务..."

# 按顺序启动服务
start_service "rag-service" 8001
sleep 2
start_service "chat-service" 8080
sleep 2
start_service "user-service" 8003
sleep 2
start_service "planning-service" 8004
sleep 2
start_service "integration-service" 8005
sleep 2
start_service "api-gateway" 8006
sleep 2

log_info "等待服务完全启动..."
sleep 5

# 检查服务健康状态
log_info "检查服务健康状态..."

declare -A health_endpoints
health_endpoints["RAG服务"]="http://localhost:8001/api/v1/health"
health_endpoints["Chat服务"]="http://localhost:8080/api/v1/health"
health_endpoints["User服务"]="http://localhost:8003/api/v1/health"
health_endpoints["Planning服务"]="http://localhost:8004/api/v1/health"
health_endpoints["Integration服务"]="http://localhost:8005/api/v1/health"
health_endpoints["API网关"]="http://localhost:8006/gateway/health"

for service_name in "${!health_endpoints[@]}"; do
    endpoint="${health_endpoints[$service_name]}"
    if curl -s "$endpoint" > /dev/null; then
        log_success "$service_name: ✅ 健康"
    else
        log_error "$service_name: ❌ 不健康"
    fi
done

log_success "✅ 服务启动完成"
