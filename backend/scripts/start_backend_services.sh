#!/bin/bash

# AI Travel Planner 后端服务统一启动脚本
# 用于开发环境下逐个启动后端服务

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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取项目根目录
get_project_root() {
    # 获取脚本的绝对路径
    local script_path="$(readlink -f "$0")"
    # 获取脚本所在目录
    local script_dir="$(dirname "$script_path")"
    # 脚本在 backend/scripts/ 目录下，所以项目根目录是 ../../
    local project_root="$(cd "$script_dir/../.." && pwd)"
    echo "$project_root"
}

# 检查Python环境
check_python_env() {
    log_info "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python 未安装，请先安装Python 3.10+"
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $python_version"
    
    # 检查是否在虚拟环境中
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        log_success "检测到虚拟环境: $VIRTUAL_ENV"
    else
        log_warning "未检测到虚拟环境，建议使用虚拟环境"
    fi
}

# 检查依赖是否安装
check_dependencies() {
    log_info "检查Python依赖..."
    
    local project_root=$(get_project_root)
    cd "$project_root/backend"
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    # 检查关键依赖
    if ! python -c "import fastapi" &> /dev/null; then
        log_warning "FastAPI未安装，正在安装依赖..."
        pip install -r requirements.txt
    else
        log_success "Python依赖检查通过"
    fi
}

# 检查基础服务
check_base_services() {
    log_info "检查基础服务状态..."

    # 检查Redis容器是否运行
    if ! docker ps | grep -q "ai-travel-redis-dev"; then
        log_error "Redis容器未运行，请先启动基础服务"
        log_info "运行: docker compose -f deployment/docker/docker-compose.dev.yml up -d redis mysql qdrant"
        exit 1
    fi

    # 检查Redis服务是否响应
    if ! docker exec ai-travel-redis-dev redis-cli ping &> /dev/null; then
        log_error "Redis服务未响应，请检查Redis容器状态"
        log_info "运行: docker compose -f deployment/docker/docker-compose.dev.yml up -d redis mysql qdrant"
        exit 1
    fi

    # 检查MySQL容器是否运行
    if ! docker ps | grep -q "ai-travel-mysql-dev"; then
        log_error "MySQL容器未运行，请先启动基础服务"
        log_info "运行: docker compose -f deployment/docker/docker-compose.dev.yml up -d redis mysql qdrant"
        exit 1
    fi

    # 检查MySQL服务是否响应
    if ! docker exec ai-travel-mysql-dev mysqladmin ping -h localhost --silent &> /dev/null; then
        log_error "MySQL服务未响应，请检查MySQL容器状态"
        exit 1
    fi

    # 检查Qdrant容器是否运行
    if ! docker ps | grep -q "ai-travel-qdrant-dev"; then
        log_error "Qdrant容器未运行，请先启动基础服务"
        log_info "运行: docker compose -f deployment/docker/docker-compose.dev.yml up -d redis mysql qdrant"
        exit 1
    fi

    # 检查Qdrant服务是否响应
    if ! curl -s http://localhost:6333/health > /dev/null; then
        log_error "Qdrant服务未响应，请检查Qdrant容器状态"
        exit 1
    fi

    log_success "基础服务检查通过"
}

# 启动单个服务
start_service() {
    local service_name=$1
    local service_port=$2
    local service_path=$3

    log_info "启动 $service_name 服务 (端口: $service_port)..."

    # 使用绝对路径
    local service_dir="/root/AI-BOX/code/fly/ai-travel-planner/backend/$service_path"

    if [ ! -d "$service_dir" ]; then
        log_error "服务目录不存在: $service_dir"
        return 1
    fi

    cd "$service_dir"

    # 检查端口是否被占用
    if netstat -tuln 2>/dev/null | grep -q ":$service_port "; then
        log_warning "$service_name 端口 $service_port 已被占用，跳过启动"
        return 0
    fi

    # 创建日志目录
    mkdir -p logs

    # 后台启动服务
    nohup python -m uvicorn main:app --host 0.0.0.0 --port $service_port --reload > "logs/${service_name}.log" 2>&1 &
    local pid=$!

    # 等待服务启动
    sleep 3

    # 检查服务是否启动成功
    if kill -0 $pid 2>/dev/null; then
        log_success "$service_name 服务启动成功 (PID: $pid)"
        echo $pid > "logs/${service_name}.pid"
    else
        log_error "$service_name 服务启动失败"
        return 1
    fi
}

# 启动所有后端服务
start_all_services() {
    local project_root=$(get_project_root)
    
    # 创建日志目录
    mkdir -p "$project_root/backend/logs"
    
    log_info "🚀 启动所有后端服务..."
    
    # 定义服务列表 (服务名 端口 路径)
    declare -a services=(
        "rag-service 8001 services/rag-service"
        "agent-service 8002 services/agent-service"
        "user-service 8003 services/user-service"
        "planning-service 8004 services/planning-service"
        "integration-service 8005 services/integration-service"
        "chat-service 8080 services/chat-service"
        "api-gateway 8006 services/api-gateway"
    )
    
    # 启动各个服务
    for service_info in "${services[@]}"; do
        read -r service_name service_port service_path <<< "$service_info"
        start_service "$service_name" "$service_port" "$service_path"
        sleep 2
    done
}

# 检查服务健康状态
check_services_health() {
    log_info "检查服务健康状态..."
    
    declare -A health_endpoints
    health_endpoints["Chat服务"]="http://localhost:8080/api/v1/health"
    health_endpoints["RAG服务"]="http://localhost:8001/api/v1/health"
    health_endpoints["Agent服务"]="http://localhost:8002/api/v1/health"
    health_endpoints["User服务"]="http://localhost:8003/api/v1/health"
    health_endpoints["Planning服务"]="http://localhost:8004/api/v1/health"
    health_endpoints["Integration服务"]="http://localhost:8005/api/v1/health"
    health_endpoints["API网关"]="http://localhost:8006/gateway/health"
    
    for service_name in "${!health_endpoints[@]}"; do
        local endpoint="${health_endpoints[$service_name]}"
        if curl -s "$endpoint" > /dev/null; then
            log_success "$service_name: ✅ 健康"
        else
            log_error "$service_name: ❌ 不健康"
        fi
    done
}

# 停止所有服务
stop_all_services() {
    log_info "停止所有后端服务..."
    
    local project_root=$(get_project_root)
    cd "$project_root/backend"
    
    # 停止所有Python服务进程
    pkill -f "uvicorn main:app" || true
    
    # 删除PID文件
    rm -f logs/*.pid
    
    log_success "所有后端服务已停止"
}

# 显示服务状态
show_status() {
    log_info "后端服务状态:"
    
    declare -a ports=(8080 8001 8002 8003 8004 8005 8006)
    
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo "  端口 $port: ✅ 运行中"
        else
            echo "  端口 $port: ❌ 未运行"
        fi
    done
}

# 显示帮助信息
show_help() {
    echo "AI Travel Planner 后端服务管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start    启动所有后端服务"
    echo "  stop     停止所有后端服务"
    echo "  restart  重启所有后端服务"
    echo "  status   显示服务状态"
    echo "  health   检查服务健康状态"
    echo "  help     显示帮助信息"
    echo ""
}

# 主函数
main() {
    local command=${1:-start}
    
    case $command in
        start)
            log_info "🚀 启动AI Travel Planner后端服务"
            check_python_env
            check_dependencies
            check_base_services
            start_all_services
            sleep 5
            check_services_health
            log_success "✅ 所有后端服务启动完成"
            ;;
        stop)
            stop_all_services
            ;;
        restart)
            stop_all_services
            sleep 3
            main start
            ;;
        status)
            show_status
            ;;
        health)
            check_services_health
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
