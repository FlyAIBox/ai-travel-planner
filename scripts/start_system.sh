#!/bin/bash

# AI Travel Planner 系统启动脚本
# 提供完整的系统启动、健康检查、日志查看等功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/deployment/docker/docker-compose.dev.yml"

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

# 打印横幅
print_banner() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "    AI Travel Planner 智能旅行规划系统"
    echo "========================================"
    echo -e "${NC}"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Docker服务状态
    if ! docker info &> /dev/null; then
        log_error "Docker服务未运行，请启动Docker服务"
        exit 1
    fi
    
    log_success "系统依赖检查通过"
}

# 检查端口占用
check_ports() {
    log_info "检查端口占用情况..."
    
    PORTS=(8080 8001 8002 8003 8080 3306 6379 6333 5678 9090 3000)
    OCCUPIED_PORTS=()
    
    for port in "${PORTS[@]}"; do
        if lsof -i :$port &> /dev/null; then
            OCCUPIED_PORTS+=($port)
        fi
    done
    
    if [ ${#OCCUPIED_PORTS[@]} -gt 0 ]; then
        log_warning "以下端口已被占用: ${OCCUPIED_PORTS[*]}"
        read -p "是否继续启动？某些服务可能无法正常运行 (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "取消启动"
            exit 0
        fi
    else
        log_success "端口检查通过"
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    DIRS=(
        "$PROJECT_ROOT/data/logs"
        "$PROJECT_ROOT/data/mysql"
        "$PROJECT_ROOT/data/redis"
        "$PROJECT_ROOT/data/qdrant"
        "$PROJECT_ROOT/data/n8n"
        "$PROJECT_ROOT/data/prometheus"
        "$PROJECT_ROOT/data/grafana"
        "$PROJECT_ROOT/data/knowledge_base"
        "$PROJECT_ROOT/data/backups"
    )
    
    for dir in "${DIRS[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "目录创建完成"
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_info "创建环境变量文件..."
        cat > "$PROJECT_ROOT/.env" << EOF
# 数据库配置
MYSQL_DATABASE=ai_travel_db
MYSQL_USER=ai_travel_user
MYSQL_PASSWORD=ai_travel_pass_2024
MYSQL_ROOT_PASSWORD=ai_travel_root_2024

# Redis配置
REDIS_PASSWORD=ai_travel_redis_2024

# JWT配置
JWT_SECRET_KEY=ai_travel_jwt_secret_key_please_change_in_production

# n8n配置
N8N_USER=admin
N8N_PASSWORD=ai_travel_n8n_2024

# 其他配置
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=info
EOF
        log_success "环境变量文件已创建"
    else
        log_info "环境变量文件已存在"
    fi
}

# 启动系统
start_system() {
    log_info "启动AI Travel Planner系统..."
    
    cd "$PROJECT_ROOT"
    
    # 拉取镜像
    log_info "拉取必要的Docker镜像..."
    docker compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # 构建镜像
    log_info "构建应用镜像..."
    docker compose -f "$DOCKER_COMPOSE_FILE" build
    
    # 启动服务
    log_info "启动所有服务..."
    docker compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    log_success "系统启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务启动..."
    
    # 定义服务和健康检查端点
    declare -A SERVICES
    SERVICES["Redis"]="redis:6379"
    SERVICES["MySQL"]="mysql:3306"
    SERVICES["Qdrant"]="qdrant:6333"
    SERVICES["Chat服务"]="localhost:8080/api/v1/health"
    SERVICES["RAG服务"]="localhost:8001/api/v1/health"
    SERVICES["智能体服务"]="localhost:8002/api/v1/health"
    SERVICES["用户服务"]="localhost:8003/api/v1/health"
    SERVICES["API网关"]="localhost:8080/gateway/health"
    
    MAX_WAIT=180  # 最大等待时间（秒）
    WAIT_TIME=0
    
    while [ $WAIT_TIME -lt $MAX_WAIT ]; do
        ALL_READY=true
        
        for service in "${!SERVICES[@]}"; do
            endpoint="${SERVICES[$service]}"
            
            if [[ $endpoint == *":"* && $endpoint != *"/"* ]]; then
                # TCP端口检查
                if ! nc -z ${endpoint/:/ } 2>/dev/null; then
                    ALL_READY=false
                    break
                fi
            else
                # HTTP健康检查
                if ! curl -s -f "http://$endpoint" >/dev/null 2>&1; then
                    ALL_READY=false
                    break
                fi
            fi
        done
        
        if [ "$ALL_READY" = true ]; then
            log_success "所有服务已就绪"
            return 0
        fi
        
        echo -n "."
        sleep 5
        WAIT_TIME=$((WAIT_TIME + 5))
    done
    
    echo
    log_warning "部分服务可能尚未完全就绪，请稍后检查"
}

# 显示服务状态
show_service_status() {
    log_info "服务状态检查..."
    
    # Docker容器状态
    echo -e "\n${BLUE}Docker容器状态:${NC}"
    docker compose -f "$DOCKER_COMPOSE_FILE" ps
    
    # 健康检查
    echo -e "\n${BLUE}服务健康状态:${NC}"
    
    declare -A HEALTH_ENDPOINTS
    HEALTH_ENDPOINTS["Chat服务"]="http://localhost:8080/api/v1/health"
    HEALTH_ENDPOINTS["RAG服务"]="http://localhost:8001/api/v1/health"
    HEALTH_ENDPOINTS["智能体服务"]="http://localhost:8002/api/v1/health"
    HEALTH_ENDPOINTS["用户服务"]="http://localhost:8003/api/v1/health"
    HEALTH_ENDPOINTS["API网关"]="http://localhost:8080/gateway/health"
    
    for service in "${!HEALTH_ENDPOINTS[@]}"; do
        endpoint="${HEALTH_ENDPOINTS[$service]}"
        
        if curl -s -f "$endpoint" >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} $service: 健康"
        else
            echo -e "  ${RED}✗${NC} $service: 不健康"
        fi
    done
}

# 显示访问信息
show_access_info() {
    echo -e "\n${GREEN}========================================"
    echo "           系统访问信息"
    echo -e "========================================${NC}"
    echo "🌐 API网关:          http://localhost:8080"
    echo "💬 聊天服务:         http://localhost:8080"
    echo "🔍 RAG服务:          http://localhost:8001"
    echo "🤖 智能体服务:       http://localhost:8002"
    echo "👤 用户服务:         http://localhost:8003"
    echo "🔧 n8n工作流:        http://localhost:5678"
    echo "📊 Prometheus:       http://localhost:9090"
    echo "📈 Grafana:         http://localhost:3000"
    echo
    echo "📚 API文档:"
    echo "  - Chat服务:        http://localhost:8080/docs"
    echo "  - RAG服务:         http://localhost:8001/docs"
    echo "  - 智能体服务:      http://localhost:8002/docs"
    echo "  - 用户服务:        http://localhost:8003/docs"
    echo
    echo "🔑 默认账号信息:"
    echo "  - n8n:            admin / ai_travel_n8n_2024"
    echo "  - Grafana:        admin / admin"
    echo
    echo -e "${YELLOW}提示: 使用 './scripts/start_system.sh logs' 查看日志${NC}"
    echo -e "${YELLOW}提示: 使用 './scripts/start_system.sh stop' 停止系统${NC}"
    echo -e "${GREEN}========================================"
    echo -e "           系统启动完成"
    echo -e "========================================${NC}"
}

# 初始化系统数据
init_system_data() {
    log_info "初始化系统数据..."
    
    # 等待服务启动
    sleep 10
    
    # 运行初始化脚本
    if [ -f "$PROJECT_ROOT/scripts/init_system.py" ]; then
        log_info "运行数据初始化脚本..."
        cd "$PROJECT_ROOT"
        python scripts/init_system.py
        log_success "数据初始化完成"
    else
        log_warning "初始化脚本不存在，跳过数据初始化"
    fi
}

# 显示日志
show_logs() {
    local service="$1"
    
    if [ -z "$service" ]; then
        log_info "显示所有服务日志..."
        docker compose -f "$DOCKER_COMPOSE_FILE" logs -f
    else
        log_info "显示 $service 服务日志..."
        docker compose -f "$DOCKER_COMPOSE_FILE" logs -f "$service"
    fi
}

# 停止系统
stop_system() {
    log_info "停止AI Travel Planner系统..."
    
    cd "$PROJECT_ROOT"
    docker compose -f "$DOCKER_COMPOSE_FILE" down
    
    log_success "系统已停止"
}

# 重启系统
restart_system() {
    log_info "重启AI Travel Planner系统..."
    
    stop_system
    sleep 5
    start_system_full
}

# 清理系统
clean_system() {
    log_warning "清理系统将删除所有数据，此操作不可恢复！"
    read -p "确定要清理系统吗？(y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "清理系统..."
        
        cd "$PROJECT_ROOT"
        docker compose -f "$DOCKER_COMPOSE_FILE" down -v --rmi all
        
        # 删除数据目录
        if [ -d "$PROJECT_ROOT/data" ]; then
            rm -rf "$PROJECT_ROOT/data"
        fi
        
        log_success "系统清理完成"
    else
        log_info "取消清理操作"
    fi
}

# 完整启动流程
start_system_full() {
    print_banner
    check_dependencies
    check_ports
    create_directories
    setup_environment
    start_system
    wait_for_services
    init_system_data
    show_service_status
    show_access_info
}

# 显示帮助信息
show_help() {
    echo "AI Travel Planner 系统管理脚本"
    echo
    echo "用法: $0 [命令] [选项]"
    echo
    echo "命令:"
    echo "  start       启动系统 (默认)"
    echo "  stop        停止系统"
    echo "  restart     重启系统"
    echo "  status      显示服务状态"
    echo "  logs [服务] 显示日志"
    echo "  clean       清理系统数据"
    echo "  help        显示帮助信息"
    echo
    echo "示例:"
    echo "  $0                    # 启动系统"
    echo "  $0 start             # 启动系统"
    echo "  $0 status            # 显示状态"
    echo "  $0 logs chat-service # 显示聊天服务日志"
    echo "  $0 stop              # 停止系统"
    echo "  $0 clean             # 清理系统"
}

# 主函数
main() {
    local command="${1:-start}"
    local arg="$2"
    
    case "$command" in
        "start")
            start_system_full
            ;;
        "stop")
            stop_system
            ;;
        "restart")
            restart_system
            ;;
        "status")
            show_service_status
            ;;
        "logs")
            show_logs "$arg"
            ;;
        "clean")
            clean_system
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 