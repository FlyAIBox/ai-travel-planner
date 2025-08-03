#!/bin/bash

# AI Travel Planner 数据库启动脚本
# 启动MySQL、Redis和Qdrant服务

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

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装Docker Compose"
        exit 1
    fi
}

# 检查Docker服务是否运行
check_docker_service() {
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行，请启动Docker服务"
        exit 1
    fi
}

# 获取项目根目录
get_project_root() {
    cd "$(dirname "$0")/.."
    pwd
}

# 启动数据库服务
start_databases() {
    local project_root=$(get_project_root)
    local compose_file="$project_root/deployment/docker/docker-compose.dev.yml"
    
    if [ ! -f "$compose_file" ]; then
        log_error "Docker Compose 文件不存在: $compose_file"
        exit 1
    fi
    
    log_info "启动数据库服务..."
    
    # 使用docker compose或docker-compose
    if command -v docker compose &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    else
        DOCKER_COMPOSE="docker-compose"
    fi
    
    # 启动数据库相关服务
    cd "$project_root"
    $DOCKER_COMPOSE -f "$compose_file" up -d mysql redis qdrant
    
    if [ $? -eq 0 ]; then
        log_success "数据库服务启动成功"
    else
        log_error "数据库服务启动失败"
        exit 1
    fi
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待MySQL
    log_info "等待MySQL服务..."
    for i in {1..30}; do
        if docker exec ai-travel-mysql-dev mysqladmin ping -h localhost --silent; then
            log_success "MySQL 服务就绪"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "MySQL 服务启动超时"
            exit 1
        fi
        sleep 2
    done
    
    # 等待Redis
    log_info "等待Redis服务..."
    for i in {1..30}; do
        if docker exec ai-travel-redis-dev redis-cli ping | grep -q PONG; then
            log_success "Redis 服务就绪"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Redis 服务启动超时"
            exit 1
        fi
        sleep 2
    done
    
    # 等待Qdrant
    log_info "等待Qdrant服务..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/health > /dev/null; then
            log_success "Qdrant 服务就绪"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Qdrant 服务启动超时"
            exit 1
        fi
        sleep 2
    done
}

# 显示服务状态
show_status() {
    log_info "服务状态:"
    echo "  MySQL:  http://localhost:3306"
    echo "  Redis:  http://localhost:6379"
    echo "  Qdrant: http://localhost:6333"
    echo ""
    log_info "可以使用以下命令查看日志:"
    echo "  docker logs ai-travel-mysql-dev"
    echo "  docker logs ai-travel-redis-dev"
    echo "  docker logs ai-travel-qdrant-dev"
}

# 主函数
main() {
    log_info "🚀 启动AI Travel Planner数据库服务"
    
    check_docker
    check_docker_service
    start_databases
    wait_for_services
    show_status
    
    log_success "✅ 所有数据库服务已启动并就绪"
    log_info "💡 现在可以运行系统初始化脚本:"
    log_info "   cd backend && python scripts/init_system.py"
}

# 运行主函数
main "$@"
