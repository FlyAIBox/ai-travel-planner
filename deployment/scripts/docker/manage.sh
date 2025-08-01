#!/bin/bash

# ==================== Docker Compose 管理脚本 ====================
# AI Travel Planner - Docker容器管理工具

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/deployment/docker"
COMPOSE_DEV="${DOCKER_DIR}/docker-compose.dev.yml"
COMPOSE_PROD="${DOCKER_DIR}/docker-compose.prod.yml"
COMPOSE_MONITORING="${DOCKER_DIR}/docker-compose.monitoring.yml"

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }

# 显示帮助信息
show_help() {
    echo -e "${CYAN}==================== AI Travel Planner Docker 管理工具 ====================${NC}"
    echo ""
    echo "用法: $0 <环境> <操作> [选项]"
    echo ""
    echo -e "${YELLOW}环境:${NC}"
    echo "  dev      开发环境"
    echo "  prod     生产环境"
    echo "  monitor  监控服务"
    echo ""
    echo -e "${YELLOW}操作:${NC}"
    echo "  up       启动服务"
    echo "  down     停止服务"
    echo "  restart  重启服务"
    echo "  build    构建镜像"
    echo "  pull     拉取镜像"
    echo "  logs     查看日志"
    echo "  ps       查看状态"
    echo "  exec     进入容器"
    echo "  clean    清理资源"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo "  --build       启动时重新构建"
    echo "  --force-rm    删除时强制移除"
    echo "  --no-cache    构建时不使用缓存"
    echo "  --follow      跟踪日志输出"
    echo "  --service     指定服务名称"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0 dev up                    # 启动开发环境"
    echo "  $0 prod up --build           # 构建并启动生产环境"
    echo "  $0 dev logs --follow         # 跟踪开发环境日志"
    echo "  $0 prod exec --service api   # 进入API服务容器"
    echo "  $0 monitor up                # 启动监控服务"
    echo "  $0 dev clean --force-rm      # 强制清理开发环境"
    echo ""
}

# 检查Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose 未安装"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行"
        exit 1
    fi
}

# 获取compose文件
get_compose_file() {
    local env=$1
    case $env in
        dev)
            echo "$COMPOSE_DEV"
            ;;
        prod)
            echo "$COMPOSE_PROD"
            ;;
        monitor)
            echo "$COMPOSE_MONITORING"
            ;;
        *)
            log_error "未知环境: $env"
            exit 1
            ;;
    esac
}

# 执行Docker Compose命令
execute_compose() {
    local env=$1
    local cmd=$2
    shift 2
    
    local compose_file=$(get_compose_file $env)
    
    if [[ ! -f "$compose_file" ]]; then
        log_error "配置文件不存在: $compose_file"
        exit 1
    fi
    
    log_info "使用配置文件: $compose_file"
    log_step "执行命令: docker compose -f $compose_file $cmd $*"
    
    cd "$PROJECT_ROOT"
    docker compose -f "$compose_file" $cmd "$@"
}

# 启动服务
start_services() {
    local env=$1
    shift
    
    local build_flag=""
    local other_args=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                build_flag="--build"
                shift
                ;;
            *)
                other_args+=("$1")
                shift
                ;;
        esac
    done
    
    log_step "启动 $env 环境服务..."
    
    if [[ -n "$build_flag" ]]; then
        log_info "构建并启动服务"
        execute_compose "$env" "up" -d --build "${other_args[@]}"
    else
        execute_compose "$env" "up" -d "${other_args[@]}"
    fi
    
    log_success "$env 环境服务启动完成"
}

# 停止服务
stop_services() {
    local env=$1
    shift
    
    local force_flag=""
    local other_args=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force-rm)
                force_flag="--remove-orphans"
                shift
                ;;
            *)
                other_args+=("$1")
                shift
                ;;
        esac
    done
    
    log_step "停止 $env 环境服务..."
    
    if [[ -n "$force_flag" ]]; then
        execute_compose "$env" "down" --remove-orphans "${other_args[@]}"
    else
        execute_compose "$env" "down" "${other_args[@]}"
    fi
    
    log_success "$env 环境服务已停止"
}

# 重启服务
restart_services() {
    local env=$1
    shift
    
    log_step "重启 $env 环境服务..."
    execute_compose "$env" "restart" "$@"
    log_success "$env 环境服务重启完成"
}

# 构建镜像
build_images() {
    local env=$1
    shift
    
    local no_cache_flag=""
    local other_args=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cache)
                no_cache_flag="--no-cache"
                shift
                ;;
            *)
                other_args+=("$1")
                shift
                ;;
        esac
    done
    
    log_step "构建 $env 环境镜像..."
    
    if [[ -n "$no_cache_flag" ]]; then
        execute_compose "$env" "build" --no-cache "${other_args[@]}"
    else
        execute_compose "$env" "build" "${other_args[@]}"
    fi
    
    log_success "$env 环境镜像构建完成"
}

# 拉取镜像
pull_images() {
    local env=$1
    shift
    
    log_step "拉取 $env 环境镜像..."
    execute_compose "$env" "pull" "$@"
    log_success "$env 环境镜像拉取完成"
}

# 查看日志
view_logs() {
    local env=$1
    shift
    
    local follow_flag=""
    local service=""
    local other_args=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --follow)
                follow_flag="-f"
                shift
                ;;
            --service)
                service="$2"
                shift 2
                ;;
            *)
                other_args+=("$1")
                shift
                ;;
        esac
    done
    
    log_step "查看 $env 环境日志..."
    
    if [[ -n "$service" ]]; then
        execute_compose "$env" "logs" $follow_flag "$service" "${other_args[@]}"
    else
        execute_compose "$env" "logs" $follow_flag "${other_args[@]}"
    fi
}

# 查看状态
view_status() {
    local env=$1
    shift
    
    log_step "查看 $env 环境状态..."
    execute_compose "$env" "ps" "$@"
}

# 进入容器
exec_container() {
    local env=$1
    shift
    
    local service=""
    local cmd="bash"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --service)
                service="$2"
                shift 2
                ;;
            --cmd)
                cmd="$2"
                shift 2
                ;;
            *)
                service="$1"
                shift
                ;;
        esac
    done
    
    if [[ -z "$service" ]]; then
        log_error "请指定服务名称"
        exit 1
    fi
    
    log_step "进入 $env 环境的 $service 容器..."
    execute_compose "$env" "exec" "$service" "$cmd"
}

# 清理资源
clean_resources() {
    local env=$1
    shift
    
    local force_flag=""
    local volumes_flag=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force-rm)
                force_flag="--remove-orphans"
                shift
                ;;
            --volumes)
                volumes_flag="-v"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_warning "清理 $env 环境资源..."
    
    if [[ -n "$volumes_flag" ]]; then
        execute_compose "$env" "down" $force_flag -v
        log_warning "已删除数据卷"
    else
        execute_compose "$env" "down" $force_flag
    fi
    
    # 清理未使用的镜像
    log_step "清理未使用的Docker资源..."
    docker system prune -f
    
    log_success "$env 环境资源清理完成"
}

# 主函数
main() {
    if [[ $# -lt 2 ]]; then
        show_help
        exit 1
    fi
    
    local env=$1
    local operation=$2
    shift 2
    
    # 检查Docker
    check_docker
    
    # 验证环境
    if [[ ! "$env" =~ ^(dev|prod|monitor)$ ]]; then
        log_error "无效的环境: $env"
        show_help
        exit 1
    fi
    
    # 执行操作
    case $operation in
        up)
            start_services "$env" "$@"
            ;;
        down)
            stop_services "$env" "$@"
            ;;
        restart)
            restart_services "$env" "$@"
            ;;
        build)
            build_images "$env" "$@"
            ;;
        pull)
            pull_images "$env" "$@"
            ;;
        logs)
            view_logs "$env" "$@"
            ;;
        ps)
            view_status "$env" "$@"
            ;;
        exec)
            exec_container "$env" "$@"
            ;;
        clean)
            clean_resources "$env" "$@"
            ;;
        help|-h|--help)
            show_help
            ;;
        *)
            log_error "未知操作: $operation"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 