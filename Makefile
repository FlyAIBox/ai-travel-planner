.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check clean dev run-dev docker-build docker-up docker-down db-init db-upgrade db-revision conda-env pip-install

# ====================================
# 项目配置变量
# ====================================
PROJECT_NAME = ai-travel-planner
PYTHON_VERSION = 3.10
CONDA_ENV = ai-travel-planner

# Python解释器和工具路径
PYTHON = python
PIP = pip
CONDA = conda
PYTEST = pytest
BLACK = black
ISORT = isort
FLAKE8 = flake8
MYPY = mypy
ALEMBIC = alembic
UVICORN = uvicorn

# Docker配置
DOCKER_COMPOSE_DEV = docker-compose.dev.yml
DOCKER_COMPOSE_PROD = docker-compose.prod.yml

# ====================================
# 帮助信息
# ====================================
help:
	@echo "🚀 AI Travel Planner - 开发工具集"
	@echo ""
	@echo "📦 环境管理:"
	@echo "  conda-env       - 创建Conda环境"
	@echo "  install         - 安装生产依赖"
	@echo "  install-dev     - 安装开发依赖"
	@echo "  pip-install     - 使用pip安装依赖"
	@echo ""
	@echo "🧪 测试相关:"
	@echo "  test           - 运行所有测试"
	@echo "  test-unit      - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-e2e       - 运行端到端测试"
	@echo "  test-cov       - 运行测试并生成覆盖率报告"
	@echo ""
	@echo "🔍 代码质量:"
	@echo "  lint           - 运行所有代码检查"
	@echo "  format         - 格式化代码"
	@echo "  type-check     - 类型检查"
	@echo "  pre-commit     - 安装pre-commit钩子"
	@echo ""
	@echo "🗄️ 数据库管理:"
	@echo "  db-init        - 初始化数据库"
	@echo "  db-upgrade     - 升级数据库到最新版本"
	@echo "  db-revision    - 创建新的数据库迁移"
	@echo "  db-reset       - 重置数据库"
	@echo ""
	@echo "🐳 Docker操作:"
	@echo "  docker-build   - 构建Docker镜像"
	@echo "  docker-up-dev  - 启动开发环境"
	@echo "  docker-up-prod - 启动生产环境"
	@echo "  docker-down    - 停止所有容器"
	@echo "  docker-logs    - 查看容器日志"
	@echo ""
	@echo "🔧 开发服务:"
	@echo "  run-dev        - 运行开发服务器"
	@echo "  run-jupyter    - 启动Jupyter Lab"
	@echo "  shell          - 进入Python交互环境"
	@echo ""
	@echo "🧹 清理操作:"
	@echo "  clean          - 清理构建文件"
	@echo "  clean-cache    - 清理缓存文件"
	@echo "  clean-docker   - 清理Docker资源"

# ====================================
# 环境管理
# ====================================
conda-env:
	@echo "🔧 创建Conda环境: $(CONDA_ENV)"
	$(CONDA) env create -f conda-environment.yml --force
	@echo "✅ Conda环境创建完成！"
	@echo "激活环境: conda activate $(CONDA_ENV)"

install:
	@echo "📦 安装生产依赖..."
	$(PIP) install -e .

install-dev: install
	@echo "📦 安装开发依赖..."
	$(PIP) install -e ".[dev]"
	@echo "✅ 开发环境安装完成！"

pip-install:
	@echo "📦 使用pip安装依赖..."
	$(PIP) install -r requirements.txt
	@echo "✅ 依赖安装完成！"

# ====================================
# 测试相关
# ====================================
test: install-dev
	@echo "🧪 运行所有测试..."
	$(PYTEST) -v

test-unit: install-dev
	@echo "🧪 运行单元测试..."
	$(PYTEST) -m unit -v

test-integration: install-dev
	@echo "🧪 运行集成测试..."
	$(PYTEST) -m integration -v

test-e2e: install-dev
	@echo "🧪 运行端到端测试..."
	$(PYTEST) -m e2e -v

test-cov: install-dev
	@echo "🧪 运行测试并生成覆盖率报告..."
	$(PYTEST) --cov=shared --cov=services --cov-report=html --cov-report=term
	@echo "📊 覆盖率报告已生成到 htmlcov/ 目录"

# ====================================
# 代码质量
# ====================================
format: install-dev
	@echo "🎨 格式化代码..."
	$(BLACK) .
	$(ISORT) .
	@echo "✅ 代码格式化完成！"

lint: install-dev
	@echo "🔍 运行代码检查..."
	$(FLAKE8) .
	@echo "✅ 代码检查完成！"

type-check: install-dev
	@echo "🔍 运行类型检查..."
	$(MYPY) shared/ services/
	@echo "✅ 类型检查完成！"

pre-commit: install-dev
	@echo "🔗 安装pre-commit钩子..."
	pre-commit install
	@echo "✅ pre-commit钩子安装完成！"

# ====================================
# 数据库管理
# ====================================
db-init: install
	@echo "🗄️ 初始化数据库..."
	$(ALEMBIC) init alembic
	@echo "✅ 数据库初始化完成！"

db-upgrade: install
	@echo "🗄️ 升级数据库到最新版本..."
	$(ALEMBIC) upgrade head
	@echo "✅ 数据库升级完成！"

db-revision: install
	@echo "🗄️ 创建新的数据库迁移..."
	@read -p "请输入迁移消息: " message; \
	$(ALEMBIC) revision --autogenerate -m "$$message"
	@echo "✅ 数据库迁移创建完成！"

db-reset: install
	@echo "⚠️ 重置数据库（危险操作）..."
	@read -p "确认重置数据库？(y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(ALEMBIC) downgrade base && $(ALEMBIC) upgrade head; \
		echo "✅ 数据库重置完成！"; \
	else \
		echo "❌ 操作已取消"; \
	fi

# ====================================
# Docker操作
# ====================================
docker-build:
	@echo "🐳 构建Docker镜像..."
	docker compose -f $(DOCKER_COMPOSE_DEV) build
	@echo "✅ Docker镜像构建完成！"

docker-up-dev:
	@echo "🐳 启动开发环境..."
	docker compose -f $(DOCKER_COMPOSE_DEV) up -d
	@echo "✅ 开发环境已启动！"
	@echo "📊 服务访问地址:"
	@echo "  - API网关: http://localhost:8000"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - n8n: http://localhost:5678"
	@echo "  - pgAdmin: http://localhost:5050"

docker-up-prod:
	@echo "🐳 启动生产环境..."
	docker compose -f $(DOCKER_COMPOSE_PROD) up -d
	@echo "✅ 生产环境已启动！"

docker-down:
	@echo "🐳 停止所有容器..."
	docker compose -f $(DOCKER_COMPOSE_DEV) down
	docker compose -f $(DOCKER_COMPOSE_PROD) down 2>/dev/null || true
	@echo "✅ 容器已停止！"

docker-logs:
	@echo "📋 查看容器日志..."
	docker compose -f $(DOCKER_COMPOSE_DEV) logs -f

clean-docker:
	@echo "🧹 清理Docker资源..."
	docker system prune -f
	docker volume prune -f
	@echo "✅ Docker资源清理完成！"

# ====================================
# 开发服务
# ====================================
run-dev:
	@echo "🚀 启动开发服务器..."
	cd services/api-gateway && $(UVICORN) main:app --reload --host 0.0.0.0 --port 8000

run-jupyter: install-dev
	@echo "📓 启动Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

shell: install-dev
	@echo "🐍 进入Python交互环境..."
	$(PYTHON) -i -c "import sys; sys.path.append('.')"

# ====================================
# 清理操作
# ====================================
clean:
	@echo "🧹 清理构建文件..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	@echo "✅ 构建文件清理完成！"

clean-cache:
	@echo "🧹 清理缓存文件..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	@echo "✅ 缓存文件清理完成！"

clean-all: clean clean-cache clean-docker
	@echo "✅ 所有清理操作完成！"

# ====================================
# 快捷命令
# ====================================
dev: install-dev pre-commit
	@echo "🎉 开发环境准备完成！"
	@echo "下一步："
	@echo "  make run-dev     - 启动开发服务器"
	@echo "  make docker-up-dev - 启动完整开发环境"

check: format lint type-check test
	@echo "✅ 所有检查通过！" 