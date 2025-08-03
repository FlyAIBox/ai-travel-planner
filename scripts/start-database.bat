@echo off
setlocal enabledelayedexpansion

REM AI Travel Planner 数据库启动脚本 (Windows版本)
REM 启动MySQL、Redis和Qdrant服务

echo [INFO] 启动AI Travel Planner数据库服务

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker 未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查Docker Compose是否可用
docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker Compose 未安装，请先安装Docker Compose
        pause
        exit /b 1
    ) else (
        set DOCKER_COMPOSE=docker-compose
    )
) else (
    set DOCKER_COMPOSE=docker compose
)

REM 检查Docker服务是否运行
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker 服务未运行，请启动Docker Desktop
    pause
    exit /b 1
)

REM 获取项目根目录
cd /d "%~dp0\.."
set PROJECT_ROOT=%CD%
set COMPOSE_FILE=%PROJECT_ROOT%\deployment\docker\docker-compose.dev.yml

REM 检查Docker Compose文件是否存在
if not exist "%COMPOSE_FILE%" (
    echo [ERROR] Docker Compose 文件不存在: %COMPOSE_FILE%
    pause
    exit /b 1
)

echo [INFO] 启动数据库服务...

REM 启动数据库相关服务
%DOCKER_COMPOSE% -f "%COMPOSE_FILE%" up -d mysql redis qdrant

if errorlevel 1 (
    echo [ERROR] 数据库服务启动失败
    pause
    exit /b 1
)

echo [SUCCESS] 数据库服务启动成功

echo [INFO] 等待服务就绪...

REM 等待MySQL服务
echo [INFO] 等待MySQL服务...
set /a count=0
:wait_mysql
set /a count+=1
docker exec ai-travel-mysql-dev mysqladmin ping -h localhost --silent >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] MySQL 服务就绪
    goto mysql_ready
)
if %count% geq 30 (
    echo [ERROR] MySQL 服务启动超时
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_mysql
:mysql_ready

REM 等待Redis服务
echo [INFO] 等待Redis服务...
set /a count=0
:wait_redis
set /a count+=1
docker exec ai-travel-redis-dev redis-cli ping | findstr "PONG" >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Redis 服务就绪
    goto redis_ready
)
if %count% geq 30 (
    echo [ERROR] Redis 服务启动超时
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_redis
:redis_ready

REM 等待Qdrant服务
echo [INFO] 等待Qdrant服务...
set /a count=0
:wait_qdrant
set /a count+=1
curl -s http://localhost:6333/health >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Qdrant 服务就绪
    goto qdrant_ready
)
if %count% geq 30 (
    echo [ERROR] Qdrant 服务启动超时
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto wait_qdrant
:qdrant_ready

echo.
echo [INFO] 服务状态:
echo   MySQL:  http://localhost:3306
echo   Redis:  http://localhost:6379
echo   Qdrant: http://localhost:6333
echo.
echo [INFO] 可以使用以下命令查看日志:
echo   docker logs ai-travel-mysql-dev
echo   docker logs ai-travel-redis-dev
echo   docker logs ai-travel-qdrant-dev
echo.
echo [SUCCESS] 所有数据库服务已启动并就绪
echo [INFO] 现在可以运行系统初始化脚本:
echo   cd backend ^&^& python scripts\init_system.py
echo.
pause
