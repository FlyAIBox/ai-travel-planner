#!/usr/bin/env python3
"""
AI Travel Planner 测试运行脚本
支持单元测试、集成测试、覆盖率报告和质量检查
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
SERVICE_DIRS = [
    "chat-service",
    "rag-service", 
    "user-service",
    "agent-service",
    "planning-service",
    "integration-service"
]

def run_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """运行命令并返回退出码"""
    print(f"运行命令: {' '.join(cmd)}")
    if cwd:
        print(f"工作目录: {cwd}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"错误: 命令未找到 {cmd[0]}")
        return 1
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 130

def setup_test_environment():
    """设置测试环境"""
    print("🔧 设置测试环境...")
    
    # 设置环境变量
    test_env = os.environ.copy()
    test_env.update({
        "ENVIRONMENT": "testing",
        "DEBUG": "false",
        "LOG_LEVEL": "WARNING",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "MYSQL_HOST": "localhost", 
        "MYSQL_PORT": "3306",
        "MYSQL_DATABASE": "test_db",
        "MYSQL_USER": "test_user",
        "MYSQL_PASSWORD": "test_pass",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
    })
    
    return test_env

def run_unit_tests(services: List[str], verbose: bool = False, coverage: bool = True) -> int:
    """运行单元测试"""
    print("🧪 运行单元测试...")
    
    test_env = setup_test_environment()
    exit_code = 0
    
    for service in services:
        service_path = PROJECT_ROOT / "services" / service
        if not service_path.exists():
            print(f"⚠️  服务目录不存在: {service}")
            continue
        
        test_path = service_path / "tests"
        if not test_path.exists():
            print(f"⚠️  测试目录不存在: {service}/tests")
            continue
        
        print(f"\n🔍 测试服务: {service}")
        
        # 构建pytest命令
        cmd = ["python", "-m", "pytest", "tests/"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=xml",
                "--cov-report=html"
            ])
        
        # 添加测试标记
        cmd.extend(["-m", "unit"])
        
        # 运行测试
        result = run_command(cmd, cwd=service_path, env=test_env)
        if result != 0:
            print(f"❌ {service} 单元测试失败")
            exit_code = result
        else:
            print(f"✅ {service} 单元测试通过")
    
    return exit_code

def run_integration_tests(verbose: bool = False) -> int:
    """运行集成测试"""
    print("🔗 运行集成测试...")
    
    test_env = setup_test_environment()
    
    # 启动测试环境
    print("启动测试环境...")
    docker_cmd = [
        "docker", "compose",
        "-f", "deployment/docker/docker-compose.test.yml",
        "up", "-d"
    ]
    
    result = run_command(docker_cmd, cwd=PROJECT_ROOT)
    if result != 0:
        print("❌ 测试环境启动失败")
        return result
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(30)
    
    try:
        # 运行集成测试
        cmd = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
        if verbose:
            cmd.append("-v")
        
        result = run_command(cmd, cwd=PROJECT_ROOT, env=test_env)
        
        if result == 0:
            print("✅ 集成测试通过")
        else:
            print("❌ 集成测试失败")
        
    finally:
        # 清理测试环境
        print("清理测试环境...")
        cleanup_cmd = [
            "docker", "compose", 
            "-f", "deployment/docker/docker-compose.test.yml",
            "down", "-v"
        ]
        run_command(cleanup_cmd, cwd=PROJECT_ROOT)
    
    return result

def run_api_tests(verbose: bool = False) -> int:
    """运行API测试"""
    print("🌐 运行API测试...")
    
    test_env = setup_test_environment()
    
    cmd = ["python", "-m", "pytest", "tests/api/", "-m", "api"]
    if verbose:
        cmd.append("-v")
    
    result = run_command(cmd, cwd=PROJECT_ROOT, env=test_env)
    
    if result == 0:
        print("✅ API测试通过")
    else:
        print("❌ API测试失败")
    
    return result

def run_e2e_tests(verbose: bool = False) -> int:
    """运行端到端测试"""
    print("🎯 运行端到端测试...")
    
    test_env = setup_test_environment()
    
    cmd = ["python", "-m", "pytest", "tests/e2e/", "-m", "e2e"]
    if verbose:
        cmd.append("-v")
    
    result = run_command(cmd, cwd=PROJECT_ROOT, env=test_env)
    
    if result == 0:
        print("✅ 端到端测试通过")
    else:
        print("❌ 端到端测试失败")
    
    return result

def run_lint_checks(services: List[str]) -> int:
    """运行代码质量检查"""
    print("📝 运行代码质量检查...")
    
    exit_code = 0
    
    for service in services:
        service_path = PROJECT_ROOT / "services" / service
        if not service_path.exists():
            continue
        
        print(f"\n🔍 检查服务: {service}")
        
        # Flake8检查
        print("运行 Flake8...")
        flake8_cmd = [
            "flake8", ".", 
            "--count", 
            "--select=E9,F63,F7,F82", 
            "--show-source", 
            "--statistics"
        ]
        result = run_command(flake8_cmd, cwd=service_path)
        if result != 0:
            print(f"❌ {service} Flake8检查失败")
            exit_code = result
        
        # Black检查
        print("运行 Black...")
        black_cmd = ["black", "--check", "."]
        result = run_command(black_cmd, cwd=service_path)
        if result != 0:
            print(f"❌ {service} Black检查失败")
            exit_code = result
        
        # isort检查
        print("运行 isort...")
        isort_cmd = ["isort", "--check-only", "."]
        result = run_command(isort_cmd, cwd=service_path)
        if result != 0:
            print(f"❌ {service} isort检查失败")
            exit_code = result
        
        if exit_code == 0:
            print(f"✅ {service} 代码质量检查通过")
    
    return exit_code

def run_security_checks() -> int:
    """运行安全检查"""
    print("🔒 运行安全检查...")
    
    exit_code = 0
    
    # Safety检查
    print("运行 Safety...")
    safety_cmd = ["safety", "check", "-r", "requirements.txt"]
    result = run_command(safety_cmd, cwd=PROJECT_ROOT)
    if result != 0:
        print("❌ Safety检查失败")
        exit_code = result
    
    # Bandit检查
    print("运行 Bandit...")
    bandit_cmd = ["bandit", "-r", "services/", "-f", "json", "-o", "bandit-report.json"]
    result = run_command(bandit_cmd, cwd=PROJECT_ROOT)
    if result != 0:
        print("❌ Bandit检查失败")
        exit_code = result
    
    if exit_code == 0:
        print("✅ 安全检查通过")
    
    return exit_code

def generate_coverage_report(services: List[str]) -> None:
    """生成覆盖率报告"""
    print("📊 生成覆盖率报告...")
    
    for service in services:
        service_path = PROJECT_ROOT / "services" / service
        coverage_file = service_path / "coverage.xml"
        
        if coverage_file.exists():
            print(f"✅ {service} 覆盖率报告: {coverage_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI Travel Planner 测试运行器")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "api", "e2e", "all"],
        default="unit",
        help="测试类型"
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=SERVICE_DIRS,
        help="要测试的服务"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="禁用覆盖率"
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="运行代码质量检查"
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="运行安全检查"
    )
    
    args = parser.parse_args()
    
    print("🚀 AI Travel Planner 测试运行器")
    print("=" * 50)
    
    exit_code = 0
    
    # 代码质量检查
    if args.lint:
        result = run_lint_checks(args.services)
        if result != 0:
            exit_code = result
    
    # 安全检查
    if args.security:
        result = run_security_checks()
        if result != 0:
            exit_code = result
    
    # 运行测试
    coverage = not args.no_coverage
    
    if args.type == "unit" or args.type == "all":
        result = run_unit_tests(args.services, args.verbose, coverage)
        if result != 0:
            exit_code = result
    
    if args.type == "integration" or args.type == "all":
        result = run_integration_tests(args.verbose)
        if result != 0:
            exit_code = result
    
    if args.type == "api" or args.type == "all":
        result = run_api_tests(args.verbose)
        if result != 0:
            exit_code = result
    
    if args.type == "e2e" or args.type == "all":
        result = run_e2e_tests(args.verbose)
        if result != 0:
            exit_code = result
    
    # 生成覆盖率报告
    if coverage and args.type in ["unit", "all"]:
        generate_coverage_report(args.services)
    
    # 总结
    print("\n" + "=" * 50)
    if exit_code == 0:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 