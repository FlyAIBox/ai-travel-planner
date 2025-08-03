#!/usr/bin/env python3
"""
测试所有服务是否能正常启动
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def test_service_import(service_name, service_path):
    """测试服务是否能正常导入"""
    print(f"\n=== 测试 {service_name} ===")
    
    # 切换到服务目录
    original_cwd = os.getcwd()
    try:
        os.chdir(service_path)
        
        # 尝试导入main模块
        result = subprocess.run([
            sys.executable, "-c", "import main; print('导入成功')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {service_name} 导入成功")
            return True
        else:
            print(f"❌ {service_name} 导入失败:")
            print(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {service_name} 导入超时")
        return False
    except Exception as e:
        print(f"❌ {service_name} 测试异常: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """主函数"""
    print("开始测试所有服务...")
    
    # 项目根目录
    project_root = Path(__file__).parent
    
    # 服务列表
    services = [
        ("Chat Service", project_root / "services" / "chat-service"),
        ("RAG Service", project_root / "services" / "rag-service"),
        ("Agent Service", project_root / "services" / "agent-service"),
    ]
    
    results = {}
    
    for service_name, service_path in services:
        if service_path.exists():
            results[service_name] = test_service_import(service_name, service_path)
        else:
            print(f"❌ {service_name} 目录不存在: {service_path}")
            results[service_name] = False
    
    # 总结结果
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    success_count = 0
    for service_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{service_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总计: {success_count}/{len(results)} 个服务可以正常导入")
    
    if success_count == len(results):
        print("🎉 所有服务都可以正常启动!")
        return 0
    else:
        print("⚠️  部分服务存在问题，需要进一步修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())
