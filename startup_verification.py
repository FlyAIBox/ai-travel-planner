#!/usr/bin/env python3
"""
后台服务启动验证脚本
验证所有三个服务是否能正常启动
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def test_service_startup(service_name, service_path, port):
    """测试服务是否能正常启动"""
    print(f"\n=== 测试 {service_name} (端口 {port}) ===")
    
    # 切换到服务目录
    original_cwd = os.getcwd()
    try:
        os.chdir(service_path)
        
        # 启动服务进程
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", "--port", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 等待启动
        time.sleep(5)
        
        # 检查进程状态
        if process.poll() is None:
            print(f"✅ {service_name} 启动成功 (PID: {process.pid})")
            
            # 停止进程
            process.terminate()
            process.wait(timeout=5)
            print(f"✅ {service_name} 正常停止")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ {service_name} 启动失败:")
            print(f"错误输出: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ {service_name} 测试异常: {e}")
        return False
    finally:
        os.chdir(original_cwd)
        # 确保进程被清理
        try:
            if process.poll() is None:
                process.kill()
        except:
            pass

def main():
    """主函数"""
    print("="*60)
    print("AI旅行规划器 - 后台服务启动验证")
    print("="*60)
    
    # 项目根目录
    project_root = Path(__file__).parent
    
    # 服务列表
    services = [
        ("Chat Service", project_root / "services" / "chat-service", 8080),
        ("RAG Service", project_root / "services" / "rag-service", 8001),
        ("Agent Service", project_root / "services" / "agent-service", 8002),
    ]
    
    results = {}
    
    for service_name, service_path, port in services:
        if service_path.exists():
            results[service_name] = test_service_startup(service_name, service_path, port)
        else:
            print(f"❌ {service_name} 目录不存在: {service_path}")
            results[service_name] = False
    
    # 总结结果
    print("\n" + "="*60)
    print("验证结果总结:")
    print("="*60)
    
    success_count = 0
    for service_name, success in results.items():
        status = "✅ 正常" if success else "❌ 异常"
        print(f"{service_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总计: {success_count}/{len(results)} 个服务可以正常启动")
    
    if success_count == len(results):
        print("\n🎉 所有后台服务启动正常！")
        print("\n启动命令:")
        print("Chat Service:  cd services/chat-service && python -m uvicorn main:app --host 0.0.0.0 --port 8080")
        print("RAG Service:   cd services/rag-service && python -m uvicorn main:app --host 0.0.0.0 --port 8001")
        print("Agent Service: cd services/agent-service && python -m uvicorn main:app --host 0.0.0.0 --port 8002")
        return 0
    else:
        print("\n⚠️  部分服务仍存在问题，需要进一步检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())
