"""健康检查 — 验证服务是否正常运行。

用法:
    python test/test_health.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

r = requests.get(f"{SERVER}/health", timeout=5)
data = r.json()
print(f"状态: {data['status']}")
assert data["status"] == "ok"
print("健康检查通过")
