"""实体详情 — get_entity_info()

获取指定实体的详细信息。

用法:
    python test/test_entity_info.py <实体名称>
"""
import os
import sys
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

name = sys.argv[1] if len(sys.argv) > 1 else "冒充公检法工作人员"
print(f"=== 实体详情: {name} ===\n")

r = requests.get(f"{SERVER}/api/v1/kg/entity/{name}", timeout=15)
data = r.json()

for key, value in data.items():
    if value:
        print(f"{key}: {value}")
