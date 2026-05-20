"""RAG 知识图谱问答 — query(question, mode)

用户以自然语言提问，系统从知识图谱中检索相关信息后由 LLM 生成回答。

用法:
    python test/test_query.py
"""
import os
import requests

SERVER = os.getenv("LIGHTSERVER_URL", "http://localhost:8001")

print("=== RAG 知识图谱问答 ===\n")

# mix 模式：知识图谱 + 向量混合检索
r = requests.post(
    f"{SERVER}/api/v1/retrieval/query",
    json={
        "question": "这起案件的诈骗手法是什么？",
        "mode": "mix",
        "top_k": 5,
    },
    timeout=60,
)
data = r.json()
print(f"问题: 这起案件的诈骗手法是什么？")
print(f"模式: mix")
print(f"回答: {data.get('answer', '无回答')}\n")
