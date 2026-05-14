from langchain_openai import ChatOpenAI
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")

print(config)

model = ChatOpenAI(model=config.llm.model, api_key=config.llm.api_key, base_url=config.llm.base_url)

prompt = """你是铜合金领域知识图谱的实体抽取专家。

你的任务是：
从{text}中，
抽取材料科学相关实体。

必须严格按照 schema 输出。

# 实体类型定义

1. MaterialSystem
铜合金体系名称
例如：
- CuNiSi
- CuCrZr
- CuNiSn
- CuFeP

2. Phase
物相名称
例如：
- α相
- β相
- δ-Ni2Si
- Al2Cu

3. ConductivityMechanism
导电机制
例如：
- 析出净化
- 溶质散射
- 晶界散射

4. StrengtheningMechanism
强化机制
例如：
- 固溶强化
- 析出强化
- 位错强化
- 细晶强化

5. PrecipitationMechanism
析出/相变机制
例如：
- 共格析出
- 失配析出
- spinodal decomposition

6. Property
材料性能
例如：
- 抗拉强度
- 屈服强度
- 导电率
- 延伸率
- 硬度

7. ServicePerformance
服役性能
例如：
- 抗氧化性
- 耐磨性
- 抗热应力松弛

8. Application
应用场景
例如：
- 电子工业
- 航空航天
- 汽车连接器

9. FailureMode
失效模式
例如：
- 应力腐蚀开裂
- 晶界蠕变
- 脱锌

# 抽取规则

1. 不允许臆造实体
2. 必须来自原文
3. 保留原文 evidence
4. 输出 normalized_name
5. 同义词统一：
   - 时效强化 → 析出强化
   - 沉淀强化 → 析出强化
6. 若不确定：
   confidence < 0.6
7. 不输出解释
8. 输出合法 JSON

# 输出格式

{{
  "entities": [
    {{
      "entity_text": "",
      "entity_type": "",
      "normalized_name": "",
      "aliases": [],
      "evidence": "",
      "confidence": 0.0
    }}
  ]
}}"""

with open(r"\\10.60.0.248\大数据超红区\b42628 仇杰芸\code\test.md", "r", encoding="utf-8") as f:
    text = f.read()

response = model.invoke(prompt.format(text=text))
print(response.text)

with open("output.json", "w", encoding="utf-8") as f:
    f.write(response.text)
