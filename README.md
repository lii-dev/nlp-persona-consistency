# 带记忆校验模块的角色人设一致性对话生成

**方法定位**：training-free 外挂式模块（人设档案 + 后置校验 + 修正重试），**不改模型权重、零额外训练成本**。  
**口径**：Persona-Chat 仅作**评测数据集**；本方案不涉及模型训练，故**无需训练集**。  
**基座**：Qwen2.5-7B-Instruct（DashScope OpenAI 兼容接口）  
**规模**：20 角色 × 10 轮 = 200 条评测用例 | 周期：11/12 – 12/20

## 快速开始

```bash
# 1) 依赖
pip install -r requirements.txt

# 2) 配置 API
copy .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 3) W1：评测人设 + 探针（可先 --offline 不调 API 归档）
python scripts/prepare_personas.py --offline
python scripts/prepare_probes.py

# 有 Key 后建议再跑一遍 LLM 归档（更干净的 5 字段）
python scripts/prepare_personas.py

# 4) 跑实验
python scripts/run_experiment.py --mode G0 --offline-annotate
python scripts/run_experiment.py --mode G1 --offline-annotate
python scripts/run_experiment.py --mode G2
python scripts/run_experiment.py --mode G2+

# 冒烟（先 2 个角色）
python scripts/run_experiment.py --mode G0 --limit 2 --offline-annotate

# 5) Judge 打分（qwen-plus）
python scripts/run_judge.py --mode G0
python scripts/run_judge.py --mode G2

# 6) 汇总表 + 多角色 demo（展望3）
python scripts/aggregate_results.py
python scripts/demo_multi_role.py G2
```

## 三档实验

| 组 | 配置 | 要证明什么 |
|---|---|---|
| G0 | 人设平铺 system，无档案无校验 | 漂移确实存在 |
| G1 | 结构化档案每轮强制读取 | 档案改善基础记忆 |
| G2 | 档案 + 后置校验 + 重试 | 校验模块是主贡献 |
| G2+ | G2 + 历史人设回溯 | 缓解长轮次记忆衰减 |

## 目录

```
data/processed/personas.jsonl   # 评测人设档案
data/processed/probes.json      # 200 条探针
src/                            # 生成器 / 校验器 / 记忆 / 评测
scripts/                        # 数据准备与实验入口
results/G0|G1|G2|G2plus|eval|demos/
docs/DESIGN.md                  # 方案与排期
```

## 指标

- 人设漂移率、一致性(1/3/5)、流畅度(1/3/5)
- 触发重试率、修正成功率
- 人工抽检一致率（报告阶段，抽 40 条）

## 六周排期（摘要）

| 周 | 交付 |
|---|---|
| W1 11/12–11/18 | personas / probes / G0 |
| W2 11/19–11/25 | verify / G1·G2 / 重试率 |
| W3 11/26–12/2 | 三指标 + 人工抽检 |
| W4 12/3–12/9 | L1+L2、G2+、多角色 demo |
| W5–W6 | 报告与答辩 PPT |
