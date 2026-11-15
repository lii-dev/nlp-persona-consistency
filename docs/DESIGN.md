# 最终方案设计（可执行版）

## 0. 定位

- **核心问题**：多轮对话人设漂移（性格矛盾、背景遗忘、风格跳变）
- **方法**：training-free 外挂（档案 → 校验 → 重试），不改权重
- **基座**：Qwen2.5-7B-Instruct（DashScope）
- **评测数据集**：Persona-Chat（只用 test；**无需训练集**）

> 本方案为 training-free 外挂模块，不涉及模型训练，故无需训练集。

## 1. 架构

结构化人设档案（JSON 5 字段）每轮强制进入 system → 生成器产出候选 → 校验器（T=0）输出 `{conflict, reason}` → 冲突则带冲突点重试（≤2），临时消息必须 pop。

## 2. 实验分解

- G0→G1：档案增量
- G1→G2：校验增量
- G2+：RAG 人设记忆（嵌入检索 + 动态 consolidate，展望2）
- L1+L2：分层校验；L2 含隐晦冲突 few-shot（展望1）
- 多角色协同：每人独立档案+校验+记忆，共享对话上下文（展望3）

## 2.1 三条展望落地

1. **语义校验增强**：VERIFY_PROMPT 增加隐含冲突类型 (e) 与 hard/soft few-shot。
2. **RAG 记忆**：`PersonaMemoryBank` 使用 `text-embedding-v3` 检索；抽取事实后动态更新，漂移事实可被 consolidate 丢弃。
3. **多角色**：`src/multi_role.py` 并行维持多人设并协同对话。

## 3. 探针设计（每人 10 轮）

自我介绍 / 细节回忆 / 反向诱导 / 风格诱导 / 虚假记忆 / 设定越界 / 长程回指（8–10）

## 4. 踩坑清单

1. 重试临时消息必须 `pop`
2. verify `temperature=0`
3. 每角色新建 `messages`

## 5. 降级线

W3 结束若 G2 未跑通：砍 G2+ 与多角色 demo，保 G0/G1/G2 + 100 条评测。
