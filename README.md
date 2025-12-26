# ReviewMirabel 文件结构概述

> 本文件对ReviewMirabel的项目结构进行概述


## 1. 项目目录结构
```text
RAG-MIA-Evaluation/
├── .venv/                      # Python 虚拟环境目录
├── .env                        # [核心配置] 环境变量文件 (API Keys, Base URLs, Model IDs)
├── requirements.txt            # 项目依赖列表
├── cache/                      # [缓存] 存放实验中间结果 (避免重复调用 API)
├── configs/                    # [配置] 实验配置文件
│   └── s2mia_splits.json       # 数据集划分配置 (定义 Member/Non-Member, Ref/Eval)
├── data/                       # [数据] 原始数据存放目录
│   └── nfcorpus/               # NFCorpus 数据集 (自动下载解压)
├── index/                      # [索引] 全局知识库索引 (包含所有文档，模拟真实 RAG 检索池)
│   ├── nf.index                # FAISS 向量索引文件 (Inner Product)
│   └── nf_docs.json            # 文档 ID 到内容的映射文件
├── index_member/               # [索引] 私有成员知识库索引 (仅包含 Member 文档，攻击目标)
│   ├── nf_member.index         # Member 数据的 FAISS 索引
│   └── nf_member_docs.json     # Member 文档映射
├── results/                    # [结果] 存放实验最终评估指标 (JSON) 和日志
└── src/                        # [源码] 核心代码目录
    ├── attacks/                # === [核心] 攻击算法实现 ===
    │   ├── ia.py               # [IA] 主入口：Interrogation Attack (提问攻击)
    │   │                       #      生成多轮问题 -> 获取 RAG 回答 -> 计算拒绝率/准确率
    │   ├── ia_pipeline.py      # [IA] RAG 流水线：集成所有防御逻辑的问答接口
    │   ├── ia_utils.py         # [IA] 工具库：问题生成、GroundTruth 生成、Cross-Encoder 筛选
    │   │
    │   ├── mba.py              # [MBA] 主入口：Mask-Based Attack (掩码攻击)
    │   │                       #      Mask 关键词 -> 询问 RAG 填空 -> 检查生成准确度
    │   ├── mba_pipeline.py     # [MBA] RAG 流水线
    │   ├── mba_utils.py        # [MBA] 工具库：TF-IDF 计算, Mask 生成策略
    │   ├── precompute_tfidf.py # [MBA] 预处理：计算语料库 TF-IDF
    │   │
    │   ├── s2mia.py            # [S2MIA] 主入口：Spectrum-to-Member Inference Attack
    │   │                       #         训练影子模型 -> 提取频谱特征 -> 分类器推断
    │   ├── s2mia_pipeline.py   # [S2MIA] RAG 流水线：支持特殊的续写式 (Continuation) Prompt
    │   ├── s2mia_utils.py      # [S2MIA] 工具库：数据加载, 影子模型训练
    │   └── s2mia_splits_and_index.py # [通用] 数据集划分与 FAISS 索引构建工具
    │
    ├── defense/                # === [核心] 防御机制实现 ===
    │   ├── mirabel/            # (目录) 参见下方 mirabel 模块
    │   ├── pad.py              # [Defense] PAD：Privacy-Adversarial Defense (Logits 扰动)
    │   ├── pad_config.py       # [Defense] PAD 的参数配置类 (epsilon, delta 等)
    │   ├── prompt_guard.py     # [Defense] Prompt Guard：基于规则/关键词拦截攻击性查询
    │   └── rewrite.py          # [Defense] Rewrite：基于 LLM 的查询改写防御
    │
    ├── mirabel/                # === [核心] Mirabel 防御特定模块 ===
    │   ├── apply_nfcorpus.py   # [Defense] Mirabel 核心算法：基于检索分数的自适应截断与隐藏
    │   ├── quick_eval_nfcorpus.py # 评估脚本
    │   └── threshold.py        # 数学工具：计算 Mirabel 的统计学阈值 (Gaussian/Gumbel 近似)
    │
    ├── rag/                    # === RAG 基础组件 ===
    │   ├── build_index_nfcorpus.py # 构建 FAISS 索引
    │   ├── download_nfcorpus.py    # 下载数据集
    │   ├── generate_llamaindex.py  # [核心] API 生成模块：封装 OpenAI/vLLM 格式的调用
    │   ├── pad_local_generate.py   # [核心] 本地生成模块：支持 PAD 防御的本地模型加载与推理
    │   └── search_nfcorpus.py      # 检索调试工具
    │
    ├── tests/                  # === 测试脚本 ===
    │   ├── analyze_ia_cache.py     # 工具：分析 IA 攻击的缓存文件 (Unknown 比例诊断)
    │   ├── download_model.py       # 工具：从 HF/ModelScope 下载模型
    │   ├── mirabel_smoke_end2end.py# 测试：Mirabel 端到端逻辑验证
    │   └── pipeline_smoke.py       # 测试：RAG Pipeline 连通性测试
    │
    └── utils/                  # === 通用工具 ===
        ├── download_model.py   # (ModelScope版) 模型下载工具
        └── vector_utils.py     # 向量处理 (L2 Normalization)
```

## 2. 环境变量配置

```py
# === 通用 LLM API 设置 (用于 RAG 回答生成) ===
LLM_API_BASE=http://localhost:8000/v1  # 例如 vLLM 或 OpenAI 地址
LLM_API_KEY=sk-xxxxxx                  # API Key
LLM_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
# === 攻击特定模型设置 (可选) ===
# 若未设置，默认回退使用 LLM_MODEL_ID
IA_QGEN_MODEL=gpt-4o                   # IA: 用于生成高质量探询问题
IA_SHADOW_MODEL=gpt-4o-mini            # IA/S2MIA: 用于生成 Ground Truth 或影子数据
# === 防御特定模型设置 ===
REWRITE_LLM_MODEL_ID=gpt-4o-mini       # Rewrite 防御: 用于改写查询的模型
PAD_MODEL_ID=/path/to/local/model      # PAD 防御: 本地加载的模型路径 (支持 huggingface id)
```

## 3. 运行时命令行参数 (CLI Arguments)

### 3.1 防御配置参数

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--defense` | Flag | `False` | **防御总开关**。加上此参数即开启防御。 |
| `--defense_mode` | String | `mirabel` | **防御模式选择** (仅当开启 `--defense` 时生效)：<br>1. `mirabel`: 检索截断与隐藏。<br>2. `prompt_guard`: 关键词拦截。<br>3. `rewrite`: 查询改写。<br>4. `pad`: 差分隐私 Logits 扰动。 |

#### 3.1.1 Mirabel 专用参数

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--rho` | `0.005` | 概率密度阈值控制参数。 |
| `--margin` | `0.02` | 决策边界 Margin。 |
| `--gap_min` | `0.03` | 最小间隙要求。 |
| `--use_gap` | `True` | 是否启用 Gap 机制增强检测鲁棒性。 |

#### 3.1.2 PAD 专用参数

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--pad_epsilon` | `2.0` | 差分隐私预算 Epsilon。 |
| `--pad_sigma_min`| `0.01` | 最小噪声标准差。 |
| `--pad_temperature`| `0.0` | 采样温度 (0.0 为 Greedy, >0 为 Sampling)。 |

### 3.2 数据集与实验规模参数

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--n_ref_m` | `50` (IA) / `60` | 参考集 (Reference) 成员样本数。 |
| `--n_ref_n` | `50` (IA) / `60` | 参考集 (Reference) 非成员样本数。 |
| `--n_eval_m` | `100` | 评估集 (Evaluation) 成员样本数 (攻击目标)。 |
| `--n_eval_n` | `100` | 评估集 (Evaluation) 非成员样本数。 |
| `--splits_path` | `configs/s2mia_splits.json` | 数据集划分配置文件路径。 |

### 3.3 RAG基础参数

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--top_k` | `3` | 检索 Top-K 文档数。 |
| `--per_ctx_clip` | `600` | 单个文档片段的最大字符数 (S2MIA 建议更短，如 300)。 |
| `--max_tokens` | `256` | 回答生成的最大 Token 数。 |
| `--index_dir_member`| `index_member` | 成员索引目录。 |

### 3.4 特定攻击独有参数

#### 3.4.1 IA

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--num_questions_generate` | `50` | 初始生成的探询问题总数。 |
| `--num_questions_select` | `30` | 筛选后保留的问题数。 |
| `--use_cross_encoder` | `True` | 是否使用 Cross-Encoder 筛选高质量问题 (推荐 True)。 |
| `--lambda_penalty` | `1.0` | RAG 拒答时的惩罚系数。 |

#### 3.4.2 MBA

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--num_masks` | `15` | 每个文档生成的 Mask 变体数量。 |
| `--idf_threshold_percentile`| `90` | 仅 Mask IDF 值排名前 10% 的关键词。 |

#### 3.4.3 S2MIA

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--cont_words` | `120` | 要求模型续写的词数。 |
| `--query_strategy` | `title_and_snippet` | 查询构造策略: `title_only`, `half_text`, `title_and_snippet`。 |
| `--shadow_train_size` | `400` | 训练影子分类器的数据量。 |

### 3.5 输出路径与缓存参数

| 参数名 | 说明 |
| :--- | :--- |
| `--cache_path` | 中间结果缓存文件路径（JSON格式）。设置此项可支持断点续跑。 |
| `--out_dir` | 最终结果输出目录（包含 metrics.json, 详情 logs, 统计图表）。 |
| `--save_every` | 每处理多少个样本保存一次缓存（防止程序崩溃数据丢失）。 |

