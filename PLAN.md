# CVPR 2026 AdvML Challenge - Phase I 完整攻击方案

## 0. 赛题核心信息速查

| 项目 | 值 |
|------|-----|
| 目标模型 | DriveLM-Agent (LLaMA-Adapter v2 Multimodal 7B) |
| 模型结构 | CLIP ViT-L/14 (视觉) + LLaMA 7B (语言) + Adapter 层 |
| 数据量 | 200 场景, 每场景 6 张 224x224 图片 + 5~8 个 QA |
| 总 QA 数 | 1446 |
| 攻击面 | 图像: 添加像素扰动; 文本: 在原问题后追加 suffix |
| 评分 | 综合考虑攻击成功率、图像余弦相似度、suffix 字符长度 |
| Phase I 截止 | 2026-04-17 20:00 (UTC+8) |
| 提交限制 | 每天 2 次, 间隔 40 分钟 |
| 硬件 | DGX Spark (GB10 Grace Blackwell, 128GB 统一内存) |

## 1. 评分公式拆解

```
SceneScore_i = (1/N_i) * Σ_j score_j

其中 score_j 综合考虑:
  - 攻击效果: 模型输出与 ground-truth 的偏离程度
  - 图像扰动代价: S̄_i = (1/6) * Σ_{k=1}^{6} S_{i,k}
    S_{i,k} = cosine_similarity(repr(img_orig_k), repr(img_adv_k)), 归一化到 [0,1]
  - 文本扰动代价: x_{i,j} = suffix 的字符长度

Final Score = (1/M) * Σ_i SceneScore_i
```

关键推论:
- 图像扰动越小 (余弦相似度越高), 扣分越少
- Suffix 越短, 扣分越少
- 攻击效果越强 (偏离 ground truth 越远), 得分越高
- 需要在"攻击强度"和"扰动隐蔽性"之间找平衡点

## 2. 攻击策略总览

### 阶段一: 图像侧白盒 PGD 攻击 (优先级最高)

目标模型是白盒的, 梯度路径完全可微:

```
pixels → CLIP ViT patch embeddings → ViT Transformer → Adapter Projection
→ Adapter Gating → LLaMA Attention → LLaMA Output Logits → Loss
```

核心方法: **带余弦相似度约束的 PGD (Projected Gradient Descent)**

```python
# 伪代码
for step in range(num_steps):
    logits = model(x_adv, question)
    loss_attack = -log_prob(ground_truth_answer, logits)  # 最小化正确答案概率
    loss_sim = 1 - cosine_similarity(clip_feat(x_adv), clip_feat(x_orig))  # 惩罚偏离
    loss = loss_attack + lambda_sim * loss_sim

    grad = backward(loss, x_adv)
    x_adv = x_adv + alpha * sign(grad)
    x_adv = clip(x_adv, 0, 1)  # 像素范围约束
```

参数建议:
- 迭代步数: 100~200 步
- 步长 alpha: 1/255 ~ 2/255
- lambda_sim: 从 0.5 开始调, 通过 grid search 确定最优
- 精度: fp16 (节省显存, DGX Spark 的 Blackwell 原生支持)

### 阶段二: 文本侧 GCG 攻击 (Greedy Coordinate Gradient)

在问题末尾追加一个短 suffix (目标 5~15 个 token):

```python
# GCG 伪代码
suffix_tokens = random_init(k=10)  # 初始化 10 个 token
for step in range(500):
    # 对每个 suffix 位置, 计算 token embedding 梯度
    grads = compute_token_gradients(model, x_adv, question + suffix_tokens, gt_answer)
    # 选 top-B 候选替换
    candidates = top_k_by_gradient(grads, B=64)
    # 逐个评估, 取 loss 最小的
    suffix_tokens = evaluate_and_select_best(candidates)
```

为什么 GCG 对这个任务有效:
- DriveLM 的问题是结构化的 (固定模式), suffix 容易干扰模型的 attention
- 白盒场景下 5~10 个 token 往往就够让回答偏离

### 阶段三: 联合优化 (图像 + 文本交替)

```python
for outer_step in range(30):
    # Step A: 固定 suffix, PGD 优化图像 (20 步)
    for img_step in range(20):
        x_adv = pgd_step(x_adv, suffix, gt_answer, lambda_sim)

    # Step B: 固定图像, GCG 优化 suffix (50 步)
    for txt_step in range(50):
        suffix = gcg_step(x_adv, suffix, gt_answer)
```

交替优化的好处:
- 两个攻击面互补, 联合效果远强于单一面
- 可以用更小的图像扰动 + 更短的 suffix 达到同样的攻击效果

### 阶段四 (可选): 迁移性增强 (为 Phase II 做准备)

在 PGD 循环中加入数据增强:
- **MI-FGSM**: 梯度加动量 (decay=1.0)
- **DI-FGSM**: 每步随机 resize + padding
- **TI-FGSM**: 梯度与高斯核卷积

这些几乎不增加计算量, 但能显著提升黑盒迁移性。

## 3. 具体实现分工

### Step 1: 环境搭建 [DGX Spark]

```bash
# 1. 克隆代码
git clone https://github.com/ychenfen/CVPR-2026-AdvML-Challenge.git
cd CVPR-2026-AdvML-Challenge

# 2. 克隆 DriveLM 和 LLaMA-Adapter
git clone https://github.com/OpenDriveLab/DriveLM.git
cd DriveLM/challenge/llama_adapter_v2_multimodal7b

# 3. 创建 conda 环境 (DGX Spark 上有 conda)
conda create -n advml python=3.10 -y
conda activate advml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft pillow scipy tqdm

# 4. 下载模型权重
# LLaMA 7B 基础权重 (从 Meta 官方或 HuggingFace 获取)
# LLaMA-Adapter v2 checkpoint (从 GitHub Release 下载)
# 具体路径见 DriveLM challenge 的 README

# 5. 验证推理能通
python demo.py --llama_dir /path/to/llama_weights --checkpoint /path/to/adapter.pth \
    --data ../test_sample.json --output test_output.json
```

### Step 2: 实现基准推理脚本 `inference.py`

功能: 加载模型, 对 200 个场景跑推理, 输出预测结果
目的: 确认模型能正常工作, 同时作为攻击效果的基线对比

### Step 3: 实现图像攻击 `attack_image.py`

核心功能:
- PGD 攻击, 支持 L_inf 和余弦相似度两种约束
- 梯度检查点 (gradient checkpointing) 节省显存
- 支持 batch 处理多个场景
- 输出: 对抗图像 + 攻击日志

### Step 4: 实现文本攻击 `attack_text.py`

核心功能:
- GCG suffix 搜索
- 可配置 suffix 长度 (token 数 / 字符数)
- 每个问题独立优化 suffix
- 输出: 修改后的 QA.json (问题末尾追加 suffix)

### Step 5: 实现联合攻击 `attack_joint.py`

核心功能:
- 交替优化图像和文本
- 可配置权重平衡参数
- 自动评估攻击效果
- 输出: 完整的提交包 (对抗图像 + 修改后的 QA.json)

### Step 6: 实现评估和打包 `evaluate.py` + `package.py`

- 本地模拟评分 (需要知道评分用的 representation extractor)
- 打包成提交格式
- 统计各场景得分分布, 找出需要重点攻击的 hard cases

## 4. DGX Spark 资源规划

**128GB 统一内存的分配估算:**

| 组件 | 显存占用 |
|------|---------|
| CLIP ViT-L/14 (fp16) | ~0.6 GB |
| LLaMA 7B (fp16) | ~14 GB |
| Adapter 权重 | ~0.1 GB |
| 梯度 (fp16, 含 checkpointing) | ~14 GB |
| 中间激活 (gradient checkpointing) | ~8 GB |
| GCG 候选 batch (B=64) | ~5 GB |
| 输入数据 + 杂项 | ~2 GB |
| **总计** | **~44 GB** |

128GB 统一内存绰绰有余。甚至可以:
- 不使用 gradient checkpointing, 直接存全部激活 (~30GB 额外)
- 同时攻击多个场景 (batch_size=4~8)
- 开大 GCG 候选数 (B=256) 提高搜索质量

**计算时间估算:**

| 任务 | 单场景耗时 | 200 场景总耗时 |
|------|-----------|--------------|
| 纯 PGD 图像攻击 (200步) | ~3 min | ~10 hr |
| 纯 GCG 文本攻击 (500步/问题) | ~5 min | ~17 hr |
| 联合攻击 (30轮交替) | ~15 min | ~50 hr |

DGX Spark 的 Blackwell GPU 算力大约是 A100 的 2~3 倍, 上述时间可能偏保守。
实际上可以先跑一小批 (10 场景) 测算真实耗时, 再决定攻击策略的复杂度。

## 5. 文件结构规划

```
CVPR-2026-AdvML-Challenge/
├── PLAN.md                    # 本文件
├── phase1/                    # 原始数据 (200 场景)
│   ├── 0000/
│   │   ├── CAM_FRONT.jpg
│   │   ├── CAM_FRONT_LEFT.jpg
│   │   ├── CAM_FRONT_RIGHT.jpg
│   │   ├── CAM_BACK.jpg
│   │   ├── CAM_BACK_LEFT.jpg
│   │   ├── CAM_BACK_RIGHT.jpg
│   │   └── QA.json
│   └── ...
├── DriveLM/                   # DriveLM 代码 (git clone)
├── src/                       # 攻击代码
│   ├── config.py              # 所有超参数配置
│   ├── model_loader.py        # 模型加载, 包装成可微分的攻击接口
│   ├── inference.py           # 基准推理
│   ├── attack_image.py        # PGD 图像攻击
│   ├── attack_text.py         # GCG 文本攻击
│   ├── attack_joint.py        # 联合攻击 (主入口)
│   ├── evaluate.py            # 本地评估
│   ├── package.py             # 打包提交
│   └── utils.py               # 工具函数
├── outputs/                   # 攻击输出
│   ├── submission_v1/         # 每次提交一个子目录
│   └── logs/                  # 攻击日志
├── scripts/                   # 运行脚本
│   ├── setup_env.sh           # 环境搭建
│   ├── run_attack.sh          # 主攻击流程
│   └── run_eval.sh            # 评估
└── .gitignore
```

## 6. 调参策略

按优先级排列:

### 第一轮: 快速验证 (Day 1-2)
- 只跑图像 PGD 攻击, eps=8/255, 50 步
- 不加 suffix
- 提交看基线分数

### 第二轮: 加入文本攻击 (Day 3-4)
- 对每个问题跑 GCG, suffix 限制 10 token
- 单独提交看文本攻击的效果

### 第三轮: 联合优化 (Day 5-7)
- 交替优化, 调 lambda_sim 和 suffix 长度的 tradeoff
- 在 10 个场景上做 grid search:
  - lambda_sim: [0.1, 0.3, 0.5, 1.0, 2.0]
  - suffix_len: [5, 10, 15, 20]
  - pgd_steps: [50, 100, 200]
- 用最优参数跑全部 200 场景

### 第四轮: 精细化 (Day 8-14)
- 分析哪些场景得分低, 单独加大攻击力度
- 尝试 targeted attack (指定错误答案) vs untargeted
- 加入迁移性增强 (MI/DI/TI-FGSM)
- 尝试 attention-guided 扰动: 把扰动集中在模型关注的图像区域

## 7. 提交策略

每天 2 次提交机会, 需要精打细算:

- **Week 1 (4/1 ~ 4/7)**: 每天 1 次, 用于验证不同攻击策略的效果
- **Week 2 (4/8 ~ 4/14)**: 每天 1~2 次, 用于参数微调
- **最后 3 天 (4/15 ~ 4/17)**: 保留 4~6 次提交, 用于冲刺

每次提交前本地先做完整评估, 确认比上次有提升再交。

## 8. 风险与备选方案

| 风险 | 应对 |
|------|------|
| DriveLM-Agent 代码不兼容 DGX Spark (ARM) | 用 NGC PyTorch 容器, 或 x86 Docker |
| PGD 效果不理想 | 换 C&W attack (直接优化评分公式) |
| GCG 搜索太慢 | 用通用 suffix (对所有问题共享一个 suffix) |
| 余弦相似度扣分太重 | 减小 eps, 主要靠文本攻击 |
| 模型加载失败 (权重获取问题) | 从 HuggingFace 找等价模型, 或联系组织者 |
| 评分函数理解有误 | 前 2 次提交专门用于验证评分机制 |

## 9. Claude Code 执行指南

这个项目完全由 Claude Code 在 DGX Spark 上执行。关键注意事项:

1. **SSH 连接**: 需要配置 DGX Spark 的 SSH 访问 (用户提供 IP 和密钥)
2. **长任务**: 所有训练/攻击脚本通过 tmux/nohup 运行, 避免 SSH 断连丢失进度
3. **检查点**: 每 10 个场景保存一次中间结果
4. **Git 同步**: 代码改动及时 push 到 GitHub, 方便追踪和回溯
5. **日志**: 详细记录每次攻击的参数和结果, 方便复盘
6. **提交**: 打包后通过比赛平台 Web UI 提交 (可能需要用户手动操作)

### 会话拆分建议

| 会话 | 内容 | 预计时长 |
|------|------|---------|
| Session 1 | 环境搭建 + 模型验证推理 | 2~3 hr |
| Session 2 | 实现 PGD 图像攻击 + 首次提交 | 3~4 hr |
| Session 3 | 实现 GCG 文本攻击 + 提交 | 3~4 hr |
| Session 4 | 实现联合攻击 + 参数搜索 | 4~6 hr |
| Session 5 | 精细化调优 + 冲刺提交 | 持续到截止 |
