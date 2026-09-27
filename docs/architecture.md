# DomainForge 架构文档（作者：晨星）

## 1. 调用图（单向无环）

```
cli.py ──→ pipeline/
             │  run / benchmark / aggregate / save_rows
             ├─→ data/synthetic     4 组合成域偏移 DGP（难度旋钮调到 source_only≠天花板）
             ├─→ registry           方法名 → 构造器（唯一解析点）
             ├─→ alignment/         coral · tca · jda · kliep（纯 numpy）
             ├─→ baselines/         source_only · adapt_coral（可选，惰性探测）
             ├─→ fusion/flagship    SAFuse（HPO + 增益门控 + 非劣守护）
             ├─→ hpo/search         Optuna TPE（目标=极小目标验证集 acc）
             └─→ eval/metrics       acc/macro-F1/linear-MMD²
                   └─→ core/        types · errors · config · interfaces
```

## 2. 方法数学核心

| 方法 | 目标 | 求解 |
|------|------|------|
| CORAL | 协方差对齐 | Xs' = (Xs−μs)·Cs^{-1/2}·Ct^{1/2}+μt，特征分解取 SPD 平方根 |
| TCA | 边际 MMD 极小 | 广义特征问题 (K·M0·K)w = λ(K·H·K)w，取 k 个最小 λ；嵌入 = K·W |
| JDA | 边际 + 类条件 MMD | M = M0 + Σ_c M_c（伪标签迭代，块矩阵按域偏移放置） |
| KLIEP-lite | 密度比实例重加权 | 判别器 p(t\|x) → w=p/(1−p)，clip 后均值归一，SVC sample_weight |
| SAFuse | 集成 | 见 README 创新点 |

统一语义：所有方法 `predict` 返回目标域类标签；`transform` 返回对齐后表征
（CORAL 对目标侧恒等映射是定义使然——只对齐源域到目标分布）。

## 3. 数据集难度标定（SOP 关键坑）

合成 DGP 的难度旋钮先扫描后定死，保证 **source_only 明显低于天花板**（否则 HPO
各 trial 目标恒定、旗舰无增益空间）：

| 数据集 | 偏移类型 | 旋钮 | seed=42 下 source_only 行为 |
|--------|----------|------|------------------------------|
| covariate_shift | 逐特征均值/尺度漂移 | shift=1.6, sep=1.2 | 明显掉点（对齐可救） |
| rotated_moons | 旋转 + 噪声增强 | angle=55°, noise 0.22→0.32 | 掉点（非线性核可救） |
| scaled_blobs | 异质轴缩放 1..3× | sep=1.0 | 掉点 |
| label_shift_digits | 类先验倾斜 | tilt=2.5 | 温和掉点 |

验证/测试不相交（行级 round9 集合断言）；验证集含两类以上样本。

## 4. 可靠性设计

- **错误码**：E100 数据 / E200 方法 / E300 HPO / E400 基准 / E500 可选后端。
- **可选后端**：`adapt`(TF) 惰性一次性探测并缓存；不可用 → `skipped` 行，绝不伪造数字。
- **配置**：`ENV_DOMAINFORGE_SEED/HPO_TRIALS/HPO_TIMEOUT/N_TARGET_VAL/...` 覆盖，
  非法值回退默认。
- **确定性**：所有 RNG 用 `default_rng(seed)`；Optuna TPESampler(seed)；SVC random_state 固定。
  基准两次运行逐位一致（有单测守护）。

## 5. 质量基线（实测，seed=42, HPO_TRIALS=16, CPU）

见 `benchmark.json`。聚合口径 = 4 数据集 acc 均值；旗舰守护状态写入行 detail。
